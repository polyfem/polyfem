#include "NLProblem.hpp"

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/OBJ_IO.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <igl/write_triangle_mesh.h>

static bool disable_collision = false;

/*
m \frac{\partial^2 u}{\partial t^2} = \psi = \text{div}(\sigma[u])\newline
u^{t+1} = u(t+\Delta t)\approx u(t) + \Delta t \dot u + \frac{\Delta t^2} 2 \ddot u \newline
= u(t) + \Delta t \dot u + \frac{\Delta t^2}{2} \psi\newline
M u^{t+1}_h \approx M u^t_h + \Delta t M v^t_h + \frac{\Delta t^2} {2} A u^{t+1}_h \newline
%
M (u^{t+1}_h - (u^t_h + \Delta t v^t_h)) - \frac{\Delta t^2} {2} A u^{t+1}_h
*/
// mü = ψ = div(σ[u])
// uᵗ⁺¹ = u(t + Δt) ≈ u(t) + Δtu̇ + ½Δt²ü = u(t) + Δtu̇ + ½Δt²ψ
// Muₕᵗ⁺¹ ≈ Muₕᵗ + ΔtMvₕᵗ ½Δt²Auₕᵗ⁺¹
// Root-finding form:
// M(uₕᵗ⁺¹ - (uₕᵗ + Δtvₕᵗ)) - ½Δt²Auₕᵗ⁺¹ = 0

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
#ifdef IPC_TOOLKIT_WITH_CUDA
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}});
#else
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"}});
#endif
} // namespace ipc

namespace polyfem
{
	using namespace assembler;
	using namespace utils;

	namespace solver
	{
		using namespace polysolve;

		NLProblem::NLProblem(const State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool no_reduced)
			: state(state), assembler(state.assembler), rhs_assembler(rhs_assembler),
			  full_size((assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
			  reduced_size(full_size - (no_reduced ? 0 : state.boundary_nodes.size())),
			  t(t), rhs_computed(false), is_time_dependent(state.problem->is_time_dependent()), ignore_inertia(state.args["solver"]["ignore_inertia"]), project_to_psd(false)
		{
			assert(!assembler.is_mixed(state.formulation()));

			_dhat = dhat;
			assert(_dhat > 0);
			_epsv = state.args["contact"]["epsv"];
			assert(_epsv > 0);
			_mu = state.args["contact"]["friction_coefficient"];
			_lagged_damping_weight = is_time_dependent ? 0 : state.args["solver"]["contact"]["lagged_damping_weight"].get<double>();
			use_adaptive_barrier_stiffness = !state.args["solver"]["contact"]["barrier_stiffness"].is_number();
			if (use_adaptive_barrier_stiffness)
			{
				_barrier_stiffness = 1;
				logger().debug("Using adaptive barrier stiffness");
			}
			else
			{
				assert(state.args["solver"]["contact"]["barrier_stiffness"].is_number());
				_barrier_stiffness = state.args["solver"]["contact"]["barrier_stiffness"];
				logger().debug("Using fixed barrier stiffness of {}", _barrier_stiffness);
			}
			_prev_distance = -1;
			if (utils::is_param_valid(state.args, "time"))
			{
				_time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
				_time_integrator->set_parameters(state.args["time"]);
			}

			_broad_phase_method = state.args["solver"]["contact"]["CCD"]["broad_phase"];
			_ccd_tolerance = state.args["solver"]["contact"]["CCD"]["tolerance"];
			_ccd_max_iterations = state.args["solver"]["contact"]["CCD"]["max_iterations"];

			forms_.push_back(std::make_shared<ElasticForm>(state));
			forms_.push_back(std::make_shared<BodyForm>(state, rhs_assembler));

			if (state.args["contact"]["enabled"])
				forms_.push_back(std::make_shared<ContactForm>(state, _dhat,
															   use_adaptive_barrier_stiffness, _barrier_stiffness, is_time_dependent,
															   _broad_phase_method, _ccd_tolerance, _ccd_max_iterations));
			if (_mu != 0)
				forms_.push_back(std::make_shared<FrictionForm>(state, _epsv, _mu,
																_dhat, _barrier_stiffness, _broad_phase_method,
																dt(), state.collision_mesh));
			if (is_time_dependent)
				forms_.push_back(std::make_shared<InertiaForm>(state.mass, *_time_integrator));
		}

		void NLProblem::init(const TVector &full)
		{
			for (auto &f : forms_)
				f->init(full);

			// TODO: DELETE ME BELOW
			if (disable_collision || !state.is_contact_enabled())
				return;

			assert(full.size() == full_size);
			if (use_adaptive_barrier_stiffness)
			{
				update_barrier_stiffness(full);
			}
			// exit(0);
		}

		void NLProblem::update_barrier_stiffness(const TVector &full)
		{
			// TODO: DELETE ME BELOW

			assert(full.size() == full_size);
			_barrier_stiffness = 1;
			if (disable_collision || !state.is_contact_enabled())
				return;

			Eigen::MatrixXd grad_energy;
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad_energy);

			if (!ignore_inertia && is_time_dependent)
			{
				// grad_energy *= time_integrator()->acceleration_scaling();
				grad_energy += state.mass * full / time_integrator()->acceleration_scaling();
			}

			grad_energy -= current_rhs();

			// TODO: HACK:
			//  grad_energy.setZero();

			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);
			update_constraint_set(displaced_surface);
			Eigen::VectorXd grad_barrier = ipc::compute_barrier_potential_gradient(
				state.collision_mesh, displaced_surface, _constraint_set, _dhat);
			grad_barrier = state.collision_mesh.to_full_dof(grad_barrier);

			_barrier_stiffness = ipc::initial_barrier_stiffness(
				ipc::world_bbox_diagonal_length(displaced), _dhat, state.avg_mass,
				grad_energy, grad_barrier, max_barrier_stiffness_);
			logger().debug("adaptive barrier stiffness {}", _barrier_stiffness);
		}

		void NLProblem::init_time_integrator(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt)
		{
			// TODO: MOVE ME OUT
			assert(dt > 0);
			if (_time_integrator)
				_time_integrator->init(x_prev, v_prev, a_prev, dt);
		}

		void NLProblem::set_project_to_psd(bool val)
		{
			for (auto &f : forms_)
				f->set_project_to_psd(val);
		}

		void NLProblem::init_lagging(const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);
			for (auto &f : forms_)
				f->init_lagging(full);

			// TODO: DELETE ME BELOW
			reduced_to_full_displaced_points(x, _displaced_prev);
			update_lagging(x);
		}

		void NLProblem::update_lagging(const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);
			for (auto &f : forms_)
				f->init_lagging(full);

			// TODO: DELETE ME BELOW
			Eigen::MatrixXd displaced;
			reduced_to_full_displaced_points(x, displaced);

			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

			if (_mu != 0)
			{
				update_constraint_set(displaced_surface);
				ipc::construct_friction_constraint_set(
					state.collision_mesh, displaced_surface, _constraint_set,
					_dhat, _barrier_stiffness, _mu, _friction_constraint_set);
			}

			// Save the variables for use in lagged damping
			reduced_to_full(x, x_lagged);
		}

		double NLProblem::compute_lagging_error(const TVector &x)
		{
			// TODO: REFACTOR ME
			//  Check || ∇B(xᵗ⁺¹) - h² Σ F(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≦ ϵ_d
			//      ≡ || ∇B(xᵗ⁺¹) + ∇D(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≤ ϵ_d
			TVector grad;
			gradient(x, grad);
			return grad.norm();
		}

		bool NLProblem::lagging_converged(const TVector &x)
		{
			// TODO: REFACTOR ME
			double tol = state.args["solver"]["contact"].value("friction_convergence_tol", 1e-2);
			double grad_norm = compute_lagging_error(x);
			logger().debug("Lagging convergece grad_norm={:g} tol={:g}", grad_norm, tol);
			return grad_norm <= tol;
		}

		void NLProblem::update_quantities(const double t, const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);

			for (auto &f : forms_)
				f->update_quantities(t, full);

			// TODO: DELETE ME BELOW

			if (is_time_dependent)
			{
				if (!ignore_inertia)
					_time_integrator->update_quantities(x);

				rhs_computed = false;
				this->t = t;

				if (use_adaptive_barrier_stiffness)
				{
					update_barrier_stiffness(x);
				}
			}
		}

		void NLProblem::substepping(const double t)
		{
			// TODO: DELETE ME BELOW
			if (is_time_dependent)
			{
				rhs_computed = false;
				this->t = t;
			}
		}

		const Eigen::MatrixXd &NLProblem::current_rhs()
		{
			// TODO: DELETE ME BELOW
			if (!rhs_computed)
			{
				rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.density, state.n_boundary_samples(), state.local_neumann_boundary, state.rhs, t, _current_rhs);
				rhs_computed = true;

				if (assembler.is_mixed(state.formulation()))
				{
					const int prev_size = _current_rhs.size();
					if (prev_size < full_size)
					{
						_current_rhs.conservativeResize(prev_size + state.n_pressure_bases, _current_rhs.cols());
						_current_rhs.block(prev_size, 0, state.n_pressure_bases, _current_rhs.cols()).setZero();
					}
				}
				assert(_current_rhs.size() == full_size);
				rhs_assembler.set_bc(std::vector<mesh::LocalBoundary>(), std::vector<int>(), state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);

				if (!ignore_inertia && is_time_dependent)
				{
					// _current_rhs *= time_integrator()->acceleration_scaling();
					_current_rhs += state.mass * time_integrator()->x_tilde() / time_integrator()->acceleration_scaling();
				}

				if (reduced_size != full_size)
				{
					// rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);
					rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), std::vector<mesh::LocalBoundary>(), _current_rhs, t);
				}
			}

			return _current_rhs;
		}

		void NLProblem::compute_displaced_points(const TVector &full, Eigen::MatrixXd &displaced)
		{
			// TODO: DELETE ME BELOW
			assert(full.size() == full_size);
			displaced = state.boundary_nodes_pos + unflatten(full, state.mesh->dimension());
		}

		void NLProblem::reduced_to_full_displaced_points(const TVector &reduced, Eigen::MatrixXd &displaced)
		{
			// TODO: DELETE ME BELOW
			TVector full;
			reduced_to_full(reduced, full);
			compute_displaced_points(full, displaced);
		}

		void NLProblem::update_constraint_set(const Eigen::MatrixXd &displaced_surface)
		{
			// TODO: DELETE ME BELOW
			//  Store the previous value used to compute the constraint set to avoid
			//  duplicate computation.
			static Eigen::MatrixXd cached_displaced_surface;
			if (cached_displaced_surface.size() == displaced_surface.size()
				&& cached_displaced_surface == displaced_surface)
				return;

			if (_use_cached_candidates)
				ipc::construct_constraint_set(
					_candidates, state.collision_mesh, displaced_surface, _dhat,
					_constraint_set);
			else
				ipc::construct_constraint_set(
					state.collision_mesh, displaced_surface, _dhat,
					_constraint_set, /*dmin=*/0, _broad_phase_method);
			cached_displaced_surface = displaced_surface;
		}

		void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
		{
			Eigen::MatrixXd full0, full1;
			reduced_to_full(x0, full0);
			reduced_to_full(x1, full1);

			for (auto &f : forms_)
				f->line_search_begin(full0, full1);

			// TODO: DELETE ME BELOW

			if (disable_collision || !state.is_contact_enabled())
				return;

			Eigen::MatrixXd displaced0, displaced1;
			reduced_to_full_displaced_points(x0, displaced0);
			reduced_to_full_displaced_points(x1, displaced1);

			ipc::construct_collision_candidates(
				state.collision_mesh,
				state.collision_mesh.vertices(displaced0),
				state.collision_mesh.vertices(displaced1),
				_candidates,
				/*inflation_radius=*/_dhat / 1.99, // divide by 1.99 instead of 2 to be conservative
				_broad_phase_method);

			_use_cached_candidates = true;
		}

		void NLProblem::line_search_end()
		{
			for (auto &f : forms_)
				f->line_search_end();

			// TODO: DELETE ME BELOW
			_candidates.clear();
			_use_cached_candidates = false;
		}

		double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
		{
			Eigen::MatrixXd full0, full1;
			reduced_to_full(x0, full0);
			reduced_to_full(x1, full1);

			double step = 1;
			for (auto &f : forms_)
				step = std::min(step, f->max_step_size(full0, full1));

			// TODO: DELETE ME BELOW and return step
			if (disable_collision || !state.is_contact_enabled())
			{
				assert(step == 1);
				return 1;
			}

			Eigen::MatrixXd displaced0, displaced1;
			reduced_to_full_displaced_points(x0, displaced0);
			reduced_to_full_displaced_points(x1, displaced1);

			// Extract surface only
			Eigen::MatrixXd V0 = state.collision_mesh.vertices(displaced0);
			Eigen::MatrixXd V1 = state.collision_mesh.vertices(displaced1);

			// OBJWriter::save("s0.obj", V0, state.collision_mesh.edges(), state.collision_mesh.faces());
			// OBJWriter::save("s1.obj", V1, state.collision_mesh.edges(), state.collision_mesh.faces());

			double max_step;
			if (_use_cached_candidates
#ifdef IPC_TOOLKIT_WITH_CUDA
				&& _broad_phase_method != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU
#endif
			)
				max_step = ipc::compute_collision_free_stepsize(
					_candidates, state.collision_mesh, V0, V1,
					_ccd_tolerance, _ccd_max_iterations);
			else
				max_step = ipc::compute_collision_free_stepsize(
					state.collision_mesh, V0, V1,
					_broad_phase_method, _ccd_tolerance, _ccd_max_iterations);
				// logger().trace("best step {}", max_step);

#ifndef NDEBUG
			// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
			Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;
			while (ipc::has_intersections(state.collision_mesh, V_toi))
			{
				double Linf = (V_toi - V0).lpNorm<Eigen::Infinity>();
				logger().error("taking max_step results in intersections (max_step={:g})", max_step);
				max_step /= 2.0;
				if (max_step <= 0 || Linf == 0)
				{
					const std::string msg = fmt::format("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf);
					logger().error(msg);
					throw std::runtime_error(msg);
				}
				V_toi = (V1 - V0) * max_step + V0;
			}
#endif

			assert(fabs(step - max_step) < 1e-10);
			return max_step;
		}

		bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
		{
			// TODO: DELETE ME BELOW
			if (disable_collision || !state.is_contact_enabled())
				return true;

			// if (!state.problem->is_time_dependent())
			// return false;

			Eigen::MatrixXd displaced0, displaced1;
			reduced_to_full_displaced_points(x0, displaced0);
			reduced_to_full_displaced_points(x1, displaced1);

			// Skip CCD if the displacement is zero.
			if ((displaced1 - displaced0).lpNorm<Eigen::Infinity>() == 0.0)
			{
				// Assumes initially intersection-free
				assert(is_intersection_free(x0));
				return true;
			}

			// OBJWriter::save("0.obj", state.collision_mesh.vertices(displaced0), state.collision_mesh.edges(), state.collision_mesh.faces());
			// OBJWriter::save("1.obj", state.collision_mesh.vertices(displaced1), state.collision_mesh.edges(), state.collision_mesh.faces());

			bool is_valid;
			if (_use_cached_candidates)
				is_valid = ipc::is_step_collision_free(
					_candidates, state.collision_mesh,
					state.collision_mesh.vertices(displaced0),
					state.collision_mesh.vertices(displaced1),
					_ccd_tolerance, _ccd_max_iterations);
			else
				is_valid = ipc::is_step_collision_free(
					state.collision_mesh,
					state.collision_mesh.vertices(displaced0),
					state.collision_mesh.vertices(displaced1),
					ipc::BroadPhaseMethod::HASH_GRID, _ccd_tolerance, _ccd_max_iterations);

			return is_valid;
		}

		bool NLProblem::is_intersection_free(const TVector &x)
		{
			// TODO: DELETE ME BELOW
			if (disable_collision || !state.is_contact_enabled())
				return true;

			Eigen::MatrixXd displaced;
			reduced_to_full_displaced_points(x, displaced);

			return !ipc::has_intersections(state.collision_mesh, state.collision_mesh.vertices(displaced));
		}

		bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
		{
			TVector full0, full1;
			reduced_to_full(x0, full0);
			reduced_to_full(x1, full1);
			for (auto &f : forms_)
				if (!f->is_step_valid(full0, full1))
					return false;

			return true;

			// TODO: DELETE ME BELOW
			TVector grad = TVector::Zero(reduced_size);
			gradient(x1, grad, true);

			if (std::isnan(grad.norm()))
				return false;

			// Check the scalar field in the output does not contain NANs.
			// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
			//          This causes small step lengths in the LS.
			// TVector x1_full;
			// reduced_to_full(x1, x1_full);
			// return state.check_scalar_value(x1_full, true, false);
			return true;
		}

		double NLProblem::value(const TVector &x)
		{
			// TODO: DELETE ME BELOW
			return value(x, false);
		}

		double NLProblem::value(const TVector &x, const bool only_elastic)
		{
			TVector full;
			reduced_to_full(x, full);

			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

			const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, state.ass_vals_cache, full);
			const double body_energy = rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.density, state.n_boundary_samples(), t);

			double intertia_energy = 0;
			if (!ignore_inertia && is_time_dependent)
			{
				const double scaling = time_integrator()->acceleration_scaling();
				const TVector tmp = full - time_integrator()->x_tilde();
				intertia_energy = (0.5 / scaling) * tmp.transpose() * state.mass * tmp;
			}

			// ½(x−(xᵗ+hvᵗ+h²M⁻¹fₑ))ᵀM(x−(xᵗ+hvᵗ+h²M⁻¹fₑ))
			// = ½ (x−xᵗ−hvᵗ−h²M⁻¹fₑ)ᵀM(x−xᵗ−hvᵗ−h²M⁻¹fₑ)
			// = ½ (t−h²M⁻¹fₑ)ᵀM(t−h²M⁻¹fₑ)
			// = ½ (tᵀM - h²fₑᵀ)(t-h²M⁻¹fₑ)
			// = ½ (tᵀMt - h²tᵀfₑ - h²fₑᵀt + h⁴fₑᵀM⁻¹fₑ)
			// = ½ (t²Mt - 2h²tᵀfₑ + h⁴fₑᵀM⁻¹fₑ)

			double collision_energy = 0;
			double friction_energy = 0;
			if (!only_elastic && !disable_collision && state.is_contact_enabled())
			{
				Eigen::MatrixXd displaced;
				compute_displaced_points(full, displaced);
				Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

				collision_energy = ipc::compute_barrier_potential(
					state.collision_mesh, displaced_surface, _constraint_set, _dhat);
				friction_energy = ipc::compute_friction_potential(
					state.collision_mesh, state.collision_mesh.vertices(displaced_prev()),
					displaced_surface, _friction_constraint_set, _epsv * dt());

				logger().trace("collision_energy {}, friction_energy {}", collision_energy, friction_energy);
			}

			double lagged_damping = _lagged_damping_weight * (full - x_lagged).squaredNorm();

			const double non_contact_terms = elastic_energy + body_energy + intertia_energy + friction_energy + lagged_damping;

			double fvalue = 0;
			{
				const double asd = elastic_energy + body_energy + friction_energy + lagged_damping + intertia_energy + _barrier_stiffness * collision_energy;
				// TODO: KEEP ONLY fvalue

				for (int i = 0; i < forms_.size(); ++i)
				{
					const auto &f = forms_[i];
					if (only_elastic && i == 2)
						continue;
					fvalue += f->value(full);
				}

				if (fabs(asd - fvalue) > 1e-9)
				{
					double xxx = 0;
					for (int i = 0; i < forms_.size(); ++i)
					{
						const auto &f = forms_[i];
						std::cout << i << "->" << f->value(full) << std::endl;
						if (i == 2)
							continue;
						xxx += f->value(full);
					}
					const double asdasd = elastic_energy + body_energy + friction_energy + lagged_damping + intertia_energy;
					std::cout << xxx - asdasd << std::endl;
				}

				assert(fabs(asd - fvalue) < 1e-9);
			}

			const double result = non_contact_terms + _barrier_stiffness * collision_energy;
			assert(fabs(result - fvalue) < 1e-9);
			return result;
		}

		void NLProblem::compute_cached_stiffness()
		{
			if (cached_stiffness.size() == 0)
			{
				const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
				if (assembler.is_linear(state.formulation()))
				{
					assembler.assemble_problem(state.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, cached_stiffness);
				}
			}
		}

		void NLProblem::gradient(const TVector &x, TVector &gradv)
		{
			gradient(x, gradv, false);
		}

		void NLProblem::gradient(const TVector &x, TVector &gradv, const bool only_elastic)
		{
			Eigen::MatrixXd grad;
			gradient_no_rhs(x, grad, only_elastic);

			grad -= current_rhs();
			// TODO delete above

			TVector full, tmp;
			reduced_to_full(x, full);
			TVector fgrad(full.size());
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				if (only_elastic && i == 2)
					continue;
				f->first_derivative(full, tmp);
				fgrad += tmp;
			}
			const double asdasd = (grad - fgrad).norm();
			assert(asdasd < 1e-9);

			full_to_reduced(grad, gradv);
		}

		void NLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad, const bool only_elastic)
		{
			TVector full;
			reduced_to_full(x, full);

			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad);

			ElasticForm asd(state);
			Eigen::VectorXd xxx;
			asd.first_derivative(full, xxx);
			const double asdasd = (xxx - grad).norm();
			assert(asdasd < 1e-10);

			if (!ignore_inertia && is_time_dependent)
			{
				// grad *= time_integrator()->acceleration_scaling();
				grad += state.mass * full / time_integrator()->acceleration_scaling();
			}

			Eigen::VectorXd grad_barrier;
			if (!only_elastic && !disable_collision && state.is_contact_enabled())
			{
				Eigen::MatrixXd displaced;
				compute_displaced_points(full, displaced);

				Eigen::MatrixXd displaced_surface_prev = state.collision_mesh.vertices(displaced_prev());
				Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

				grad_barrier = ipc::compute_barrier_potential_gradient(
					state.collision_mesh, displaced_surface, _constraint_set, _dhat);
				grad_barrier = state.collision_mesh.to_full_dof(grad_barrier);

				Eigen::VectorXd grad_friction = ipc::compute_friction_potential_gradient(
					state.collision_mesh, displaced_surface_prev, displaced_surface,
					_friction_constraint_set, _epsv * dt());
				grad += state.collision_mesh.to_full_dof(grad_friction);
			}
			else
			{
				grad_barrier.setZero(full_size);
			}

			grad += _lagged_damping_weight * (full - x_lagged);
			grad += _barrier_stiffness * grad_barrier;

			assert(grad.size() == full_size);
		}

		void NLProblem::hessian(const TVector &x, THessian &hessian)
		{
			THessian full_hessian;
			hessian_full(x, full_hessian);
			full_hessian_to_reduced_hessian(full_hessian, hessian);
		}

		void NLProblem::hessian_full(const TVector &x, THessian &hessian)
		{
			// scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

			TVector full;
			reduced_to_full(x, full);

			THessian energy_hessian(full_size, full_size);
			{
				POLYFEM_SCOPED_TIMER("\telastic hessian time");

				const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
				if (assembler.is_linear(rhs_assembler.formulation()))
				{
					compute_cached_stiffness();
					energy_hessian = cached_stiffness;
				}
				else
				{
					assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, project_to_psd, state.bases, gbases, state.ass_vals_cache, full, mat_cache, energy_hessian);
				}

				if (!ignore_inertia && is_time_dependent)
				{
					// energy_hessian *= time_integrator()->acceleration_scaling();
				}
			}

			THessian inertia_hessian(full_size, full_size);
			if (!ignore_inertia && is_time_dependent)
			{
				POLYFEM_SCOPED_TIMER("\tinertia hessian time");
				inertia_hessian = state.mass / time_integrator()->acceleration_scaling();
			}

			THessian barrier_hessian(full_size, full_size), friction_hessian(full_size, full_size);
			if (!disable_collision && state.is_contact_enabled())
			{
				POLYFEM_SCOPED_TIMER("\tipc hessian(s) time");

				Eigen::MatrixXd displaced;
				{
					POLYFEM_SCOPED_TIMER("\t\tdisplace pts time");
					compute_displaced_points(full, displaced);
				}

				// {
				// 	POLYFEM_SCOPED_TIMER("\t\tconstraint set time");
				// }

				Eigen::MatrixXd displaced_surface_prev = state.collision_mesh.vertices(displaced_prev());
				Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

				{
					POLYFEM_SCOPED_TIMER("\t\tbarrier hessian time");
					barrier_hessian = ipc::compute_barrier_potential_hessian(
						state.collision_mesh, displaced_surface, _constraint_set,
						_dhat, project_to_psd);
					barrier_hessian = state.collision_mesh.to_full_dof(barrier_hessian);
				}

				{
					POLYFEM_SCOPED_TIMER("\t\tfriction hessian time");
					friction_hessian = ipc::compute_friction_potential_hessian(
						state.collision_mesh, displaced_surface_prev, displaced_surface,
						_friction_constraint_set, _epsv * dt(), project_to_psd);
					friction_hessian = state.collision_mesh.to_full_dof(friction_hessian);
				}
			}

			THessian lagged_damping_hessian = _lagged_damping_weight * sparse_identity(full.size(), full.size());

			// Summing the hessian matrices like this might be less efficient than multiple `hessian += ...`, but
			// it is much easier to read and export the individual matrices for inspection.
			THessian non_contact_hessian = energy_hessian + inertia_hessian + friction_hessian + lagged_damping_hessian;
			hessian = non_contact_hessian + _barrier_stiffness * barrier_hessian;

			assert(hessian.rows() == full_size);
			assert(hessian.cols() == full_size);
		}

		void NLProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
		{
			POLYFEM_SCOPED_TIMER("\tfull hessian to reduced hessian");

			if (reduced_size == full_size || reduced_size == full.rows())
			{
				assert(reduced_size == full.rows() && reduced_size == full.cols());
				reduced = full;
				return;
			}

			Eigen::VectorXi indices(full_size);
			int index = 0;
			size_t kk = 0;
			for (int i = 0; i < full_size; ++i)
			{
				if (kk < state.boundary_nodes.size() && state.boundary_nodes[kk] == i)
				{
					++kk;
					indices(i) = -1;
				}
				else
				{
					indices(i) = index++;
				}
			}
			assert(index == reduced_size);

			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(full.nonZeros()); // Conservative estimate
			for (int k = 0; k < full.outerSize(); ++k)
			{
				if (indices(k) < 0)
					continue;

				for (THessian::InnerIterator it(full, k); it; ++it)
				{
					assert(it.col() == k);
					if (indices(it.row()) < 0 || indices(it.col()) < 0)
						continue;

					assert(indices(it.row()) >= 0);
					assert(indices(it.col()) >= 0);

					entries.emplace_back(indices(it.row()), indices(it.col()), it.value());
				}
			}

			reduced.resize(reduced_size, reduced_size);
			reduced.setFromTriplets(entries.begin(), entries.end());
			reduced.makeCompressed();
		}

		void NLProblem::solution_changed(const TVector &newX)
		{
			Eigen::MatrixXd newFull;
			reduced_to_full(newX, newFull);

			for (auto &f : forms_)
				f->solution_changed(newFull);

			if (disable_collision || !state.is_contact_enabled())
				return;

			Eigen::MatrixXd displaced;
			reduced_to_full_displaced_points(newX, displaced);

			update_constraint_set(state.collision_mesh.vertices(displaced));
		}

		double NLProblem::heuristic_max_step(const TVector &dx)
		{
			// if (disable_collision || !state.is_contact_enabled())
			// 	return 1;

			// //pSize = average(searchDir)
			// const double pSize = dx.lpNorm<1>() / dx.size();
			// const double voxelSize = state.average_edge_length / 3.0;

			// const double spanSize = pSize / voxelSize;
			// std::cout << "pSize " << pSize << " spanSize " << spanSize << " voxelSize " << voxelSize << " avg " << state.average_edge_length << std::endl;
			// if (spanSize > 1)
			// {
			// 	return 1 / spanSize;
			// }

			return 1;
		}

		void NLProblem::post_step(const int iter_num, const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);

			for (auto &f : forms_)
				f->post_step(iter_num, full);

			if (disable_collision || !state.is_contact_enabled())
				return;

			reduced_to_full(x, full);

			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

			if (state.args["output"]["advanced"]["save_nl_solve_sequence"])
			{
				OBJWriter::save(state.resolve_output_path(fmt::format("step{:03d}.obj", iter_num)),
								displaced_surface, state.collision_mesh.edges(), state.collision_mesh.faces());
			}

			const double dist_sqr = ipc::compute_minimum_distance(state.collision_mesh, displaced_surface, _constraint_set);
			logger().trace("min_dist {}", sqrt(dist_sqr));

			if (use_adaptive_barrier_stiffness)
			{
				if (is_time_dependent)
				{
					double prev_barrier_stiffness = _barrier_stiffness;
					ipc::update_barrier_stiffness(
						_prev_distance, dist_sqr, max_barrier_stiffness_,
						_barrier_stiffness, ipc::world_bbox_diagonal_length(displaced_surface));
					if (prev_barrier_stiffness != _barrier_stiffness)
					{
						logger().debug(
							"updated barrier stiffness from {:g} to {:g}",
							prev_barrier_stiffness, _barrier_stiffness);
					}
				}
				else
				{
					update_barrier_stiffness(full);
				}
			}
			_prev_distance = dist_sqr;
		}

		void NLProblem::save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const
		{
			time_integrator()->save_raw(x_path, v_path, a_path);
		}
	} // namespace solver
} // namespace polyfem
