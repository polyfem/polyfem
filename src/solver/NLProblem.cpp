#include <polyfem/NLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>
#include <polyfem/Timer.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <igl/write_triangle_mesh.h>

static bool disable_collision = false;

/*
m \frac{\partial^2 u}{\partial t^2} = \psi = \text{div}(\sigma[u])\\
u^{t+1} = u(t+\Delta t)\approx u(t) + \Delta t \dot u + \frac{\Delta t^2} 2 \ddot u \\
= u(t) + \Delta t \dot u + \frac{\Delta t^2}{2} \psi\\
M u^{t+1}_h \approx M u^t_h + \Delta t M v^t_h + \frac{\Delta t^2} {2} A u^{t+1}_h \\
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
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{
			{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
			{ipc::BroadPhaseMethod::HASH_GRID, "HG"},
			{ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
			{ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
			{ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
			{ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		});
}

namespace polyfem
{
	namespace
	{
		bool writeOBJ(const std::string &path, const Eigen::MatrixXd &v, const Eigen::MatrixXi &e, const Eigen::MatrixXi &f)
		{
			std::ofstream obj(path, std::ios::out);
			if (!obj.is_open())
				return false;

			obj.precision(15);

			for (int i = 0; i < v.rows(); ++i)
				obj << "v " << v(i, 0) << " " << v(i, 1) << " " << (v.cols() > 2 ? v(i, 2) : 0) << "\n";

			for (int i = 0; i < e.rows(); ++i)
				obj << "l " << e(i, 0) + 1 << " " << e(i, 1) + 1 << "\n";

			for (int i = 0; i < f.rows(); ++i)
				obj << "f " << f(i, 0) + 1 << " " << f(i, 1) + 1 << " " << f(i, 2) + 1 << "\n";

			return true;
		}
	} // namespace

	using namespace polysolve;

	NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const bool no_reduced)
		: state(state), assembler(state.assembler), rhs_assembler(rhs_assembler),
		  full_size((assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
		  reduced_size(full_size - (no_reduced ? 0 : state.boundary_nodes.size())),
		  t(t), rhs_computed(false), is_time_dependent(state.problem->is_time_dependent()), ignore_inertia(state.args["ignore_inertia"]), project_to_psd(project_to_psd)
	{
		assert(!assembler.is_mixed(state.formulation()));

		_dhat = dhat;
		_epsv = state.args["epsv"];
		_mu = state.args["mu"];
		use_adaptive_barrier_stiffness = !state.args["barrier_stiffness"].is_number();
		if (use_adaptive_barrier_stiffness)
		{
			_barrier_stiffness = 1;
			logger().debug("Using adaptive barrier stiffness");
		}
		else
		{
			assert(state.args["barrier_stiffness"].is_number());
			_barrier_stiffness = state.args["barrier_stiffness"];
			logger().debug("Using fixed barrier stiffness of {}", _barrier_stiffness);
		}
		_prev_distance = -1;
		time_integrator = ImplicitTimeIntegrator::construct_time_integrator(state.args["time_integrator"]);
		time_integrator->set_parameters(state.args["time_integrator_params"]);

		_broad_phase_method = state.args["solver_params"]["broad_phase_method"];
		_ccd_tolerance = state.args["solver_params"]["ccd_tolerance"];
		_ccd_max_iterations = state.args["solver_params"]["ccd_max_iterations"];
	}

	void NLProblem::init(const TVector &full)
	{
		if (disable_collision || !state.args["has_collision"])
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
		assert(full.size() == full_size);
		_barrier_stiffness = 1;
		if (disable_collision || !state.args["has_collision"])
			return;

		Eigen::MatrixXd grad_energy;
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad_energy);

		if (!ignore_inertia && is_time_dependent)
		{
			grad_energy *= time_integrator->acceleration_scaling();
			grad_energy += state.mass * full;
		}

		grad_energy -= current_rhs();

		Eigen::MatrixXd displaced;
		compute_displaced_points(full, displaced);

		update_constraint_set(displaced);
		Eigen::VectorXd grad_barrier = ipc::compute_barrier_potential_gradient(
			displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat);

		_barrier_stiffness = ipc::initial_barrier_stiffness(
			ipc::world_bbox_diagonal_length(displaced), _dhat, state.avg_mass,
			grad_energy, grad_barrier, max_barrier_stiffness_);
		polyfem::logger().debug("adaptive barrier stiffness {}", _barrier_stiffness);
	}

	void NLProblem::init_time_integrator(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt)
	{
		time_integrator->init(x_prev, v_prev, a_prev, dt);
	}

	void NLProblem::init_lagging(const TVector &x)
	{
		reduced_to_full_displaced_points(x, displaced_prev);
		update_lagging(x);
	}

	void NLProblem::update_lagging(const TVector &x)
	{
		Eigen::MatrixXd displaced;
		reduced_to_full_displaced_points(x, displaced);

		if (_mu != 0)
		{
			update_constraint_set(displaced);
			ipc::construct_friction_constraint_set(
				displaced, state.boundary_edges, state.boundary_triangles,
				_constraint_set, _dhat, _barrier_stiffness, _mu,
				_friction_constraint_set);
		}

	}

	double NLProblem::compute_lagging_error(const TVector &x)
	{
		// Check || ∇B(xᵗ⁺¹) - h² Σ F(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≦ ϵ_d
		//     ≡ || ∇B(xᵗ⁺¹) + ∇D(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≤ ϵ_d
		TVector grad;
		gradient(x, grad);
		return grad.norm();
	}

	bool NLProblem::lagging_converged(const TVector &x)
	{
		double tol = state.args.value("friction_convergence_tol", 1e-2);
		double grad_norm = compute_lagging_error(x);
		logger().debug("Lagging convergece grad_norm={:g} tol={:g}", grad_norm, tol);
		return grad_norm <= tol;
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		if (is_time_dependent)
		{
			if (!ignore_inertia)
				time_integrator->update_quantities(x);

			// rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t);
			// rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t);

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
		if (is_time_dependent)
		{
			rhs_computed = false;
			this->t = t;

			// rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t);
			// rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t);
		}
	}

	const Eigen::MatrixXd &NLProblem::current_rhs()
	{
		if (!rhs_computed)
		{
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.density, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, _current_rhs);
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
			rhs_assembler.set_bc(std::vector<LocalBoundary>(), std::vector<int>(), state.args["n_boundary_samples"], state.local_neumann_boundary, _current_rhs, t);

			if (!ignore_inertia && is_time_dependent)
			{
				_current_rhs *= time_integrator->acceleration_scaling();
				_current_rhs += state.mass * time_integrator->x_tilde();
			}

			if (reduced_size != full_size)
			{
				// rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, _current_rhs, t);
				rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], std::vector<LocalBoundary>(), _current_rhs, t);
			}
		}

		return _current_rhs;
	}

	void NLProblem::compute_displaced_points(const TVector &full, Eigen::MatrixXd &displaced)
	{
		assert(full.size() == full_size);

		const int problem_dim = state.mesh->dimension();
		displaced.resize(full.size() / problem_dim, problem_dim);
		assert(displaced.rows() * problem_dim == full.size());
		// Unflatten rowwises, so every problem_dim elements in full become a row
		for (int i = 0; i < full.size(); ++i)
		{
			displaced(i / problem_dim, i % problem_dim) = full(i);
		}

		assert(displaced(0, 0) == full(0));
		assert(displaced(0, 1) == full(1));

		displaced += state.boundary_nodes_pos;
	}

	void NLProblem::reduced_to_full_displaced_points(const TVector &reduced, Eigen::MatrixXd &displaced)
	{
		TVector full;
		reduced_to_full(reduced, full);
		compute_displaced_points(full, displaced);
	}

	void NLProblem::update_constraint_set(const Eigen::MatrixXd &displaced)
	{
		// Store the previous value used to compute the constraint set to avoid
		// duplicate computation.
		static Eigen::MatrixXd cached_displaced;
		if (cached_displaced.size() == displaced.size() && cached_displaced == displaced)
		{
			return;
		}
		if (_candidates.size() > 0)
			ipc::construct_constraint_set(
				_candidates, state.boundary_nodes_pos, displaced,
				state.boundary_edges, state.boundary_triangles, _dhat,
				_constraint_set, state.boundary_faces_to_edges);
		else
			ipc::construct_constraint_set(
				state.boundary_nodes_pos, displaced, state.codimensional_nodes,
				state.boundary_edges, state.boundary_triangles, _dhat,
				_constraint_set, state.boundary_faces_to_edges, /*dmin=*/0,
				_broad_phase_method, [&](size_t vi, size_t vj) {
					return can_vertices_collide(vi, vj);
				});
		cached_displaced = displaced;
	}

	void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		if (disable_collision || !state.args["has_collision"])
			return;

		Eigen::MatrixXd displaced0, displaced1;
		reduced_to_full_displaced_points(x0, displaced0);
		reduced_to_full_displaced_points(x1, displaced1);

		ipc::construct_collision_candidates(
			displaced0, displaced1, state.codimensional_nodes, state.boundary_edges,
			state.boundary_triangles, _candidates,
			/*inflation_radius=*/_dhat / 1.99, // divide by 1.99 instead of 2 to be conservative
			_broad_phase_method, [&](size_t vi, size_t vj) { return can_vertices_collide(vi, vj); });
	}

	void NLProblem::line_search_end()
	{
		_candidates.clear();
	}

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		if (disable_collision || !state.args["has_collision"])
			return 1;

		Eigen::MatrixXd displaced0, displaced1;
		reduced_to_full_displaced_points(x0, displaced0);
		reduced_to_full_displaced_points(x1, displaced1);

		// if (displaced0.cols() == 3)
		// {
		// 	igl::write_triangle_mesh("s0.obj", displaced0, state.boundary_triangles);
		// 	igl::write_triangle_mesh("s1.obj", displaced1, state.boundary_triangles);
		// }

		double max_step = ipc::compute_collision_free_stepsize(
			_candidates, displaced0, displaced1, state.boundary_edges,
			state.boundary_triangles, _ccd_tolerance, _ccd_max_iterations);
		// polyfem::logger().trace("best step {}", max_step);

#ifndef NDEBUG
		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		Eigen::MatrixXd displaced_toi = (displaced1 - displaced0) * max_step + displaced0;
		while (ipc::has_intersections(displaced_toi, state.boundary_edges, state.boundary_triangles, [&](size_t vi, size_t vj) { return can_vertices_collide(vi, vj); }))
		{
			double Linf = (displaced_toi - displaced0).lpNorm<Eigen::Infinity>();
			logger().warn("taking max_step results in intersections (max_step={:g})", max_step);
			max_step /= 2.0;
			if (max_step <= 0 || Linf == 0)
			{
				std::string msg = fmt::format("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf);
				logger().error(msg);
				throw msg;
			}
			displaced_toi = (displaced1 - displaced0) * max_step + displaced0;
		}
#endif

		return max_step;
	}

	bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		if (disable_collision || !state.args["has_collision"])
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

		// if (displaced0.cols() == 3)
		// {
		// 	igl::write_triangle_mesh("0.obj", displaced0, state.boundary_triangles);
		// 	igl::write_triangle_mesh("1.obj", displaced1, state.boundary_triangles);
		// }
		// else
		// {
		// 	Eigen::MatrixXd asd(displaced0.rows(), 3);
		// 	asd.setZero();
		// 	asd.block(0, 0, displaced0.rows(), 2) = displaced0;
		// 	igl::write_triangle_mesh("0.obj", asd, state.boundary_triangles);
		// 	asd.block(0, 0, displaced1.rows(), 2) = displaced1;
		// 	igl::write_triangle_mesh("1.obj", asd, state.boundary_triangles);
		// }

		const bool is_valid = ipc::is_step_collision_free(
			_candidates, displaced0, displaced1, state.boundary_edges,
			state.boundary_triangles, _ccd_tolerance, _ccd_max_iterations);

		return is_valid;
	}

	bool NLProblem::is_intersection_free(const TVector &x)
	{
		if (disable_collision || !state.args["has_collision"])
			return true;

		Eigen::MatrixXd displaced;
		reduced_to_full_displaced_points(x, displaced);

		return !ipc::has_intersections(
			displaced, state.boundary_edges, state.boundary_triangles,
			[&](size_t vi, size_t vj) { return can_vertices_collide(vi, vj); });
	}

	bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		TVector grad = TVector::Zero(reduced_size);
		gradient(x1, grad, true);

		if (std::isnan(grad.norm()))
			return false;

		return true;
	}

	double NLProblem::value(const TVector &x)
	{
		return value(x, false);
	}

	double NLProblem::value(const TVector &x, const bool only_elastic)
	{
		TVector full;
		reduced_to_full(x, full);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, state.ass_vals_cache, full);
		const double body_energy = rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.density, state.args["n_boundary_samples"], t);

		double intertia_energy = 0;
		double scaling = 1;
		if (!ignore_inertia && is_time_dependent)
		{
			scaling = time_integrator->acceleration_scaling();
			const TVector tmp = full - time_integrator->x_tilde();
			intertia_energy = 0.5 * tmp.transpose() * state.mass * tmp;
		}

		/*
		\frac 1 2 (x-(x^t+hv^t+h^2M^{-1}f_e))^TM(x-(x^t+hv^t+h^2M^{-1}f_e))=\\
		\frac 1 2 (x-x^t-hv^t-h^2M^{-1}f_e)^TM(x-x^t-hv^t-h^2M^{-1}f_e)=\\
		\frac 1 2 (t-h^2M^{-1}f_e)^TM(t-h^2M^{-1}f_e)=\\
		\frac 1 2 (t^T M-h^2f_e^T)(t-h^2M^{-1}f_e)=\\
		\frac 1 2 (t^T M t - h^2 t^T f_e
		-h^2f_e^T t + h^4f_e^T M^{-1}f_e)=\\
		\frac 1 2 (t^T M t - 2 h^2 t^T f_e + h^4f_e^T M^{-1}f_e)
		*/

		double collision_energy = 0;
		double friction_energy = 0;
		if (!only_elastic && !disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			collision_energy = ipc::compute_barrier_potential(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat);
			friction_energy = ipc::compute_friction_potential(displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set, _epsv * dt());

			polyfem::logger().trace("collision_energy {}, friction_energy {}", collision_energy, friction_energy);
		}

		// logger().trace("|constraints|={} |friction_constraints|={}", _constraint_set.size(), _friction_constraint_set.size());
		// logger().trace(
		// 	"elastic_energy={:.16g} body_energy={:.16g} intertia_energy={:.16g} collision_energy={:.16g} friction_energy={:.16g}",
		// 	scaling * elastic_energy, scaling * body_energy, intertia_energy, _barrier_stiffness * collision_energy, friction_energy);

#ifdef POLYFEM_DIV_BARRIER_STIFFNESS
		return (scaling * (elastic_energy + body_energy) + intertia_energy + friction_energy) / _barrier_stiffness + collision_energy;
#else
		return scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy + friction_energy;
#endif
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

#ifdef POLYFEM_DIV_BARRIER_STIFFNESS
		grad -= current_rhs() / _barrier_stiffness;
#else
		grad -= current_rhs();
#endif

		full_to_reduced(grad, gradv);

		// std::cout<<"gradv\n"<<gradv<<"\n--------------\n"<<std::endl;
	}

	void NLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad, const bool only_elastic)
	{
		// scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

		TVector full;
		reduced_to_full(x, full);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad);

		if (!ignore_inertia && is_time_dependent)
		{
			grad *= time_integrator->acceleration_scaling();
			grad += state.mass * full;
		}

		// logger().trace("grad norm {}", grad.norm());

#ifdef POLYFEM_DIV_BARRIER_STIFFNESS
		grad /= _barrier_stiffness;
#endif

		if (!only_elastic && !disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

#ifdef POLYFEM_DIV_BARRIER_STIFFNESS
			grad += ipc::compute_barrier_potential_gradient(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat);
			grad += ipc::compute_friction_potential_gradient(
						displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set, _epsv * dt())
					/ _barrier_stiffness;
#else
			grad += _barrier_stiffness * ipc::compute_barrier_potential_gradient(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat);
			grad += ipc::compute_friction_potential_gradient(
				displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set, _epsv * dt());
#endif
			// logger().trace("ipc grad norm {}", ipc::compute_barrier_potential_gradient(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat).norm());
		}

		assert(grad.size() == full_size);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian)
	{
		THessian tmp;
		hessian_full(x, tmp);

		if (reduced_size == full_size)
		{
			hessian = tmp;
			return;
		}

		POLYFEM_SCOPED_TIMER("\tremoving costraint time {}s");

		std::vector<Eigen::Triplet<double>> entries;

		Eigen::VectorXi indices(full_size);

		int index = 0;
		size_t kk = 0;
		for (int i = 0; i < full_size; ++i)
		{
			if (kk < state.boundary_nodes.size() && state.boundary_nodes[kk] == i)
			{
				++kk;
				indices(i) = -1;
				continue;
			}

			indices(i) = index++;
		}
		assert(index == reduced_size);

		for (int k = 0; k < tmp.outerSize(); ++k)
		{
			if (indices(k) < 0)
			{
				continue;
			}

			for (THessian::InnerIterator it(tmp, k); it; ++it)
			{
				// std::cout<<it.row()<<" "<<it.col()<<" "<<k<<std::endl;
				assert(it.col() == k);
				if (indices(it.row()) < 0 || indices(it.col()) < 0)
				{
					continue;
				}

				assert(indices(it.row()) >= 0);
				assert(indices(it.col()) >= 0);

				entries.emplace_back(indices(it.row()), indices(it.col()), it.value());
			}
		}

		hessian.resize(reduced_size, reduced_size);
		hessian.setFromTriplets(entries.begin(), entries.end());
		hessian.makeCompressed();
		// write_sparse_matrix_csv("hessian.csv", hessian);
	}

	void NLProblem::hessian_full(const TVector &x, THessian &hessian)
	{
		// scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

		TVector full;
		reduced_to_full(x, full);

		THessian energy_hessian(full_size, full_size);
		{
			POLYFEM_SCOPED_TIMER("\telastic hessian time {}s");

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
				energy_hessian *= time_integrator->acceleration_scaling();
			}
		}

		THessian inertia_hessian(full_size, full_size);
		if (!ignore_inertia && is_time_dependent)
		{
			POLYFEM_SCOPED_TIMER("\tinertia hessian time {}s");
			inertia_hessian = state.mass;
		}

		THessian barrier_hessian(full_size, full_size), friction_hessian(full_size, full_size);
		if (!disable_collision && state.args["has_collision"])
		{
			POLYFEM_SCOPED_TIMER("\tipc hessian(s) time {}s");

			Eigen::MatrixXd displaced;
			{
				POLYFEM_SCOPED_TIMER("\t\tdisplace pts time {}s");
				compute_displaced_points(full, displaced);
			}

			// {
			// 	POLYFEM_SCOPED_TIMER("\t\tconstraint set time {}s");
			// }

			{
				POLYFEM_SCOPED_TIMER("\t\tbarrier hessian time {}s");
				barrier_hessian = ipc::compute_barrier_potential_hessian(
					displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat, project_to_psd);
			}

			{
				POLYFEM_SCOPED_TIMER("\t\tfriction hessian time {}s");
				friction_hessian = ipc::compute_friction_potential_hessian(
					displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set,
					_epsv * dt(), project_to_psd);
			}
		}

		// Summing the hessian matrices like this might be less efficient than multiple `hessian += ...`, but
		// it is much easier to read and export the individual matrices for inspection.
#ifdef POLYFEM_DIV_BARRIER_STIFFNESS
		hessian = (energy_hessian + inertia_hessian + friction_hessian) / _barrier_stiffness + barrier_hessian;
#else
		hessian = energy_hessian + inertia_hessian + _barrier_stiffness * barrier_hessian + friction_hessian;
#endif
		assert(hessian.rows() == full_size);
		assert(hessian.cols() == full_size);

		// write_sparse_matrix_csv("energy_hessian.csv", energy_hessian);
		// write_sparse_matrix_csv("inertia_hessian.csv", inertia_hessian);
		// write_sparse_matrix_csv("barrier_hessian.csv", _barrier_stiffness * barrier_hessian);
		// write_sparse_matrix_csv("friction_hessian.csv", friction_hessian);
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		if (disable_collision || !state.args["has_collision"])
			return;

		Eigen::MatrixXd displaced;
		reduced_to_full_displaced_points(newX, displaced);

		update_constraint_set(displaced);
	}

	double NLProblem::heuristic_max_step(const TVector &dx)
	{
		// if (disable_collision || !state.args["has_collision"])
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
		if (disable_collision || !state.args["has_collision"])
			return;

		TVector full;
		reduced_to_full(x, full);

		Eigen::MatrixXd displaced;
		compute_displaced_points(full, displaced);

		if (state.args["save_nl_solve_sequence"])
		{
			writeOBJ(state.resolve_output_path(fmt::format("step{:03d}.obj", iter_num)),
					 displaced, state.boundary_edges, state.boundary_triangles);
		}

		const double dist_sqr = ipc::compute_minimum_distance(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set);
		polyfem::logger().trace("min_dist {}", sqrt(dist_sqr));

		if (use_adaptive_barrier_stiffness)
		{
			if (is_time_dependent)
			{
				double prev_barrier_stiffness = _barrier_stiffness;
				ipc::update_barrier_stiffness(
					_prev_distance, dist_sqr, max_barrier_stiffness_,
					_barrier_stiffness, ipc::world_bbox_diagonal_length(displaced));
				if (prev_barrier_stiffness != _barrier_stiffness)
				{
					polyfem::logger().debug(
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
		time_integrator->save_raw(x_path, v_path, a_path);
	}
} // namespace polyfem
