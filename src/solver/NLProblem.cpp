#include <polyfem/NLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <igl/write_triangle_mesh.h>

#include <unsupported/Eigen/SparseExtra>

#include <igl/Timer.h>

static bool disable_collision = false;

// #define USE_DIV_BARRIER_STIFFNESS

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

namespace polyfem
{
	using namespace polysolve;

	NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const bool no_reduced)
		: state(state), assembler(state.assembler), rhs_assembler(rhs_assembler),
		  full_size((assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
		  reduced_size(full_size - (no_reduced ? 0 : state.boundary_nodes.size())),
		  t(t), rhs_computed(false), is_time_dependent(state.problem->is_time_dependent()), project_to_psd(project_to_psd)
	{
		assert(!assembler.is_mixed(state.formulation()));

		_dhat = dhat;
		_epsv = state.args["epsv"];
		_mu = state.args["mu"];
		_barrier_stiffness = 1;
		_prev_distance = -1;
		time_integrator = ImplicitTimeIntegrator::construct_time_integrator(state.args["time_integrator"]);
		time_integrator->set_parameters(state.args["time_integrator_params"]);

		if (state.args["solver_params"].contains("ccd_method"))
		{
			const std::string ccd_method = state.args["solver_params"]["ccd_method"];

			if (ccd_method == "brute_force")
				_broad_phase_method = ipc::BroadPhaseMethod::BRUTE_FORCE;
			else if (ccd_method == "spatial_hash")
				_broad_phase_method = ipc::BroadPhaseMethod::SPATIAL_HASH;
			else
				_broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
		}

		else
			_broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;

		if (state.args["solver_params"].contains("ccd_tolerance"))
			_ccd_tolerance = state.args["solver_params"]["ccd_tolerance"];
		else
			_ccd_tolerance = 1e-6;
		if (state.args["solver_params"].contains("ccd_max_iterations"))
			_max_ccd_max_iterations = state.args["solver_params"]["ccd_max_iterations"];
		else
			_max_ccd_max_iterations = 1e6;

		_ccd_max_iterations = _max_ccd_max_iterations;
	}

	void NLProblem::init(const TVector &full)
	{
		if (disable_collision || !state.args["has_collision"])
			return;

		assert(full.size() == full_size);
		update_barrier_stiffness(full);
		// exit(0);
	}

	void NLProblem::update_barrier_stiffness(const TVector &full)
	{
		assert(full.size() == full_size);
		_barrier_stiffness = 1;
		if (disable_collision || !state.args["has_collision"])
			return;

		Eigen::MatrixXd grad;
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad);

		if (is_time_dependent)
		{
			grad *= time_integrator->acceleration_scaling();
			grad += state.mass * full;
		}

		grad -= current_rhs();

		Eigen::MatrixXd displaced;
		compute_displaced_points(full, displaced);

		_barrier_stiffness = ipc::initial_barrier_stiffness(
			state.boundary_nodes_pos,
			displaced,
			state.boundary_edges, state.boundary_triangles,
			_dhat,
			state.avg_mass,
			grad,
			max_barrier_stiffness_);
		polyfem::logger().debug("adaptive stiffness {}", _barrier_stiffness);
	}

	void NLProblem::init_time_integrator(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt)
	{
		time_integrator->init(x_prev, v_prev, a_prev, dt);
	}

	void NLProblem::update_lagging(const TVector &x, bool start_of_timestep)
	{
		Eigen::MatrixXd displaced;
		reduced_to_full_displaced_points(x, displaced);

		if (_mu != 0)
		{
			ipc::construct_constraint_set(
				state.boundary_nodes_pos, displaced, state.boundary_edges,
				state.boundary_triangles, _dhat, _constraint_set, _ignore_codimensional_vertices,
				_broad_phase_method, Eigen::VectorXi(),
				state.boundary_faces_to_edges);
			ipc::construct_friction_constraint_set(
				displaced, state.boundary_edges, state.boundary_triangles,
				_constraint_set, _dhat, _barrier_stiffness, _mu,
				_friction_constraint_set);
		}

		if (start_of_timestep)
		{
			displaced_prev = displaced;
		}
	}

	bool NLProblem::lagging_converged(const TVector &x, bool do_lagging_update)
	{
		if (do_lagging_update)
		{
			update_lagging(x, /*start_of_timestep=*/false);
		}
		// else assume the lagging was updated before

		// Check || ∇B(xᵗ⁺¹) - h² Σ F(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≦ ϵ_d
		//     ≡ || ∇B(xᵗ⁺¹) + ∇D(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≤ ϵ_d
		TVector grad;
		gradient(x, grad);
		double tol = state.args.value("friction_convergence_tol", 1e-2);
		double grad_norm = grad.norm();
		logger().debug("Lagging convergece grad_norm={:g} tol={:g}", grad_norm, tol);
		return grad.norm() <= tol;
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		if (is_time_dependent)
		{
			time_integrator->update_quantities(x);

			// rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t);
			// rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t);

			rhs_computed = false;
			this->t = t;

			update_barrier_stiffness(x);
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

			if (is_time_dependent)
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

	void NLProblem::compute_displaced_points(const Eigen::MatrixXd &full, Eigen::MatrixXd &displaced)
	{
		assert(full.size() == full_size);

		const int problem_dim = state.mesh->dimension();
		displaced.resize(full.size() / problem_dim, problem_dim);
		assert(displaced.rows() * problem_dim == full.size());
		for (int i = 0; i < full.size(); i += problem_dim)
		{
			for (int d = 0; d < problem_dim; ++d)
			{
				displaced(i / problem_dim, d) = full(i + d);
			}
		}

		assert(displaced(0, 0) == full(0));
		assert(displaced(0, 1) == full(1));

		displaced += state.boundary_nodes_pos;
	}

	void NLProblem::reduced_to_full_displaced_points(const TVector &reduced, Eigen::MatrixXd &displaced)
	{
		Eigen::MatrixXd full;
		if (reduced.size() == reduced_size)
			reduced_to_full(reduced, full);
		else
			full = reduced;
		assert(full.size() == full_size);

		compute_displaced_points(full, displaced);
	}

	void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		if (disable_collision)
			return;
		if (!state.args["has_collision"])
			return;

		Eigen::MatrixXd displaced0, displaced1;
		reduced_to_full_displaced_points(x0, displaced0);
		reduced_to_full_displaced_points(x1, displaced1);

		ipc::construct_collision_candidates(
			displaced0, displaced1, state.boundary_edges,
			state.boundary_triangles, _candidates, /*inflation_radius=*/0,
			_broad_phase_method, _ignore_codimensional_vertices,
			/*vertex_group_ids=*/Eigen::VectorXi());
	}

	void NLProblem::line_search_end()
	{
		_candidates.clear();
	}

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		if (disable_collision)
			return 1;
		if (!state.args["has_collision"])
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
			_candidates,
			displaced0, displaced1,
			state.boundary_edges, state.boundary_triangles,
			_ccd_tolerance, _ccd_max_iterations);
		polyfem::logger().trace("best step {}", max_step);

		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		// Eigen::MatrixXd displaced_toi = (displaced1 - displaced0) * max_step + displaced0;
		// while (ipc::has_intersections(displaced_toi, state.boundary_edges, state.boundary_triangles))
		// {
		// 	double Linf = (displaced_toi - displaced0).lpNorm<Eigen::Infinity>();
		// 	logger().warn("taking max_step results in intersections (max_step={:g})", max_step);
		// 	max_step /= 2.0;
		// 	if (max_step <= 0 || Linf == 0)
		// 	{
		// 		logger().error("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf);
		// 	}
		// 	displaced_toi = (displaced1 - displaced0) * max_step + displaced0;
		// }

		return max_step;
	}

	bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		if (disable_collision)
			return true;
		if (!state.args["has_collision"])
			return true;

		// if (!state.problem->is_time_dependent())
		// return false;

		Eigen::MatrixXd displaced0, displaced1;
		reduced_to_full_displaced_points(x0, displaced0);
		reduced_to_full_displaced_points(x1, displaced1);

		// if (displaced0.cols() == 3)
		// {
		// 	igl::write_triangle_mesh("0.obj", displaced0, state.boundary_triangles);
		// 	igl::write_triangle_mesh("1.obj", displaced1, state.boundary_triangles);
		// }

		const bool is_valid = ipc::is_step_collision_free(_candidates,
														  displaced0, displaced1,
														  state.boundary_edges, state.boundary_triangles,
														  _ccd_tolerance, _ccd_max_iterations);

		return is_valid;
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
		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, state.ass_vals_cache, full);
		const double body_energy = rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.density, state.args["n_boundary_samples"], t);

		double intertia_energy = 0;
		double scaling = 1;
		if (is_time_dependent)
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

#ifdef USE_DIV_BARRIER_STIFFNESS
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

#ifdef USE_DIV_BARRIER_STIFFNESS
		grad -= current_rhs() / _barrier_stiffness;
#else
		grad -= current_rhs();
#endif

		full_to_reduced(grad, gradv);

		// std::cout<<"gradv\n"<<gradv<<"\n--------------\n"<<std::endl;
	}

	void NLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad, const bool only_elastic)
	{
		//scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad);

		if (is_time_dependent)
		{
			grad *= time_integrator->acceleration_scaling();
			grad += state.mass * full;
		}

		// logger().trace("grad norm {}", grad.norm());

#ifdef USE_DIV_BARRIER_STIFFNESS
		grad /= _barrier_stiffness;
#endif

		if (!only_elastic && !disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

#ifdef USE_DIV_BARRIER_STIFFNESS
			grad += ipc::compute_barrier_potential_gradient(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat);
			grad += ipc::compute_friction_potential_gradient(
						displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set, _epsv * dt())
					/ barrier_stiffness;
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
	}

	void NLProblem::hessian_full(const TVector &x, THessian &hessian)
	{
		//scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

		igl::Timer timer;
		timer.start();

		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;

		timer.stop();
		polyfem::logger().trace("\treduced to full time {}s", timer.getElapsedTimeInSec());
		timer.start();

		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		if (assembler.is_linear(rhs_assembler.formulation()))
		{
			compute_cached_stiffness();
			hessian = cached_stiffness;
		}
		else
			assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, project_to_psd, state.bases, gbases, state.ass_vals_cache, full, mat_cache, hessian);

		timer.stop();
		polyfem::logger().trace("\telastic hessian time {}s", timer.getElapsedTimeInSec());
		timer.start();
		if (is_time_dependent)
		{
			hessian *= time_integrator->acceleration_scaling();
			hessian += state.mass;
		}

		timer.stop();
		polyfem::logger().trace("\tinertia hessian time {}s", timer.getElapsedTimeInSec());

#ifdef USE_DIV_BARRIER_STIFFNESS
		hessian /= _barrier_stiffness;
#endif

		if (!disable_collision && state.args["has_collision"])
		{
			timer.start();

			igl::Timer timeri;
			timeri.start();

			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);
			timeri.stop();
			polyfem::logger().trace("\tdisplace pts time {}s", timeri.getElapsedTimeInSec());
			timeri.start();

			timeri.stop();
			polyfem::logger().trace("\tconstraint set time {}s", timeri.getElapsedTimeInSec());
			timeri.start();
#ifdef USE_DIV_BARRIER_STIFFNESS
			hessian += ipc::compute_barrier_potential_hessian(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat, project_to_psd);
			hessian += ipc::compute_friction_potential_hessian(
						   displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set, _epsv * dt(), project_to_psd)
					   / _barrier_stiffness;
#else
			hessian += _barrier_stiffness * ipc::compute_barrier_potential_hessian(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set, _dhat, project_to_psd);
			hessian += ipc::compute_friction_potential_hessian(
				displaced_prev, displaced, state.boundary_edges, state.boundary_triangles, _friction_constraint_set, _epsv * dt(), project_to_psd);
#endif
			timeri.stop();
			polyfem::logger().trace("\tonly ipc hessian time {}s", timeri.getElapsedTimeInSec());

			timer.stop();
			polyfem::logger().trace("\tipc hessian time {}s", timer.getElapsedTimeInSec());
		}

		assert(hessian.rows() == full_size);
		assert(hessian.cols() == full_size);
		// Eigen::saveMarket(tmp, "tmp.mat");
		// exit(0);
	}

	void NLProblem::full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const
	{
		full_to_reduced_aux(state, full_size, reduced_size, full, reduced);
	}

	void NLProblem::reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full)
	{
		reduced_to_full_aux(state, full_size, reduced_size, reduced, current_rhs(), full);
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		if (disable_collision)
			return;
		if (!state.args["has_collision"])
			return;

		Eigen::MatrixXd displaced;
		reduced_to_full_displaced_points(newX, displaced);

		if (_candidates.size() > 0)
			ipc::construct_constraint_set(_candidates, state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles,
										  _dhat, _constraint_set, state.boundary_faces_to_edges);
		else
			ipc::construct_constraint_set(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles,
										  _dhat, _constraint_set, _ignore_codimensional_vertices, _broad_phase_method, Eigen::VectorXi(), state.boundary_faces_to_edges);
	}

	double NLProblem::heuristic_max_step(const TVector &dx)
	{
		// if (disable_collision)
		// 	return 1;
		// if (!state.args["has_collision"])
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

	void NLProblem::post_step(const TVector &x0)
	{
		if (disable_collision)
			return;
		if (!state.args["has_collision"])
			return;

		Eigen::MatrixXd full;
		if (x0.size() == reduced_size)
			reduced_to_full(x0, full);
		else
			full = x0;

		assert(full.size() == full_size);

		Eigen::MatrixXd displaced;

		compute_displaced_points(full, displaced);

		const double dist_sqr = ipc::compute_minimum_distance(displaced, state.boundary_edges, state.boundary_triangles, _constraint_set);
		polyfem::logger().trace("min_dist {}", sqrt(dist_sqr));
		// igl::write_triangle_mesh("step.obj", displaced, state.boundary_triangles);

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
		_prev_distance = dist_sqr;
	}

	void NLProblem::save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path)
	{
		time_integrator->save_raw(x_path, v_path, a_path);
	}
} // namespace polyfem
