#include "polyfem/optimization/StateDiff.hpp"

#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Types.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <polysolve/linear/FEMSolver.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
// Because below types in SolverData are forward declared, we need to include them explicitly.
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <ipc/ipc.hpp>
#include <ipc/potentials/friction_potential.hpp>

#include <Eigen/Dense>
#include <algorithm>

using namespace polyfem::basis;

namespace polyfem
{
	namespace
	{
		void replace_rows_by_identity(StiffnessMatrix &reduced_mat, const StiffnessMatrix &mat, const std::vector<int> &rows)
		{
			reduced_mat.resize(mat.rows(), mat.cols());

			std::vector<bool> mask(mat.rows(), false);
			for (int i : rows)
				mask[i] = true;

			std::vector<Eigen::Triplet<double>> coeffs;
			for (int k = 0; k < mat.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mat, k); it; ++it)
				{
					if (mask[it.row()])
					{
						if (it.row() == it.col())
							coeffs.emplace_back(it.row(), it.col(), 1.0);
					}
					else
						coeffs.emplace_back(it.row(), it.col(), it.value());
				}
			}
			reduced_mat.setFromTriplets(coeffs.begin(), coeffs.end());
		}
	} // namespace

	void cache_transient_adjoint_quantities(State &state, const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad)
	{
		auto &s = state;

		StiffnessMatrix gradu_h(sol.size(), sol.size());
		if (current_step == 0)
			s.diff_cached.init(s.mesh->dimension(), s.ndof(), s.problem->is_time_dependent() ? s.args["time"]["time_steps"].get<int>() : 0);

		ipc::NormalCollisions cur_collision_set;
		ipc::SmoothCollisions cur_smooth_collision_set;
		ipc::TangentialCollisions cur_friction_set;
		ipc::NormalCollisions cur_normal_adhesion_set;
		ipc::TangentialCollisions cur_tangential_adhesion_set;

		if (s.optimization_enabled == solver::CacheLevel::Derivatives)
		{
			if (!s.problem->is_time_dependent() || current_step > 0)
				compute_force_jacobian(s, sol, disp_grad, gradu_h);

			if (s.solve_data.contact_form)
			{
				if (const auto barrier_contact = dynamic_cast<const solver::BarrierContactForm *>(s.solve_data.contact_form.get()))
					cur_collision_set = barrier_contact->collision_set();
				else if (const auto smooth_contact = dynamic_cast<const solver::SmoothContactForm *>(s.solve_data.contact_form.get()))
					cur_smooth_collision_set = smooth_contact->collision_set();
			}
			cur_friction_set = s.solve_data.friction_form ? s.solve_data.friction_form->friction_collision_set() : ipc::TangentialCollisions();
			cur_normal_adhesion_set = s.solve_data.normal_adhesion_form ? s.solve_data.normal_adhesion_form->collision_set() : ipc::NormalCollisions();
			cur_tangential_adhesion_set = s.solve_data.tangential_adhesion_form ? s.solve_data.tangential_adhesion_form->tangential_collision_set() : ipc::TangentialCollisions();
		}

		if (s.problem->is_time_dependent())
		{
			if (s.args["time"]["quasistatic"].get<bool>())
			{
				s.diff_cached.cache_quantities_quasistatic(current_step, sol, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_normal_adhesion_set, disp_grad);
			}
			else
			{
				Eigen::MatrixXd vel, acc;
				if (current_step == 0)
				{
					if (dynamic_cast<time_integrator::BDF *>(s.solve_data.time_integrator.get()))
					{
						const auto bdf_integrator = dynamic_cast<time_integrator::BDF *>(s.solve_data.time_integrator.get());
						vel = bdf_integrator->weighted_sum_v_prevs();
					}
					else if (dynamic_cast<time_integrator::ImplicitEuler *>(s.solve_data.time_integrator.get()))
					{
						const auto euler_integrator = dynamic_cast<time_integrator::ImplicitEuler *>(s.solve_data.time_integrator.get());
						vel = euler_integrator->v_prev();
					}
					else
						log_and_throw_error("Differentiable code doesn't support this time integrator!");

					acc.setZero(s.ndof(), 1);
				}
				else
				{
					vel = s.solve_data.time_integrator->compute_velocity(sol);
					acc = s.solve_data.time_integrator->compute_acceleration(vel);
				}

				s.diff_cached.cache_quantities_transient(current_step, s.solve_data.time_integrator->steps(), sol, vel, acc, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_friction_set);
			}
		}
		else
		{
			s.diff_cached.cache_quantities_static(sol, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_friction_set, cur_normal_adhesion_set, cur_tangential_adhesion_set, disp_grad);
		}
	}

	void compute_force_jacobian(State &state, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, StiffnessMatrix &hessian)
	{
		auto &s = state;

		if (s.problem->is_time_dependent())
		{
			if (s.assembler->is_linear() && !s.is_contact_enabled())
				log_and_throw_adjoint_error("Differentiable transient linear solve is not supported!");

			StiffnessMatrix tmp_hess;
			s.solve_data.nl_problem->set_project_to_psd(false);
			s.solve_data.nl_problem->FullNLProblem::solution_changed(sol);
			s.solve_data.nl_problem->FullNLProblem::hessian(sol, tmp_hess);
			hessian.setZero();
			replace_rows_by_identity(hessian, tmp_hess, s.boundary_nodes);
		}
		else // static formulation
		{
			if (s.assembler->is_linear() && !s.is_contact_enabled() && !s.is_homogenization())
			{
				hessian.setZero();
				StiffnessMatrix stiffness;
				s.build_stiffness_mat(stiffness);
				replace_rows_by_identity(hessian, stiffness, s.boundary_nodes);
			}
			else
			{
				s.solve_data.nl_problem->set_project_to_psd(false);
				if (s.is_homogenization())
				{
					Eigen::VectorXd reduced;
					std::shared_ptr<solver::NLHomoProblem> homo_problem = std::dynamic_pointer_cast<solver::NLHomoProblem>(s.solve_data.nl_problem);
					reduced = homo_problem->full_to_reduced(sol, disp_grad);
					s.solve_data.nl_problem->solution_changed(reduced);
					s.solve_data.nl_problem->hessian(reduced, hessian);
				}
				else
				{
					StiffnessMatrix tmp_hess;
					s.solve_data.nl_problem->FullNLProblem::solution_changed(sol);
					s.solve_data.nl_problem->FullNLProblem::hessian(sol, tmp_hess);
					hessian.setZero();
					replace_rows_by_identity(hessian, tmp_hess, s.boundary_nodes);
				}
			}
		}
	}

	void compute_force_jacobian_prev(const State &state, const int force_step, const int sol_step, StiffnessMatrix &hessian_prev)
	{
		assert(force_step > 0);
		assert(force_step > sol_step);

		auto &s = state;

		if (s.assembler->is_linear() && !s.is_contact_enabled())
		{
			hessian_prev = StiffnessMatrix(s.ndof(), s.ndof());
		}
		else
		{
			const Eigen::MatrixXd u = s.diff_cached.u(force_step);
			const Eigen::MatrixXd u_prev = s.diff_cached.u(sol_step);
			const double beta = time_integrator::BDF::betas(s.diff_cached.bdf_order(force_step) - 1);
			const double dt = s.solve_data.time_integrator->dt();

			hessian_prev = StiffnessMatrix(u.size(), u.size());
			if (s.problem->is_time_dependent())
			{
				if (s.solve_data.friction_form)
				{
					if (sol_step == force_step - 1)
					{
						Eigen::MatrixXd surface_solution_prev = s.collision_mesh.vertices(utils::unflatten(u_prev, s.mesh->dimension()));
						Eigen::MatrixXd surface_solution = s.collision_mesh.vertices(utils::unflatten(u, s.mesh->dimension()));

						// TODO: use the time integration to compute the velocity
						const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / dt;
						const double dv_dut = -1 / dt;

						if (const auto barrier_contact = dynamic_cast<const solver::BarrierContactForm *>(s.solve_data.contact_form.get()))
						{
							hessian_prev =
								s.solve_data.friction_form->friction_potential().force_jacobian(
									s.diff_cached.friction_collision_set(force_step),
									s.collision_mesh,
									s.collision_mesh.rest_positions(),
									/*lagged_displacements=*/surface_solution_prev,
									surface_velocities,
									barrier_contact->barrier_potential(),
									barrier_contact->barrier_stiffness(),
									ipc::FrictionPotential::DiffWRT::LAGGED_DISPLACEMENTS)
								+ s.solve_data.friction_form->friction_potential().force_jacobian(
									  s.diff_cached.friction_collision_set(force_step),
									  s.collision_mesh,
									  s.collision_mesh.rest_positions(),
									  /*lagged_displacements=*/surface_solution_prev,
									  surface_velocities,
									  barrier_contact->barrier_potential(),
									  barrier_contact->barrier_stiffness(),
									  ipc::FrictionPotential::DiffWRT::VELOCITIES)
									  * dv_dut;
						}

						hessian_prev *= -1;

						// {
						// 	Eigen::MatrixXd X = collision_mesh.rest_positions();
						// 	Eigen::VectorXd x = utils::flatten(surface_solution_prev);
						// 	const double barrier_stiffness = solve_data.contact_form->barrier_stiffness();
						// 	const double dhat = solve_data.contact_form->dhat();
						// 	const double mu = solve_data.friction_form->mu();
						// 	const double epsv = solve_data.friction_form->epsv();

						// 	Eigen::MatrixXd fgrad;
						// 	fd::finite_jacobian(
						// 		x, [&](const Eigen::VectorXd &y) -> Eigen::VectorXd
						// 		{
						// 			Eigen::MatrixXd fd_Ut = utils::unflatten(y, surface_solution_prev.cols());

						// 			ipc::TangentialCollisions fd_friction_constraints;
						// 			ipc::NormalCollisions fd_constraints;
						// 			fd_constraints.set_use_convergent_formulation(solve_data.contact_form->use_convergent_formulation());
						// 			fd_constraints.set_enable_shape_derivatives(true);
						// 			fd_constraints.build(collision_mesh, X + fd_Ut, dhat);

						// 			fd_friction_constraints.build(
						// 				collision_mesh, X + fd_Ut, fd_constraints, dhat, barrier_stiffness,
						// 				mu);

						// 			return fd_friction_constraints.compute_potential_gradient(collision_mesh, (surface_solution - fd_Ut) / dt, epsv);

						// 		}, fgrad, fd::AccuracyOrder::SECOND, 1e-8);

						// 	logger().trace("force Ut derivative error {} {}", (fgrad - hessian_prev).norm(), hessian_prev.norm());
						// }

						hessian_prev = s.collision_mesh.to_full_dof(hessian_prev); // / (beta * dt) / (beta * dt);
					}
					else
					{
						// const double alpha = time_integrator::BDF::alphas(std::min(diff_cached.bdf_order(force_step), force_step) - 1)[force_step - sol_step - 1];
						// Eigen::MatrixXd velocity = collision_mesh.map_displacements(utils::unflatten(diff_cached.v(force_step), collision_mesh.dim()));
						// hessian_prev = diff_cached.friction_collision_set(force_step).compute_potential_hessian( //
						// 			collision_mesh, velocity, solve_data.friction_form->epsv(), false) * (-alpha / beta / dt);

						// hessian_prev = collision_mesh.to_full_dof(hessian_prev);
					}
				}

				if (s.solve_data.tangential_adhesion_form)
				{

					if (sol_step == force_step - 1)
					{
						StiffnessMatrix adhesion_hessian_prev(u.size(), u.size());

						Eigen::MatrixXd surface_solution_prev = s.collision_mesh.vertices(utils::unflatten(u_prev, s.mesh->dimension()));
						Eigen::MatrixXd surface_solution = s.collision_mesh.vertices(utils::unflatten(u, s.mesh->dimension()));

						// TODO: use the time integration to compute the velocity
						const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / dt;
						const double dv_dut = -1 / dt;

						adhesion_hessian_prev =
							s.solve_data.tangential_adhesion_form->tangential_adhesion_potential().force_jacobian(
								s.diff_cached.tangential_adhesion_collision_set(force_step),
								s.collision_mesh,
								s.collision_mesh.rest_positions(),
								/*lagged_displacements=*/surface_solution_prev,
								surface_velocities,
								s.solve_data.normal_adhesion_form->normal_adhesion_potential(),
								1,
								ipc::TangentialPotential::DiffWRT::LAGGED_DISPLACEMENTS)
							+ s.solve_data.tangential_adhesion_form->tangential_adhesion_potential().force_jacobian(
								  s.diff_cached.tangential_adhesion_collision_set(force_step),
								  s.collision_mesh,
								  s.collision_mesh.rest_positions(),
								  /*lagged_displacements=*/surface_solution_prev,
								  surface_velocities,
								  s.solve_data.normal_adhesion_form->normal_adhesion_potential(),
								  1,
								  ipc::TangentialPotential::DiffWRT::VELOCITIES)
								  * dv_dut;

						adhesion_hessian_prev *= -1;

						adhesion_hessian_prev = s.collision_mesh.to_full_dof(adhesion_hessian_prev); // / (beta * dt) / (beta * dt);

						hessian_prev += adhesion_hessian_prev;
					}
				}

				if (s.damping_assembler->is_valid() && sol_step == force_step - 1) // velocity in damping uses BDF1
				{
					utils::SparseMatrixCache mat_cache;
					StiffnessMatrix damping_hessian_prev(u.size(), u.size());
					s.damping_prev_assembler->assemble_hessian(s.mesh->is_volume(), s.n_bases, false, s.bases, s.geom_bases(), s.ass_vals_cache, force_step * s.args["time"]["dt"].get<double>() + s.args["time"]["t0"].get<double>(), dt, u, u_prev, mat_cache, damping_hessian_prev);

					hessian_prev += damping_hessian_prev;
				}

				if (sol_step == force_step - 1)
				{
					StiffnessMatrix body_force_hessian(u.size(), u.size());
					s.solve_data.body_form->hessian_wrt_u_prev(u_prev, force_step * dt, body_force_hessian);
					hessian_prev += body_force_hessian;
				}
			}
		}
	}

	void solve_adjoint_cached(State state, const Eigen::MatrixXd &rhs)
	{
		state.diff_cached.cache_adjoints(solve_adjoint(state, rhs));
	}

	Eigen::MatrixXd solve_adjoint(const State &state, const Eigen::MatrixXd &rhs)
	{
		if (state.problem->is_time_dependent())
			return solve_transient_adjoint(state, rhs);
		else
			return solve_static_adjoint(state, rhs);
	}

	Eigen::MatrixXd get_adjoint_mat(const State &state, int type)
	{
		assert(state.diff_cached.adjoint_mat().size() > 0);

		auto &s = state;

		if (s.problem->is_time_dependent())
		{
			if (type == 0)
				return s.diff_cached.adjoint_mat().leftCols(s.diff_cached.adjoint_mat().cols() / 2);
			else if (type == 1)
				return s.diff_cached.adjoint_mat().middleCols(s.diff_cached.adjoint_mat().cols() / 2, s.diff_cached.adjoint_mat().cols() / 2);
			else
				log_and_throw_adjoint_error("Invalid adjoint type!");
		}

		return s.diff_cached.adjoint_mat();
	}

	Eigen::MatrixXd solve_static_adjoint(const State &state, const Eigen::MatrixXd &adjoint_rhs)
	{
		auto &s = state;

		Eigen::MatrixXd b = adjoint_rhs;

		Eigen::MatrixXd adjoint;
		if (s.lin_solver_cached)
		{
			b(s.boundary_nodes, Eigen::all).setZero();

			StiffnessMatrix A = s.diff_cached.gradu_h(0);
			const int full_size = A.rows();
			const int problem_dim = s.problem->is_scalar() ? 1 : s.mesh->dimension();
			int precond_num = problem_dim * s.n_bases;

			b.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), b.cols()));

			std::vector<int> boundary_nodes_tmp;
			if (s.has_periodic_bc())
			{
				boundary_nodes_tmp = s.periodic_bc->full_to_periodic(s.boundary_nodes);
				precond_num = s.periodic_bc->full_to_periodic(A);
				b = s.periodic_bc->full_to_periodic(b, true);
			}
			else
				boundary_nodes_tmp = s.boundary_nodes;

			adjoint.setZero(s.ndof(), adjoint_rhs.cols());
			for (int i = 0; i < b.cols(); i++)
			{
				Eigen::VectorXd x, tmp;
				tmp = b.col(i);
				dirichlet_solve_prefactorized(*s.lin_solver_cached, A, tmp, boundary_nodes_tmp, x);

				if (s.has_periodic_bc())
					adjoint.col(i) = s.periodic_bc->periodic_to_full(full_size, x);
				else
					adjoint.col(i) = x;
			}
		}
		else
		{
			auto solver = polysolve::linear::Solver::create(s.args["solver"]["adjoint_linear"], adjoint_logger());

			StiffnessMatrix A = s.diff_cached.gradu_h(0); // This should be transposed, but A is symmetric in hyper-elastic and diffusion problems

			/*
			For non-periodic problems, the adjoint solution p's size is the full size in NLProblem
			For periodic problems, the adjoint solution p's size is the reduced size in NLProblem
			*/
			if (!s.is_homogenization())
			{
				adjoint.setZero(s.ndof(), adjoint_rhs.cols());
				for (int i = 0; i < b.cols(); i++)
				{
					Eigen::VectorXd tmp = b.col(i);
					tmp(s.boundary_nodes).setZero();

					Eigen::VectorXd x;
					x.setZero(tmp.size());
					dirichlet_solve(*solver, A, tmp, s.boundary_nodes, x, A.rows(), "", false, false, false);

					adjoint.col(i) = x;
					adjoint(s.boundary_nodes, i) = -b(s.boundary_nodes, i);
				}
			}
			else
			{
				solver->analyze_pattern(A, A.rows());
				solver->factorize(A);

				adjoint.setZero(adjoint_rhs.rows(), adjoint_rhs.cols());
				for (int i = 0; i < b.cols(); i++)
				{
					Eigen::MatrixXd tmp = b.col(i);

					Eigen::VectorXd x;
					x.setZero(tmp.size());
					solver->solve(tmp, x);
					x.conservativeResize(adjoint.rows());

					adjoint.col(i) = x;
				}
			}
		}

		return adjoint;
	}

	Eigen::MatrixXd solve_transient_adjoint(const State &state, const Eigen::MatrixXd &adjoint_rhs)
	{
		auto &s = state;

		const double dt = s.args["time"]["dt"];
		const int time_steps = s.args["time"]["time_steps"];

		int bdf_order = 1;
		if (s.args["time"]["integrator"].is_string())
			bdf_order = 1;
		else if (s.args["time"]["integrator"]["type"] == "ImplicitEuler")
			bdf_order = 1;
		else if (s.args["time"]["integrator"]["type"] == "BDF")
			bdf_order = s.args["time"]["integrator"]["steps"].get<int>();
		else
			log_and_throw_adjoint_error("Integrator type not supported for differentiability.");

		assert(adjoint_rhs.cols() == time_steps + 1);

		const int cols_per_adjoint = time_steps + 1;
		Eigen::MatrixXd adjoints;
		adjoints.setZero(s.ndof(), cols_per_adjoint * 2);

		// set dirichlet rows of mass to identity
		StiffnessMatrix reduced_mass;
		replace_rows_by_identity(reduced_mass, s.mass, s.boundary_nodes);

		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		for (int i = time_steps; i >= 0; --i)
		{
			{
				sum_alpha_p.setZero(s.ndof(), 1);
				sum_alpha_nu.setZero(s.ndof(), 1);

				const int num = std::min(bdf_order, time_steps - i);

				Eigen::VectorXd bdf_coeffs = Eigen::VectorXd::Zero(num);
				for (int j = 0; j < bdf_order && i + j < time_steps; ++j)
					bdf_coeffs(j) = -time_integrator::BDF::alphas(std::min(bdf_order - 1, i + j))[j];

				sum_alpha_p = adjoints.middleCols(i + 1, num) * bdf_coeffs;
				sum_alpha_nu = adjoints.middleCols(cols_per_adjoint + i + 1, num) * bdf_coeffs;
			}

			Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu - adjoint_rhs.col(i);
			for (int j = 1; j <= bdf_order; j++)
			{
				if (i + j > time_steps)
					break;

				StiffnessMatrix gradu_h_prev;
				compute_force_jacobian_prev(state, i + j, i, gradu_h_prev);
				Eigen::VectorXd tmp = adjoints.col(i + j) * (time_integrator::BDF::betas(s.diff_cached.bdf_order(i + j) - 1) * dt);
				tmp(s.boundary_nodes).setZero();
				rhs_ += -gradu_h_prev.transpose() * tmp;
			}

			if (i > 0)
			{
				double beta_dt = time_integrator::BDF::betas(s.diff_cached.bdf_order(i) - 1) * dt;

				rhs_ += (1. / beta_dt) * (s.diff_cached.gradu_h(i) - reduced_mass).transpose() * sum_alpha_p;

				{
					StiffnessMatrix A = s.diff_cached.gradu_h(i).transpose();
					Eigen::VectorXd b_ = rhs_;
					b_(s.boundary_nodes).setZero();

					auto solver = polysolve::linear::Solver::create(s.args["solver"]["adjoint_linear"], adjoint_logger());

					Eigen::VectorXd x;
					dirichlet_solve(*solver, A, b_, s.boundary_nodes, x, A.rows(), "", false, false, false);
					adjoints.col(i + cols_per_adjoint) = x;
				}

				// TODO: generalize to BDFn
				Eigen::VectorXd tmp = rhs_(s.boundary_nodes);
				if (i + 1 < cols_per_adjoint)
					tmp += (-2. / beta_dt) * adjoints(s.boundary_nodes, i + 1);
				if (i + 2 < cols_per_adjoint)
					tmp += (1. / beta_dt) * adjoints(s.boundary_nodes, i + 2);

				tmp -= (s.diff_cached.gradu_h(i).transpose() * adjoints.col(i + cols_per_adjoint))(s.boundary_nodes);
				adjoints(s.boundary_nodes, i + cols_per_adjoint) = tmp;
				adjoints.col(i) = beta_dt * adjoints.col(i + cols_per_adjoint) - sum_alpha_p;
			}
			else
			{
				adjoints.col(i) = -reduced_mass.transpose() * sum_alpha_p;
				adjoints.col(i + cols_per_adjoint) = rhs_; // adjoint_nu[0] actually stores adjoint_mu[0]
			}
		}
		return adjoints;
	}

	void compute_surface_node_ids(const State &state, const int surface_selection, std::vector<int> &node_ids)
	{
		auto &s = state;

		node_ids = {};

		const auto &gbases = s.geom_bases();
		for (const auto &lb : s.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = s.mesh->get_boundary_id(primitive_global_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *s.mesh);

				if (boundary_id == surface_selection)
				{
					for (long n = 0; n < nodes.size(); ++n)
					{
						const int g_id = gbases[e].bases[nodes(n)].global()[0].index;

						if (std::count(node_ids.begin(), node_ids.end(), g_id) == 0)
							node_ids.push_back(g_id);
					}
				}
			}
		}
	}

	void compute_total_surface_node_ids(const State &state, std::vector<int> &node_ids)
	{
		auto &s = state;

		node_ids = {};

		const auto &gbases = s.geom_bases();
		for (const auto &lb : s.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *s.mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					const int g_id = gbases[e].bases[nodes(n)].global()[0].index;

					if (std::count(node_ids.begin(), node_ids.end(), g_id) == 0)
						node_ids.push_back(g_id);
				}
			}
		}
	}

	void compute_volume_node_ids(const State &state, const int volume_selection, std::vector<int> &node_ids)
	{
		auto &s = state;

		node_ids = {};

		const auto &gbases = s.geom_bases();
		for (int e = 0; e < gbases.size(); e++)
		{
			const int body_id = s.mesh->get_body_id(e);
			if (body_id == volume_selection)
				for (const auto &gbs : gbases[e].bases)
					for (const auto &g : gbs.global())
						node_ids.push_back(g.index);
		}
	}

} // namespace polyfem
