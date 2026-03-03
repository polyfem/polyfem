#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/linear/FEMSolver.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/utils/local_to_global.hpp>
#include <ipc/potentials/friction_potential.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <deque>
#include <map>
#include <algorithm>

#include <fstream>

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

	void State::get_vertices(Eigen::MatrixXd &vertices) const
	{
		vertices.setZero(mesh->n_vertices(), mesh->dimension());

		for (int v = 0; v < mesh->n_vertices(); v++)
			vertices.row(v) = mesh->point(v);
	}

	void State::get_elements(Eigen::MatrixXi &elements) const
	{
		assert(mesh->is_simplicial());

		auto node_to_primitive_map = node_to_primitive();

		const auto &gbases = geom_bases();
		int dim = mesh->dimension();
		elements.setZero(gbases.size(), dim + 1);
		for (int e = 0; e < gbases.size(); e++)
		{
			int i = 0;
			for (const auto &gbs : gbases[e].bases)
				elements(e, i++) = node_to_primitive_map[gbs.global()[0].index];
		}
	}

	void State::set_mesh_vertex(int v_id, const Eigen::VectorXd &vertex)
	{
		assert(vertex.size() == mesh->dimension());
		mesh->set_point(v_id, vertex);
	}

	void State::cache_transient_adjoint_quantities(const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad)
	{
		StiffnessMatrix gradu_h(sol.size(), sol.size());
		if (current_step == 0)
			diff_cached.init(mesh->dimension(), ndof(), problem->is_time_dependent() ? args["time"]["time_steps"].get<int>() : 0);

		ipc::NormalCollisions cur_collision_set;
		ipc::SmoothCollisions cur_smooth_collision_set;
		ipc::TangentialCollisions cur_friction_set;
		ipc::NormalCollisions cur_normal_adhesion_set;
		ipc::TangentialCollisions cur_tangential_adhesion_set;

		if (optimization_enabled == solver::CacheLevel::Derivatives)
		{
			if (!problem->is_time_dependent() || current_step > 0)
				compute_force_jacobian(sol, disp_grad, gradu_h);

			if (solve_data.contact_form)
			{
				if (const auto barrier_contact = dynamic_cast<const solver::BarrierContactForm *>(solve_data.contact_form.get()))
					cur_collision_set = barrier_contact->collision_set();
				else if (const auto smooth_contact = dynamic_cast<const solver::SmoothContactForm *>(solve_data.contact_form.get()))
					cur_smooth_collision_set = smooth_contact->collision_set();
			}
			cur_friction_set = solve_data.friction_form ? solve_data.friction_form->friction_collision_set() : ipc::TangentialCollisions();
			cur_normal_adhesion_set = solve_data.normal_adhesion_form ? solve_data.normal_adhesion_form->collision_set() : ipc::NormalCollisions();
			cur_tangential_adhesion_set = solve_data.tangential_adhesion_form ? solve_data.tangential_adhesion_form->tangential_collision_set() : ipc::TangentialCollisions();
		}

		if (problem->is_time_dependent())
		{
			if (args["time"]["quasistatic"].get<bool>())
			{
				diff_cached.cache_quantities_quasistatic(current_step, sol, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_normal_adhesion_set, disp_grad);
			}
			else
			{
				Eigen::MatrixXd vel, acc;
				if (current_step == 0)
				{
					if (dynamic_cast<time_integrator::BDF *>(solve_data.time_integrator.get()))
					{
						const auto bdf_integrator = dynamic_cast<time_integrator::BDF *>(solve_data.time_integrator.get());
						vel = bdf_integrator->weighted_sum_v_prevs();
					}
					else if (dynamic_cast<time_integrator::ImplicitEuler *>(solve_data.time_integrator.get()))
					{
						const auto euler_integrator = dynamic_cast<time_integrator::ImplicitEuler *>(solve_data.time_integrator.get());
						vel = euler_integrator->v_prev();
					}
					else
						log_and_throw_error("Differentiable code doesn't support this time integrator!");

					acc.setZero(ndof(), 1);
				}
				else
				{
					vel = solve_data.time_integrator->compute_velocity(sol);
					acc = solve_data.time_integrator->compute_acceleration(vel);
				}

				diff_cached.cache_quantities_transient(current_step, solve_data.time_integrator->steps(), sol, vel, acc, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_friction_set);
			}
		}
		else
		{
			diff_cached.cache_quantities_static(sol, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_friction_set, cur_normal_adhesion_set, cur_tangential_adhesion_set, disp_grad);
		}
	}

	void State::compute_force_jacobian(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, StiffnessMatrix &hessian)
	{
		if (problem->is_time_dependent())
		{
			if (assembler->is_linear() && !is_contact_enabled())
				log_and_throw_adjoint_error("Differentiable transient linear solve is not supported!");

			StiffnessMatrix tmp_hess;
			solve_data.nl_problem->set_project_to_psd(false);
			solve_data.nl_problem->FullNLProblem::solution_changed(sol);
			solve_data.nl_problem->FullNLProblem::hessian(sol, tmp_hess);
			hessian.setZero();
			replace_rows_by_identity(hessian, tmp_hess, boundary_nodes);
		}
		else // static formulation
		{
			if (assembler->is_linear() && !is_contact_enabled() && !is_homogenization())
			{
				hessian.setZero();
				StiffnessMatrix stiffness;
				build_stiffness_mat(stiffness);
				replace_rows_by_identity(hessian, stiffness, boundary_nodes);
			}
			else
			{
				solve_data.nl_problem->set_project_to_psd(false);
				if (is_homogenization())
				{
					Eigen::VectorXd reduced;
					std::shared_ptr<solver::NLHomoProblem> homo_problem = std::dynamic_pointer_cast<solver::NLHomoProblem>(solve_data.nl_problem);
					reduced = homo_problem->full_to_reduced(sol, disp_grad);
					solve_data.nl_problem->solution_changed(reduced);
					solve_data.nl_problem->hessian(reduced, hessian);
				}
				else
				{
					StiffnessMatrix tmp_hess;
					solve_data.nl_problem->FullNLProblem::solution_changed(sol);
					solve_data.nl_problem->FullNLProblem::hessian(sol, tmp_hess);
					hessian.setZero();
					replace_rows_by_identity(hessian, tmp_hess, boundary_nodes);
				}
			}
		}
	}

	void State::compute_force_jacobian_prev(const int force_step, const int sol_step, StiffnessMatrix &hessian_prev) const
	{
		assert(force_step > 0);
		assert(force_step > sol_step);
		if (assembler->is_linear() && !is_contact_enabled())
		{
			hessian_prev = StiffnessMatrix(ndof(), ndof());
		}
		else
		{
			const Eigen::MatrixXd u = diff_cached.u(force_step);
			const Eigen::MatrixXd u_prev = diff_cached.u(sol_step);
			const double beta = time_integrator::BDF::betas(diff_cached.bdf_order(force_step) - 1);
			const double dt = solve_data.time_integrator->dt();

			hessian_prev = StiffnessMatrix(u.size(), u.size());
			if (problem->is_time_dependent())
			{
				if (solve_data.friction_form)
				{
					if (sol_step == force_step - 1)
					{
						Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(u_prev, mesh->dimension()));
						Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(u, mesh->dimension()));

						// TODO: use the time integration to compute the velocity
						const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / dt;
						const double dv_dut = -1 / dt;

						if (const auto barrier_contact = dynamic_cast<const solver::BarrierContactForm *>(solve_data.contact_form.get()))
						{
							hessian_prev =
								solve_data.friction_form->friction_potential().force_jacobian(
									diff_cached.friction_collision_set(force_step),
									collision_mesh,
									collision_mesh.rest_positions(),
									/*lagged_displacements=*/surface_solution_prev,
									surface_velocities,
									barrier_contact->barrier_potential(),
									barrier_contact->barrier_stiffness(),
									ipc::FrictionPotential::DiffWRT::LAGGED_DISPLACEMENTS)
								+ solve_data.friction_form->friction_potential().force_jacobian(
									  diff_cached.friction_collision_set(force_step),
									  collision_mesh,
									  collision_mesh.rest_positions(),
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

						hessian_prev = collision_mesh.to_full_dof(hessian_prev); // / (beta * dt) / (beta * dt);
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

				if (solve_data.tangential_adhesion_form)
				{

					if (sol_step == force_step - 1)
					{
						StiffnessMatrix adhesion_hessian_prev(u.size(), u.size());

						Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(u_prev, mesh->dimension()));
						Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(u, mesh->dimension()));

						// TODO: use the time integration to compute the velocity
						const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / dt;
						const double dv_dut = -1 / dt;

						adhesion_hessian_prev =
							solve_data.tangential_adhesion_form->tangential_adhesion_potential().force_jacobian(
								diff_cached.tangential_adhesion_collision_set(force_step),
								collision_mesh,
								collision_mesh.rest_positions(),
								/*lagged_displacements=*/surface_solution_prev,
								surface_velocities,
								solve_data.normal_adhesion_form->normal_adhesion_potential(),
								1,
								ipc::TangentialPotential::DiffWRT::LAGGED_DISPLACEMENTS)
							+ solve_data.tangential_adhesion_form->tangential_adhesion_potential().force_jacobian(
								  diff_cached.tangential_adhesion_collision_set(force_step),
								  collision_mesh,
								  collision_mesh.rest_positions(),
								  /*lagged_displacements=*/surface_solution_prev,
								  surface_velocities,
								  solve_data.normal_adhesion_form->normal_adhesion_potential(),
								  1,
								  ipc::TangentialPotential::DiffWRT::VELOCITIES)
								  * dv_dut;

						adhesion_hessian_prev *= -1;

						adhesion_hessian_prev = collision_mesh.to_full_dof(adhesion_hessian_prev); // / (beta * dt) / (beta * dt);

						hessian_prev += adhesion_hessian_prev;
					}
				}

				if (damping_assembler->is_valid() && sol_step == force_step - 1) // velocity in damping uses BDF1
				{
					utils::SparseMatrixCache mat_cache;
					StiffnessMatrix damping_hessian_prev(u.size(), u.size());
					damping_prev_assembler->assemble_hessian(mesh->is_volume(), n_bases, false, bases, geom_bases(), ass_vals_cache, force_step * args["time"]["dt"].get<double>() + args["time"]["t0"].get<double>(), dt, u, u_prev, mat_cache, damping_hessian_prev);

					hessian_prev += damping_hessian_prev;
				}

				if (sol_step == force_step - 1)
				{
					StiffnessMatrix body_force_hessian(u.size(), u.size());
					solve_data.body_form->hessian_wrt_u_prev(u_prev, force_step * dt, body_force_hessian);
					hessian_prev += body_force_hessian;
				}
			}
		}
	}

	void State::solve_adjoint_cached(const Eigen::MatrixXd &rhs)
	{
		diff_cached.cache_adjoints(solve_adjoint(rhs));
	}

	Eigen::MatrixXd State::solve_adjoint(const Eigen::MatrixXd &rhs) const
	{
		if (problem->is_time_dependent())
			return solve_transient_adjoint(rhs);
		else
			return solve_static_adjoint(rhs);
	}

	Eigen::MatrixXd State::solve_static_adjoint(const Eigen::MatrixXd &adjoint_rhs) const
	{
		Eigen::MatrixXd b = adjoint_rhs;

		Eigen::MatrixXd adjoint;
		if (lin_solver_cached)
		{
			b(boundary_nodes, Eigen::all).setZero();

			StiffnessMatrix A = diff_cached.gradu_h(0);
			const int full_size = A.rows();
			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			int precond_num = problem_dim * n_bases;

			b.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), b.cols()));

			std::vector<int> boundary_nodes_tmp;
			if (has_periodic_bc())
			{
				boundary_nodes_tmp = periodic_bc->full_to_periodic(boundary_nodes);
				precond_num = periodic_bc->full_to_periodic(A);
				b = periodic_bc->full_to_periodic(b, true);
			}
			else
				boundary_nodes_tmp = boundary_nodes;

			adjoint.setZero(ndof(), adjoint_rhs.cols());
			for (int i = 0; i < b.cols(); i++)
			{
				Eigen::VectorXd x, tmp;
				tmp = b.col(i);
				dirichlet_solve_prefactorized(*lin_solver_cached, A, tmp, boundary_nodes_tmp, x);

				if (has_periodic_bc())
					adjoint.col(i) = periodic_bc->periodic_to_full(full_size, x);
				else
					adjoint.col(i) = x;
			}
		}
		else
		{
			auto solver = polysolve::linear::Solver::create(args["solver"]["adjoint_linear"], adjoint_logger());

			StiffnessMatrix A = diff_cached.gradu_h(0); // This should be transposed, but A is symmetric in hyper-elastic and diffusion problems

			/*
			For non-periodic problems, the adjoint solution p's size is the full size in NLProblem
			For periodic problems, the adjoint solution p's size is the reduced size in NLProblem
			*/
			if (!is_homogenization())
			{
				adjoint.setZero(ndof(), adjoint_rhs.cols());
				for (int i = 0; i < b.cols(); i++)
				{
					Eigen::VectorXd tmp = b.col(i);
					tmp(boundary_nodes).setZero();

					Eigen::VectorXd x;
					x.setZero(tmp.size());
					dirichlet_solve(*solver, A, tmp, boundary_nodes, x, A.rows(), "", false, false, false);

					adjoint.col(i) = x;
					adjoint(boundary_nodes, i) = -b(boundary_nodes, i);
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

	Eigen::MatrixXd State::solve_transient_adjoint(const Eigen::MatrixXd &adjoint_rhs) const
	{
		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = 1;
		if (args["time"]["integrator"].is_string())
			bdf_order = 1;
		else if (args["time"]["integrator"]["type"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"]["type"] == "BDF")
			bdf_order = args["time"]["integrator"]["steps"].get<int>();
		else
			log_and_throw_adjoint_error("Integrator type not supported for differentiability.");

		assert(adjoint_rhs.cols() == time_steps + 1);

		const int cols_per_adjoint = time_steps + 1;
		Eigen::MatrixXd adjoints;
		adjoints.setZero(ndof(), cols_per_adjoint * 2);

		// set dirichlet rows of mass to identity
		StiffnessMatrix reduced_mass;
		replace_rows_by_identity(reduced_mass, mass, boundary_nodes);

		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		for (int i = time_steps; i >= 0; --i)
		{
			{
				sum_alpha_p.setZero(ndof(), 1);
				sum_alpha_nu.setZero(ndof(), 1);

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
				compute_force_jacobian_prev(i + j, i, gradu_h_prev);
				Eigen::VectorXd tmp = adjoints.col(i + j) * (time_integrator::BDF::betas(diff_cached.bdf_order(i + j) - 1) * dt);
				tmp(boundary_nodes).setZero();
				rhs_ += -gradu_h_prev.transpose() * tmp;
			}

			if (i > 0)
			{
				double beta_dt = time_integrator::BDF::betas(diff_cached.bdf_order(i) - 1) * dt;

				rhs_ += (1. / beta_dt) * (diff_cached.gradu_h(i) - reduced_mass).transpose() * sum_alpha_p;

				{
					StiffnessMatrix A = diff_cached.gradu_h(i).transpose();
					Eigen::VectorXd b_ = rhs_;
					b_(boundary_nodes).setZero();

					auto solver = polysolve::linear::Solver::create(args["solver"]["adjoint_linear"], adjoint_logger());

					Eigen::VectorXd x;
					dirichlet_solve(*solver, A, b_, boundary_nodes, x, A.rows(), "", false, false, false);
					adjoints.col(i + cols_per_adjoint) = x;
				}

				// TODO: generalize to BDFn
				Eigen::VectorXd tmp = rhs_(boundary_nodes);
				if (i + 1 < cols_per_adjoint)
					tmp += (-2. / beta_dt) * adjoints(boundary_nodes, i + 1);
				if (i + 2 < cols_per_adjoint)
					tmp += (1. / beta_dt) * adjoints(boundary_nodes, i + 2);

				tmp -= (diff_cached.gradu_h(i).transpose() * adjoints.col(i + cols_per_adjoint))(boundary_nodes);
				adjoints(boundary_nodes, i + cols_per_adjoint) = tmp;
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

	void State::compute_surface_node_ids(const int surface_selection, std::vector<int> &node_ids) const
	{
		node_ids = {};

		const auto &gbases = geom_bases();
		for (const auto &lb : total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

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

	void State::compute_total_surface_node_ids(std::vector<int> &node_ids) const
	{
		node_ids = {};

		const auto &gbases = geom_bases();
		for (const auto &lb : total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					const int g_id = gbases[e].bases[nodes(n)].global()[0].index;

					if (std::count(node_ids.begin(), node_ids.end(), g_id) == 0)
						node_ids.push_back(g_id);
				}
			}
		}
	}

	void State::compute_volume_node_ids(const int volume_selection, std::vector<int> &node_ids) const
	{
		node_ids = {};

		const auto &gbases = geom_bases();
		for (int e = 0; e < gbases.size(); e++)
		{
			const int body_id = mesh->get_body_id(e);
			if (body_id == volume_selection)
				for (const auto &gbs : gbases[e].bases)
					for (const auto &g : gbs.global())
						node_ids.push_back(g.index);
		}
	}

} // namespace polyfem
