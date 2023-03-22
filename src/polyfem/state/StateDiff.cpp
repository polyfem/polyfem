#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/solver/NLProblem.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/utils/local_to_global.hpp>

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
		elements.setZero(mesh->n_elements(), mesh->dimension() + 1);

		for (int e = 0; e < mesh->n_elements(); e++)
			for (int i = 0; i < mesh->dimension() + 1; i++)
				elements(e, i) = mesh->element_vertex(e, i);
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
			diff_cached.init(ndof(), problem->is_time_dependent() ? args["time"]["time_steps"].get<int>() : 0);
		if (!problem->is_time_dependent() || current_step > 0)
			compute_force_hessian(sol, gradu_h);

		StiffnessMatrix tmp = gradu_h;
		gradu_h.setZero();
		replace_rows_by_identity(gradu_h, tmp, boundary_nodes);

		auto cur_contact_set = solve_data.contact_form ? solve_data.contact_form->get_constraint_set() : ipc::Constraints();
		auto cur_friction_set = solve_data.friction_form ? solve_data.friction_form->get_friction_constraint_set() : ipc::FrictionConstraints();

		if (problem->is_time_dependent())
		{
			Eigen::MatrixXd vel, acc;
			if (current_step == 0)
			{
				if (dynamic_cast<time_integrator::BDF*>(solve_data.time_integrator.get()))
				{
					const auto bdf_integrator = dynamic_cast<time_integrator::BDF*>(solve_data.time_integrator.get());
					vel = bdf_integrator->weighted_sum_v_prevs();
				}
				else if (dynamic_cast<time_integrator::ImplicitEuler*>(solve_data.time_integrator.get()))
				{
					const auto euler_integrator = dynamic_cast<time_integrator::ImplicitEuler*>(solve_data.time_integrator.get());
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

			diff_cached.cache_quantities_transient(current_step, solve_data.time_integrator->steps(), sol, vel, acc, gradu_h, cur_contact_set, cur_friction_set);
		}
		else
			diff_cached.cache_quantities_static(sol, gradu_h, cur_contact_set, cur_friction_set);
	}

	void State::compute_force_hessian(const Eigen::MatrixXd &sol, StiffnessMatrix &hessian) const
	{
		if (assembler.is_linear(formulation()) && !is_contact_enabled())
		{
			hessian = stiffness;
		}
		else
		{
			solve_data.nl_problem->set_project_to_psd(false);
			solve_data.nl_problem->FullNLProblem::solution_changed(sol);
			solve_data.nl_problem->FullNLProblem::hessian(sol, hessian);
		}
	}

	void State::compute_force_hessian_prev(const int step, StiffnessMatrix &hessian_prev) const
	{
		assert(step > 0);
		if (assembler.is_linear(formulation()) && !is_contact_enabled())
		{
			hessian_prev = StiffnessMatrix(stiffness.rows(), stiffness.cols());
		}
		else
		{
			const Eigen::MatrixXd u = diff_cached.u(step);
			const Eigen::MatrixXd u_prev = diff_cached.u(step - 1);
			const double beta = time_integrator::BDF::betas(diff_cached.bdf_order(step) - 1);
			const double dt = solve_data.time_integrator->dt();
			
			hessian_prev = StiffnessMatrix(u.size(), u.size());
			if (problem->is_time_dependent())
			{
				if (solve_data.friction_form)
				{
					Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(u_prev, mesh->dimension()));
					Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(u, mesh->dimension()));

					hessian_prev = -ipc::compute_friction_force_jacobian(
						collision_mesh,
						collision_mesh.vertices_at_rest(),
						surface_solution_prev,
						surface_solution,
						diff_cached.friction_constraint_set(step),
						solve_data.contact_form->dhat(), solve_data.contact_form->barrier_stiffness(), solve_data.friction_form->epsv_dt(),
						ipc::FrictionConstraint::DiffWRT::Ut);
					hessian_prev = collision_mesh.to_full_dof(hessian_prev) / (beta * dt) / (beta * dt);
				}

				if (assembler.has_damping())
				{
					utils::SpareMatrixCache mat_cache;
					StiffnessMatrix damping_hessian_prev(u.size(), u.size());
					assembler.assemble_energy_hessian("DampingPrev", mesh->is_volume(), n_bases, false, bases, geom_bases(), ass_vals_cache, dt, u, u_prev, mat_cache, damping_hessian_prev);

					hessian_prev += damping_hessian_prev;
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
		for (int i : boundary_nodes)
			b.row(i).setZero();

		Eigen::MatrixXd adjoint;
		adjoint.setZero(ndof(), adjoint_rhs.cols());
		if (lin_solver_cached)
		{
			StiffnessMatrix A = diff_cached.gradu_h(0);
			const int full_size = A.rows();
			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			int precond_num = problem_dim * n_bases;
			apply_lagrange_multipliers(A);

			b.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), b.cols()));

			std::vector<int> boundary_nodes_tmp = boundary_nodes;
			full_to_periodic(boundary_nodes_tmp);
			if (need_periodic_reduction())
			{
				precond_num = full_to_periodic(A);
				Eigen::MatrixXd tmp = b;
				full_to_periodic(tmp, true);
				b = tmp;
			}

			for (int i = 0; i < b.cols(); i++)
			{
				Eigen::VectorXd x, tmp;
				tmp = b.col(i);
				dirichlet_solve_prefactorized(*lin_solver_cached, A, tmp, boundary_nodes_tmp, x);
				x.conservativeResize(x.size() - n_lagrange_multipliers());

				if (need_periodic_reduction())
					adjoint.col(i) = periodic_to_full(full_size, x);
				else
					adjoint.col(i) = x;
			}
		}
		else
		{
			auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["adjoint_solver"], args["solver"]["linear"]["precond"]);
			solver->setParameters(args["solver"]["linear"]);

			StiffnessMatrix A;
			solve_data.nl_problem->full_hessian_to_reduced_hessian(diff_cached.gradu_h(0), A);
			solver->analyzePattern(A, A.rows());
			solver->factorize(A);

			for (int i = 0; i < b.cols(); i++)
			{
				Eigen::MatrixXd tmp = solve_data.nl_problem->full_to_reduced_grad(b.col(i));

				Eigen::VectorXd x;
				x.setZero(tmp.size());
				// dirichlet_solve(*solver, A, tmp, {}, x, A.rows(), "", false, false, false);
				solver->solve(tmp, x);
				x.conservativeResize(x.size() - n_lagrange_multipliers());

				adjoint.col(i) = solve_data.nl_problem->reduced_to_full(x);
			}
			adjoint(boundary_nodes, Eigen::all).setZero();
		}

		return adjoint;
	}

	Eigen::MatrixXd State::solve_transient_adjoint(const Eigen::MatrixXd &adjoint_rhs) const
	{
		const int bdf_order = get_bdf_order();
		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

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

				Eigen::VectorXd bdf_coeffs(num);
				for (int j = 0; j < bdf_order && i + j < time_steps; ++j)
					bdf_coeffs(j) = -time_integrator::BDF::alphas(std::min(bdf_order - 1, i + j))[j];

				sum_alpha_p = adjoints.middleCols(i + 1, num) * bdf_coeffs;
				sum_alpha_nu = adjoints.middleCols(cols_per_adjoint + i + 1, num) * bdf_coeffs;
			}

			Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu - adjoint_rhs.col(i);
			if (i < time_steps) 
			{
				StiffnessMatrix gradu_h_prev;
				compute_force_hessian_prev(i + 1, gradu_h_prev);
				Eigen::VectorXd tmp = adjoints.col(i + 1) * (time_integrator::BDF::betas(diff_cached.bdf_order(i + 1) - 1) * dt);
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

					auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["adjoint_solver"], args["solver"]["linear"]["precond"]);
					solver->setParameters(args["solver"]["linear"]);

					Eigen::VectorXd x;
					dirichlet_solve(*solver, A, b_, boundary_nodes, x, A.rows(), "", false, false, false);
					adjoints.col(i + cols_per_adjoint) = x;
				}

				// TODO: generalize to BDFn
				Eigen::VectorXd tmp = rhs_(boundary_nodes);
				if (i + 1 < cols_per_adjoint)
					tmp += -2. / beta_dt * adjoints(boundary_nodes, i + 1);
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
