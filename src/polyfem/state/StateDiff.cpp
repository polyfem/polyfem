#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>

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
		void solve_zero_dirichlet(const json &args, StiffnessMatrix &A, Eigen::VectorXd &b, const std::vector<int> &indices, Eigen::MatrixXd &adjoint_solution)
		{
			auto solver = polysolve::LinearSolver::create(args["solver"], args["precond"]);
			solver->setParameters(args);
			const int precond_num = A.rows();

			for (int i : indices)
				b(i) = 0;

			Eigen::Vector4d adjoint_spectrum;
			Eigen::VectorXd x;
			adjoint_spectrum = dirichlet_solve(*solver, A, b, indices, x, precond_num, "", false, false, false);
			adjoint_solution = x;
		}

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

		StiffnessMatrix replace_rows_by_zero(const StiffnessMatrix &mat, const std::vector<int> &rows)
		{
			StiffnessMatrix reduced_mat;
			reduced_mat.resize(mat.rows(), mat.cols());

			std::vector<bool> mask(mat.rows(), false);
			for (int i : rows)
				mask[i] = true;

			std::vector<Eigen::Triplet<double>> coeffs;
			for (int k = 0; k < mat.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mat, k); it; ++it)
				{
					if (!mask[it.row()])
						coeffs.emplace_back(it.row(), it.col(), it.value());
				}
			}
			reduced_mat.setFromTriplets(coeffs.begin(), coeffs.end());

			return reduced_mat;
		}
	} // namespace

	void State::get_vf(Eigen::MatrixXd &vertices, Eigen::MatrixXi &faces, const bool geometric) const
	{
		const auto &cur_bases = geometric ? geom_bases() : bases;
		const int n_elements = int(cur_bases.size());
		vertices = Eigen::MatrixXd::Zero(geometric ? n_geom_bases : n_bases, mesh->dimension());
		faces = Eigen::MatrixXi::Zero(cur_bases.size(), cur_bases[0].bases.size());
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < n_elements; ++e)
		{
			const int n_loc_bases = cur_bases[e].bases.size();
			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto &v = cur_bases[e].bases[i];
				assert(v.global().size() == 1);

				vertices.row(v.global()[0].index) = v.global()[0].node;
				faces(e, i) = v.global()[0].index;
			}
		}
	}

	void State::set_mesh_vertices(const Eigen::MatrixXd &vertices)
	{
		assert(vertices.cols() == mesh->dimension());

		const auto &primitive_to_node = iso_parametric() ? primitive_to_bases_node : primitive_to_geom_bases_node;
		for (int v = 0; v < mesh->n_vertices(); v++)
			if (primitive_to_node[v] >= 0 && primitive_to_node[v] < vertices.rows())
				mesh->set_point(v, vertices.block(primitive_to_node[v], 0, 1, mesh->dimension()));
	}

	void State::cache_transient_adjoint_quantities(const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad)
	{
		adjoint_solved_ = false;

		StiffnessMatrix gradu_h(sol.size(), sol.size()), gradu_h_prev(sol.size(), sol.size());
		if (current_step == 0)
			diff_cached.clear();
		if (!problem->is_time_dependent() || current_step > 0)
			compute_force_hessian(sol, gradu_h, gradu_h_prev);
		if (diff_cached.size() > 0)
			diff_cached.back().gradu_h_next = replace_rows_by_zero(gradu_h_prev, boundary_nodes);

		StiffnessMatrix tmp = gradu_h;
		gradu_h.setZero();
		replace_rows_by_identity(gradu_h, tmp, boundary_nodes);

		auto cur_contact_set = solve_data.contact_form ? solve_data.contact_form->get_constraint_set() : ipc::Constraints();
		auto cur_friction_set = solve_data.friction_form ? solve_data.friction_form->get_friction_constraint_set() : ipc::FrictionConstraints();
		diff_cached.push_back({gradu_h, StiffnessMatrix(sol.size(), sol.size()), sol, disp_grad, cur_contact_set, cur_friction_set, Eigen::MatrixXd(), Eigen::MatrixXd()});
	}

	void State::compute_force_hessian(const Eigen::MatrixXd &sol, StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev) const
	{
		if (assembler.is_linear(formulation()) && !is_contact_enabled())
		{
			hessian = stiffness;
			hessian_prev = StiffnessMatrix(stiffness.rows(), stiffness.cols());
		}
		else
		{
			solve_data.nl_problem->set_project_to_psd(false);
			solve_data.nl_problem->FullNLProblem::solution_changed(sol);
			solve_data.nl_problem->FullNLProblem::hessian(sol, hessian);
			// if (problem->is_time_dependent())
			// {
			// 	hessian -= mass;
			// 	hessian /= sqrt(solve_data.time_integrator->acceleration_scaling());
			// }

			hessian_prev = StiffnessMatrix(sol.size(), sol.size());
			if (problem->is_time_dependent() && diff_cached.size() > 0)
			{
				if (solve_data.friction_form)
				{
					Eigen::MatrixXd displaced = boundary_nodes_pos + utils::unflatten(sol, mesh->dimension());

					Eigen::MatrixXd displaced_prev;
					if (diff_cached.size())
						displaced_prev = boundary_nodes_pos + utils::unflatten(diff_cached.back().u, mesh->dimension());
					else
						displaced_prev = displaced;

					Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(displaced_prev - boundary_nodes_pos);
					Eigen::MatrixXd surface_solution = collision_mesh.vertices(displaced - boundary_nodes_pos);

					hessian_prev = -ipc::compute_friction_force_jacobian(
						collision_mesh,
						collision_mesh.vertices_at_rest(),
						surface_solution_prev,
						surface_solution,
						solve_data.friction_form->get_friction_constraint_set(),
						solve_data.contact_form->dhat(), solve_data.contact_form->barrier_stiffness(), solve_data.friction_form->epsv_dt(),
						ipc::FrictionConstraint::DiffWRT::Ut);
					hessian_prev = collision_mesh.to_full_dof(hessian_prev) / solve_data.time_integrator->acceleration_scaling();
				}

				if (assembler.has_damping())
				{
					utils::SpareMatrixCache mat_cache;
					StiffnessMatrix damping_hessian_prev(sol.size(), sol.size());
					assembler.assemble_energy_hessian("DampingPrev", mesh->is_volume(), n_bases, false, bases, geom_bases(), ass_vals_cache, solve_data.time_integrator->dt(), sol, diff_cached.back().u, mat_cache, damping_hessian_prev);

					hessian_prev += damping_hessian_prev;
				}
			}
		}
	}

	void State::solve_adjoint(const Eigen::MatrixXd &rhs)
	{
		if (problem->is_time_dependent())
		{
			solve_transient_adjoint(rhs);
		}
		else
		{
			assert(rhs.cols() == 1);
			solve_static_adjoint(Eigen::VectorXd(rhs));
		}
	}

	void State::solve_static_adjoint(const Eigen::VectorXd &adjoint_rhs)
	{
		Eigen::VectorXd b = adjoint_rhs;
		for (int i : boundary_nodes)
			b(i) = 0;

		if (lin_solver_cached)
		{
			StiffnessMatrix A = diff_cached[0].gradu_h;
			const int full_size = A.rows();
			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			int precond_num = problem_dim * n_bases;
			apply_lagrange_multipliers(A);

			b.conservativeResizeLike(Eigen::VectorXd::Zero(A.rows()));

			std::vector<int> boundary_nodes_tmp = boundary_nodes;
			full_to_periodic(boundary_nodes_tmp);
			if (need_periodic_reduction())
			{
				precond_num = full_to_periodic(A);
				Eigen::MatrixXd tmp = b;
				full_to_periodic(tmp, true);
				b = tmp;
			}
			
			Eigen::VectorXd x;
			dirichlet_solve_prefactorized(*lin_solver_cached, A, b, boundary_nodes_tmp, x);
			x.conservativeResize(x.size() - n_lagrange_multipliers());

			if (need_periodic_reduction())
				diff_cached[0].p = periodic_to_full(full_size, x);
			else
				diff_cached[0].p = x;
		}
		else
		{
			auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["adjoint_solver"], args["solver"]["linear"]["precond"]);
			solver->setParameters(args["solver"]["linear"]);

			StiffnessMatrix A;
			solve_data.nl_problem->full_hessian_to_reduced_hessian(diff_cached[0].gradu_h, A);
			b = solve_data.nl_problem->full_to_reduced_grad(b);

			Eigen::VectorXd x;
			dirichlet_solve(*solver, A, b, {}, x, A.rows(), "", false, false, false);
			x.conservativeResize(x.size() - n_lagrange_multipliers());

			diff_cached[0].p = solve_data.nl_problem->reduced_to_full(x);
			for (int i : boundary_nodes)
				diff_cached[0].p(i) = 0;
		}

		adjoint_solved_ = true;
	}

	void State::solve_transient_adjoint(const Eigen::MatrixXd &adjoint_rhs)
	{
		const int bdf_order = get_bdf_order();
		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		assert(adjoint_rhs.cols() == time_steps + 1);

		std::vector<Eigen::MatrixXd> adjoint_p, adjoint_nu;
		adjoint_p.assign(time_steps + 2, Eigen::MatrixXd::Zero(diff_cached[0].u.size(), 1));
		adjoint_nu.assign(time_steps + 2, Eigen::MatrixXd::Zero(diff_cached[0].u.size(), 1));

		// set dirichlet rows of mass to identity
		StiffnessMatrix reduced_mass;
		replace_rows_by_identity(reduced_mass, mass, boundary_nodes);

		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		for (int i = time_steps; i >= 0; --i)
		{
			double beta, beta_dt;
			{
				int order = std::min(bdf_order, i);
				if (order >= 1)
					beta = time_integrator::BDF::betas(order - 1);
				else
					beta = std::nan("");
				beta_dt = beta * dt;
			}
			
			{
				sum_alpha_p.setZero(adjoint_p[0].size(), 1);
				sum_alpha_nu.setZero(adjoint_nu[0].size(), 1);

				int num = std::min(bdf_order, time_steps - i);
				for (int j = 0; j < num; ++j)
				{
					int order = std::min(bdf_order - 1, i + j);
					sum_alpha_p -= time_integrator::BDF::alphas(order)[j] * adjoint_p[i + j + 1];
					sum_alpha_nu -= time_integrator::BDF::alphas(order)[j] * adjoint_nu[i + j + 1];
				}
			}

			StiffnessMatrix gradu_h_next = -beta_dt * diff_cached[i].gradu_h_next;

			if (i > 0)
			{
				Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu + (1. / beta_dt) * (diff_cached[i].gradu_h - reduced_mass).transpose() * sum_alpha_p + gradu_h_next.transpose() * adjoint_p[i + 1] - adjoint_rhs.col(i);
				
				// TODO: generalize to BDFn
				for (const auto &b : boundary_nodes)
				{
					rhs_(b) += (1. / beta_dt) * (-2 * adjoint_p[i + 1](b));
					if ((i + 2) < adjoint_p.size() - 1)
						rhs_(b) += (1. / beta_dt) * adjoint_p[i + 2](b);
				}
				
				{
					StiffnessMatrix A = diff_cached[i].gradu_h.transpose();
					Eigen::VectorXd b_ = rhs_;
					solve_zero_dirichlet(args["solver"]["linear"], A, b_, boundary_nodes, adjoint_nu[i]);
				}

				Eigen::VectorXd tmp = rhs_ - diff_cached[i].gradu_h.transpose() * adjoint_nu[i];
				for (const auto &b : boundary_nodes)
					adjoint_nu[i](b) = tmp(b);
				adjoint_p[i] = beta_dt * adjoint_nu[i] - sum_alpha_p;
			}
			else
			{
				adjoint_p[i] = -reduced_mass.transpose() * sum_alpha_p;
				adjoint_nu[i] = -adjoint_rhs.col(i) - reduced_mass.transpose() * sum_alpha_nu - gradu_h_next * adjoint_p[i + 1]; // adjoint_nu[0] actually stores adjoint_mu[0]
			}
			diff_cached[i].p = adjoint_p[i];
			diff_cached[i].nu = adjoint_nu[i];
		}

		adjoint_solved_ = true;
	}

} // namespace polyfem
