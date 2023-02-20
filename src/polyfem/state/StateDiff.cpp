#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/time_integrator/ImplicitEuler.hpp>

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

		const auto &primitive_to_node = iso_parametric() ? mesh_nodes->primitive_to_node() : geom_mesh_nodes->primitive_to_node();
		for (int v = 0; v < mesh->n_vertices(); v++)
			if (primitive_to_node[v] >= 0 && primitive_to_node[v] < vertices.rows())
				mesh->set_point(v, vertices.block(primitive_to_node[v], 0, 1, mesh->dimension()));
	}

	void State::cache_transient_adjoint_quantities(const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad)
	{
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

		Eigen::MatrixXd vel, acc;
		if (problem->is_time_dependent())
		{
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
		}

		auto cur_contact_set = solve_data.contact_form ? solve_data.contact_form->get_constraint_set() : ipc::Constraints();
		auto cur_friction_set = solve_data.friction_form ? solve_data.friction_form->get_friction_constraint_set() : ipc::FrictionConstraints();
		diff_cached.push_back({gradu_h, StiffnessMatrix(sol.size(), sol.size()), sol, vel, acc, disp_grad, cur_contact_set, cur_friction_set});
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
					Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(diff_cached.back().u, mesh->dimension()));
					Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(sol, mesh->dimension()));

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

	Eigen::MatrixXd State::solve_adjoint(const Eigen::MatrixXd &rhs) const
	{
		if (problem->is_time_dependent())
		{
			return solve_transient_adjoint(rhs);
		}
		else
		{
			return solve_static_adjoint(rhs);
		}
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
			StiffnessMatrix A = diff_cached[0].gradu_h;
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
			solve_data.nl_problem->full_hessian_to_reduced_hessian(diff_cached[0].gradu_h, A);
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
				for (int j : boundary_nodes)
					adjoint(j, i) = 0;
			}
		}

		return adjoint;
	}

	Eigen::MatrixXd State::solve_transient_adjoint(const Eigen::MatrixXd &adjoint_rhs) const
	{
		const int bdf_order = get_bdf_order();
		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		assert(adjoint_rhs.cols() == time_steps + 1);

		Eigen::MatrixXd adjoints;
		adjoints.setZero(ndof(), (time_steps + 2) * 2);

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
				sum_alpha_p.setZero(ndof(), 1);
				sum_alpha_nu.setZero(ndof(), 1);

				int num = std::min(bdf_order, time_steps - i);
				for (int j = 0; j < num; ++j)
				{
					int order = std::min(bdf_order - 1, i + j);
					sum_alpha_p -= time_integrator::BDF::alphas(order)[j] * adjoints.col((i + j + 1) * 2);
					sum_alpha_nu -= time_integrator::BDF::alphas(order)[j] * adjoints.col((i + j + 1) * 2 + 1);
				}
			}

			StiffnessMatrix gradu_h_next = -beta_dt * diff_cached[i].gradu_h_next;

			if (i > 0)
			{
				Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu + (1. / beta_dt) * (diff_cached[i].gradu_h - reduced_mass).transpose() * sum_alpha_p + gradu_h_next.transpose() * adjoints.col((i+1)*2) - adjoint_rhs.col(i);
				
				// TODO: generalize to BDFn
				for (const auto &b : boundary_nodes)
				{
					rhs_(b) += -2. / beta_dt * adjoints(b, (i+1)*2);
					if ((i+2)*2 < adjoints.cols())
						rhs_(b) += (1. / beta_dt) * adjoints(b, (i+2)*2);
				}
				
				{
					StiffnessMatrix A = diff_cached[i].gradu_h.transpose();
					Eigen::VectorXd b_ = rhs_;
					Eigen::MatrixXd x;
					solve_zero_dirichlet(args["solver"]["linear"], A, b_, boundary_nodes, x);
					adjoints.col(2*i+1) = x;
				}

				Eigen::VectorXd tmp = rhs_ - diff_cached[i].gradu_h.transpose() * adjoints.col(2*i+1);
				for (const auto &b : boundary_nodes)
					adjoints(b, 2*i+1) = tmp(b);
				adjoints.col(2*i) = beta_dt * adjoints.col(2*i+1) - sum_alpha_p;
			}
			else
			{
				adjoints.col(2*i) = -reduced_mass.transpose() * sum_alpha_p;
				adjoints.col(2*i+1) = -adjoint_rhs.col(i) - reduced_mass.transpose() * sum_alpha_nu - gradu_h_next * adjoints.col((i+1)*2); // adjoint_nu[0] actually stores adjoint_mu[0]
			}
		}
		return adjoints.block(0, 0, adjoints.rows(), adjoints.cols() - 2);
	}

} // namespace polyfem
