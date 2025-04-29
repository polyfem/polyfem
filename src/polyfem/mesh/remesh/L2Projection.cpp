#include "L2Projection.hpp"

#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/problems/StaticBoundaryNLProblem.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/InversionBarrierForm.hpp>
#include <polyfem/solver/forms/L2ProjectionForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <ipc/ipc.hpp>

#include <polysolve/linear/Solver.hpp>

namespace polyfem::mesh
{
	Eigen::MatrixXd unconstrained_L2_projection(
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::Ref<const Eigen::MatrixXd> &y)
	{
		// Construct a linear solver for M
		std::unique_ptr<polysolve::linear::Solver> solver;
#ifdef POLYSOLVE_WITH_MKL
		solver = polysolve::linear::Solver::create("Eigen::PardisoLDLT", "");
#elif defined(POLYSOLVE_WITH_CHOLMOD)
		solver = polysolve::linear::Solver::create("Eigen::CholmodSimplicialLDLT", "");
#else
		solver = polysolve::linear::Solver::create("Eigen::SimplicialLDLT", "");
#endif

		solver->analyze_pattern(M, 0);
		solver->factorize(M);

		const Eigen::MatrixXd rhs = A * y;
		Eigen::MatrixXd x(rhs.rows(), rhs.cols());
		for (int i = 0; i < x.cols(); ++i)
			solver->solve(rhs.col(i), x.col(i));

		double residual_error = (M * x - rhs).norm();
		logger().debug("residual error in L2 projection: {}", residual_error);
		assert(residual_error < std::max(1e-12 * rhs.norm(), 1e-12));

		return x;
	}

	void reduced_L2_projection(
		const Eigen::MatrixXd &M,
		const Eigen::MatrixXd &A,
		const Eigen::Ref<const Eigen::MatrixXd> &y,
		const std::vector<int> &boundary_nodes,
		Eigen::Ref<Eigen::MatrixXd> x)
	{
		std::vector<int> free_nodes;
		for (int i = 0, j = 0; i < y.rows(); ++i)
		{
			if (boundary_nodes[j] == i)
				++j;
			else
				free_nodes.push_back(i);
		}

		const Eigen::MatrixXd H = M(free_nodes, free_nodes);

		const Eigen::MatrixXd g = -((M * x - A * y)(free_nodes, Eigen::all));
		const Eigen::MatrixXd sol = H.llt().solve(g);
		x(free_nodes, Eigen::all) += sol;
	}

	Eigen::VectorXd constrained_L2_projection(
		// Nonlinear solver
		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver,
		// L2 projection form
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::VectorXd &y,
		// Inversion-free form
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXi &elements,
		const int dim,
		// Contact form
		const ipc::CollisionMesh &collision_mesh,
		const double dhat,
		const double barrier_stiffness,
		const bool use_convergent_formulation,
		const ipc::BroadPhaseMethod broad_phase_method,
		const double ccd_tolerance,
		const int ccd_max_iterations,
		// Augmented lagrangian form
		const std::vector<int> &boundary_nodes,
		const size_t obstacle_ndof,
		const Eigen::VectorXd &target_x,
		// Initial guess
		const Eigen::VectorXd &x0)
	{
		using namespace polyfem::solver;

		assert(M.rows() == M.cols());
		assert(A.rows() == M.rows());
		assert(A.cols() == y.size());

		std::vector<std::shared_ptr<Form>> forms;

		forms.push_back(std::make_shared<L2ProjectionForm>(M, A, y));

		forms.push_back(std::make_shared<InversionBarrierForm>(
			rest_positions, elements, dim, /*vhat=*/1e-12));
		forms.back()->set_weight(barrier_stiffness); // use same weight as barrier stiffness
		assert(forms.back()->is_step_valid(x0, x0));

		if (collision_mesh.num_vertices() != 0)
		{
			forms.push_back(std::make_shared<BarrierContactForm>(
				collision_mesh, dhat, /*avg_mass=*/1.0, use_convergent_formulation,
				/*use_adaptive_barrier_stiffness=*/false, /*is_time_dependent=*/true,
				/*enable_shape_derivatives=*/false, broad_phase_method, ccd_tolerance,
				ccd_max_iterations));
			forms.back()->set_weight(barrier_stiffness);
			assert(!ipc::has_intersections(collision_mesh, collision_mesh.displace_vertices(utils::unflatten(x0, dim))));
		}

		const int ndof = x0.size();

		std::shared_ptr<BCLagrangianForm> bc_lagrangian_form = std::make_shared<BCLagrangianForm>(
			ndof, boundary_nodes, M, obstacle_ndof, target_x);
		forms.push_back(bc_lagrangian_form);

		// --------------------------------------------------------------------

		std::vector<std::shared_ptr<AugmentedLagrangianForm>> bc_forms;
		bc_forms.push_back(bc_lagrangian_form);

		StaticBoundaryNLProblem problem(ndof, target_x, forms, bc_forms);

		// --------------------------------------------------------------------

		// Create augmented Lagrangian solver
		// AL parameters
		constexpr double al_initial_weight = 1e6;
		constexpr double al_scaling = 2.0;
		constexpr int al_max_weight = 100 * al_initial_weight;
		constexpr double al_eta_tol = 0.99;
		constexpr size_t al_max_solver_iter = 1000;
		ALSolver al_solver(
			bc_forms, al_initial_weight,
			al_scaling, al_max_weight, al_eta_tol,
			/*update_barrier_stiffness=*/[&](const Eigen::MatrixXd &x) {});

		Eigen::MatrixXd sol = x0;

		const size_t default_max_iterations = nl_solver->stop_criteria().iterations;
		nl_solver->stop_criteria().iterations = al_max_solver_iter;
		al_solver.solve_al(nl_solver, problem, sol);

		nl_solver->stop_criteria().iterations = default_max_iterations;
		al_solver.solve_reduced(nl_solver, problem, sol);

#ifndef NDEBUG
		assert(forms[1]->is_step_valid(sol, sol)); // inversion-free
		if (collision_mesh.num_vertices() != 0)
			assert(!ipc::has_intersections(collision_mesh, collision_mesh.displace_vertices(utils::unflatten(sol, dim))));
#endif

		return sol;
	}
} // namespace polyfem::mesh