#include "L2Projection.hpp"

#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/forms/InversionBarrierForm.hpp>
#include <polyfem/solver/forms/L2ProjectionForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/PardisoSupport>

namespace polyfem::mesh
{
	Eigen::MatrixXd unconstrained_L2_projection(
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::MatrixXd &y)
	{
		// Construct a linear solver for M
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> solver;
		solver.analyzePattern(M);
		solver.factorize(M);

		const Eigen::MatrixXd rhs = A * y;
		const Eigen::MatrixXd x = solver.solve(rhs);

		double residual_error = (M * x - rhs).norm();
		logger().debug("residual error in L2 projection: {}", residual_error);
		assert(residual_error < 1e-12);

		return x;
	}

	Eigen::VectorXd constrained_L2_projection(
		// L2 projection form
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::VectorXd &y,
		// Inversion-free form
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
		const Obstacle &obstacle,
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

		forms.push_back(std::make_shared<InversionBarrierForm>(elements, dim, /*vhat=*/dhat));
		forms.back()->set_weight(barrier_stiffness); // use same weight as barrier stiffness

		forms.push_back(std::make_shared<ContactForm>(
			collision_mesh, dhat, /*avg_mass=*/1.0, use_convergent_formulation,
			/*use_adaptive_barrier_stiffness=*/false, /*is_time_dependent=*/true,
			broad_phase_method, ccd_tolerance, ccd_max_iterations));
		forms.back()->set_weight(barrier_stiffness);

		const int ndof = x0.size();
		std::shared_ptr<ALForm> al_form = std::make_shared<ALForm>(
			ndof, boundary_nodes, M, obstacle, target_x);
		forms.push_back(al_form);

		// --------------------------------------------------------------------

		StaticBoundaryNLProblem problem(ndof, boundary_nodes, target_x, forms);

		// --------------------------------------------------------------------

		// Create Newton solver
		using NLSolver = cppoptlib::NonlinearSolver<decltype(problem)>;
		std::shared_ptr<NLSolver> nl_solver;
		{
			// TODO: expose these parameters
			const json newton_args = R"({
				"f_delta": 1e-7,
				"grad_norm": 1e-7,
				"use_grad_norm": true,
				"first_grad_norm_tol": 1e-10,
				"max_iterations": 100,
				"relative_gradient": false,
				"line_search": {
					"method": "backtracking",
					"use_grad_norm_tol": 0.0001
				}
			})"_json;
			const json linear_solver_args = R"({
				"solver": "Eigen::PardisoLDLT",
				"precond": "Eigen::IdentityPreconditioner"
			})"_json;
			using NewtonSolver = cppoptlib::SparseNewtonDescentSolver<decltype(problem)>;
			nl_solver = std::make_shared<NewtonSolver>(newton_args, linear_solver_args);
		}

		// --------------------------------------------------------------------

		// TODO: Make these parameters
		const double al_initial_weight = 0.5;
		const double al_scaling = 10.0;
		const int al_max_steps = 20;
		const bool force_al = false;

		// Create augmented Lagrangian solver
		ALSolver<StaticBoundaryNLProblem> al_solver = ALSolver<StaticBoundaryNLProblem>(
			nl_solver, al_form, al_initial_weight, al_scaling, al_max_steps,
			/*update_barrier_stiffness=*/[&](const Eigen::MatrixXd &x) {});

		Eigen::MatrixXd sol = x0;
		al_solver.solve(problem, sol, force_al);

		return sol;
	}
} // namespace polyfem::mesh