#include "L2Projection.hpp"

#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/problems/StaticBoundaryNLProblem.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/forms/InversionBarrierForm.hpp>
#include <polyfem/solver/forms/L2ProjectionForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#ifdef POLYSOLVE_WITH_MKL
#include <Eigen/PardisoSupport>
#else
#include <Eigen/CholmodSupport>
#endif

namespace polyfem::mesh
{
	Eigen::MatrixXd unconstrained_L2_projection(
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::MatrixXd &y)
	{
		// Construct a linear solver for M
#ifdef POLYSOLVE_WITH_MKL
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> solver;
#else
		Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrix<double>> solver;
#endif
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
		// Nonlinear solver
		std::shared_ptr<cppoptlib::NonlinearSolver<polyfem::solver::NLProblem>> nl_solver,
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
		const Obstacle &obstacle,
		const Eigen::VectorXd &target_x,
		// Initial guess
		const Eigen::VectorXd &x0,
		// AL parameters
		const double al_initial_weight,
		const double al_scaling,
		const int al_max_steps,
		const bool force_al)
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

		forms.push_back(std::make_shared<ContactForm>(
			collision_mesh, dhat, /*avg_mass=*/1.0, use_convergent_formulation,
			/*use_adaptive_barrier_stiffness=*/false, /*is_time_dependent=*/true,
			broad_phase_method, ccd_tolerance, ccd_max_iterations));
		forms.back()->set_weight(barrier_stiffness);
		assert(!ipc::has_intersections(collision_mesh, collision_mesh.displace_vertices(utils::unflatten(x0, dim))));

		const int ndof = x0.size();
		std::shared_ptr<ALForm> al_form = std::make_shared<ALForm>(
			ndof, boundary_nodes, M, obstacle, target_x);
		forms.push_back(al_form);

		// --------------------------------------------------------------------

		StaticBoundaryNLProblem problem(ndof, boundary_nodes, target_x, forms);

		// --------------------------------------------------------------------

		// Create augmented Lagrangian solver
		ALSolver al_solver(
			nl_solver, al_form, al_initial_weight, al_scaling, al_max_steps,
			/*update_barrier_stiffness=*/[&](const Eigen::MatrixXd &x) {});

		Eigen::MatrixXd sol = x0;
		al_solver.solve(problem, sol, force_al);

		assert(forms[1]->is_step_valid(sol, sol));
		assert(!ipc::has_intersections(collision_mesh, collision_mesh.displace_vertices(utils::unflatten(sol, dim))));

		return sol;
	}
} // namespace polyfem::mesh