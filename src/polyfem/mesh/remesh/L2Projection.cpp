#include "L2Projection.hpp"

#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/PardisoSupport>

namespace polyfem::mesh
{
	using namespace polyfem::assembler;
	using namespace polyfem::basis;
	using namespace polyfem::solver;
	using namespace polyfem::utils;

	void L2_projection(
		const bool is_volume,
		const int size,
		const int n_from_basis,
		const std::vector<ElementBases> &from_bases,
		const std::vector<ElementBases> &from_gbases,
		const int n_to_basis,
		const std::vector<ElementBases> &to_bases,
		const std::vector<ElementBases> &to_gbases,
		const std::vector<int> &boundary_nodes,
		const Obstacle &obstacle,
		const Eigen::MatrixXd &target_x,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const bool lump_mass_matrix)
	{
		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			MassMatrixAssembler assembler;
			Density no_density; // Density of one (i.e., no scaling of mass matrix)
			AssemblyValsCache cache;

			assembler.assemble(
				is_volume, size,
				n_to_basis, no_density, to_bases, to_gbases,
				cache, M);

			assembler.assemble_cross(
				is_volume, size,
				n_from_basis, from_bases, from_gbases,
				n_to_basis, to_bases, to_gbases,
				cache, A);
		}

		if (lump_mass_matrix)
		{
			M = lump_matrix(M);
		}

		std::shared_ptr<L2ProjectionForm> l2_projection_form =
			std::make_shared<L2ProjectionForm>(M, A, y.col(0));
		const int ndof = n_to_basis * size;
		std::shared_ptr<ALForm> al_form = std::make_shared<ALForm>(
			ndof, boundary_nodes, M, obstacle, target_x);
		// std::shared_ptr<ElasticForm> elastic_form = std::make_shared<ElasticForm>(state);
		// const bool use_adaptive_barrier_stiffness = !state.args["solver"]["contact"]["barrier_stiffness"].is_number();
		// std::shared_ptr<ContactForm> contact_form = std::make_shared<ContactForm>(
		// 	state,
		// 	state.args["contact"]["dhat"],
		// 	use_adaptive_barrier_stiffness,
		// 	/*is_time_dependent=*/true,
		// 	state.args["solver"]["contact"]["CCD"]["broad_phase"],
		// 	state.args["solver"]["contact"]["CCD"]["tolerance"],
		// 	state.args["solver"]["contact"]["CCD"]["max_iterations"]);

		// if (use_adaptive_barrier_stiffness)
		// {
		// 	contact_form->set_weight(1);
		// 	logger().debug("Using adaptive barrier stiffness");
		// }
		// else
		// {
		// 	contact_form->set_weight(state.args["solver"]["contact"]["barrier_stiffness"]);
		// 	logger().debug("Using fixed barrier stiffness of {}", contact_form->barrier_stiffness());
		// }

		// std::vector<std::shared_ptr<Form>> forms = {l2_projection_form, al_form, elastic_form, contact_form};
		std::vector<std::shared_ptr<Form>> forms = {l2_projection_form, al_form};
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

		// Create a lambda function to update the barrier stiffness
		auto update_barrier_stiffness = [&](const Eigen::MatrixXd &x) {
			// if (!contact_form->use_adaptive_barrier_stiffness())
			// 	return;

			// Eigen::VectorXd grad_energy(x.size(), 1);
			// grad_energy.setZero();
			// elastic_form->first_derivative(x, grad_energy);

			// Eigen::VectorXd grad_L2(x.size());
			// l2_projection_form->first_derivative(x, grad_L2);
			// grad_energy += grad_L2;

			// contact_form->update_barrier_stiffness(x, grad_energy);
		};

		// --------------------------------------------------------------------

		// TODO: Make these parameters
		const double al_initial_weight = 0.5;
		const double al_scaling = 10.0;
		const int al_max_steps = 20;
		const bool force_al = false;

		// Create augmented Lagrangian solver
		ALSolver al_solver(
			nl_solver, al_form, al_initial_weight, al_scaling, al_max_steps,
			update_barrier_stiffness);

		Eigen::MatrixXd sol = Eigen::VectorXd::Zero(M.rows());
		al_solver.solve(problem, sol, force_al);

		// --------------------------------------------------------------------

		// Construct a linear solver for M
		Eigen::PardisoLU<Eigen::SparseMatrix<double>> linear_solver;
		// linear_solver->setParameters(solver_params);
		// NOTE: remove & if you want to have a more complicated LHS
		const Eigen::SparseMatrix<double> &LHS = M;
		linear_solver.analyzePattern(LHS);
		linear_solver.factorize(LHS);

		const Eigen::MatrixXd rhs = A * y.rightCols(2);
		x.resize(rhs.rows(), y.cols());
		x.col(0) = sol;
		x.rightCols(2) = linear_solver.solve(rhs);
		// x = linear_solver.solve(rhs);
		double residual_error = (LHS * x.rightCols(2) - rhs).norm();
		// double residual_error = (LHS * x - rhs).norm();
		logger().critical("residual error in L2 projection: {}", residual_error);
		assert(residual_error < 1e-12);
	}

	L2ProjectionForm::L2ProjectionForm(
		const StiffnessMatrix &M,
		const StiffnessMatrix &A,
		const Eigen::VectorXd &x_prev)
		: M_(M), rhs_(A * x_prev)
	{
	}

	double L2ProjectionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return x.transpose() * (0.5 * M_ * x - rhs_);
	}

	void L2ProjectionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = M_ * x - rhs_;
	}

	void L2ProjectionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = M_;
	}

} // namespace polyfem::mesh