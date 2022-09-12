#include "L2Projection.hpp"

#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/PardisoSupport>

namespace polyfem::mesh
{
	using namespace polyfem::utils;

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const State &state,
		const RhsAssembler &rhs_assembler,
		const bool is_volume,
		const int size,
		const int n_from_basis,
		const std::vector<ElementBases> &from_bases,
		const std::vector<ElementBases> &from_gbases,
		const int n_to_basis,
		const std::vector<ElementBases> &to_bases,
		const std::vector<ElementBases> &to_gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const bool lump_mass_matrix)
	{
		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			MassMatrixAssembler assembler;
			Density no_density; // Density of one (i.e., no scaling of mass matrix)
			assembler.assemble(
				is_volume, size,
				n_to_basis, no_density, to_bases, to_gbases,
				cache, M);

			assembler.assemble_cross(
				is_volume, size,
				n_from_basis, from_bases, from_gbases,
				n_to_basis, to_bases, to_gbases,
				cache, A);

			// write_sparse_matrix_csv("M.csv", M);
			// write_sparse_matrix_csv("A.csv", A);
			// logger().critical("M =\n{}", Eigen::MatrixXd(M));
			// logger().critical("A =\n{}", Eigen::MatrixXd(A));
		}

		if (lump_mass_matrix)
		{
			M = lump_matrix(M);
		}

		// --------------------------------------------------------------------

		double al_weight = state.args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = state.args["solver"]["augmented_lagrangian"]["max_weight"];

		const double t = 0;

		std::shared_ptr<L2ProjectionForm> l2_projection_form = std::make_shared<L2ProjectionForm>(M, A, y.col(0));
		std::shared_ptr<ALForm> al_form = std::make_shared<ALForm>(state, rhs_assembler, t);
		std::vector<std::shared_ptr<Form>> forms = {l2_projection_form, al_form};
		NLProblem problem(state, rhs_assembler, t, forms);

		/*
		Eigen::VectorXd sol = Eigen::VectorXd::Zero(M.rows());
		Eigen::VectorXd tmp_sol;

		problem.full_to_reduced(sol, tmp_sol);

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		json newton_args = state.args["solver"]["nonlinear"];
		newton_args["f_delta"] = 0;
		newton_args["grad_norm"] = 1e-8;
		newton_args["use_grad_norm"] = true;
		// newton_args["relative_gradient"] = true;

		problem.line_search_begin(sol, tmp_sol);
		while (
			!std::isfinite(problem.value(tmp_sol))
			|| !problem.is_step_valid(sol, tmp_sol)
			|| !problem.is_step_collision_free(sol, tmp_sol))
		{
			problem.line_search_end();
			problem.set_weight(al_weight);
			logger().debug("Solving L2 Projection with weight {}", al_weight);

			cppoptlib::SparseNewtonDescentSolver<L2ProjectionOptimizationProblem> solver(
				newton_args,
				state.args["solver"]["linear"]["solver"],
				state.args["solver"]["linear"]["precond"]);
			solver.setLineSearch(state.args["solver"]["nonlinear"]["line_search"]["method"]);
			problem.init(sol);
			tmp_sol = sol;
			solver.minimize(problem, tmp_sol);

			sol = tmp_sol;
			problem.full_to_reduced(sol, tmp_sol);
			problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
				log_and_throw_error(fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight));
		}
		problem.line_search_end();

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		cppoptlib::SparseNewtonDescentSolver<L2ProjectionOptimizationProblem> solver(
			newton_args,
			state.args["solver"]["linear"]["solver"],
			state.args["solver"]["linear"]["precond"]);
		solver.setLineSearch(state.args["solver"]["nonlinear"]["line_search"]["method"]);
		problem.init(sol);
		solver.minimize(problem, tmp_sol);

		problem.reduced_to_full(tmp_sol, sol);

		// --------------------------------------------------------------------

		// Construct a linear solver for M
		Eigen::PardisoLU<decltype(M)> linear_solver;
		// linear_solver->setParameters(solver_params);
		const Eigen::SparseMatrix<double> &LHS = M; // NOTE: remove & if you want to have a more complicated LHS
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
		*/
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

	void L2ProjectionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		hessian = M_;
	}

} // namespace polyfem::mesh