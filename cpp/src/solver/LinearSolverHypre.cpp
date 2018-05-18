#ifdef USE_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include "LinearSolverHypre.hpp"

#include <HYPRE_krylov.h>

////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

////////////////////////////////////////////////////////////////////////////////

LinearSolverHypre::LinearSolverHypre()
{
}


// Set solver parameters
void LinearSolverHypre::setParameters(const json &params) {
// if (params.count("mtype")) {
// 	setType(params["mtype"].get<int>());
// }
}

void LinearSolverHypre::getInfo(json &params) const {
	params["num_iterations"] = num_iterations;
	params["final_res_norm"] = final_res_norm;
}



////////////////////////////////////////////////////////////////////////////////

void LinearSolverHypre::analyzePattern(const SparseMatrixXd &Ain) {
	HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, Ain.rows(), 0, Ain.cols(), &A);
	HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
	HYPRE_IJMatrixInitialize(A);

	// HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);

	for (int k = 0; k < Ain.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(Ain, k); it; ++it) {
			const int i = it.row();
			const int j = it.col();
			int n_cols = 1;
			const double v = it.value();

			HYPRE_IJMatrixSetValues(A, 1, &n_cols, &i, &j, &v);
		}
	}

	HYPRE_IJMatrixAssemble(A);
	HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);



}


////////////////////////////////////////////////////////////////////////////////

void LinearSolverHypre::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result) {
	HYPRE_IJVector b;
	HYPRE_ParVector par_b;
	HYPRE_IJVector x;
	HYPRE_ParVector par_x;

	HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size(), &b);
	HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
	HYPRE_IJVectorInitialize(b);

	HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size(), &x);
	HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
	HYPRE_IJVectorInitialize(x);

	Eigen::VectorXi indices(rhs.size());

	for(int i = 0; i < indices.size(); ++i)
		indices(i) = i;
	Eigen::VectorXd initial_guess(rhs.size()); initial_guess.setZero();

	HYPRE_IJVectorSetValues(b, rhs.size(), indices.data(), rhs.data());
	HYPRE_IJVectorSetValues(x, rhs.size(), indices.data(), initial_guess.data());

	HYPRE_IJVectorAssemble(b);
	HYPRE_IJVectorGetObject(b, (void **) &par_b);

	HYPRE_IJVectorAssemble(x);
	HYPRE_IJVectorGetObject(x, (void **) &par_x);


	/* PCG with AMG preconditioner */

	/* Create solver */
	HYPRE_Solver solver, precond;
	HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

	/* Set some parameters (See Reference Manual for more parameters) */
	HYPRE_PCGSetMaxIter(solver, max_iter_); /* max iterations */
	HYPRE_PCGSetTol(solver, conv_tol_); /* conv. tolerance */
	HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
	// HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
	HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

	/* Now set up the AMG preconditioner and specify any parameters */
	HYPRE_BoomerAMGCreate(&precond);
	// HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
	HYPRE_BoomerAMGSetCoarsenType(precond, 6);
	HYPRE_BoomerAMGSetOldDefault(precond);
	HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
	HYPRE_BoomerAMGSetNumSweeps(precond, 1);
	HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
	HYPRE_BoomerAMGSetMaxIter(precond, pre_max_iter_); /* do only one iteration! */

	/* Set the PCG preconditioner */
	HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

	/* Now setup and solve! */
	HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
	HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

	/* Run info - needed logging turned on */
	HYPRE_PCGGetNumIterations(solver, &num_iterations);
	HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

	// printf("\n");
	// printf("Iterations = %d\n", num_iterations);
	// printf("Final Relative Residual Norm = %e\n", final_res_norm);
	// printf("\n");

	/* Destroy solver and preconditioner */
	HYPRE_ParCSRPCGDestroy(solver);
	HYPRE_BoomerAMGDestroy(precond);


	assert(result.size() == rhs.size());
	HYPRE_IJVectorGetValues(x, indices.size(), indices.data(), result.data());


	HYPRE_IJVectorDestroy(b);
	HYPRE_IJVectorDestroy(x);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

LinearSolverHypre::~LinearSolverHypre() {
	HYPRE_IJMatrixDestroy(A);
}

#endif
