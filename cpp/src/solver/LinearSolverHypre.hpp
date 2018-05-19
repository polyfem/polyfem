#pragma once

#ifdef USE_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include "Common.hpp"
#include "LinearSolver.hpp"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <_hypre_utilities.h>
#include <HYPRE.h>
#include <IJ_matrix.h>
#include <HYPRE_parcsr_ls.h>
////////////////////////////////////////////////////////////////////////////////
//
// https://computation.llnl.gov/sites/default/files/public/hypre-2.11.2_usr_manual.pdf
//

namespace poly_fem {

	class LinearSolverHypre : public LinearSolver {

	public:
		LinearSolverHypre();
		~LinearSolverHypre();

	private:
		POLYFEM_DELETE_MOVE_COPY(LinearSolverHypre)

	public:
	//////////////////////
	// Public interface //
	//////////////////////

	// Set solver parameters
		virtual void setParameters(const json &params) override;

	// Retrieve memory information from Pardiso
		virtual void getInfo(json &params) const override;

	// Analyze sparsity pattern
		virtual void analyzePattern(const SparseMatrixXd &A) override;

	// Factorize system matrix
		virtual void factorize(const SparseMatrixXd &) override { }

	// Solve the linear system Ax = b
		virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

	// Name of the solver type (for debugging purposes)
		virtual std::string name() const override { return "Hypre"; }

	protected:
		int max_iter_ = 1000;
		int pre_max_iter_ = 5;
		double conv_tol_ = 1e-8;

		int num_iterations;
		double final_res_norm;
	private:
		bool has_matrix_ = false;

		HYPRE_IJMatrix A;
		HYPRE_ParCSRMatrix parcsr_A;

	};

} // namespace poly_fem

#endif
