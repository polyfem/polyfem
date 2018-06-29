#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <polyfem/Common.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
////////////////////////////////////////////////////////////////////////////////
// TODO:
// - [ ] Support both RowMajor + ColumnMajor sparse matrices
// - [ ] Wrapper around MUMPS
// - [ ] Wrapper around other iterative solvers (AMGCL, ViennaCL, etc.)
// - [ ] Document the json parameters for each
////////////////////////////////////////////////////////////////////////////////

namespace poly_fem {

/**
 * @brief      Base class for linear solver.
 */
class LinearSolver {

public:
	// Shortcut alias
	typedef Eigen::VectorXd VectorXd;
	typedef Eigen::SparseMatrix<double> SparseMatrixXd;
	template<typename T> using Ref = Eigen::Ref<T>;

public:
	//////////////////
	// Constructors //
	//////////////////

	// Virtual destructor
	virtual ~LinearSolver() = default;

	// Static constructor
	//
	// @param[in]  solver   Solver type
	// @param[in]  precond  Preconditioner for iterative solvers
	//
	static std::unique_ptr<LinearSolver> create(const std::string &solver, const std::string &precond);

	// List available solvers
	static std::vector<std::string> availableSolvers();
	static std::string defaultSolver();

	// List available preconditioners
	static std::vector<std::string> availablePrecond();
	static std::string defaultPrecond();

protected:
	// Default constructor
	LinearSolver() = default;

public:
	//////////////////////
	// Public interface //
	//////////////////////

	// Set solver parameters
	virtual void setParameters(const json &params) {}

	// Get info on the last solve step
	virtual void getInfo(json &params) const {};

	// Analyze sparsity pattern
	virtual void analyzePattern(const SparseMatrixXd &A) {}

	// Factorize system matrix
	virtual void factorize(const SparseMatrixXd &A) {}

	//
	// @brief         { Solve the linear system Ax = b }
	//
	// @param[in]     b     { Right-hand side. }
	// @param[in,out] x     { Unknown to compute. When using an iterative
	//                      solver, the input unknown vector is used as an
	//                      initial guess, and must thus be properly allocated
	//                      and initialized. }
	//
	virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) = 0;

public:
	///////////
	// Debug //
	///////////

	// Name of the solver type (for debugging purposes)
	virtual std::string name() const { return ""; }
};

} // namespace poly_fem
