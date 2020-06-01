#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <polyfem/LinearSolver.hpp>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem {

// -----------------------------------------------------------------------------

template<typename SparseSolver>
class LinearSolverEigenDirect : public LinearSolver {
protected:

	// Solver class
	SparseSolver m_Solver;

public:
	// Name of the solver type (for debugging purposes)
	virtual std::string name() const override { return typeid(m_Solver).name(); }

public:
	// Get info on the last solve step
	virtual void getInfo(json &params) const override;

	// Analyze sparsity pattern
	virtual void analyzePattern(const StiffnessMatrix &K, const int precond_num) override;

	// Factorize system matrix
	virtual void factorize(const StiffnessMatrix &K) override;

	// Solve the linear system
	virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
};

// -----------------------------------------------------------------------------

template<typename SparseSolver>
class LinearSolverEigenIterative : public LinearSolver {
protected:

	// Solver class
	SparseSolver m_Solver;

public:
	// Name of the solver type (for debugging purposes)
	virtual std::string name() const override { return typeid(m_Solver).name(); }

public:
	// Set solver parameters
	virtual void setParameters(const json &params) override;

	// Get info on the last solve step
	virtual void getInfo(json &params) const override;

	// Analyze sparsity pattern
	virtual void analyzePattern(const StiffnessMatrix &K, const int precond_num) override;

	// Factorize system matrix
	virtual void factorize(const StiffnessMatrix &K) override;

	// Solve the linear system
	virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
};

// -----------------------------------------------------------------------------

} // namespace polyfem

////////////////////////////////////////////////////////////////////////////////

#include <polyfem/LinearSolverEigen.hpp>
