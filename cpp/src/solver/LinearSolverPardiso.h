#pragma once

#ifdef POLYFEM_WITH_PARDISO

////////////////////////////////////////////////////////////////////////////////
#include <polyfem/Common.hpp>
#include <polyfem/LinearSolver.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
////////////////////////////////////////////////////////////////////////////////
//
// http://pardiso-project.org/manual/manual.pdf
//
// See page 29 for instruction on installing and running Pardiso
// The following environment variables must be set:
// - OMP_NUM_THREADS: number of threads
// - PARDISO_LIC_PATH: path to the folder containing the license file

namespace polyfem {

class LinearSolverPardiso : public LinearSolver {

public:
	LinearSolverPardiso();
	~LinearSolverPardiso();

private:
	POLYFEM_DELETE_MOVE_COPY(LinearSolverPardiso)

protected:
	void setType(int _mtype);
	void init();
	void freeNumericalFactorizationMemory();

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
	virtual void factorize(const SparseMatrixXd &A) override;

	// Solve the linear system Ax = b
	virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

	// Name of the solver type (for debugging purposes)
	virtual std::string name() const override { return "Pardiso"; }

protected:
	Eigen::VectorXi ia, ja;
	VectorXd a;

protected:
	int numRows;

	///////////////////
	// Pardiso stuff //
	///////////////////

	// |-------|-----------------------------------------|
	// | mtype | matrix type                             |
	// |-------|-----------------------------------------|
	// |    1  | real and structurally symmetric         |
	// |    2  | real and symmetric positive definite    |
	// |   -2  | real and symmetric indefinite           |
	// |    3  | complex and structurally symmetric      |
	// |    4  | complex and Hermitian positive definite |
	// |   -4  | complex and Hermitian indefinite        |
	// |    6  | complex and symmetric                   |
	// |   11  | real and nonsymmetric                   |
	// |   13  | complex and nonsymmetric                |
	// |-------|-----------------------------------------|

	bool isSymmetric() {
		switch (mtype) {
			case 2:
			case -2:
			case 4:
			case -4:
			case 6:
				return true;
			default:
				return false;
		}
	}

	int mtype; // Matrix type
	int nrhs = 1; // Number of right hand sides.

	// Internal solver memory pointer pt,
	// 32-bit: int pt[64]; 64-bit: long int pt[64]
	// or void *pt[64] should be OK on both architectures
	void *pt[64];

	// Pardiso control parameters.
	int iparm[64];
	double dparm[64];
	int maxfct, mnum, phase, error, msglvl, solver = 0;

	// Auxiliary variables.
	double ddum; // Double dummy
	int idum; // Integer dummy

	int numUniqueElements;
};

} // namespace polyfem

#endif
