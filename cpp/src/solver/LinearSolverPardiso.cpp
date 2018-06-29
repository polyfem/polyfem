#ifdef POLYFEM_WITH_PARDISO

////////////////////////////////////////////////////////////////////////////////
#include "LinearSolverPardiso.h"
#include <thread>
////////////////////////////////////////////////////////////////////////////////
extern "C" {
// PARDISO prototype.
void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
							 double *, int    *,    int *, int *,   int *, int *,
							 int *, double *, double *, int *, double *);
void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
void pardiso_chkvec     (int *, int *, double *, int *);
void pardiso_printstats (int *, int *, double *, int *, int *, int *,
											double *, int *);
}
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
// #define PLOTS_PARDISO

////////////////////////////////////////////////////////////////////////////////

LinearSolverPardiso::LinearSolverPardiso()
	: mtype(-1)
{
	setType(11);
}

void LinearSolverPardiso::setType(int _mtype) {
	mtype = _mtype;
	init();
}

// Set solver parameters
void LinearSolverPardiso::setParameters(const json &params) {
	if (params.count("mtype")) {
		setType(params["mtype"].get<int>());
	}
}

void LinearSolverPardiso::getInfo(json &params) const {
	params["mem_symbolic_peak"] = iparm[14];
	params["mem_symbolic_perm"] = iparm[15];
	params["mem_numerical_fact"] = iparm[16];
	params["mem_total_peak"] = std::max(iparm[14], iparm[15] + iparm[16]);
	params["num_nonzero_factors"] = iparm[17];
}

////////////////////////////////////////////////////////////////////////////////

void LinearSolverPardiso::init() {
	if (mtype ==-1) { throw std::runtime_error("[Pardiso] mtype not set."); }

	// --------------------------------------------------------------------
	// ..  Setup Pardiso control parameters.
	// --------------------------------------------------------------------

	error = 0;
	solver = 0; // Use sparse direct solver
	pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);

	if (error != 0) {
		if (error == -10) {
			throw std::runtime_error("[Pardiso] No license file found");
		} else if (error == -11) {
			throw std::runtime_error("[Pardiso] License is expired");
		} else if (error == -12) {
			throw std::runtime_error("[Pardiso] Wrong username or hostname");
		}
		throw std::runtime_error("[Pardiso] Unknown error in pardisoinit");
	} else {
		printf("[Pardiso] License check was successful ... \n");
	}

	// Numbers of processors, value of OMP_NUM_THREADS
	char * var = getenv("OMP_NUM_THREADS");
	int num_procs = 1;
	if (var != NULL) {
		sscanf(var, "%d", &num_procs);
	} else {
		throw std::runtime_error("[Pardiso] Set environment OMP_NUM_THREADS to 1");
	}
	iparm[2] = num_procs;

	maxfct = 1; // Maximum number of numerical factorizations
	mnum = 1;   // Which factorization to use

	msglvl = 0; // Print statistical information
	error = 0;  // Initialize error flag

	//  --------------------------------------------------------------------
	//  .. Initialize the internal solver memory pointer. This is only
	//  necessary for the FIRST call of the PARDISO solver.
	//  --------------------------------------------------------------------
	//  for ( i = 0; i < 64; i++ )
	//  {
	//    pt[i] = 0;
	//  }
}

////////////////////////////////////////////////////////////////////////////////

// - For symmetric matrices the solver needs only the upper triangular part of the system
// - Make sure diagonal terms are included, even as zeros (pardiso claims this is
//   necessary for best performance)

namespace {

typedef Eigen::SparseMatrix<double> SparseMatrixXd;

// Count number of non-zeros
int countNonZeros(const SparseMatrixXd &K, bool upperOnly) {
	int count = 0;
	for (int r = 0; r < K.rows(); ++r) {
		for (int j = K.outerIndexPtr()[r]; j < K.outerIndexPtr()[r+1]; ++j) {
			int c = K.innerIndexPtr()[j];
			if (!upperOnly || r <= c) {
				++count;
			}
		}
	}
	return count;
}

// Compute indices of matrix coeffs in CRS format
void computeIndices(const SparseMatrixXd &K, Eigen::VectorXi &ia, Eigen::VectorXi &ja, bool upperOnly) {
	int count = 0;
	for (int r = 0; r < K.rows(); ++r) {
		ia(r) = count;
		for (int j = K.outerIndexPtr()[r]; j < K.outerIndexPtr()[r+1]; ++j) {
			int c = K.innerIndexPtr()[j];
			if (!upperOnly || r <= c) {
				ja(count++) = c;
			}
		}
	}
	ia.tail<1>()[0] = count;
}

// Compue non-zero coefficients and put them in 'a'
void computeCoeffs(const SparseMatrixXd &K, Eigen::VectorXd &a, bool upperOnly) {
	int count = 0;
	for (int r = 0; r < K.rows(); ++r) {
		for (int j = K.outerIndexPtr()[r]; j < K.outerIndexPtr()[r+1]; ++j) {
			int c = K.innerIndexPtr()[j];
			if (!upperOnly || r <= c) {
				a(count++) = K.valuePtr()[j];
			}
		}
	}
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void LinearSolverPardiso::analyzePattern(const SparseMatrixXd &A) {
	if (mtype ==-1) { throw std::runtime_error("[Pardiso] mtype not set."); }
	assert(A.isCompressed());

	numRows = (int) A.rows();
	int nnz = countNonZeros(A, isSymmetric());
	ia.resize(numRows+1);
	ja.resize(nnz);
	a.resize(nnz);
	computeIndices(A, ia, ja, isSymmetric());
	computeCoeffs(A, a, isSymmetric());

	// Convert matrix from 0-based C-notation to Fortran 1-based notation.
	ia = ia.array() + 1;
	ja = ja.array() + 1;

#ifdef PLOTS_PARDISO
	// --------------------------------------------------------------------
	//  .. pardiso_chk_matrix(...)
	//     Checks the consistency of the given matrix.
	//     Use this functionality only for debugging purposes
	// --------------------------------------------------------------------

	pardiso_chkmatrix(&mtype, &numRows, a.data(), ia.data(), ja.data(), &error);
	if (error != 0) {
		throw std::runtime_error("[Pardiso] ERROR in consistency of matrix: " + std::to_string(error));
	}
#endif

	// --------------------------------------------------------------------
	// ..  Reordering and Symbolic Factorization.  This step also allocates
	//     all memory that is necessary for the factorization.
	// --------------------------------------------------------------------
	phase = 11;

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &numRows, a.data(), ia.data(),
		ja.data(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error, dparm);

	if (error != 0) {
		throw std::runtime_error("[Pardiso] ERROR during symbolic factorization: " + std::to_string(error));
	}

#ifdef PLOTS_PARDISO
	printf("\nReordering completed ... ");
	printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
	printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
#endif
}

// -----------------------------------------------------------------------------

void LinearSolverPardiso::factorize(const SparseMatrixXd &A) {
	if (mtype ==-1) { throw std::runtime_error("[Pardiso] mtype not set."); }
	assert(ia.size() == A.rows() + 1);

	// Update cached coefficients
	computeCoeffs(A, a, isSymmetric());

	// --------------------------------------------------------------------
	// ..  Numerical factorization.
	// --------------------------------------------------------------------
	phase = 22;
	// iparm[32] = 1; // Compute determinant
	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &numRows, a.data(), ia.data(),
		ja.data(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error, dparm);

	if (error != 0) {
		throw std::runtime_error("[Pardiso] ERROR during numerical factorization: " + std::to_string(error));
	}
#ifdef PLOTS_PARDISO
	printf("\nFactorization completed ... ");
#endif
}

////////////////////////////////////////////////////////////////////////////////

void LinearSolverPardiso::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result) {
	if (mtype ==-1) { throw std::runtime_error("[Pardiso] mtype not set."); }
	assert(numRows == rhs.size());
	assert(result.size() == rhs.size());


	double *rhs_ptr = const_cast<double *>(rhs.data());
	result = VectorXd(rhs.size());

#ifdef PLOTS_PARDISO
	// --------------------------------------------------------------------
	// ..  pardiso_chkvec(...)
	//     Checks the given vectors for infinite and NaN values
	//     Input parameters (see PARDISO user manual for a description):
	//     Use this functionality only for debugging purposes
	// --------------------------------------------------------------------

	pardiso_chkvec(&numRows, &nrhs, rhs_ptr, &error);
	if (error != 0) {
		throw std::runtime_error("[Pardiso] ERROR in right hand side: " + std::to_string(error));
	}

	// --------------------------------------------------------------------
	// .. pardiso_printstats(...)
	//    prints information on the matrix to STDOUT.
	//    Use this functionality only for debugging purposes
	// --------------------------------------------------------------------

	pardiso_printstats(&mtype, &numRows, a.data(), ia.data(), ja.data(), &nrhs, rhs_ptr, &error);
	if (error != 0) {
		throw std::runtime_error("[Pardiso] ERROR in right hand side: " + std::to_string(error));
	}

#endif
	result.resize(numRows, 1);
	// --------------------------------------------------------------------
	// ..  Back substitution and iterative refinement.
	// --------------------------------------------------------------------
	phase = 33;

	iparm[7] = 1; // Max numbers of iterative refinement steps

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &numRows, a.data(), ia.data(), ja.data(),
		&idum, &nrhs, iparm, &msglvl, rhs_ptr, result.data(), &error, dparm);

	if (error != 0) {
		throw std::runtime_error("[Pardiso] ERROR during solution: " + std::to_string(error));
	}
#ifdef PLOTS_PARDISO
	printf("\nSolve completed ... ");
	printf("\nThe solution of the system is: ");
	for (int i = 0; i < numRows; i++) {
		printf("\n x [%d] = % f", i, result.data()[i]);
	}
	printf("\n\n");
#endif

	// --------------------------------------------------------------------
	// ..  Back substitution with transposed matrix A^t x=b
	// --------------------------------------------------------------------
	if (!isSymmetric()) {
		phase = 33;

		iparm[7]  = 1; // Max numbers of iterative refinement steps.
		iparm[11] = 1; // Solving with transpose matrix.

		pardiso (pt, &maxfct, &mnum, &mtype, &phase, &numRows, a.data(), ia.data(),
			ja.data(), &idum, &nrhs, iparm, &msglvl, rhs_ptr, result.data(), &error, dparm);

		if (error != 0) {
			throw std::runtime_error("[Pardiso] ERROR during solution: " + std::to_string(error));
		}

#ifdef PLOTS_PARDISO
		printf("\nSolve completed ... ");
		printf("\nThe solution of the system is: ");
		for (int i = 0; i < numRows; i++) {
			printf("\n x [%d] = % f", i, result.data()[i] );
		}
		printf ("\n");
#endif
	}
}

////////////////////////////////////////////////////////////////////////////////

void LinearSolverPardiso::freeNumericalFactorizationMemory() {
	phase = 0; // Release internal memory

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &numRows, &ddum, ia.data(),
		ja.data(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error, dparm);
}

////////////////////////////////////////////////////////////////////////////////

LinearSolverPardiso::~LinearSolverPardiso() {
	if (mtype == -1) {
		return;
	}

	// --------------------------------------------------------------------
	// ..  Termination and release of memory.
	// --------------------------------------------------------------------
	phase = -1; // Release internal memory

	pardiso(pt, &maxfct, &mnum, &mtype, &phase, &numRows, &ddum, ia.data(),
		ja.data(), &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error, dparm);
}

#endif
