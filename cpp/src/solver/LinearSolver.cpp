////////////////////////////////////////////////////////////////////////////////
#include "LinearSolver.hpp"
#include "LinearSolverEigen.hpp"
// -----------------------------------------------------------------------------
#include <Eigen/Sparse>
#ifdef POLYFEM_WITH_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#ifdef POLYFEM_WITH_UMFPACK
#include <Eigen/UmfPackSupport>
#endif
#ifdef POLYFEM_WITH_SUPERLU
#include <Eigen/SuperLUSupport>
#endif
#ifdef POLYFEM_WITH_MKL
#include <Eigen/PardisoSupport>
#endif
#ifdef POLYFEM_WITH_PARDISO
#include "LinearSolverPardiso.h"
#endif
#include <unsupported/Eigen/IterativeSolvers>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

////////////////////////////////////////////////////////////////////////////////

#if EIGEN_VERSION_AT_LEAST(3,3,0)

// Magic macro because C++ has no introspection
#define ENUMERATE_PRECOND(HelperFunctor, SolverType, DefaultPrecond, precond) \
    do {                                                                      \
        using namespace Eigen;                                                \
        if (precond == "Eigen::IdentityPreconditioner") {                     \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                IdentityPreconditioner>::type>();                             \
        } else if (precond == "Eigen::DiagonalPreconditioner") {              \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                DiagonalPreconditioner<double>>::type>();                     \
        } else if (precond == "Eigen::IncompleteCholesky") {                  \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                IncompleteCholesky<double>>::type>();                         \
        } else if (precond == "Eigen::LeastSquareDiagonalPreconditioner") {   \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                LeastSquareDiagonalPreconditioner<double>>::type>();          \
        } else if (precond == "Eigen::IncompleteLUT") {                       \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                IncompleteLUT<double>>::type>();                              \
        } else {                                                              \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                DefaultPrecond>::type>();                                     \
        }                                                                     \
    } while (0)

#else

// Magic macro because C++ has no introspection
#define ENUMERATE_PRECOND(HelperFunctor, SolverType, DefaultPrecond, precond) \
    do {                                                                      \
        using namespace Eigen;                                                \
        if (precond == "Eigen::IdentityPreconditioner") {                     \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                IdentityPreconditioner>::type>();                             \
        } else if (precond == "Eigen::DiagonalPreconditioner") {              \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                DiagonalPreconditioner<double>>::type>();                     \
        } else if (precond == "Eigen::IncompleteCholesky") {                  \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                IncompleteCholesky<double>>::type>();                         \
        } else if (precond == "Eigen::IncompleteLUT") {                       \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                IncompleteLUT<double>>::type>();                              \
        } else {                                                              \
            return std::make_unique<typename HelperFunctor<SolverType,        \
                DefaultPrecond>::type>();                                     \
        }                                                                     \
    } while (0)

#endif

// -----------------------------------------------------------------------------

#define RETURN_DIRECT_SOLVER_PTR(EigenSolver)                        \
    do {                                                             \
        return std::make_unique<LinearSolverEigenDirect<EigenSolver< \
            SparseMatrixXd > > >();                                  \
    } while (0)

////////////////////////////////////////////////////////////////////////////////

namespace {

template<template<class, class> class SparseSolver, typename Precond>
struct MakeSolver {
	typedef LinearSolverEigenIterative<SparseSolver<LinearSolver::SparseMatrixXd, Precond>> type;
};

template<template<class, int, class> class SparseSolver, typename Precond>
struct MakeSolverSym {
	typedef LinearSolverEigenIterative<SparseSolver<LinearSolver::SparseMatrixXd,
		Eigen::Lower|Eigen::Upper, Precond> > type;
};

// -----------------------------------------------------------------------------

template<
	template<class, class> class SolverType,
	typename DefaultPrecond = Eigen::DiagonalPreconditioner<double> >
struct PrecondHelper {
	static std::unique_ptr<LinearSolver> create(const std::string &arg) {
		ENUMERATE_PRECOND(MakeSolver, SolverType, DefaultPrecond, arg);
	}
};

template<
	template<class, int, class> class SolverType,
	typename DefaultPrecond = Eigen::DiagonalPreconditioner<double> >
struct PrecondHelperSym {
	static std::unique_ptr<LinearSolver> create(const std::string &arg) {
		ENUMERATE_PRECOND(MakeSolverSym, SolverType, DefaultPrecond, arg);
	}
};

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

// Static constructor
std::unique_ptr<LinearSolver> LinearSolver::create(const std::string &solver, const std::string &precond) {
	using namespace Eigen;

	if (solver.empty() || solver == "Eigen::SimplicialLDLT") {
		RETURN_DIRECT_SOLVER_PTR(SimplicialLDLT);
	} else if (solver == "Eigen::SparseLU") {
		RETURN_DIRECT_SOLVER_PTR(SparseLU);
#ifdef POLYFEM_WITH_CHOLMOD
	} else if (solver == "Eigen::CholmodSupernodalLLT") {
		RETURN_DIRECT_SOLVER_PTR(CholmodSupernodalLLT);
#endif
#ifdef POLYFEM_WITH_UMFPACK
	} else if (solver == "Eigen::UmfPackLU") {
		RETURN_DIRECT_SOLVER_PTR(UmfPackLU);
#endif
#ifdef POLYFEM_WITH_SUPERLU
	} else if (solver == "Eigen::SuperLU") {
		RETURN_DIRECT_SOLVER_PTR(SuperLU);
#endif
#ifdef POLYFEM_WITH_MKL
	} else if (solver == "Eigen::PardisoLDLT") {
		RETURN_DIRECT_SOLVER_PTR(PardisoLDLT);
	} else if (solver == "Eigen::PardisoLU") {
		RETURN_DIRECT_SOLVER_PTR(PardisoLU);
#endif
#ifdef POLYFEM_WITH_PARDISO
	} else if (solver == "Pardiso") {
		return std::make_unique<LinearSolverPardiso>();
#endif
#if EIGEN_VERSION_AT_LEAST(3,3,0)
	// Available only with Eigen 3.3.0 and newer
	} else if (solver == "Eigen::LeastSquaresConjugateGradient") {
		return PrecondHelper<BiCGSTAB, LeastSquareDiagonalPreconditioner<double>>::create(precond);
	} else if (solver == "Eigen::DGMRES") {
		return PrecondHelper<DGMRES>::create(precond);
#endif
	} else if (solver == "Eigen::ConjugateGradient") {
		return PrecondHelperSym<ConjugateGradient>::create(precond);
	} else if (solver == "Eigen::BiCGSTAB") {
		return PrecondHelper<BiCGSTAB>::create(precond);
	} else if (solver == "Eigen::GMRES") {
		return PrecondHelper<GMRES>::create(precond);
	} else if (solver == "Eigen::MINRES") {
		return PrecondHelperSym<MINRES>::create(precond);
	}
	throw std::runtime_error("Unrecognized solver type: " + solver);
}

////////////////////////////////////////////////////////////////////////////////

// List available solvers
std::vector<std::string> LinearSolver::availableSolvers() {
	return {{
		"Eigen::SimplicialLDLT",
		"Eigen::SparseLU",
#ifdef POLYFEM_WITH_CHOLMOD
		"Eigen::CholmodSupernodalLLT",
#endif
#ifdef POLYFEM_WITH_UMFPACK
		"Eigen::UmfPackLU",
#endif
#ifdef POLYFEM_WITH_SUPERLU
		"Eigen::SuperLU",
#endif
#ifdef POLYFEM_WITH_MKL
		"Eigen::PardisoLDLT",
		"Eigen::PardisoLU",
#endif
#ifdef POLYFEM_WITH_PARDISO
		"Pardiso",
#endif
#if EIGEN_VERSION_AT_LEAST(3,3,0)
		"Eigen::LeastSquaresConjugateGradient",
		"Eigen::DGMRES",
#endif
		"Eigen::ConjugateGradient",
		"Eigen::BiCGSTAB",
		"Eigen::GMRES",
		"Eigen::MINRES",
	}};
}

std::string LinearSolver::defaultSolver() {
	// return "Eigen::BiCGSTAB";
#ifdef POLYFEM_WITH_PARDISO
	return "Pardiso";
#else
	return "Eigen::BiCGSTAB";
#endif
}

// -----------------------------------------------------------------------------

// List available preconditioners
std::vector<std::string> LinearSolver::availablePrecond() {
	return {{
		"Eigen::IdentityPreconditioner",
		"Eigen::DiagonalPreconditioner",
		"Eigen::IncompleteCholesky",
#if EIGEN_VERSION_AT_LEAST(3,3,0)
		"Eigen::LeastSquareDiagonalPreconditioner",
#endif
		"Eigen::IncompleteLUT",
	}};
}

std::string LinearSolver::defaultPrecond() {
	return "Eigen::DiagonalPreconditioner";
}
