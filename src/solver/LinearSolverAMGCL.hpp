#pragma once

#ifdef POLYFEM_WITH_AMGCL

////////////////////////////////////////////////////////////////////////////////
#include <polyfem/Common.hpp>
#include <polyfem/LinearSolver.hpp>

#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>


////////////////////////////////////////////////////////////////////////////////
//
//

namespace polyfem {

	class LinearSolverAMGCL : public LinearSolver {

	public:
		LinearSolverAMGCL();
		~LinearSolverAMGCL();

	private:
		POLYFEM_DELETE_MOVE_COPY(LinearSolverAMGCL)

	public:
		//////////////////////
		// Public interface //
		//////////////////////

		// Set solver parameters
		virtual void setParameters(const json &params) override;

		// Retrieve memory information from Pardiso
		virtual void getInfo(json &params) const override;

		// Analyze sparsity pattern
		virtual void analyzePattern(const StiffnessMatrix &A) override;

		// Factorize system matrix
		virtual void factorize(const StiffnessMatrix &) override { }

		// Solve the linear system Ax = b
		virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

		// Name of the solver type (for debugging purposes)
		virtual std::string name() const override { return "AMGCL"; }

	private:
		// typedef amgcl::backend::eigen<double> Backend;
		typedef amgcl::backend::builtin<double> Backend;

		// typedef amgcl::make_solver<
		// 	amgcl::amg<
		// 		Backend,
		// 		amgcl::coarsening::aggregation,
		// 		amgcl::relaxation::gauss_seidel>,
		// 	amgcl::solver::cg<Backend>> Solver;

		// typedef amgcl::make_solver<
		// 	// Use AMG as preconditioner:
		// 	amgcl::amg<
		// 		Backend,
		// 		amgcl::coarsening::smoothed_aggregation,
		// 		amgcl::relaxation::spai0>,
		// 	// And BiCGStab as iterative solver:
		// 	amgcl::solver::bicgstab<Backend>>
		// 	Solver;

		// Use AMG as preconditioner:
		typedef amgcl::make_solver<
			// Use AMG as preconditioner:
			amgcl::amg<
				Backend,
				amgcl::coarsening::smoothed_aggregation,
				amgcl::relaxation::gauss_seidel>,
			// And CG as iterative solver:
			amgcl::solver::cg<Backend>>
			Solver;

		Solver *solver_ = nullptr;
		Solver::params params_;
		// StiffnessMatrix mat;

		std::vector<int> ia_, ja_;
		std::vector<double> a_;

		size_t iterations_;
		double residual_error_;
	};

} // namespace polyfem

#endif
