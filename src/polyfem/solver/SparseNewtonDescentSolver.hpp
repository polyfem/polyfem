#pragma once

#include <polyfem/Common.hpp>
#include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

namespace cppoptlib
{
	template <typename ProblemType>
	class SparseNewtonDescentSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		SparseNewtonDescentSolver(
			const json &solver_params,
			const std::string &linear_solver_type,
			const std::string &linear_precond_type);

		std::string name() const override { return "Newton"; }

	protected:
		virtual int default_descent_strategy() override { return 0; }
		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy) const override;
		void increase_descent_strategy() override;

	protected:
		void reset(const ProblemType &objFunc, const TVector &x) override;

		bool compute_update_direction(
			ProblemType &objFunc,
			const TVector &x,
			const TVector &grad,
			TVector &direction) override;

		void update_solver_info() override;

		static bool has_hessian_nans(const polyfem::StiffnessMatrix &hessian);

		std::unique_ptr<polysolve::LinearSolver> linear_solver;
		const std::string linear_solver_type;
		const std::string linear_precond_type;

		json internal_solver_info = json::array();

		polyfem::StiffnessMatrix hessian; // Cached version of hessian

		// Regularization Coefficients
		double reg_weight = 0;
		static constexpr double reg_weight_min = 1e-8; // needs to be greater than zero
		static constexpr double reg_weight_max = 1e8;
		static constexpr double reg_weight_inc = 10;
		static constexpr double reg_weight_dec = 2;
	};

} // namespace cppoptlib

#include "SparseNewtonDescentSolver.tpp"