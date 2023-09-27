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

		SparseNewtonDescentSolver(const json &solver_params, const json &linear_solver_params, const double dt, const double characteristic_length);

		std::string name() const override { return "Newton"; }

	protected:
		const double characteristic_length;

		bool compute_update_direction(ProblemType &objFunc, const TVector &x, const TVector &grad, TVector &direction) override;

		void assemble_hessian(ProblemType &objFunc, const TVector &x, polyfem::StiffnessMatrix &hessian);
		bool solve_linear_system(const polyfem::StiffnessMatrix &hessian, const TVector &grad, TVector &direction);
		bool check_direction(const polyfem::StiffnessMatrix &hessian, const TVector &grad, const TVector &direction);

		static bool has_hessian_nans(const polyfem::StiffnessMatrix &hessian);

		// ====================================================================
		//                        Solver parameters
		// ====================================================================

		static constexpr double reg_weight_min = 1e-8; // needs to be greater than zero
		static constexpr double reg_weight_max = 1e8;
		static constexpr double reg_weight_inc = 10;
		static constexpr double reg_weight_dec = 2;

		// ====================================================================
		//                           Solver state
		// ====================================================================

		void reset(const int ndof) override;

		virtual int default_descent_strategy() override { return force_psd_projection ? 1 : 0; }
		void increase_descent_strategy() override;

		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy) const override;

		spdlog::level::level_enum log_level() const
		{
			return this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug;
		}

		std::unique_ptr<polysolve::LinearSolver> linear_solver; ///< Linear solver used to solve the linear system
		bool force_psd_projection = false;                      ///< Whether to force the Hessian to be positive semi-definite
		double reg_weight = 0;                                  ///< Regularization Coefficients

		// ====================================================================
		//                            Solver info
		// ====================================================================

		void update_solver_info(const double energy) override;

		json internal_solver_info = json::array();

		// ====================================================================
		//                                END
		// ====================================================================
	};

} // namespace cppoptlib

#include "SparseNewtonDescentSolver.tpp"