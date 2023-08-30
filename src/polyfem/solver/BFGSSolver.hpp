// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include <polyfem/Common.hpp>
#include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

#include <LBFGSpp/BFGSMat.h>

namespace cppoptlib
{
	template <typename ProblemType>
	class BFGSSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		BFGSSolver(const json &solver_params, const double dt, const double characteristic_length)
			: Superclass(solver_params, dt, characteristic_length)
		{
		}

		std::string name() const override { return "BFGS"; }

	protected:
		virtual int default_descent_strategy() override { return 1; }

		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy) const override
		{
			switch (descent_strategy)
			{
			case 1:
				return "BFGS";
			case 2:
				return "gradient descent";
			default:
				throw std::invalid_argument("invalid descent strategy");
			}
		}

		void increase_descent_strategy() override
		{
			if (this->descent_strategy == 1)
				this->descent_strategy++;

			assert(this->descent_strategy <= 2);
		}

	protected:
		TVector m_prev_x;    // Previous x
		TVector m_prev_grad; // Previous gradient

		Eigen::MatrixXd hess;

		void reset(const int ndof) override
		{
			Superclass::reset(ndof);

			reset_history(ndof);
		}

		void reset_history(const int ndof)
		{
			m_prev_x.resize(ndof);
			m_prev_grad.resize(ndof);

			hess.setIdentity(ndof, ndof);

			// Use gradient descent for first iteration
			this->descent_strategy = 2;
		}

		virtual bool compute_update_direction(
			ProblemType &objFunc,
			const TVector &x,
			const TVector &grad,
			TVector &direction) override
		{
			if (this->descent_strategy == 2)
			{
				direction = -grad;
			}
			else
			{
				direction = hess.ldlt().solve(-grad);

				TVector y = grad - m_prev_grad;
				TVector s = x - m_prev_x;

				double y_s = y.dot(s);
				TVector Bs = hess * s;
				double sBs = s.transpose() * Bs;

				hess += (y * y.transpose()) / y_s - (Bs * Bs.transpose()) / sBs;
			}

			m_prev_x = x;
			m_prev_grad = grad;

			if (std::isnan(direction.squaredNorm()))
			{
				reset_history(x.size());
				increase_descent_strategy();
				polyfem::logger().log(
					this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
					"nan in direction {} (||∇f||={}); reverting to {}",
					direction.dot(grad), this->descent_strategy_name());
				return compute_update_direction(objFunc, x, grad, direction);
			}
			else if (grad.squaredNorm() != 0 && direction.dot(grad) >= 0)
			{
				reset_history(x.size());
				increase_descent_strategy();
				polyfem::logger().log(
					this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
					"L-BFGS direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
					direction.dot(grad), this->descent_strategy_name());
				return compute_update_direction(objFunc, x, grad, direction);
			}

			return true;
		}
	};
} // namespace cppoptlib
