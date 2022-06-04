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
	class LBFGSSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		LBFGSSolver(const json &solver_params)
			: Superclass(solver_params)
		{
		}

		std::string name() const override { return "L-BFGS"; }

	protected:
		virtual int default_descent_strategy() override { return 1; }

		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy) const override
		{
			switch (descent_strategy)
			{
			case 1:
				return "L-BFGS";
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

			m_bfgs.reset(m_prev_x.size(), m_history_size);

			assert(this->descent_strategy <= 2);
		}

	protected:
		LBFGSpp::BFGSMat<Scalar> m_bfgs; // Approximation to the Hessian matrix

		/// The number of corrections to approximate the inverse Hessian matrix.
		/// The L-BFGS routine stores the computation results of previous \ref m
		/// iterations to approximate the inverse Hessian matrix of the current
		/// iteration. This parameter controls the size of the limited memories
		/// (corrections). The default value is \c 6. Values less than \c 3 are
		/// not recommended. Large values will result in excessive computing time.
		int m_history_size = 6;

		TVector m_prev_x;    // Previous x
		TVector m_prev_grad; // Previous gradient

		void reset(const ProblemType &objFunc, const TVector &x) override
		{
			Superclass::reset(objFunc, x);

			m_bfgs.reset(x.size(), m_history_size);
			m_prev_x.resize(x.size());
			m_prev_grad.resize(x.size());

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
				// Use gradient descent in the first iteration or if the previous iteration failed
				direction = -grad;
			}
			else
			{
				// Update s and y
				// s_{i+1} = x_{i+1} - x_i
				// y_{i+1} = g_{i+1} - g_i
				m_bfgs.add_correction(x - m_prev_x, grad - m_prev_grad);

				// Recursive formula to compute d = -H * g
				m_bfgs.apply_Hv(grad, -Scalar(1), direction);
			}

			m_prev_x = x;
			m_prev_grad = grad;

			return true;
		}
	};
} // namespace cppoptlib
