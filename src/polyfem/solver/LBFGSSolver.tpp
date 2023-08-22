// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include "LBFGSSolver.hpp"

namespace cppoptlib
{
	template <typename ProblemType>
	LBFGSSolver<ProblemType>::LBFGSSolver(const json &solver_params, const double dt, const double characteristic_length)
		: Superclass(solver_params, dt, characteristic_length)
	{
		m_history_size = solver_params.value("history_size", 6);
	}

	template <typename ProblemType>
	std::string LBFGSSolver<ProblemType>::descent_strategy_name(int descent_strategy) const
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

	template <typename ProblemType>
	void LBFGSSolver<ProblemType>::increase_descent_strategy()
	{
		if (this->descent_strategy == 1)
			this->descent_strategy++;

		m_bfgs.reset(m_prev_x.size(), m_history_size);

		assert(this->descent_strategy <= 2);
	}

	template <typename ProblemType>
	void LBFGSSolver<ProblemType>::reset(const int ndof)
	{
		Superclass::reset(ndof);

		m_bfgs.reset(ndof, m_history_size);

		// Use gradient descent for first iteration
		this->descent_strategy = 2;
	}

	template <typename ProblemType>
	bool LBFGSSolver<ProblemType>::compute_update_direction(
		ProblemType &objFunc,
		const TVector &x,
		const TVector &grad,
		TVector &direction)
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
			assert(m_prev_x.size() == x.size());
			assert(m_prev_grad.size() == grad.size());
			m_bfgs.add_correction(x - m_prev_x, grad - m_prev_grad);

			// Recursive formula to compute d = -H * g
			m_bfgs.apply_Hv(grad, -Scalar(1), direction);
		}

		m_prev_x = x;
		m_prev_grad = grad;

		return true;
	}
} // namespace cppoptlib
