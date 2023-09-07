#pragma once

#include <polyfem/Common.hpp>
#include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

namespace cppoptlib
{
	template <typename ProblemType>
	class GradientDescentSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		GradientDescentSolver(const json &solver_params_, const double dt, const double characteristic_length)
			: Superclass(solver_params_, dt, characteristic_length)
		{
		}

		std::string name() const override { return "GradientDescent"; }

	protected:
		virtual int default_descent_strategy() override { return 1; }

		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy_) const override
		{
			switch (descent_strategy_)
			{
			case 1:
				return "gradient descent";
			default:
				throw std::invalid_argument("invalid descent strategy");
			}
		}

		void increase_descent_strategy() override
		{
			assert(this->descent_strategy <= 1);
		}

	protected:
		void reset(const int ndof) override
		{
			Superclass::reset(ndof);
			this->descent_strategy = 1;
		}

		virtual bool compute_update_direction(
			ProblemType &objFunc,
			const TVector &x,
			const TVector &grad,
			TVector &direction) override
		{
			direction = -grad;

			return true;
		}
	};
} // namespace cppoptlib
