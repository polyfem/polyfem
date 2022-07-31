// L-BFGS solver (Using the LBFGSpp under MIT License).

#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/NonlinearSolver.hpp>
#include <polysolve/LinearSolver.hpp>
#include <polyfem/MatrixUtils.hpp>

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

		GradientDescentSolver(const json &solver_params_)
			: Superclass(solver_params_)
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
		void reset(const ProblemType &objFunc, const TVector &x) override
		{
			Superclass::reset(objFunc, x);
			this->descent_strategy = 1;
		}

		void remesh_reset(const ProblemType &objFunc, const TVector &x) override
		{
			Superclass::remesh_reset(objFunc, x);
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
