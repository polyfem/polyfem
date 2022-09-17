#pragma once

#include <polyfem/Common.hpp>
#include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

#include "MMASolverAux.hpp"

namespace cppoptlib
{
	template <typename ProblemType>
	class MMASolver : public NonlinearSolver<ProblemType>
    {
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		MMASolver(const json &solver_params)
			: Superclass(solver_params)
		{
		}

		std::string name() const override { return "MMA"; }

	protected:
		virtual int default_descent_strategy() override { return 1; }

		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy) const override
		{
			switch (descent_strategy)
			{
			case 1:
				return "MMA";
			default:
				throw std::invalid_argument("invalid descent strategy");
			}
		}

		void increase_descent_strategy() override
		{
			assert(this->descent_strategy <= 1);
		}

	protected:
		std::shared_ptr<MMASolverAux> mma;

		void reset(const int ndof) override
		{
			Superclass::reset(ndof);
            mma.reset();
		}

		void remesh_reset(const ProblemType &objFunc, const TVector &x) override
		{
			Superclass::remesh_reset(objFunc, x);
            mma.reset();
		}

		virtual bool compute_update_direction(
			ProblemType &objFunc,
			const TVector &x,
			const TVector &grad,
			TVector &direction) override
		{
			TVector lower_bound = objFunc.get_lower_bound(x);
			TVector upper_bound = objFunc.get_upper_bound(x);

			const int m = objFunc.n_inequality_constraints();

            if (!mma)
                mma = std::make_shared<MMASolverAux>(x.size(), m);

            Eigen::VectorXd g, dg;
            g.setZero(m);
            dg.setZero(m * x.size());
			for (int i = 0; i < m; i++)
			{
				g(i) = objFunc.inequality_constraint_val(x, i);
				auto gradg = objFunc.inequality_constraint_grad(x, i);
				for (int j = 0; j < gradg.size(); j++)
					dg(j * m + i) = gradg(j);
			}
			auto y = x;
			mma->Update(y.data(), grad.data(), g.data(), dg.data(), lower_bound.data(), upper_bound.data());
            direction = y - x;

			if (std::isnan(direction.squaredNorm()))
			{
                polyfem::logger().error("nan in direction.");
                throw std::runtime_error("nan in direction.");
			}
			// else if (grad.squaredNorm() != 0 && direction.dot(grad) > 0)
			// {
            //     polyfem::logger().error("Direction is not a descent direction, stop.");
            //     throw std::runtime_error("Direction is not a descent direction, stop.");
			// }

			return true;
		}
	};

}