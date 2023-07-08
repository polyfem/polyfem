#pragma once

#include "MMASolver.hpp"

namespace cppoptlib
{
    template <typename ProblemType>
    void MMASolver<ProblemType>::reset(const int ndof)
    {
        Superclass::reset(ndof);
        mma.reset();
    }

    template <typename ProblemType>
    bool MMASolver<ProblemType>::compute_update_direction(
        ProblemType &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        TVector lower_bound = Superclass::get_lower_bound(x);
        TVector upper_bound = Superclass::get_upper_bound(x);

        const int m = constraints_.size();

        if (!mma)
            mma = std::make_shared<MMASolverAux>(x.size(), m);

        Eigen::VectorXd g, gradv, dg;
        g.setZero(m);
        gradv.setZero(x.size());
        dg.setZero(m * x.size());
        for (int i = 0; i < m; i++)
        {
        	g(i) = constraints_[i]->value(x);
            for (int j = 0; j < objFunc.n_states(); j++)
            {
                objFunc.get_state(j)->solve_adjoint_cached(constraints_[i]->compute_adjoint_rhs(x, *(objFunc.get_state(j)))); // caches inside state
            }

        	constraints_[i]->first_derivative(x, gradv);
            dg(Eigen::seqN(0, gradv.size(), m)) = gradv;
        }
        polyfem::logger().info("Constraint values are {}", g.transpose());
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
}