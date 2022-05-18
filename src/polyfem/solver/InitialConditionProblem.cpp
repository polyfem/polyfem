#include <polyfem/InitialConditionProblem.hpp>

#include <polyfem/Types.hpp>
#include <polyfem/Timer.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <igl/writeOBJ.h>

#include <filesystem>

namespace polyfem
{
    InitialConditionProblem::InitialConditionProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args): OptimizationProblem(state_, j_, args)
    {
        assert(state.problem->is_time_dependent());
        optimization_name = "initial";

        const int dof = state.n_bases;
        const int dim_ = dim;
        x_to_param = [dim_, dof](const TVector& x, Eigen::MatrixXd& init_sol, Eigen::MatrixXd& init_vel)
        {
            init_sol.setZero(dof * dim_, 1);
            init_vel.setZero(dof * dim_, 1);
            for (int i = 0; i < dof; i++)
                for (int d = 0; d < x.size(); d++)
                    init_vel(i * dim_ + d) = x(d);
        };

        param_to_x = [dim_](TVector& x, const Eigen::MatrixXd& init_sol, const Eigen::MatrixXd& init_vel)
        {
            x = init_vel.block(0, 0, dim_, 1);
        };

        dparam_to_dx = [dim_, dof](TVector& x, const Eigen::MatrixXd& init_sol, const Eigen::MatrixXd& init_vel)
        {
            x.setZero(dim_);
            for (int i = 0; i < dof; i++)
                x += init_vel.block(i * dim_, 0, dim_, 1);
        };
    }

    void InitialConditionProblem::line_search_begin(const TVector &x0, const TVector &x1)
    {
        descent_direction = x1 - x0;

        // debug
        if (opt_params.contains("debug_fd") && opt_params["debug_fd"].get<bool>()) {
            double t = 1e-5;
            TVector new_x = x0 + descent_direction * t;

            solution_changed(new_x);
            double J2 = value(new_x);

            solution_changed(x0);
            double J1 = value(x0);
            TVector gradv;
            gradient(x0, gradv);

            logger().debug("finite difference: {}, derivative: {}", (J2 - J1) / t, gradv.dot(descent_direction));
        }
    }

    void InitialConditionProblem::line_search_end(bool failed)
    {
        if (!failed)
            return;
    }

    double InitialConditionProblem::value(const TVector &x)
    {
        return j->energy(state);
    }

    void InitialConditionProblem::gradient(const TVector &x, TVector &gradv)
    {
        TVector tmp = j->gradient(state, "initial-condition");

        Eigen::MatrixXd init_sol(tmp.size() / 2, 1), init_vel(tmp.size() / 2, 1);
        for (int i = 0; i < init_sol.size(); i++) {
            init_sol(i) = tmp(i);
            init_vel(i) = tmp(i + init_sol.size());
        }
        
        dparam_to_dx(gradv, init_sol, init_vel);
    }

    void InitialConditionProblem::post_step(const int iter_num, const TVector &x0)
    {
        iter++;
    }

    void InitialConditionProblem::solution_changed(const TVector &newX)
    {
        static TVector cache_x;
        if (cache_x.size() == newX.size() && cache_x == newX)
            return;
        
        Eigen::MatrixXd init_sol, init_vel;
        x_to_param(newX, init_sol, init_vel);
        state.initial_sol_update = init_sol;
        state.initial_vel_update = init_vel;

        solve_pde(newX);
        cache_x = newX;
    }
}