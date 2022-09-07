#include "InitialConditionProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/writeOBJ.h>

#include <filesystem>

namespace polyfem
{
	InitialConditionProblem::InitialConditionProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
	{
		assert(state.problem->is_time_dependent());
		optimization_name = "initial";

		const int dof = state.n_bases;
		const int dim_ = dim;
		x_to_param = [dim_, dof](const TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
			init_sol.setZero(dof * dim_, 1);
			init_vel.setZero(dof * dim_, 1);
			for (int i = 0; i < dof; i++)
				for (int d = 0; d < x.size(); d++)
					init_vel(i * dim_ + d) = x(d);
		};

		param_to_x = [dim_](TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
			x = init_vel.block(0, 0, dim_, 1);
		};

		dparam_to_dx = [dim_, dof](TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
			x.setZero(dim_);
			for (int i = 0; i < dof; i++)
				x += init_vel.block(i * dim_, 0, dim_, 1);
		};
	}

	void InitialConditionProblem::line_search_end(bool failed)
	{
		if (!failed)
			return;
	}

	double InitialConditionProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
			cur_val = target_value(x);
		return cur_val;
	}

	void InitialConditionProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		TVector tmp = j->gradient(state, "initial-condition");

		Eigen::MatrixXd init_sol(tmp.size() / 2, 1), init_vel(tmp.size() / 2, 1);
		for (int i = 0; i < init_sol.size(); i++)
		{
			init_sol(i) = tmp(i);
			init_vel(i) = tmp(i + init_sol.size());
		}

		dparam_to_dx(gradv, init_sol, init_vel, state);
	}

	bool InitialConditionProblem::solution_changed_pre(const TVector &newX)
	{
		x_to_param(newX, state.initial_sol_update, state.initial_vel_update, state);

		return true;
	}

} // namespace polyfem