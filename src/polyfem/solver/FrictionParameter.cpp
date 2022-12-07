#include "FrictionParameter.hpp"

namespace polyfem
{
	FrictionParameter::FrictionParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
	{
		parameter_name_ = "friction";

		full_dim_ = 1;
		optimization_dim_ = 1;

		max_change_ = args["max_change"];

		for (auto state : states_ptr_)
			if (!state->problem->is_time_dependent())
				log_and_throw_error("Friction parameter optimization is only supported in transient simulations!");

		if (args["bound"].get<std::vector<double>>().size() == 0)
		{
			min_fric = 0.0;
			max_fric = std::numeric_limits<double>::max();
		}
		else
		{
			min_fric = args["bound"][0];
			max_fric = args["bound"][1];
		}
	}

	bool FrictionParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change_)
			return false;

		const double mu = x1(0);

		bool flag = true;
		if (min_fric > mu || max_fric < mu)
			flag = false;

		return flag;
	}

	bool FrictionParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		for (auto state : states_ptr_)
			state->args["contact"]["friction_coefficient"] = newX(0);
		return true;
	}

	Eigen::VectorXd FrictionParameter::get_lower_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd min(1);
		min(0) = min_fric;
		return min;
	}

	Eigen::VectorXd FrictionParameter::get_upper_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd max(1);
		max(0) = max_fric;
		return max;
	}

} // namespace polyfem