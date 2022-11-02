#include "FrictionParameter.hpp"

namespace polyfem
{
	FrictionParameter::FrictionParameter(std::vector<std::shared_ptr<State>> states_ptr) : Parameter(states_ptr)
	{
		parameter_name_ = "friction";

		optimization_dim_ = 1;

		for (auto state : states_ptr_)
			if (!state->problem->is_time_dependent())
				log_and_throw_error("Friction parameter optimization is only supported in transient simulations!");

		json opt_params;
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "friction")
			{
				material_params = param;
				break;
			}
		}

		if (material_params["bound"].get<std::vector<double>>().size() == 0)
		{
			min_fric = 0.0;
			max_fric = std::numeric_limits<double>::max();
		}
		else
		{
			min_fric = material_params["bound"][0];
			max_fric = material_params["bound"][1];
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

} // namespace polyfem