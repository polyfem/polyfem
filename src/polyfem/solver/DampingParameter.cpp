#include "DampingParameter.hpp"

namespace polyfem
{
	DampingParameter::DampingParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
	{
		parameter_name_ = "damping";

		optimization_dim_ = 2;
		full_dim_ = 2;

		for (auto state : states_ptr_)
			if (!state->problem->is_time_dependent())
				log_and_throw_error("Damping parameter optimization is only supported in transient simulations!");

		max_change_ = args["max_change"];

		if (args["phi_bound"].get<std::vector<double>>().size() == 0)
		{
			min_phi = 0.0;
			max_phi = std::numeric_limits<double>::max();
		}
		else
		{
			min_phi = args["phi_bound"][0];
			max_phi = args["phi_bound"][1];
		}

		if (args["psi_bound"].get<std::vector<double>>().size() == 0)
		{
			min_psi = 0.0;
			max_psi = std::numeric_limits<double>::max();
		}
		else
		{
			min_psi = args["psi_bound"][0];
			max_psi = args["psi_bound"][1];
		}
	}

	bool DampingParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change_)
			return false;

		const double psi = x1(0);
		const double phi = x1(1);

		bool flag = true;
		if (min_phi > phi || max_phi < phi)
			flag = false;
		if (min_psi > psi || max_psi < psi)
			flag = false;

		return flag;
	}

	bool DampingParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		json damping_param = {
			{"psi", newX(0)},
			{"phi", newX(1)},
		};
		for (auto state : states_ptr_)
			state->assembler.add_multimaterial(0, damping_param);
		logger().info("Current damping params: {}, {}", newX(0), newX(1));
		return true;
	}

	Eigen::VectorXd DampingParameter::get_lower_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd min(2);
		min(0) = min_psi;
		min(1) = min_phi;
		return min;
	}

	Eigen::VectorXd DampingParameter::get_upper_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd max(2);
		max(0) = max_psi;
		max(1) = max_phi;
		return max;
	}
} // namespace polyfem