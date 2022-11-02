#include "DampingParameter.hpp"

namespace polyfem
{
	DampingParameter::DampingParameter(std::vector<std::shared_ptr<State>> states_ptr) : Parameter(states_ptr)
	{
		parameter_name_ = "damping";

		optimization_dim_ = 2;

		for (auto state : states_ptr_)
			if (!state->problem->is_time_dependent())
				log_and_throw_error("Damping parameter optimization is only supported in transient simulations!");

		json opt_params;
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "damping")
			{
				material_params = param;
				break;
			}
		}

		if (material_params["phi_bound"].get<std::vector<double>>().size() == 0)
		{
			min_phi = 0.0;
			max_phi = std::numeric_limits<double>::max();
		}
		else
		{
			min_phi = material_params["phi_bound"][0];
			max_phi = material_params["phi_bound"][1];
		}

		if (material_params["psi_bound"].get<std::vector<double>>().size() == 0)
		{
			min_psi = 0.0;
			max_psi = std::numeric_limits<double>::max();
		}
		else
		{
			min_psi = material_params["psi_bound"][0];
			max_psi = material_params["psi_bound"][1];
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
} // namespace polyfem