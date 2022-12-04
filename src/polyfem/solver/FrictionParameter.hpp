#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class FrictionParameter : public Parameter
	{
	public:
		FrictionParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override
		{
			Eigen::VectorXd x(1);
			x(0) = get_state().args["contact"]["friction_coefficient"];
			return x;
		}
		
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		bool remesh(Eigen::VectorXd &x) override { return false; }

		bool pre_solve(const Eigen::VectorXd &newX) override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override;

	private:
		double target_weight = 1;

		double min_fric;
		double max_fric;
	};
} // namespace polyfem