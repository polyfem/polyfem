#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class FrictionParameter : Parameter
	{
	public:
		FrictionParameter(std::vector<std::shared_ptr<State>> states_ptr);

		void update() override
		{
		}

		void map(const Eigen::MatrixXd &x, Eigen::MatrixXd &q) override
		{
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		bool remesh(Eigen::VectorXd &x) override { return false; }

		bool pre_solve(const Eigen::VectorXd &newX) override;

	private:
		json material_params;

		double target_weight = 1;

		double min_fric;
		double max_fric;
	};
} // namespace polyfem