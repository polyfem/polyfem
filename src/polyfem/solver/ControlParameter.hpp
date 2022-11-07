#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class ControlParameter : public Parameter
	{
	public:
		ControlParameter(std::vector<std::shared_ptr<State>> states_ptr);

		void update() override
		{
		}

		void map(const Eigen::MatrixXd &x, Eigen::MatrixXd &q) const override
		{
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override { return true; };

		bool remesh(Eigen::VectorXd &x) override { return false; };

		bool pre_solve(const Eigen::VectorXd &newX) override;

		const std::map<int, int> &get_optimize_boundary_ids_to_position() { return optimize_boundary_ids_to_position; }

	private:
		double target_weight = 1;
		double smoothing_weight;
		std::vector<int> boundary_ids_list;

		int time_steps;

		json control_params;
		json smoothing_params;

		std::map<int, int> optimize_boundary_ids_to_position;
	};
} // namespace polyfem