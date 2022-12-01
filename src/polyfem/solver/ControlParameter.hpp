#pragma once

#include "Parameter.hpp"
#include "constraints/ControlConstraints.hpp"

namespace polyfem
{
	class ControlParameter : public Parameter
	{
	public:
		ControlParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;

		Eigen::VectorXd initial_guess() const override
		{
			return starting_dirichlet;
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override { return true; };

		bool remesh(Eigen::VectorXd &x) override { return false; };

		bool pre_solve(const Eigen::VectorXd &newX) override;

		const std::map<int, int> &get_boundary_id_to_reduced_param() { return boundary_id_to_reduced_param; }

	private:
		double target_weight = 1;
		double smoothing_weight;
		std::vector<int> boundary_ids_list;

		int time_steps;
		int dim;

		Eigen::VectorXd starting_dirichlet;

		json control_params;

		std::map<int, int> boundary_id_to_reduced_param;

		std::shared_ptr<ControlConstraints> control_constraints_;
	};
} // namespace polyfem