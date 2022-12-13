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

		const std::map<int, int> &get_boundary_id_to_reduced_param() const { return boundary_id_to_reduced_param; }

		int get_timestep_dim() const { return boundary_ids_list.size(); }

		Eigen::VectorXd inverse_map_grad_timestep(const Eigen::VectorXd &reduced_grad) const;

		Eigen::VectorXd get_current_dirichlet(const int time_step) const
		{
			if (time_step == 0)
				return Eigen::VectorXd::Zero(boundary_id_to_reduced_param.size() * dim);
			else
				return current_dirichlet.segment((time_step - 1) * boundary_id_to_reduced_param.size() * dim, boundary_id_to_reduced_param.size() * dim);
		}

	private:
		std::vector<int> boundary_ids_list;

		int time_steps;
		int dim;

		Eigen::VectorXd starting_dirichlet;
		Eigen::VectorXd current_dirichlet;

		json control_params;

		std::map<int, int> boundary_id_to_reduced_param;

		std::shared_ptr<ControlConstraints> control_constraints_;
	};
} // namespace polyfem