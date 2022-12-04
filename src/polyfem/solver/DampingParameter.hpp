#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class DampingParameter : public Parameter
	{
	public:
		DampingParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override
		{
			Eigen::VectorXd x(2);
			x << get_state().assembler.damping_params()[0], get_state().assembler.damping_params()[1];
			return x;
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		bool remesh(Eigen::VectorXd &x) override { return false; }

		bool pre_solve(const Eigen::VectorXd &newX) override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override;

	private:
		double min_phi, min_psi;
		double max_phi, max_psi;
	};
} // namespace polyfem