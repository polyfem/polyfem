#pragma once

#include "Parameter.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem
{
	class MacroStrainParameter : public Parameter
	{
	public:
		MacroStrainParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override;
		
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		bool remesh(Eigen::VectorXd &x) override { return false; }

		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;

		bool pre_solve(const Eigen::VectorXd &newX) override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override;

    private:
        int dim;
		std::vector<int> inactive_entries;

		Eigen::VectorXd initial_disp_grad;
	};
} // namespace polyfem