#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class ElasticParameter : public Parameter
	{
	public:
		ElasticParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override;

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override;

		bool pre_solve(const Eigen::VectorXd &newX) override;

	private:
		double min_mu, min_lambda;
		double max_mu, max_lambda;
		double min_E, min_nu;
		double max_E, max_nu;

		std::string design_variable_name = "lambda_mu";
	};
} // namespace polyfem