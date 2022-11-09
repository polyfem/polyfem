#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class ElasticParameter : public Parameter
	{
	public:
		ElasticParameter(std::vector<std::shared_ptr<State>> states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override
		{
			assert(false);
			return Eigen::VectorXd();
		}

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const override
		{
			return x;
		}

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		bool remesh(Eigen::VectorXd &x) override { return false; }

		bool pre_solve(const Eigen::VectorXd &newX) override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override
		{
			Eigen::VectorXd min(x.size());
			// min.setConstant(std::numeric_limits<double>::min());
			if (design_variable_name == "lambda_mu")
			{
				for (int i = 0; i < x.size(); i++)
				{
					if (i % 2 == 0)
						min(i) = min_lambda;
					else
						min(i) = min_mu;
				}
			}
			else if (design_variable_name == "E_nu")
			{
				for (int i = 0; i < x.size(); i++)
				{
					if (i % 2 == 0)
						min(i) = min_E;
					else
						min(i) = min_nu;
				}
			}
			return min;
		}
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override
		{
			Eigen::VectorXd max(x.size());
			// max.setConstant(std::numeric_limits<double>::max());
			if (design_variable_name == "lambda_mu")
			{
				for (int i = 0; i < x.size(); i++)
				{
					if (i % 2 == 0)
						max(i) = max_lambda;
					else
						max(i) = max_mu;
				}
			}
			else if (design_variable_name == "E_nu")
			{
				for (int i = 0; i < x.size(); i++)
				{
					if (i % 2 == 0)
						max(i) = max_E;
					else
						max(i) = max_nu;
				}
			}
			return max;
		}

		std::string design_variable_name = "lambda_mu";

	private:
		double min_mu, min_lambda;
		double max_mu, max_lambda;
		double min_E, min_nu;
		double max_E, max_nu;

		bool has_material_smoothing = false;
		json material_params;
		json smoothing_params;

		double smoothing_weight;
		double target_weight = 1;

		int optimization_dim_ = -1;

		Eigen::SparseMatrix<bool> tt_adjacency;
	};
} // namespace polyfem