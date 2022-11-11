#include "ElasticParameter.hpp"

#include <polyfem/mesh/Mesh.hpp>

namespace polyfem
{
	ElasticParameter::ElasticParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
	{
		parameter_name_ = "material";
		full_dim_ = states_ptr_[0]->bases.size() * 2;
		optimization_dim_ = full_dim_;

		if (args["mu_bound"].get<std::vector<double>>().size() == 0)
		{
			min_mu = 0.0;
			max_mu = std::numeric_limits<double>::max();
		}
		else
		{
			min_mu = args["mu_bound"][0];
			max_mu = args["mu_bound"][1];
		}

		if (args["lambda_bound"].get<std::vector<double>>().size() == 0)
		{
			min_lambda = 0.0;
			max_lambda = std::numeric_limits<double>::max();
		}
		else
		{
			min_lambda = args["lambda_bound"][0];
			max_lambda = args["lambda_bound"][1];
		}

		if (args["E_bound"].get<std::vector<double>>().size() == 0)
		{
			min_E = 0.0;
			max_E = std::numeric_limits<double>::max();
		}
		else
		{
			min_E = args["E_bound"][0];
			max_E = args["E_bound"][1];
		}

		if (args["nu_bound"].get<std::vector<double>>().size() == 0)
		{
			min_nu = 0.0;
			max_nu = std::numeric_limits<double>::max();
		}
		else
		{
			min_nu = args["nu_bound"][0];
			max_nu = args["nu_bound"][1];
		}

	}

	bool ElasticParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change_)
			return false;
		pre_solve(x1);

		const auto &cur_lambdas = states_ptr_[0]->assembler.lame_params().lambda_mat_;
		const auto &cur_mus = states_ptr_[0]->assembler.lame_params().mu_mat_;

		bool flag = true;

		if (cur_lambdas.minCoeff() < min_lambda || cur_mus.minCoeff() < min_mu)
			flag = false;
		if (cur_lambdas.maxCoeff() > max_lambda || cur_mus.maxCoeff() > max_mu)
			flag = false;

		for (int e = 0; e < cur_lambdas.size(); e++)
		{
			const double E = cur_mus(e) * (3 * cur_lambdas(e) + 2 * cur_mus(e)) / (cur_lambdas(e) + cur_mus(e));
			const double nu = cur_lambdas(e) / (2 * (cur_lambdas(e) + cur_mus(e)));

			if (E < min_E || E > max_E || nu < min_nu || E > max_E)
				flag = false;
		}

		pre_solve(x0);

		return flag;
	}

	Eigen::VectorXd ElasticParameter::initial_guess() const
	{
		// TODO: apply constraints
		const auto &state = get_state();
		Eigen::VectorXd x(state.bases.size() * 2);
		x.head(state.bases.size()) = state.assembler.lame_params().lambda_mat_;
		x.tail(state.bases.size()) = state.assembler.lame_params().mu_mat_;
		return x;
	}

	Eigen::MatrixXd ElasticParameter::map(const Eigen::VectorXd &x) const
	{
		assert(false);
		return x;
	}

	Eigen::VectorXd ElasticParameter::map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const
	{
		return full_grad;
	}

	bool ElasticParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		// TODO: apply constraints
		Eigen::VectorXd lambda = newX.head(newX.size() / 2);
		Eigen::VectorXd mu     = newX.tail(newX.size() / 2);
        for (auto &state : states_ptr_)
        {
            state->assembler.update_lame_params(lambda, mu);
        }

		return true;
	}

	Eigen::VectorXd ElasticParameter::get_lower_bound(const Eigen::VectorXd &x) const
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
		else
			log_and_throw_error("Box constraints for current parameter is not supported!");
		return min;
	}
	
	Eigen::VectorXd ElasticParameter::get_upper_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd max(x.size());
		max.setConstant(std::numeric_limits<double>::max());
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
		else
			log_and_throw_error("Box constraints for current parameter is not supported!");
		return max;
	}

} // namespace polyfem