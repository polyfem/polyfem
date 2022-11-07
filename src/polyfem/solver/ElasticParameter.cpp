#include "ElasticParameter.hpp"

#include <polyfem/mesh/Mesh.hpp>

namespace polyfem
{
	ElasticParameter::ElasticParameter(std::vector<std::shared_ptr<State>> states_ptr) : Parameter(states_ptr)
	{
		parameter_name_ = "material";
		full_dim_ = states_ptr_[0]->bases.size() * 2;

		json opt_params = states_ptr_[0]->args["optimization"];
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "material")
			{
				material_params = param;
				break;
			}
		}

		if (material_params["mu_bound"].get<std::vector<double>>().size() == 0)
		{
			min_mu = 0.0;
			max_mu = std::numeric_limits<double>::max();
		}
		else
		{
			min_mu = material_params["mu_bound"][0];
			max_mu = material_params["mu_bound"][1];
		}

		if (material_params["lambda_bound"].get<std::vector<double>>().size() == 0)
		{
			min_lambda = 0.0;
			max_lambda = std::numeric_limits<double>::max();
		}
		else
		{
			min_lambda = material_params["lambda_bound"][0];
			max_lambda = material_params["lambda_bound"][1];
		}

		if (material_params["E_bound"].get<std::vector<double>>().size() == 0)
		{
			min_E = 0.0;
			max_E = std::numeric_limits<double>::max();
		}
		else
		{
			min_E = material_params["E_bound"][0];
			max_E = material_params["E_bound"][1];
		}

		if (material_params["nu_bound"].get<std::vector<double>>().size() == 0)
		{
			min_nu = 0.0;
			max_nu = std::numeric_limits<double>::max();
		}
		else
		{
			min_nu = material_params["nu_bound"][0];
			max_nu = material_params["nu_bound"][1];
		}

		has_material_smoothing = false;
		for (const auto &param : opt_params["functionals"])
		{
			if (param["type"] == "material_smoothing")
			{
				smoothing_params = param;
				has_material_smoothing = true;
				smoothing_weight = smoothing_params.value("weight", 1.0);
				break;
			}
			else
				target_weight = param.value("weight", 1.0);
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

	bool ElasticParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		return true;
	}
} // namespace polyfem