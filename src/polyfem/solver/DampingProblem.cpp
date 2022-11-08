#include "DampingProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/writeOBJ.h>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <filesystem>

namespace polyfem
{
	using namespace utils;

	DampingProblem::DampingProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
	{
		optimization_name = "damping";

		if (!state.problem->is_time_dependent())
			log_and_throw_error("Damping parameter optimization is only supported in transient simulations!");

		param_to_x = [](TVector &x, const State &state) {
			x.setZero(2);
			x(0) = state.assembler.damping_params()[0];
			x(1) = state.assembler.damping_params()[1];
		};

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "damping")
			{
				material_params = param;
				break;
			}
		}

		if (material_params["phi_bound"].get<std::vector<double>>().size() == 0)
		{
			min_phi = 0.0;
			max_phi = std::numeric_limits<double>::max();
		}
		else
		{
			min_phi = material_params["phi_bound"][0];
			max_phi = material_params["phi_bound"][1];
		}

		if (material_params["psi_bound"].get<std::vector<double>>().size() == 0)
		{
			min_psi = 0.0;
			max_psi = std::numeric_limits<double>::max();
		}
		else
		{
			min_psi = material_params["psi_bound"][0];
			max_psi = material_params["psi_bound"][1];
		}
	}

	void DampingProblem::line_search_end()
	{
	}

	double DampingProblem::target_value(const TVector &x)
	{
		return target_weight * j->energy(state);
	}

	double DampingProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			cur_val = target_value(x);
			logger().debug("damping: target = {}", cur_val);
		}
		return cur_val;
	}

	void DampingProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		gradv = j->gradient(state, "damping") * target_weight;
	}

	void DampingProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			target_gradient(x, cur_grad);
			logger().debug("damping: ∇ target = {}, {}", cur_grad(0), cur_grad(1));
		}

		gradv = cur_grad;
	}

	bool DampingProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change)
			return false;

		const double psi = x1(0);
		const double phi = x1(1);

		bool flag = true;
		if (min_phi > phi || max_phi < phi)
			flag = false;
		if (min_psi > psi || max_psi < psi)
			flag = false;

		return flag;
	}

	bool DampingProblem::solution_changed_pre(const TVector &newX)
	{
		json damping_param = {
		{"psi", newX(0)},
		{"phi", newX(1)},
		};
		state.assembler.add_multimaterial(0, damping_param);
		logger().info("Current damping params: {}, {}", newX(0), newX(1));
		return true;
	}

} // namespace polyfem