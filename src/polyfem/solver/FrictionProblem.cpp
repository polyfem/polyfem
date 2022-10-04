#include "FrictionProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/writeOBJ.h>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <filesystem>

namespace polyfem
{
	using namespace utils;

	FrictionProblem::FrictionProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
	{
		optimization_name = "friction";

		if (!state.problem->is_time_dependent())
			log_and_throw_error("Friction parameter optimization is only supported in transient simulations!");

		x_to_param = [](const TVector &x, State &state) {
			state.args["contact"]["friction_coefficient"] = x(0);
			logger().info("Current friction coefficient: {}", x(0));
		};

		param_to_x = [](TVector &x, State &state) {
			x.setZero(1);
			x(0) = state.args["contact"]["friction_coefficient"];
		};

		dparam_to_dx = [](TVector &dx, const Eigen::VectorXd &dparams, State &state) {
			dx = dparams;
		};

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "friction")
			{
				material_params = param;
				break;
			}
		}

		if (material_params["bound"].get<std::vector<double>>().size() == 0)
		{
			min_fric = 0.0;
			max_fric = std::numeric_limits<double>::max();
		}
		else
		{
			min_fric = material_params["bound"][0];
			max_fric = material_params["bound"][1];
		}
	}

	void FrictionProblem::line_search_end(bool failed)
	{
	}

	double FrictionProblem::target_value(const TVector &x)
	{
		return target_weight * j->energy(state);
	}

	double FrictionProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			cur_val = target_value(x);
			logger().debug("friction: target = {}", cur_val);
		}
		return cur_val;
	}

	void FrictionProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd dparam = j->gradient(state, "friction-coefficient");

		dparam_to_dx(gradv, dparam, state);
		gradv *= target_weight;
	}

	void FrictionProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			target_gradient(x, cur_grad);
			logger().debug("friction: ∇ target = {}", cur_grad(0));
		}

		gradv = cur_grad;
	}

	bool FrictionProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change)
			return false;

		const double mu = x1(0);

		bool flag = true;
		if (min_fric > mu || max_fric < mu)
			flag = false;

		return flag;
	}

	bool FrictionProblem::solution_changed_pre(const TVector &newX)
	{
		x_to_param(newX, state);
		return true;
	}

} // namespace polyfem