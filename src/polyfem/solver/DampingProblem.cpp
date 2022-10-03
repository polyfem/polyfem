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

		x_to_param = [](const TVector &x, State &state) {
			json damping_param = {
			{"psi", x(0)},
			{"phi", x(1)},
			};
			state.assembler.add_multimaterial(0, damping_param);
		};

		param_to_x = [](TVector &x, State &state) {
			x.setZero(2);
			x(0) = state.assembler.damping_params()[0];
			x(1) = state.assembler.damping_params()[1];
		};

		dparam_to_dx = [](TVector &dx, const Eigen::VectorXd &dparams, State &state) {
			dx = dparams;
		};

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "damping")
			{
				material_params = param;
				break;
			}
		}
	}

	void DampingProblem::line_search_end(bool failed)
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
			logger().debug("target = {}", cur_val);
		}
		return cur_val;
	}

	void DampingProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd dparam = j->gradient(state, "damping-parameter");

		dparam_to_dx(gradv, dparam, state);
		gradv *= target_weight;
	}

	void DampingProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			target_gradient(x, cur_grad);
			logger().debug("‖∇ target‖ = {}", cur_grad.norm());
		}

		gradv = cur_grad;
	}

	bool DampingProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change)
			return false;
		solution_changed_pre(x1);

		const double psi = state.assembler.damping_params()[0];
		const double phi = state.assembler.damping_params()[1];

		bool flag = true;
		if (psi < 0 || phi < 0)
			flag = false;

		if (material_params.contains("min_phi") && material_params["min_phi"].get<double>() > phi)
			flag = false;
		if (material_params.contains("max_phi") && material_params["max_phi"].get<double>() < phi)
			flag = false;
		if (material_params.contains("min_psi") && material_params["min_psi"].get<double>() > psi)
			flag = false;
		if (material_params.contains("max_psi") && material_params["max_psi"].get<double>() < psi)
			flag = false;

		solution_changed_pre(x0);

		return flag;
	}

	bool DampingProblem::solution_changed_pre(const TVector &newX)
	{
		x_to_param(newX, state);
		return true;
	}

} // namespace polyfem