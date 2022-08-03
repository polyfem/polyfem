#include "TopologyOptimizationProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <filesystem>

namespace polyfem
{
    TopologyOptimizationProblem::TopologyOptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
    {
        optimization_name = "topology";
        state.args["output"]["paraview"]["options"]["material"] = true;

        if (opt_params.contains("min_density"))
            min_density = opt_params["min_density"];
        if (opt_params.contains("max_density"))
            max_density = opt_params["max_density"];

		// volume constraint
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "topology")
			{
				if (param.contains("bound"))
				{
					min_density = param["bound"][0];
					max_density = param["bound"][1];
				}
				break;
			}
		}

		// mass constraint
		bool has_mass_constraint = false;
		for (const auto &param : opt_params["functionals"])
		{
			if (param["type"] == "mass_constraint")
			{
				mass_params = param;
				has_mass_constraint = true;
				break;
			}
		}

		if (has_mass_constraint)
		{
			j_mass = CompositeFunctional::create("Mass");
			auto &func_mass = *dynamic_cast<MassFunctional *>(j_mass.get());
			func_mass.set_max_mass(mass_params["soft_bound"][1]);
			func_mass.set_min_mass(mass_params["soft_bound"][0]);
		}
    }

	double TopologyOptimizationProblem::mass_value(const TVector &x)
	{
		if (has_mass_constraint)
			return j_mass->energy(state) * mass_params["weight"].get<double>();
		else
			return 0.;
	}

	double TopologyOptimizationProblem::smooth_value(const TVector &x)
	{
        // TODO
		return 0.;
	}

	double TopologyOptimizationProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			double target_val, mass_val, smooth_val;
			target_val = target_value(x);
			mass_val = mass_value(x);
			smooth_val = smooth_value(x);
			logger().debug("target = {}, vol = {}, smooth = {}", target_val, mass_val, smooth_val);
			cur_val = target_val + mass_val + smooth_val;
		}
		return cur_val;
	}

	void TopologyOptimizationProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		gradv = j->gradient(state, "topology");
	}

	void TopologyOptimizationProblem::mass_gradient(const TVector &x, TVector &gradv)
	{
		gradv.setZero(x.size());
		if (!has_mass_constraint)
			return;

		gradv = j_mass->gradient(state, "topology") * mass_params["weight"];
	}

	void TopologyOptimizationProblem::smooth_gradient(const TVector &x, TVector &gradv)
	{
        // TODO
		gradv.setZero(x.size());
	}

	void TopologyOptimizationProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			Eigen::VectorXd grad_target, grad_smoothing, grad_mass;
			target_gradient(x, grad_target);
			smooth_gradient(x, grad_smoothing);
			mass_gradient(x, grad_mass);
			logger().debug("‖∇ target‖ = {}, ‖∇ vol‖ = {}, ‖∇ smooth‖ = {}", grad_target.norm(), grad_mass.norm(), grad_smoothing.norm());
			cur_grad = grad_target + grad_mass + grad_smoothing;
		}

		gradv = cur_grad;
	}

    bool TopologyOptimizationProblem::is_step_valid(const TVector &x0, const TVector &x1)
    {
        const auto &cur_density = state.assembler.lame_params().density_mat_;

        if (cur_density.minCoeff() < min_density || cur_density.maxCoeff() > max_density)
            return false;
        
        return true;
    }

	void TopologyOptimizationProblem::solution_changed(const TVector &newX)
	{
		if (cur_x.size() == newX.size() && cur_x == newX)
			return;

		state.assembler.update_lame_params_density(newX);
		solve_pde(newX);

		cur_x = newX;
	}

	double TopologyOptimizationProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double size = 1;
		const auto lambda0 = state.assembler.lame_params().lambda_mat_;
		const auto mu0 = state.assembler.lame_params().mu_mat_;
		while (size > 0)
		{
			auto newX = x0 + (x1 - x0) * size;
			state.assembler.update_lame_params_density(newX);

			if (!is_step_valid(x0, newX))
				size /= 2.;
			else
				break;
		}
		state.assembler.update_lame_params_density(x0);

		return size;
	}

	void TopologyOptimizationProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		descent_direction = x1 - x0;

		// debug
		if (opt_nonlinear_params.contains("debug_fd") && opt_nonlinear_params["debug_fd"].get<bool>())
		{
			double t = 1e-6;
			TVector new_x = x0 + descent_direction * t;

			solution_changed(new_x);
			double J2 = target_value(new_x);

			solution_changed(x0);
			double J1 = target_value(x0);
			TVector gradv;
			target_gradient(x0, gradv);

			logger().debug("step size: {}, finite difference: {}, derivative: {}", t, (J2 - J1) / t, gradv.dot(descent_direction));
		}
		state.descent_direction = descent_direction;

		sol_at_ls_begin = state.sol;
	}

	void TopologyOptimizationProblem::line_search_end(bool failed)
	{
		if (opt_output_params.contains("export_energies"))
		{
			std::ofstream outfile;
			outfile.open(opt_output_params["export_energies"], std::ofstream::out | std::ofstream::app);

			outfile << value(cur_x) << ", " << target_value(cur_x) << ", " << smooth_value(cur_x) << ", " << mass_value(cur_x) << "\n";
			outfile.close();
		}
	}
}