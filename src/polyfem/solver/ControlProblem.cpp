#include "ControlProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <filesystem>

namespace polyfem
{

	ControlProblem::ControlProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
	{
		optimization_name = "control";

		for (const auto &functional_params : opt_params["functionals"])
		{
			if (functional_params["type"] == "control_smoothing")
			{
				smoothing_params = functional_params;
			}
		}
		for (const auto &params : opt_params["parameters"])
		{
			if (params["type"] == "control")
			{
				control_params = params;
			}
		}

		if (control_params.contains("surface_selection"))
		{
			int count = 0;
			for (int i : control_params["surface_selection"])
			{
				optimize_boundary_ids_to_position[i] = count;
				count++;
			}
		}

		boundary_ids_list.resize(state.boundary_nodes.size());
		int dim = state.mesh->dimension();
		for (auto it = state.local_boundary.begin(); it != state.local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = state.bases[lb.element_id()];
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = b.local_nodes_for_primitive(primitive_global_id, *state.mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					auto &bs = b.bases[nodes(n)];
					for (size_t g = 0; g < bs.global().size(); ++g)
					{
						const int base_index = bs.global()[g].index * dim;
						for (int d = 0; d < dim; ++d)
						{
							int boundary_id = state.mesh->get_boundary_id(primitive_global_id);
							if (state.problem->is_dimension_dirichet(boundary_id, d))
							{
								auto result = std::find(state.boundary_nodes.begin(), state.boundary_nodes.end(), base_index + d);
								assert(result != state.boundary_nodes.end());
								int index = result - state.boundary_nodes.begin();
								boundary_ids_list[index] = boundary_id;
							}
						}
					}
				}
			}
		}
		assert(state.boundary_nodes.size() == boundary_ids_list.size());

		smoothing_weight = smoothing_params.contains("weight") ? smoothing_params["weight"].get<double>() : 1.;

		time_steps = state.args["time"]["time_steps"].get<int>();
		if (time_steps < 1)
			logger().error("Set time_steps for control optimization, currently {}!", time_steps);

		x_to_param = [&](const TVector &x, TVector &param) {
			param.setZero(boundary_ids_list.size());
			for (int t = 0; t < time_steps; ++t)
			{
				for (int b = 0; b < boundary_ids_list.size(); ++b)
				{
					int boundary_id = boundary_ids_list[b];
					if (optimize_boundary_ids_to_position.count(boundary_id) == 0)
						continue;
					int dim = b % state.mesh->dimension();
					int position = optimize_boundary_ids_to_position[boundary_id];
					param(t * boundary_ids_list.size() + b) = x(t * optimize_boundary_ids_to_position.size() * state.mesh->dimension() + position * state.mesh->dimension() + dim);
				}
			}
		};
		param_to_x = [&](TVector &x, const TVector &param) {
			x.setZero(time_steps * optimize_boundary_ids_to_position.size() * state.mesh->dimension());
			assert(param.size() == (boundary_ids_list.size() * time_steps));
			for (int t = 0; t < time_steps; ++t)
				for (int b = 0; b < boundary_ids_list.size(); ++b)
				{
					int boundary_id = boundary_ids_list[b];
					if (optimize_boundary_ids_to_position.count(boundary_id) == 0)
						continue;
					int dim = b % state.mesh->dimension();
					int position = optimize_boundary_ids_to_position[boundary_id];
					x(t * optimize_boundary_ids_to_position.size() * state.mesh->dimension() + position * state.mesh->dimension() + dim) = param(t * boundary_ids_list.size() + b);
				}
		};
		dparam_to_dx = [&](TVector &dx, const TVector &dparam) {
			dx.setZero(time_steps * optimize_boundary_ids_to_position.size() * state.mesh->dimension());
			assert(dparam.size() == (boundary_ids_list.size() * time_steps));
			for (int t = 0; t < time_steps; ++t)
				for (int b = 0; b < boundary_ids_list.size(); ++b)
				{
					int boundary_id = boundary_ids_list[b];
					if (optimize_boundary_ids_to_position.count(boundary_id) == 0)
						continue;
					int dim = b % state.mesh->dimension();
					int position = optimize_boundary_ids_to_position[boundary_id];
					dx(t * optimize_boundary_ids_to_position.size() * state.mesh->dimension() + position * state.mesh->dimension() + dim) += dparam(t * boundary_ids_list.size() + b);
				}
		};
	}

	double ControlProblem::smooth_value(const TVector &x)
	{
		double val = 0;
		double dt = state.args["time"]["dt"].get<double>();
		int dim_per_timestep = optimize_boundary_ids_to_position.size() * state.mesh->dimension();

		Eigen::VectorXd prev;
		prev.setZero(dim_per_timestep);
		for (int t = 0; t < time_steps; ++t)
		{
			Eigen::VectorXd curr = x.segment(t * dim_per_timestep, dim_per_timestep);
			val += ((curr - prev) / dt).array().pow(2).sum();

			prev = curr;
		}

		return smoothing_weight * val;
	}

	double ControlProblem::value(const TVector &x)
	{
		double target_val, smooth_val;
		target_val = target_value(x);
		smooth_val = smooth_value(x);
		logger().debug("target = {}, smooth = {}", target_val, smooth_val);
		return target_val + smooth_val;
	}

	void ControlProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd dparam = j->gradient(state, "dirichlet");
		logger().info("target dparam norm {}", dparam.norm());

		dparam_to_dx(gradv, dparam);
		gradv *= target_weight;
	}

	void ControlProblem::smooth_gradient(const TVector &x, TVector &gradv)
	{
		gradv.setZero(x.size());
		double dt = state.args["time"]["dt"].get<double>();
		double dim_per_timestep = optimize_boundary_ids_to_position.size() * state.mesh->dimension();

		Eigen::VectorXd prev;
		prev.setZero(dim_per_timestep);
		Eigen::VectorXd curr = x.segment(0, dim_per_timestep);
		for (int t = 0; t < time_steps; ++t)
		{
			gradv.segment(t * dim_per_timestep, dim_per_timestep) += 2 * (curr - prev) / pow(dt, 2);
			prev = curr;
			if (t < time_steps - 1)
			{
				Eigen::VectorXd next = x.segment((t + 1) * dim_per_timestep, dim_per_timestep);
				gradv.segment(t * dim_per_timestep, dim_per_timestep) -= 2 * (next - curr) / pow(dt, 2);
				curr = next;
			}
		}

		gradv *= smoothing_weight;

		// // FD checking.
		// double eps = 1e-8;
		// double val = smooth_value(x);
		// TVector gradv_fd(x.size());
		// for (int i = 0; i < x.size(); ++i)
		// {
		// 	TVector x_ = x;
		// 	x_(i) += eps;
		// 	double val_eps = smooth_value(x_);
		// 	gradv_fd(i) = (val_eps - val) / eps;
		// }
		// std::cout << "smoothing accuracy " << (gradv_fd - gradv).norm() << std::endl;
	}

	void ControlProblem::gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd grad_target, grad_smoothing;
		target_gradient(x, grad_target);
		smooth_gradient(x, grad_smoothing);
		logger().debug("‖∇ target‖ = {}, ‖∇ smooth‖ = {}", grad_target.norm(), grad_smoothing.norm());

		gradv = grad_target + grad_smoothing;
	}

	bool ControlProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		// Implement some max on velocity of dirichlet.
		return true;
	}

	bool ControlProblem::solution_changed_pre(const TVector &newX)
	{
		auto &problem = *dynamic_cast<assembler::GenericTensorProblem *>(state.problem.get());
		for (const auto &kv : optimize_boundary_ids_to_position)
		{
			json dirichlet_bc;
			if (state.mesh->dimension() == 2)
				dirichlet_bc = {{}, {}};
			else
				dirichlet_bc = {{}, {}, {}};
			for (int k = 0; k < state.mesh->dimension(); ++k)
			{
				for (int t = 0; t < time_steps; ++t)
				{
					dirichlet_bc[k].push_back(newX(t * optimize_boundary_ids_to_position.size() * state.mesh->dimension() + kv.second * state.mesh->dimension() + k));
				}
				// Need time_steps + 1 entry, though unused.
				dirichlet_bc[k].push_back(dirichlet_bc[k][time_steps - 1]);
			}
			// std::cout << kv.first << "\t" << kv.second << std::endl;
			// std::cout << "time steps: " << time_steps << std::endl;
			logger().trace("Updating boundary id {} to dirichlet bc {}", kv.first, dirichlet_bc);
			problem.update_dirichlet_boundary(kv.first, dirichlet_bc, true, true, true, "");
		}

		return true;
	}

	void ControlProblem::line_search_end(bool failed)
	{
		if (opt_params.contains("export_energies"))
		{
			std::ofstream outfile;
			outfile.open(opt_params["export_energies"].get<std::string>(), std::ofstream::out | std::ofstream::app);

			outfile << value(cur_x) << ", " << target_value(cur_x) << ", " << smooth_value(cur_x) << "\n";
			outfile.close();
		}
	}
} // namespace polyfem