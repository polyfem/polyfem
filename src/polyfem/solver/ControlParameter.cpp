#include "ControlParameter.hpp"

#include <polyfem/assembler/GenericProblem.hpp>

namespace polyfem
{
	ControlParameter::ControlParameter(std::vector<std::shared_ptr<State>> states_ptr_) : Parameter(states_ptr_)
	{
		assert(states_ptr_[0]->problem->is_time_dependent());
		parameter_name_ = "control";

		if (states_ptr_.size() > 1)
		{
			logger().error("One control parameter cannot be tied to more than 1 state!");
			return;
		}

		full_dim_ = states_ptr_[0]->boundary_nodes.size() * states_ptr_[0]->args["time"]["time_steps"].get<int>();

		json opt_params = states_ptr_[0]->args["optimization"];
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

		boundary_ids_list.resize(states_ptr_[0]->boundary_nodes.size());
		int dim = states_ptr_[0]->mesh->dimension();
		for (auto it = states_ptr_[0]->local_boundary.begin(); it != states_ptr_[0]->local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = states_ptr_[0]->bases[lb.element_id()];
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = b.local_nodes_for_primitive(primitive_global_id, *states_ptr_[0]->mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					auto &bs = b.bases[nodes(n)];
					for (size_t g = 0; g < bs.global().size(); ++g)
					{
						const int base_index = bs.global()[g].index * dim;
						for (int d = 0; d < dim; ++d)
						{
							int boundary_id = states_ptr_[0]->mesh->get_boundary_id(primitive_global_id);
							if (states_ptr_[0]->problem->is_dimension_dirichet(boundary_id, d))
							{
								auto result = std::find(states_ptr_[0]->boundary_nodes.begin(), states_ptr_[0]->boundary_nodes.end(), base_index + d);
								assert(result != states_ptr_[0]->boundary_nodes.end());
								int index = result - states_ptr_[0]->boundary_nodes.begin();
								boundary_ids_list[index] = boundary_id;
							}
						}
					}
				}
			}
		}
		assert(states_ptr_[0]->boundary_nodes.size() == boundary_ids_list.size());

		smoothing_weight = smoothing_params.contains("weight") ? smoothing_params["weight"].get<double>() : 1.;

		time_steps = states_ptr_[0]->args["time"]["time_steps"].get<int>();
		if (time_steps < 1)
			logger().error("Set time_steps for control optimization, currently {}!", time_steps);
	}

	bool ControlParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		// Implement some max on velocity of dirichlet.
		return true;
	}

	bool ControlParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		auto &problem = *dynamic_cast<assembler::GenericTensorProblem *>(states_ptr_[0]->problem.get());
		for (const auto &kv : optimize_boundary_ids_to_position)
		{
			json dirichlet_bc;
			if (states_ptr_[0]->mesh->dimension() == 2)
				dirichlet_bc = {{}, {}};
			else
				dirichlet_bc = {{}, {}, {}};
			for (int k = 0; k < states_ptr_[0]->mesh->dimension(); ++k)
			{
				for (int t = 0; t < time_steps; ++t)
				{
					dirichlet_bc[k].push_back(newX(t * optimize_boundary_ids_to_position.size() * states_ptr_[0]->mesh->dimension() + kv.second * states_ptr_[0]->mesh->dimension() + k));
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

} // namespace polyfem