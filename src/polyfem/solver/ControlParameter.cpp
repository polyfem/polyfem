#include "ControlParameter.hpp"

#include <polyfem/assembler/GenericProblem.hpp>

namespace polyfem
{
	ControlParameter::ControlParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
	{
		assert(states_ptr_[0]->problem->is_time_dependent());
		parameter_name_ = "dirichlet";

		if (states_ptr_.size() > 1)
		{
			logger().error("One control parameter cannot be tied to more than 1 state!");
			return;
		}

		if (get_state().get_bdf_order() > 1)
			logger().error("Dirichlet derivative only works for BDF1!");

		dim = states_ptr_[0]->mesh->dimension();

		time_steps = states_ptr_[0]->args["time"]["time_steps"].get<int>();

		full_dim_ = states_ptr_[0]->boundary_nodes.size() * time_steps;

		if (args.contains("surface_selection"))
		{
			int count = 0;
			for (int i : args["surface_selection"])
			{
				boundary_id_to_reduced_param[i] = count;
				count++;
			}
		}

		starting_dirichlet.setZero(time_steps * boundary_id_to_reduced_param.size() * (dim - 1) * 3);
		for (auto dirichlet : states_ptr_[0]->args["boundary_conditions"]["dirichlet_boundary"])
			for (int k = 0; k < dim; ++k)
				for (int i = 0; i < time_steps; ++i)
				{
					if (boundary_id_to_reduced_param.count(dirichlet["id"].get<int>()) == 0)
						continue;
					starting_dirichlet(i * boundary_id_to_reduced_param.size() * (dim - 1) * 3 + boundary_id_to_reduced_param[dirichlet["id"].get<int>()] * (dim - 1) * 3 + k) = dirichlet["value"][k][i];
				}

		boundary_ids_list.resize(states_ptr_[0]->boundary_nodes.size());
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
						int boundary_id = states_ptr_[0]->mesh->get_boundary_id(primitive_global_id);
						if (boundary_id_to_reduced_param.count(boundary_id) == 0)
							continue;
						for (int d = 0; d < dim; ++d)
						{
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

		Eigen::VectorXd boundary_nodes_rest(states_ptr_[0]->boundary_nodes.size(), 1);
		for (int i = 0; i < states_ptr_[0]->boundary_nodes.size(); ++i)
		{
			int k = i % dim;
			if (k != 0)
				continue;
			boundary_nodes_rest.segment(i, dim) = states_ptr_[0]->mesh_nodes->node_position(states_ptr_[0]->boundary_nodes[i] / dim);
		}

		if (time_steps < 1)
			logger().error("Set time_steps for control optimization, currently {}!", time_steps);

		control_constraints_ = std::make_shared<ControlConstraints>(args, time_steps, states_ptr[0]->mesh->dimension(), boundary_ids_list, boundary_id_to_reduced_param, boundary_nodes_rest);
		optimization_dim_ = control_constraints_->get_optimization_dim();
		current_dirichlet = starting_dirichlet;
	}

	bool ControlParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		// Implement some max on velocity of dirichlet.
		return true;
	}

	// Eigen::MatrixXd ControlParameter::map(const Eigen::VectorXd &x) const
	// {
	// 	Eigen::MatrixXd dirichlet_full;
	// 	control_constraints_->reduced_to_full(x, dirichlet_full);
	// 	return dirichlet_full;
	// }

	Eigen::VectorXd ControlParameter::map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const
	{
		Eigen::VectorXd dreduced;
		control_constraints_->dfull_to_dreduced(x, full_grad, dreduced);
		return dreduced;
	}

	Eigen::VectorXd ControlParameter::inverse_map_grad_timestep(const Eigen::VectorXd &reduced_grad) const
	{
		Eigen::VectorXd full_grad_timestep;
		control_constraints_->dreduced_to_dfull_timestep(reduced_grad, full_grad_timestep);
		return full_grad_timestep;
	}

	bool ControlParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		auto &problem = *dynamic_cast<assembler::GenericTensorProblem *>(states_ptr_[0]->problem.get());
		// This should eventually update dirichlet boundaries per boundary element, using the shape constraint.
		auto constraint_string = control_constraints_->constraint_to_string(newX);
		for (const auto &kv : boundary_id_to_reduced_param)
		{
			json dirichlet_bc = constraint_string[kv.first];
			// Need time_steps + 1 entry, though unused.
			for (int k = 0; k < states_ptr_[0]->mesh->dimension(); ++k)
				dirichlet_bc[k].push_back(dirichlet_bc[k][time_steps - 1]);
			logger().trace("Updating boundary id {} to dirichlet bc {}", kv.first, dirichlet_bc);
			problem.update_dirichlet_boundary(kv.first, dirichlet_bc, true, true, true, "");
		}

		current_dirichlet = newX;

		return true;
	}

} // namespace polyfem