#include <polyfem/optimization/var2sims/ActiveSelectionUtils.hpp>

#include <polyfem/legacy/State.hpp>

#include <Eigen/Core>

#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <unordered_set>

#include <spdlog/fmt/fmt.h>

#include <polyfem/assembler/GenericProblem.hpp>

namespace polyfem::solver
{

	bool is_active_geom_nodes_valid(const Eigen::VectorXi &active_geom_nodes,
									const std::vector<std::shared_ptr<legacy::State>> &states,
									std::string &reason)
	{
		assert(!states.empty());

		// Check state vertex num consistency.
		int num = states[0]->mesh->n_vertices();
		for (auto &s : states)
		{
			if (s->mesh->n_vertices() != num)
			{
				reason = "Mesh vertex num mismatch between states";
				return false;
			}
		}

		if (active_geom_nodes.size() == 0)
		{
			return true;
		}

		// Check active node index range.
		int min_id = active_geom_nodes.minCoeff();
		int max_id = active_geom_nodes.maxCoeff();
		if (min_id < 0 || max_id >= num)
		{
			reason = fmt::format("Invalid active node range [{}, {}]", min_id, max_id);
			return false;
		}

		// Check duplication.
		std::unordered_set<int> id_set(active_geom_nodes.begin(), active_geom_nodes.end());
		if (id_set.size() != active_geom_nodes.size())
		{
			reason = "Duplicate active node id";
			return false;
		}

		return true;
	}

	bool is_active_dims_valid(const Eigen::VectorXi &active_dimensions,
							  const std::vector<std::shared_ptr<legacy::State>> &states,
							  std::string &reason)
	{
		assert(!states.empty());

		// Check state dim consistency.
		int dim = states[0]->mesh->dimension();
		for (auto &s : states)
		{
			if (s->mesh->dimension() != dim)
			{
				reason = "Mesh dimension mismatch between states";
				return false;
			}
		}

		if (active_dimensions.size() == 0)
		{
			return true;
		}

		// Check active dim range.
		int min_dim = active_dimensions.minCoeff();
		int max_dim = active_dimensions.maxCoeff();
		if (min_dim < 0 || max_dim >= dim)
		{
			reason = fmt::format("Invalid active dimension range [{}, {}]", min_dim, max_dim);
			return false;
		}

		// Check duplication.
		std::unordered_set<int> dim_set(active_dimensions.begin(), active_dimensions.end());
		if (dim_set.size() != active_dimensions.size())
		{
			reason = "Duplicate dimensions";
			return false;
		}

		return true;
	}

	bool is_active_dofs_valid(const Eigen::VectorXi &active_dofs,
							  const std::vector<std::shared_ptr<legacy::State>> &states,
							  std::string &reason)
	{
		assert(!states.empty());

		// Check state ndof consistency.
		int ndof = states[0]->ndof();
		for (auto &s : states)
		{
			if (s->ndof() != ndof)
			{
				reason = "legacy::State ndof mismatch between states";
				return false;
			}
		}

		if (active_dofs.size() == 0)
		{
			return true;
		}

		// Check active dof index range.
		const int min_id = active_dofs.minCoeff();
		const int max_id = active_dofs.maxCoeff();
		if (min_id < 0 || max_id >= ndof)
		{
			reason = fmt::format("Invalid active dof range [{}, {}]", min_id, max_id);
			return false;
		}

		// Check duplication.
		std::unordered_set<int> id_set(active_dofs.begin(), active_dofs.end());
		if (id_set.size() != active_dofs.size())
		{
			reason = "Duplicate active dof id";
			return false;
		}

		return true;
	}

	bool is_active_time_slices_valid(const Eigen::VectorXi &active_time_slices,
									 const std::vector<std::shared_ptr<legacy::State>> &states,
									 std::string &reason)
	{
		assert(!states.empty());

		// Check state time_steps consistency.
		int time_steps = states[0]->args["time"]["time_steps"];
		for (auto &s : states)
		{
			if (int(s->args["time"]["time_steps"]) != time_steps)
			{
				reason = "time_steps mismatch between states";
				return false;
			}
		}

		if (active_time_slices.size() == 0)
		{
			return true;
		}

		// Check range.
		const int min_id = active_time_slices.minCoeff();
		const int max_id = active_time_slices.maxCoeff();
		if (min_id < 0 || max_id >= time_steps)
		{
			reason = fmt::format("Invalid active time slice range [{}, {}]", min_id, max_id);
			return false;
		}

		// Check duplication.
		std::unordered_set<int> id_set(active_time_slices.begin(), active_time_slices.end());
		if (id_set.size() != active_time_slices.size())
		{
			reason = "Duplicate active time slices";
			return false;
		}

		return true;
	}

	bool is_active_dirichlet_boundary_ids_valid(const Eigen::VectorXi &active_boundary_ids,
												const std::vector<std::shared_ptr<legacy::State>> &states,
												std::string &reason)
	{
		assert(!states.empty());

		if (active_boundary_ids.size() == 0)
		{
			return true;
		}

		// Check duplication.
		std::unordered_set<int> id_set(active_boundary_ids.begin(), active_boundary_ids.end());
		if (id_set.size() != active_boundary_ids.size())
		{
			reason = "Duplicate active boundary id";
			return false;
		}

		// Validate ids exist and dimensions are all active.
		int dim = states[0]->mesh->dimension();
		for (auto &s : states)
		{
			// boundary_dims is a map where
			// key: boundary id
			// value: array<bool, 3>, true means that dimension is active.
			auto boundry_dims = s->boundary_conditions_ids("dirichlet_boundary");
			for (int i = 0; i < active_boundary_ids.size(); ++i)
			{
				// Check boundary id exists.
				int id = active_boundary_ids(i);
				auto iter = boundry_dims.find(id);
				if (iter == boundry_dims.end())
				{
					reason = fmt::format("Invalid dirichlet boundary id {}", id);
					return false;
				}

				if (s->mesh->dimension() != dim)
				{
					reason = "Inconsistent boundary node dimension";
					return false;
				}

				// Check boundary does not have inactive dimension.
				for (int d = 0; d < dim; ++d)
				{
					if (!iter->second[d])
					{
						reason = fmt::format("Dirichlet boundary id {} has inactive dimensions (not supported)", id);
						return false;
					}
				}
			}
		}

		return true;
	}

	bool is_active_dirichlet_node_valid(const Eigen::VectorXi &active_dirichlet_nodes,
										const std::vector<std::shared_ptr<legacy::State>> &states,
										std::string &reason)
	{
		assert(!states.empty());

		// Basic node index checks (range/duplicates and mesh consistency).
		if (!is_active_geom_nodes_valid(active_dirichlet_nodes, states, reason))
		{
			return false;
		}

		int vertex_num = states[0]->mesh->n_vertices();
		int dim = states[0]->mesh->dimension();

		for (auto &s : states)
		{
			if (s->mesh->n_vertices() != vertex_num)
			{
				reason = "Mesh vertex num mismatch between states";
				return false;
			}

			if (s->mesh->dimension() != dim)
			{
				reason = "Mesh dimension mismatch between states";
				return false;
			}

			if (!s->problem->has_nodal_dirichlet())
			{
				reason = "Nodal Dirichlet matrix is missing (cannot update nodal Dirichlet values)";
				return false;
			}

			for (int i = 0; i < active_dirichlet_nodes.size(); ++i)
			{
				int v_in = active_dirichlet_nodes(i);
				int v = s->in_node_to_node(v_in);
				if (v < 0 || v >= vertex_num)
				{
					reason = fmt::format("Invalid in_node_to_node mapping: input vertex {} -> {}", v_in, v);
					return false;
				}

				int tag = s->mesh->get_node_id(v);
				if (!s->problem->is_nodal_dirichlet_boundary(v, tag))
				{
					reason = fmt::format("Input vertex {} is not a nodal Dirichlet node", v_in);
					return false;
				}

				for (int d = 0; d < dim; ++d)
				{
					if (!s->problem->is_nodal_dimension_dirichlet(v, tag, d))
					{
						reason = fmt::format("Nodal Dirichlet at input vertex {} has inactive dimensions (not supported)", v_in);
						return false;
					}
				}
			}
		}

		return true;
	}

	bool is_active_pressure_boundary_ids_valid(const Eigen::VectorXi &active_boundary_ids,
											   const std::vector<std::shared_ptr<legacy::State>> &states,
											   std::string &reason)
	{
		assert(!states.empty());

		if (active_boundary_ids.size() == 0)
		{
			return true;
		}

		// Check duplication.
		std::unordered_set<int> id_set(active_boundary_ids.begin(), active_boundary_ids.end());
		if (id_set.size() != active_boundary_ids.size())
		{
			reason = "Duplicate pressure boundary id";
			return false;
		}

		for (auto &s : states)
		{
			for (auto id : active_boundary_ids)
			{
				if (!s->problem->is_boundary_pressure(id))
				{
					reason = fmt::format("Invalid pressure boundary id {}", id);
					return false;
				}
			}
		}

		return true;
	}

} // namespace polyfem::solver
