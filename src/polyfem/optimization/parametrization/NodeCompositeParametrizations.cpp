#include "NodeCompositeParametrizations.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <map>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	VariableToNodes::VariableToNodes(const State &state, const std::vector<int> &active_dimensions) : active_dimensions_(active_dimensions)
	{
		dim = state.mesh->dimension();
		if (active_dimensions_.size() == 0)
			for (int d = 0; d < dim; d++)
				active_dimensions_.push_back(d);
	}

	void VariableToNodes::set_output_indexing(const std::vector<int> &node_ids)
	{
		output_indexing_.resize(node_ids.size() * active_dimensions_.size());
		for (int i = 0; i < node_ids.size(); ++i)
			for (int k = 0; k < active_dimensions_.size(); k++)
				output_indexing_(i * active_dimensions_.size() + k) = node_ids[i] * dim + active_dimensions_[k];
	}

	VariableToInteriorNodes::VariableToInteriorNodes(const State &state, const std::vector<int> &active_dimensions, const std::vector<int> &volume_selection) : VariableToNodes(state, active_dimensions)
	{
		const auto &mesh = state.mesh;

		std::set<int> node_ids;
		for (int e = 0; e < mesh->n_elements(); e++)
		{
			const int body_id = mesh->get_body_id(e);
			if (volume_selection.size() == 0 || std::find(volume_selection.begin(), volume_selection.end(), body_id) != volume_selection.end())
			{
				for (int i = 0; i < mesh->dimension() + 1; i++)
				{
					const int vid = mesh->element_vertex(e, i);
					if (!mesh->is_boundary_vertex(vid))
						node_ids.insert(vid);
				}
			}
		}

		set_output_indexing(std::vector(node_ids.begin(), node_ids.end()));
	}

	VariableToBoundaryNodes::VariableToBoundaryNodes(const State &state, const std::vector<int> &active_dimensions, const std::vector<int> &surface_selection) : VariableToNodes(state, active_dimensions)
	{
		const auto &mesh = state.mesh;
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();

		std::set<int> node_ids;
		for (const auto &lb : state.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);

				if (surface_selection.size() == 0 || std::find(surface_selection.begin(), surface_selection.end(), boundary_id) != surface_selection.end())
					for (long n = 0; n < mesh->dimension(); ++n)
						node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
			}
		}

		set_output_indexing(std::vector(node_ids.begin(), node_ids.end()));
	}

	VariableToBoundaryNodesExclusive::VariableToBoundaryNodesExclusive(const State &state, const std::vector<int> &active_dimensions, const std::vector<int> &exclude_surface_selections) : VariableToNodes(state, active_dimensions)
	{
		const auto &mesh = state.mesh;
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();

		if (!mesh->is_simplicial())
			log_and_throw_adjoint_error("VariableToBoundaryNodesExclusive only supports simplices!");

		std::set<int> excluded_node_ids;
		std::set<int> all_node_ids;
		for (const auto &lb : state.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);

				assert(mesh->is_simplex(e));
				if (std::count(exclude_surface_selections.begin(), exclude_surface_selections.end(), boundary_id))
					for (long n = 0; n < mesh->dimension(); ++n)
						excluded_node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
				for (long n = 0; n < mesh->dimension(); ++n)
					all_node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
			}
		}

		std::vector<int> node_ids;
		std::set_difference(all_node_ids.begin(), all_node_ids.end(), excluded_node_ids.begin(), excluded_node_ids.end(), std::back_inserter(node_ids));

		set_output_indexing(node_ids);
	}
} // namespace polyfem::solver