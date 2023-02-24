#pragma once

#include "Parametrization.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/State.hpp>

#include <Eigen/Core>
#include <map>

namespace polyfem::solver
{
	class VariableToNodes : public CompositeParametrization
	{
	public:
		using CompositeParametrization::CompositeParametrization;

		virtual void set_output_indexing(const std::vector<int> node_ids) final
		{
			output_indexing_.resize(node_ids.size() * dim);
			for (int i = 0; i < node_ids.size(); ++i)
				for (int k = 0; k < dim; ++k)
					output_indexing_(i * dim + k) = node_ids[i] * dim + k;
		}

	protected:
		int dim;
	};

	class VariableToInteriorNodes : public VariableToNodes
	{
	public:
		VariableToInteriorNodes(const std::vector<std::shared_ptr<Parametrization>> &parametrizations, const State &state, const int volume_selection) : VariableToNodes(parametrizations)
		{
			const auto &mesh = state.mesh;
			const auto &bases = state.bases;
			const auto &gbases = state.geom_bases();

			std::set<int> total_bnode_ids;
			for (const auto &lb : state.total_local_boundary)
			{
				const int e = lb.element_id();
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const int boundary_id = mesh->get_boundary_id(primitive_global_id);
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

					for (long n = 0; n < nodes.size(); ++n)
						total_bnode_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
				}
			}

			std::set<int> node_ids;
			for (int e = 0; e < gbases.size(); e++)
			{
				const int body_id = mesh->get_body_id(e);
				if (volume_selection == body_id)
					for (const auto &gbs : gbases[e].bases)
						for (const auto &g : gbs.global())
							if (!total_bnode_ids.count(g.index))
								node_ids.insert(g.index);
			}

			dim = mesh->dimension();
			set_output_indexing(std::vector(node_ids.begin(), node_ids.end()));
		}
	};

	class VariableToBoundaryNodes : public VariableToNodes
	{
	public:
		VariableToBoundaryNodes(const std::vector<std::shared_ptr<Parametrization>> &parametrizations, const State &state, const int surface_selection) : VariableToNodes(parametrizations)
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
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

					if (surface_selection == boundary_id)
						for (long n = 0; n < nodes.size(); ++n)
							node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
				}
			}

			dim = mesh->dimension();
			set_output_indexing(std::vector(node_ids.begin(), node_ids.end()));
		}
	};

	class VariableToBoundaryNodesExclusive : public VariableToNodes
	{
	public:
		VariableToBoundaryNodesExclusive(const std::vector<std::shared_ptr<Parametrization>> &parametrizations, const State &state, const std::vector<int> &exclude_surface_selections) : VariableToNodes(parametrizations)
		{
			const auto &mesh = state.mesh;
			const auto &bases = state.bases;
			const auto &gbases = state.geom_bases();

			std::set<int> excluded_node_ids;
			std::set<int> all_node_ids;
			for (const auto &lb : state.total_local_boundary)
			{
				const int e = lb.element_id();
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const int boundary_id = mesh->get_boundary_id(primitive_global_id);
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

					if (std::count(exclude_surface_selections.begin(), exclude_surface_selections.end(), boundary_id))
						for (long n = 0; n < nodes.size(); ++n)
							excluded_node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
					for (long n = 0; n < nodes.size(); ++n)
						all_node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
				}
			}

			std::vector<int> node_ids;
			std::set_difference(all_node_ids.begin(), all_node_ids.end(), excluded_node_ids.begin(), excluded_node_ids.end(), std::back_inserter(node_ids));

			dim = mesh->dimension();
			set_output_indexing(node_ids);
		}
	};
} // namespace polyfem::solver