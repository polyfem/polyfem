#include "NodeCompositeParametrizations.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <map>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
    VariableToNodes::VariableToNodes(const State &state)
    {
        dim = state.mesh->dimension();
    }

    void VariableToNodes::set_output_indexing(const std::vector<int> node_ids)
    {
        output_indexing_.resize(node_ids.size() * dim);
        for (int i = 0; i < node_ids.size(); ++i)
            for (int k = 0; k < dim; ++k)
                output_indexing_(i * dim + k) = node_ids[i] * dim + k;
    }

    VariableToInteriorNodes::VariableToInteriorNodes(const State &state, const int volume_selection) : VariableToNodes(state)
    {
        const auto &mesh = state.mesh;
        const auto &bases = state.bases;
        const auto &gbases = state.geom_bases();

        std::set<int> node_ids;
        for (int e = 0; e < mesh->n_elements(); e++)
        {
            const int body_id = mesh->get_body_id(e);
            if (volume_selection == body_id)
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

    VariableToBoundaryNodes::VariableToBoundaryNodes(const State &state, const int surface_selection) : VariableToNodes(state)
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

                if (surface_selection == boundary_id)
                    for (long n = 0; n < mesh->dimension(); ++n)
                        node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
            }
        }

        set_output_indexing(std::vector(node_ids.begin(), node_ids.end()));
    }

    VariableToBoundaryNodesExclusive::VariableToBoundaryNodesExclusive(const State &state, const std::vector<int> &exclude_surface_selections) : VariableToNodes(state)
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
}