#include <polyfem/optimization/NodeSelectionUtils.hpp>

#include <polyfem/State.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/utils/Logger.hpp>

#include <cassert>
#include <algorithm>
#include <iterator>
#include <set>

namespace polyfem
{
	namespace
	{
		template <typename T>
		Eigen::VectorXi eigen_vec_from_iterable(const T &iterable)
		{
			Eigen::VectorXi out;
			out.resize(iterable.size());

			int counter = 0;
			for (int v : iterable)
			{
				out[counter] = v;
				++counter;
			}
			return out;
		}
	} // namespace

	Eigen::VectorXi select_interior_nodes(
		const State &state,
		const std::vector<int> &volume_selection)
	{
		auto &mesh = state.mesh;

		std::set<int> node_ids{};
		for (int e = 0; e < mesh->n_elements(); ++e)
		{
			int body_id = mesh->get_body_id(e);
			if (std::find(volume_selection.begin(), volume_selection.end(), body_id) != volume_selection.end())
			{
				for (int i = 0; i < mesh->dimension() + 1; ++i)
				{
					const int vid = mesh->element_vertex(e, i);
					if (!mesh->is_boundary_vertex(vid))
					{
						node_ids.insert(vid);
					}
				}
			}
		}

		Eigen::VectorXi nodes = eigen_vec_from_iterable(node_ids);
		std::sort(nodes.begin(), nodes.end());
		return nodes;
	}

	Eigen::VectorXi select_boundary_nodes(
		const State &state,
		const std::vector<int> &surface_selection)
	{
		auto &mesh = state.mesh;

		std::set<int> node_ids{};
		for (const auto &lb : state.total_local_boundary)
		{
			for (int i = 0; i < lb.size(); ++i)
			{
				int primitive_global_id = lb.global_primitive_id(i);
				int boundary_id = mesh->get_boundary_id(primitive_global_id);

				if (std::find(surface_selection.begin(), surface_selection.end(), boundary_id) != surface_selection.end())
				{
					for (int n = 0; n < mesh->dimension(); ++n)
					{
						node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
					}
				}
			}
		}

		Eigen::VectorXi nodes = eigen_vec_from_iterable(node_ids);
		std::sort(nodes.begin(), nodes.end());
		return nodes;
	}

	Eigen::VectorXi select_boundary_nodes_excluding_surfaces(
		const State &state,
		const std::vector<int> &exclude_surface_selections)
	{
		auto &mesh = state.mesh;

		if (!mesh->is_simplicial())
		{
			log_and_throw_adjoint_error("select_boundary_node_dofs_excluding_surfaces only supports simplices!");
		}

		std::set<int> excluded_node_ids{};
		std::set<int> all_node_ids{};
		for (const auto &lb : state.total_local_boundary)
		{
			int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				int primitive_global_id = lb.global_primitive_id(i);
				int boundary_id = mesh->get_boundary_id(primitive_global_id);

				assert(mesh->is_simplex(e));
				if (std::count(exclude_surface_selections.begin(), exclude_surface_selections.end(), boundary_id) != 0)
				{
					for (int n = 0; n < mesh->dimension(); ++n)
					{
						excluded_node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
					}
				}

				for (int n = 0; n < mesh->dimension(); ++n)
				{
					all_node_ids.insert(mesh->boundary_element_vertex(primitive_global_id, n));
				}
			}
		}

		std::vector<int> node_ids;
		std::set_difference(all_node_ids.begin(), all_node_ids.end(),
							excluded_node_ids.begin(), excluded_node_ids.end(),
							std::back_inserter(node_ids));

		std::sort(node_ids.begin(), node_ids.end());
		return eigen_vec_from_iterable(node_ids);
	}

} // namespace polyfem
