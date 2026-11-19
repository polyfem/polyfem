#include "Problem.hpp"

namespace polyfem
{
	using namespace basis;
	using namespace mesh;
	using namespace utils;

	namespace assembler
	{
		Problem::Problem(const std::string &name)
			: name_(name)
		{
		}

		bool Problem::has_boundary(const BoundaryKind kind, const int tag, const int fe_space_id)
		{
			(void)fe_space_id;
			if (tag <= 0)
				return false;

			switch (kind)
			{
			case BoundaryKind::Dirichlet:
				return (!might_have_no_dirichlet() && boundary_ids_.empty())
					   || std::find(boundary_ids_.begin(), boundary_ids_.end(), -1) != boundary_ids_.end()
					   || std::find(boundary_ids_.begin(), boundary_ids_.end(), tag) != boundary_ids_.end();
			case BoundaryKind::Neumann:
				return std::find(neumann_boundary_ids_.begin(), neumann_boundary_ids_.end(), tag) != neumann_boundary_ids_.end()
					   || std::find(normal_aligned_neumann_boundary_ids_.begin(), normal_aligned_neumann_boundary_ids_.end(), tag) != normal_aligned_neumann_boundary_ids_.end();
			case BoundaryKind::NormalAlignedNeumann:
				return is_boundary_pressure(tag);
			case BoundaryKind::PressureCavity:
				return std::find(pressure_cavity_ids_.begin(), pressure_cavity_ids_.end(), tag) != pressure_cavity_ids_.end();
			default:
				return false;
			}

			return false;
		}

		void Problem::setup_bc(
			const Mesh &mesh,
			const BoundaryKind kind,
			const int fe_space_id,
			const std::vector<LocalBoundary> &local_boundary,
			std::vector<LocalBoundary> &selected_local_boundary)
		{
			selected_local_boundary.clear();

			for (const LocalBoundary &lb : local_boundary)
			{
				LocalBoundary selected_lb(lb.element_id(), lb.type());

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_g_id = lb.global_primitive_id(i);
					const int tag = mesh.get_boundary_id(primitive_g_id);

					if (has_boundary(kind, tag, fe_space_id))
						selected_lb.add_boundary_primitive(primitive_g_id, lb[i]);
				}

				if (!selected_lb.empty())
					selected_local_boundary.emplace_back(selected_lb);
			}
		}

		void Problem::setup_bc(
			const Mesh &mesh,
			const BoundaryKind kind,
			const int fe_space_id,
			const std::vector<ElementBases> &bases,
			const std::vector<LocalBoundary> &local_boundary,
			std::vector<LocalBoundary> &selected_local_boundary,
			std::vector<int> &boundary_nodes,
			const int value_dim)
		{
			setup_bc(mesh, kind, fe_space_id, local_boundary, selected_local_boundary);
			boundary_nodes.clear();

			if (kind != BoundaryKind::Dirichlet)
				return;

			const int dim = value_dim > 0 ? value_dim : (is_scalar() ? 1 : mesh.dimension());

			for (const LocalBoundary &lb : selected_local_boundary)
			{
				const auto &b = bases[lb.element_id()];
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = b.local_nodes_for_primitive(primitive_global_id, mesh);

					for (long n = 0; n < nodes.size(); ++n)
					{
						auto &bs = b.bases[nodes(n)];
						for (size_t g = 0; g < bs.global().size(); ++g)
						{
							const int base_index = bs.global()[g].index * dim;
							for (int d = 0; d < dim; ++d)
							{
								if (is_dimension_dirichet(mesh.get_boundary_id(primitive_global_id), d, fe_space_id))
									boundary_nodes.push_back(base_index + d);
							}
						}
					}
				}
			}

			std::sort(boundary_nodes.begin(), boundary_nodes.end());
			const auto end = std::unique(boundary_nodes.begin(), boundary_nodes.end());
			boundary_nodes.resize(std::distance(boundary_nodes.begin(), end));
		}

		void Problem::setup_bc(
			const Mesh &mesh,
			const BoundaryKind kind,
			const int fe_space_id,
			const std::vector<LocalBoundary> &local_boundary,
			std::unordered_map<int, std::vector<LocalBoundary>> &selected_by_tag)
		{
			std::vector<LocalBoundary> selected;
			setup_bc(mesh, kind, fe_space_id, local_boundary, selected);
			selected_by_tag.clear();
			for (const LocalBoundary &lb : selected)
			{
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_id = lb.global_primitive_id(i);
					const int tag = mesh.get_boundary_id(primitive_id);
					LocalBoundary tagged_boundary(lb.element_id(), lb.type());
					tagged_boundary.add_boundary_primitive(primitive_id, lb[i]);
					selected_by_tag[tag].emplace_back(tagged_boundary);
				}
			}
		}

		void Problem::setup_nodal_bc(
			const Mesh &mesh,
			const BoundaryKind kind,
			const int fe_space_id,
			const int n_bases,
			std::vector<int> &nodes)
		{
			nodes.clear();

			if (kind == BoundaryKind::Dirichlet && !(mesh.has_node_ids() || has_nodal_dirichlet(fe_space_id)))
				return;
			if (kind == BoundaryKind::Neumann && !(mesh.has_node_ids() || has_nodal_neumann(fe_space_id)))
				return;
			for (int n_id = 0; n_id < n_bases; ++n_id)
			{
				const int tag = mesh.get_node_id(n_id);

				if (kind == BoundaryKind::Dirichlet && is_nodal_dirichlet_boundary(n_id, tag, fe_space_id))
					nodes.push_back(n_id);
				else if (kind == BoundaryKind::Neumann
						 && !is_nodal_dirichlet_boundary(n_id, tag, fe_space_id)
						 && is_nodal_neumann_boundary(n_id, tag, fe_space_id))
					nodes.push_back(n_id);
			}
		}
	} // namespace assembler
} // namespace polyfem
