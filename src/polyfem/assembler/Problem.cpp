#include "Problem.hpp"

#include <memory>
#include <iostream>

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

		void Problem::setup_bc(const Mesh &mesh, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &pressure_bases, std::vector<LocalBoundary> &local_boundary, std::vector<int> &boundary_nodes, std::vector<LocalBoundary> &local_neumann_boundary, std::vector<int> &pressure_boundary_nodes)
		{
			std::vector<LocalBoundary> new_local_boundary;
			std::vector<LocalBoundary> new_local_pressure_dirichlet_boundary;
			local_neumann_boundary.clear();
			for (auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
			{
				const auto &lb = *it;
				LocalBoundary new_lb(lb.element_id(), lb.type());
				LocalBoundary new_neumann_lb(lb.element_id(), lb.type());
				LocalBoundary new_pressure_dirichlet_lb(lb.element_id(), lb.type());
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_g_id = lb.global_primitive_id(i);
					const int tag = mesh.get_boundary_id(primitive_g_id);

					if (tag <= 0)
						continue;

					if ((!might_have_no_dirichlet() && boundary_ids_.empty()) || std::find(boundary_ids_.begin(), boundary_ids_.end(), tag) != boundary_ids_.end())
						new_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
					if (std::find(neumann_boundary_ids_.begin(), neumann_boundary_ids_.end(), tag) != neumann_boundary_ids_.end())
						new_neumann_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
					if (std::find(pressure_boundary_ids_.begin(), pressure_boundary_ids_.end(), tag) != pressure_boundary_ids_.end())
						new_neumann_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
					if (std::find(splitting_pressure_boundary_ids_.begin(), splitting_pressure_boundary_ids_.end(), tag) != splitting_pressure_boundary_ids_.end())
						new_pressure_dirichlet_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
				}

				if (!new_lb.empty())
					new_local_boundary.emplace_back(new_lb);
				if (!new_neumann_lb.empty())
					local_neumann_boundary.emplace_back(new_neumann_lb);
				if (!new_pressure_dirichlet_lb.empty())
					new_local_pressure_dirichlet_boundary.emplace_back(new_pressure_dirichlet_lb);
			}
			local_boundary.clear();
			std::swap(local_boundary, new_local_boundary);

			boundary_nodes.clear();
			pressure_boundary_nodes.clear();

			const int dim = is_scalar() ? 1 : mesh.dimension();

			for (auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
			{
				const auto &lb = *it;
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
								if (is_dimension_dirichet(mesh.get_boundary_id(primitive_global_id), d))
									boundary_nodes.push_back(base_index + d);
							}
						}
					}
				}
			}

			for (auto it = new_local_pressure_dirichlet_boundary.begin(); it != new_local_pressure_dirichlet_boundary.end(); ++it)
			{
				const auto &lb = *it;
				const auto &b = pressure_bases[lb.element_id()];
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = b.local_nodes_for_primitive(primitive_global_id, mesh);

					for (long n = 0; n < nodes.size(); ++n)
					{
						auto &bs = b.bases[nodes(n)];
						for (size_t g = 0; g < bs.global().size(); ++g)
						{
							const int base_index = bs.global()[g].index;
							pressure_boundary_nodes.push_back(base_index);
						}
					}
				}
			}

			std::sort(boundary_nodes.begin(), boundary_nodes.end());
			auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
			boundary_nodes.resize(std::distance(boundary_nodes.begin(), it));

			std::sort(pressure_boundary_nodes.begin(), pressure_boundary_nodes.end());
			auto it_ = std::unique(pressure_boundary_nodes.begin(), pressure_boundary_nodes.end());
			pressure_boundary_nodes.resize(std::distance(pressure_boundary_nodes.begin(), it_));
		}
	} // namespace assembler
} // namespace polyfem
