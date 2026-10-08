#include "MeshData.hpp"

#include <polyfem/utils/Logger.hpp>
#include <set>
#include <map>
#include <array>
#include <algorithm>

namespace polyfem::mesh
{
	void NCMeshData::append(const NCMeshData &b)
	{
		auto &a = *this;
		if (!(a.label_flags & 2) && (b.label_flags & 2))
			a.element_state.col(5).setZero();
		const int nv = a.vertices.rows(), ne = a.cells.rows(), ned = a.edges.rows(), nf = a.faces.rows();
		auto join = [](auto &dst, const auto &src) { const int rows = dst.rows(); dst.conservativeResize(rows + src.rows(), Eigen::NoChange); dst.bottomRows(src.rows()) = src; };
		join(a.vertices, b.vertices);
		join(a.cells, (b.cells.array() + nv).matrix().eval());
		join(a.ordered_cells, (b.ordered_cells.array() + nv).matrix().eval());
		join(a.cell_edges, (b.cell_edges.array() + ned).matrix().eval());
		join(a.cell_faces, (b.cell_faces.array() + nf).matrix().eval());
		auto children = b.children;
		for (int &id : children.reshaped())
			if (id >= 0)
				id += ne;
		join(a.children, children);
		auto state = b.element_state;
		if ((a.label_flags & 2) && !(b.label_flags & 2))
			state.col(5).setZero();
		for (int i = 0; i < state.rows(); ++i)
			if (state(i, 1) >= 0)
				state(i, 1) += ne;
		join(a.element_state, state);
		auto edges = b.edges;
		edges.leftCols(2).array() += nv;
		join(a.edges, edges);
		auto faces = b.faces;
		faces.leftCols(3).array() += nv;
		join(a.faces, faces);
		join(a.midpoints, (b.midpoints.array() + nv).matrix().eval());
		a.node_ids.insert(a.node_ids.end(), b.node_ids.begin(), b.node_ids.end());
		for (int id : b.refinement_history)
			a.refinement_history.push_back(id + ne);
		a.label_flags |= b.label_flags;
	}

	void NCMeshData::validate(const MeshData &active) const
	{
		const int d = active.dimension(), count = cells.rows(), nv = vertices.rows();
		auto require = [](bool valid, const char *message) { if (!valid) log_and_throw_error("Invalid NC mesh: {}", message); };
		require(nv > 0 && vertices.cols() == d && vertices.allFinite(), "vertices");
		require(count > 0 && cells.cols() == d + 1 && ordered_cells.rows() == count && ordered_cells.cols() == d + 1, "cells");
		require(element_state.rows() == count && element_state.cols() == 6 && children.rows() == count && children.cols() == (1 << d), "hierarchy dimensions");
		require(cell_edges.rows() == count && cell_edges.cols() == 3 * (d - 1) && cell_faces.rows() == count && cell_faces.cols() == 4 * (d - 2), "incidence dimensions");
		require(edges.cols() == 3 && faces.cols() == 4 && (d == 3 || faces.rows() == 0) && midpoints.cols() == 3, "primitive dimensions");
		require(node_ids.size() == size_t(nv) && label_flags >= 0 && label_flags < 16, "labels");
		require(!active.has_polyhedral_topology() && active.higher_order_connectivity.empty() && active.elements.cols() == d + 1, "NC requires linear simplices");
		require(bool(label_flags & 1) == !active.body_ids.empty() && bool(label_flags & 2) == !active.geometry_ids.empty()
					&& bool(label_flags & 4) == !active.node_ids.empty() && bool(label_flags & 8) == !active.boundary_ids.empty(),
				"active label categories");
		std::set<std::vector<int>> edge_keys, face_keys;
		auto check_primitives = [&](const Eigen::MatrixXi &primitives, int width, auto &keys) {
			for (int i = 0; i < primitives.rows(); ++i)
			{
				std::vector<int> key;
				for (int j = 0; j < width; ++j)
				{
					const int v = primitives(i, j);
					require(v >= 0 && v < nv, "primitive vertex reference");
					key.push_back(v);
				}
				require(std::is_sorted(key.begin(), key.end()) && std::adjacent_find(key.begin(), key.end()) == key.end() && keys.insert(key).second, "duplicate or unordered primitive");
			}
		};
		check_primitives(edges, 2, edge_keys);
		check_primitives(faces, 3, face_keys);
		std::set<std::pair<int, int>> midpoint_edges;
		std::set<int> midpoint_vertices;
		std::map<std::pair<int, int>, int> midpoint_lookup;
		for (int i = 0; i < midpoints.rows(); ++i)
		{
			const int a = midpoints(i, 0), b = midpoints(i, 1), v = midpoints(i, 2);
			require(a >= 0 && a < b && b < v && v < nv, "midpoint references or cycle");
			require(midpoint_edges.emplace(a, b).second && midpoint_vertices.insert(v).second, "duplicate midpoint");
			require(vertices.row(v).isApprox((vertices.row(a) + vertices.row(b)) / 2., 1e-12), "midpoint position");
			midpoint_lookup.emplace(std::make_pair(a, b), v);
		}
		std::vector<int> active_cells;
		std::set<int> active_vertices;
		for (int i = 0; i < count; ++i)
		{
			std::vector<int> geom, ordered;
			for (int j = 0; j < d + 1; ++j)
			{
				require(cells(i, j) >= 0 && cells(i, j) < nv, "cell vertex");
				geom.push_back(cells(i, j));
				ordered.push_back(ordered_cells(i, j));
			}
			std::sort(geom.begin(), geom.end());
			std::sort(ordered.begin(), ordered.end());
			require(geom == ordered && std::adjacent_find(geom.begin(), geom.end()) == geom.end(), "cell ordering");
			const int parent = element_state(i, 1), level = element_state(i, 0), refined = element_state(i, 2), ghost = element_state(i, 3);
			require((refined == 0 || refined == 1) && (ghost == 0 || ghost == 1) && level >= 0, "element flags");
			require(parent >= -1 && parent < i, "parent reference or cycle");
			if (parent >= 0)
			{
				require(level == element_state(parent, 0) + 1, "refinement level");
				require((children.row(parent).array() == i).count() == 1, "parent-child reciprocity");
				if (!ghost)
					require(element_state(parent, 2) && !element_state(parent, 3), "active descendant of inactive parent");
			}
			else
				require(level == 0, "root level");
			const bool has_children = children(i, 0) >= 0;
			require(!refined || has_children, "refined element without children");
			std::set<int> child_ids;
			for (int j = 0; j < children.cols(); ++j)
			{
				const int child = children(i, j);
				if (has_children)
					require(child > i && child < count && element_state(child, 1) == i && child_ids.insert(child).second, "child reference");
				else
					require(child == -1, "partial children");
			}
			if (has_children)
			{
				std::set<int> expected_vertices(geom.begin(), geom.end()), child_vertices;
				for (int a = 0; a < d + 1; ++a)
					for (int b = a + 1; b < d + 1; ++b)
					{
						const auto it = midpoint_lookup.find({geom[a], geom[b]});
						require(it != midpoint_lookup.end(), "missing refinement midpoint");
						expected_vertices.insert(it->second);
					}
				for (int child : child_ids)
					for (int j = 0; j < d + 1; ++j)
						child_vertices.insert(cells(child, j));
				require(child_vertices == expected_vertices, "child refinement connectivity");
			}
			auto check_incidence = [&](const Eigen::MatrixXi &incidence, const Eigen::MatrixXi &primitives, int width) {
				std::set<int> ids;
				for (int j = 0; j < incidence.cols(); ++j)
				{
					const int id = incidence(i, j);
					require(id >= 0 && id < primitives.rows() && ids.insert(id).second, "cell incidence");
					for (int k = 0; k < width; ++k)
						require(std::binary_search(geom.begin(), geom.end(), primitives(id, k)), "cell incidence vertices");
				}
			};
			check_incidence(cell_edges, edges, 2);
			check_incidence(cell_faces, faces, 3);
			// Navigation uses a fixed local edge/face order, not just incidence sets.
			const std::array<std::array<int, 2>, 6> local_edges = {{{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}};
			const std::array<std::array<int, 3>, 4> local_faces = {{{0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}}};
			for (int j = 0; j < cell_edges.cols(); ++j)
			{
				const int a = ordered_cells(i, local_edges[j][0]), b = ordered_cells(i, local_edges[j][1]);
				const int edge = cell_edges(i, j);
				require(edges(edge, 0) == std::min(a, b) && edges(edge, 1) == std::max(a, b), "local edge order");
			}
			for (int j = 0; j < cell_faces.cols(); ++j)
			{
				std::array<int, 3> expected;
				for (int k = 0; k < 3; ++k)
					expected[k] = ordered_cells(i, local_faces[j][k]);
				std::sort(expected.begin(), expected.end());
				for (int k = 0; k < 3; ++k)
					require(faces(cell_faces(i, j), k) == expected[k], "local face order");
			}
			if (!refined && !ghost)
			{
				active_cells.push_back(i);
				active_vertices.insert(geom.begin(), geom.end());
			}
		}
		for (int id : refinement_history)
			require(id >= 0 && id < count, "history reference");
		require(active_cells.size() == size_t(active.elements.rows()) && active_vertices.size() == size_t(active.vertices.rows()), "active counts");
		std::vector<int> full_to_active(nv, -1);
		int v = 0;
		for (int id : active_vertices)
		{
			full_to_active[id] = v;
			require(active.vertices.row(v).isApprox(vertices.row(id), 1e-12), "active vertex order");
			if (!active.node_ids.empty())
				require(active.node_ids[v] == node_ids[id], "active node labels");
			++v;
		}
		for (int i = 0; i < int(active_cells.size()); ++i)
		{
			const int id = active_cells[i];
			std::vector<int> expected, actual;
			for (int j = 0; j < d + 1; ++j)
			{
				expected.push_back(full_to_active[cells(id, j)]);
				actual.push_back(active.elements(i, j));
			}
			std::sort(expected.begin(), expected.end());
			std::sort(actual.begin(), actual.end());
			require(expected == actual, "active cells");
			if (!active.body_ids.empty())
				require(active.body_ids[i] == element_state(id, 4), "active body labels");
			if (!active.geometry_ids.empty())
				require(active.geometry_ids[i] == element_state(id, 5), "active geometry labels");
		}
		const auto &boundary = d == 2 ? edges : faces;
		const std::vector<int> active_to_full(active_vertices.begin(), active_vertices.end());
		std::map<std::vector<int>, int> boundary_labels;
		for (int i = 0; i < boundary.rows(); ++i)
		{
			std::vector<int> key;
			for (int j = 0; j < d; ++j)
				key.push_back(boundary(i, j));
			boundary_labels.emplace(std::move(key), boundary(i, d));
		}
		for (int i = 0; i < int(active.boundary_elements.size()); ++i)
		{
			require(active.boundary_elements[i].size() == size_t(d), "active boundary width");
			std::vector<int> key;
			for (int id : active.boundary_elements[i])
				key.push_back(active_to_full[id]);
			std::sort(key.begin(), key.end());
			const auto it = boundary_labels.find(key);
			require(it != boundary_labels.end(), "active boundary connectivity");
			require(active.boundary_ids[i] == it->second, "active boundary label");
		}
	}

	void MeshData::validate() const
	{
		if (vertices.rows() == 0 || (vertices.cols() != 2 && vertices.cols() != 3))
			log_and_throw_error("MeshData vertices must be a nonempty n x 2 or n x 3 matrix.");
		if (elements.rows() == 0 || elements.cols() < dimension() + 1)
			log_and_throw_error("MeshData elements have invalid dimensions.");

		for (int i = 0; i < elements.rows(); ++i)
		{
			int count = 0;
			bool found_padding = false;
			for (int j = 0; j < elements.cols(); ++j)
			{
				const int vertex = elements(i, j);
				if (vertex == -1)
				{
					found_padding = true;
					continue;
				}
				if (found_padding || vertex < 0 || vertex >= vertices.rows())
					log_and_throw_error("MeshData element {} contains invalid connectivity.", i);
				++count;
			}
			if (count < dimension() + 1)
				log_and_throw_error("MeshData element {} has too few vertices.", i);
		}

		const auto require_elements = [&](const size_t size, const std::string &name) {
			if (size != 0 && size != size_t(elements.rows()))
				log_and_throw_error("MeshData {} has {} entries; expected {}.", name, size, elements.rows());
		};
		require_elements(body_ids.size(), "body_ids");
		require_elements(geometry_ids.size(), "geometry_ids");
		require_elements(higher_order_connectivity.size(), "higher_order_connectivity");
		require_elements(higher_order_weights.size(), "higher_order_weights");
		if (!node_ids.empty() && node_ids.size() != size_t(vertices.rows()))
			log_and_throw_error("MeshData node_ids has {} entries; expected {}.", node_ids.size(), vertices.rows());

		if (boundary_ids.empty() != boundary_elements.empty())
			log_and_throw_error("MeshData boundary_elements and boundary_ids must be provided together.");
		if (!boundary_ids.empty() && boundary_ids.size() != boundary_elements.size())
			log_and_throw_error("MeshData boundary_elements and boundary_ids have different sizes.");
		for (const auto &element : boundary_elements)
		{
			if (element.size() < size_t(dimension()))
				log_and_throw_error("MeshData contains a boundary element with too few vertices.");
			for (const int vertex : element)
				if (vertex < 0 || vertex >= vertices.rows())
					log_and_throw_error("MeshData boundary connectivity references an invalid vertex.");
		}

		if (higher_order_connectivity.empty() != (higher_order_nodes.rows() == 0))
			log_and_throw_error("MeshData higher-order nodes and connectivity must be provided together.");
		if (!higher_order_connectivity.empty()
			&& (higher_order_nodes.cols() != dimension() || higher_order_nodes.rows() < vertices.rows()))
			log_and_throw_error("MeshData higher-order nodes have invalid dimensions.");
		for (const auto &connectivity : higher_order_connectivity)
			for (const int node : connectivity)
				if (node < 0 || node >= higher_order_nodes.rows())
					log_and_throw_error("MeshData higher-order connectivity references an invalid node.");
		if (!higher_order_weights.empty())
		{
			if (higher_order_connectivity.empty())
				log_and_throw_error("MeshData higher-order weights require higher-order connectivity.");
			for (int i = 0; i < higher_order_weights.size(); ++i)
				if (!higher_order_weights[i].empty()
					&& higher_order_weights[i].size() != higher_order_connectivity[i].size())
					log_and_throw_error("MeshData element {} has incompatible higher-order weights.", i);
		}

		if (has_polyhedral_topology())
		{
			if (dimension() != 3)
				log_and_throw_error("MeshData polyhedral topology requires three-dimensional vertices.");
			require_elements(cell_faces.size(), "cell_faces");
			require_elements(cell_face_orientations.size(), "cell_face_orientations");
			require_elements(cell_is_hex.size(), "cell_is_hex");
			if (cell_kernel_points.rows() != elements.rows() || cell_kernel_points.cols() != 3)
				log_and_throw_error("MeshData polyhedral cells require one 3D kernel point per element.");
			for (const auto &face : faces)
			{
				if (face.size() < 3)
					log_and_throw_error("MeshData contains a polyhedral face with fewer than three vertices.");
				for (const int vertex : face)
					if (vertex < 0 || vertex >= vertices.rows())
						log_and_throw_error("MeshData polyhedral face references an invalid vertex.");
			}
			for (int i = 0; i < elements.rows(); ++i)
			{
				if (cell_faces[i].size() != cell_face_orientations[i].size())
					log_and_throw_error("MeshData cell {} has inconsistent face orientations.", i);
				for (const int orientation : cell_face_orientations[i])
					if (orientation != 0 && orientation != 1)
						log_and_throw_error("MeshData cell {} has an invalid face orientation.", i);
				for (const int face : cell_faces[i])
					if (face < 0 || face >= faces.size())
						log_and_throw_error("MeshData cell {} references an invalid face.", i);
			}
		}
		else if (!cell_faces.empty() || !cell_face_orientations.empty() || !cell_is_hex.empty() || cell_kernel_points.size())
			log_and_throw_error("MeshData has incomplete polyhedral topology.");
		if (nc)
			nc->validate(*this);
	}
} // namespace polyfem::mesh
