////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem::mesh
{
	namespace
	{

		std::vector<bool> interface_edges(const Mesh2D &mesh)
		{
			std::vector<bool> is_interface(mesh.n_edges(), false);
			for (int f = 0; f < mesh.n_faces(); ++f)
			{
				if (mesh.is_cube(f))
				{
					continue;
				} // Skip quads

				auto index = mesh.get_index_from_face(f, 0);
				for (int lv = 0; lv < mesh.n_face_vertices(f); ++lv)
				{
					auto index2 = mesh.switch_face(index);
					if (index2.face >= 0)
					{
						is_interface[index.edge] = true;
					}
					index = mesh.next_around_face(index);
				}
			}
			return is_interface;
		}

		std::vector<bool> interface_faces(const Mesh3D &mesh)
		{
			std::vector<bool> is_interface(mesh.n_faces(), false);
			for (int c = 0; c < mesh.n_cells(); ++c)
			{
				if (mesh.is_cube(c))
				{
					continue;
				} // Skip hexes

				for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
				{
					auto index = mesh.get_index_from_element(c, lf, 0);
					auto index2 = mesh.switch_element(index);
					if (index2.element >= 0)
					{
						is_interface[index.face] = true;
					}
				}
			}
			return is_interface;
		}

	} // anonymous namespace

	// -----------------------------------------------------------------------------

	MeshNodes::MeshNodes(const Mesh &mesh, const bool has_poly, const bool connect_nodes, const int max_nodes_per_edge, const int max_nodes_per_face, const int max_nodes_per_cell)
		: mesh_(mesh), connect_nodes_(connect_nodes), edge_offset_(mesh.n_vertices()), face_offset_(edge_offset_ + mesh.n_edges() * max_nodes_per_edge), cell_offset_(face_offset_ + mesh.n_faces() * max_nodes_per_face)

		  ,
		  max_nodes_per_edge_(max_nodes_per_edge), max_nodes_per_face_(max_nodes_per_face), max_nodes_per_cell_(max_nodes_per_cell)
	{
		// Initialization
		int n_nodes = cell_offset_ + mesh.n_cells() * max_nodes_per_cell;
		primitive_to_node_.assign(n_nodes, -1); // #v + #e + #f + #c
		nodes_.resize(n_nodes, mesh.dimension());
		is_boundary_.assign(n_nodes, false);

		if (has_poly)
			is_interface_.assign(n_nodes, false);

		// Vertex nodes
		for (int v = 0; v < mesh.n_vertices(); ++v)
		{
			// nodes_.row(v) = mesh.point(v);
			is_boundary_[v] = mesh.is_boundary_vertex(v);
		}
		// if (!vertices_only) {
		// Eigen::MatrixXd bary;
		// Edge nodes
		// mesh.edge_barycenters(bary);

		for (int e = 0; e < mesh.n_edges(); ++e)
		{
			// nodes_.row(edge_offset_ + e) = bary.row(e);
			const bool is_boundary = mesh.is_boundary_edge(e);
			for (int tmp = 0; tmp < max_nodes_per_edge; ++tmp)
				is_boundary_[edge_offset_ + max_nodes_per_edge * e + tmp] = is_boundary;
		}
		// Face nodes
		// mesh.face_barycenters(bary);
		for (int f = 0; f < mesh.n_faces(); ++f)
		{
			// nodes_.row(face_offset_ + f) = bary.row(f);
			for (int tmp = 0; tmp < max_nodes_per_face; ++tmp)
			{
				if (mesh.is_volume())
				{
					is_boundary_[face_offset_ + max_nodes_per_face * f + tmp] = mesh.is_boundary_face(f);
				}
				else
				{
					is_boundary_[face_offset_ + max_nodes_per_face * f + tmp] = false;
				}
			}
		}
		// Cell nodes
		// mesh.cell_barycenters(bary);
		for (int c = 0; c < mesh.n_cells(); ++c)
		{
			for (int tmp = 0; tmp < max_nodes_per_cell; ++tmp)
				is_boundary_[cell_offset_ + max_nodes_per_cell * c + tmp] = false;
		}
		// }

		// Vertices only, no need to compute interface marker
		// if (vertices_only) { return; }

		// Compute edges/faces that are at interface with a polytope
		if (has_poly)
		{
			if (mesh.is_volume())
			{
				const Mesh3D *mesh3d = dynamic_cast<const Mesh3D *>(&mesh);
				assert(mesh3d);
				auto is_interface_face = interface_faces(*mesh3d);
				for (int f = 0; f < mesh.n_faces(); ++f)
				{
					for (int tmp = 0; tmp < max_nodes_per_face; ++tmp)
						is_interface_[face_offset_ + max_nodes_per_face * f + tmp] = is_interface_face[f];
				}
			}
			else
			{
				const Mesh2D *mesh2d = dynamic_cast<const Mesh2D *>(&mesh);
				assert(mesh2d);
				auto is_interface_edge = interface_edges(*mesh2d);
				for (int e = 0; e < mesh.n_edges(); ++e)
				{
					for (int tmp = 0; tmp < max_nodes_per_edge; ++tmp)
						is_interface_[edge_offset_ + max_nodes_per_edge * e + tmp] = is_interface_edge[e];
				}
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////

	int MeshNodes::node_id_from_primitive(int primitive_id)
	{
		if (primitive_to_node_[primitive_id] < 0 || !connect_nodes_)
		{
			primitive_to_node_[primitive_id] = n_nodes();

			in_ordered_vertices_.push_back(primitive_id);
			node_to_primitive_.push_back(primitive_id);
			node_to_primitive_gid_.push_back(primitive_id);
			assert(in_ordered_vertices_.size() == n_nodes());

			if (primitive_id < edge_offset_)
				nodes_.row(primitive_id) = mesh_.point(primitive_id);
			else if (primitive_id < face_offset_)
				nodes_.row(primitive_id) = mesh_.edge_barycenter(primitive_id - edge_offset_);
			else if (primitive_id < cell_offset_)
				nodes_.row(primitive_id) = mesh_.face_barycenter(primitive_id - face_offset_);
			else
				nodes_.row(primitive_id) = mesh_.cell_barycenter(primitive_id - cell_offset_);
		}
		return primitive_to_node_[primitive_id];
	}

	std::vector<int> MeshNodes::node_ids_from_edge(const Navigation::Index &index, const int n_new_nodes)
	{
		std::vector<int> res;
		if (n_new_nodes <= 0)
			return res;

		const Mesh2D *mesh2d = dynamic_cast<const Mesh2D *>(&mesh_);
		int start;

		if (connect_nodes_)
			start = edge_offset_ + index.edge * max_nodes_per_edge_;
		else
		{
			if (mesh2d->is_boundary_edge(index.edge) || mesh2d->switch_face(index).face > index.face)
				start = edge_offset_ + index.edge * max_nodes_per_edge_;
			else
				start = edge_offset_ + index.edge * max_nodes_per_edge_ + max_nodes_per_edge_ / 2;
		}

		// const auto v1 = mesh2d->point(index.vertex);
		// const auto v2 = mesh2d->point(mesh2d->switch_vertex(index).vertex);
		assert(start < primitive_to_node_.size());
		const int start_node_id = primitive_to_node_[start];
#ifndef NDEBUG
		if (!connect_nodes_)
		{
			assert(start_node_id < 0);
		}
#endif
		if (start_node_id < 0 || !connect_nodes_)
		{
			for (int i = 1; i <= n_new_nodes; ++i)
			{
				// const double t = i/(n_new_nodes + 1.0);

				const int primitive_id = start + i - 1;
				assert(primitive_id < primitive_to_node_.size());

				const auto [node, node_id] = mesh2d->edge_node(index, n_new_nodes, i);

				primitive_to_node_[primitive_id] = n_nodes();

				in_ordered_vertices_.push_back(node_id);
				node_to_primitive_.push_back(primitive_id);
				node_to_primitive_gid_.push_back(index.edge);

				nodes_.row(primitive_id) = node;

				res.push_back(primitive_to_node_[primitive_id]);
				assert(in_ordered_vertices_.size() == n_nodes());
			}
		}
		else
		{
			const auto [v, _] = mesh2d->edge_node(index, n_new_nodes, 1);

			if ((node_position(start_node_id) - v).norm() < 1e-10)
			{
				for (int i = 0; i < n_new_nodes; ++i)
				{
					const int primitive_id = start + i;
					res.push_back(primitive_to_node_[primitive_id]);
				}
			}
			else
			{
				for (int i = n_new_nodes - 1; i >= 0; --i)
				{
					const int primitive_id = start + i;
					res.push_back(primitive_to_node_[primitive_id]);
				}
			}
		}

		assert(res.size() == size_t(n_new_nodes));
		return res;
	}

	std::vector<int> MeshNodes::node_ids_from_edge(const Navigation3D::Index &index, const int n_new_nodes)
	{
		std::vector<int> res;
		if (n_new_nodes <= 0)
			return res;

		const int start = edge_offset_ + index.edge * max_nodes_per_edge_;

		const Mesh3D *mesh3d = dynamic_cast<const Mesh3D *>(&mesh_);

		const int start_node_id = primitive_to_node_[start];
		if (start_node_id < 0)
		{
			for (int i = 1; i <= n_new_nodes; ++i)
			{
				const auto [node, node_id] = mesh3d->edge_node(index, n_new_nodes, i);

				const int primitive_id = start + i - 1;
				primitive_to_node_[primitive_id] = n_nodes();

				in_ordered_vertices_.push_back(node_id);
				node_to_primitive_.push_back(primitive_id);
				node_to_primitive_gid_.push_back(index.edge);

				nodes_.row(primitive_id) = node;

				res.push_back(primitive_to_node_[primitive_id]);
				assert(in_ordered_vertices_.size() == n_nodes());
			}
		}
		else
		{
			const auto [v, _] = mesh3d->edge_node(index, n_new_nodes, 1);
			if ((node_position(start_node_id) - v).norm() < 1e-10)
			{
				for (int i = 0; i < n_new_nodes; ++i)
				{
					const int primitive_id = start + i;
					res.push_back(primitive_to_node_[primitive_id]);
				}
			}
			else
			{
				for (int i = n_new_nodes - 1; i >= 0; --i)
				{
					const int primitive_id = start + i;
					res.push_back(primitive_to_node_[primitive_id]);
				}
			}
		}

		assert(res.size() == size_t(n_new_nodes));
		return res;
	}

	std::vector<int> MeshNodes::node_ids_from_face(const Navigation::Index &index, const int n_new_nodes)
	{
		std::vector<int> res;
		if (n_new_nodes <= 0)
			return res;

		// assert(mesh_.is_simplex(index.face));
		const int start = face_offset_ + index.face * max_nodes_per_face_;
		const int start_node_id = primitive_to_node_[start];

		const Mesh2D *mesh2d = dynamic_cast<const Mesh2D *>(&mesh_);

		if (start_node_id < 0 || !connect_nodes_)
		{
			int loc_index = 0;
			for (int i = 1; i <= n_new_nodes; ++i)
			{
				const int end = mesh2d->is_simplex(index.face) ? (n_new_nodes - i + 1) : n_new_nodes;
				for (int j = 1; j <= end; ++j)
				{
					const auto [node, node_id] = mesh2d->face_node(index, n_new_nodes, i, j);

					const int primitive_id = start + loc_index;
					primitive_to_node_[primitive_id] = n_nodes();

					in_ordered_vertices_.push_back(node_id);
					node_to_primitive_.push_back(primitive_id);
					node_to_primitive_gid_.push_back(index.face);

					nodes_.row(primitive_id) = node;

					res.push_back(primitive_to_node_[primitive_id]);

					assert(in_ordered_vertices_.size() == n_nodes());

					++loc_index;
				}
			}
		}
		else
		{
			assert(false);
		}

#ifndef NDEBUG
		if (mesh2d->is_simplex(index.face))
			assert(res.size() == size_t(n_new_nodes * (n_new_nodes + 1) / 2));
		else
			assert(res.size() == size_t(n_new_nodes * n_new_nodes));
#endif
		return res;
	}

	std::vector<int> MeshNodes::node_ids_from_face(const Navigation3D::Index &index, const int n_new_nodes)
	{
		std::vector<int> res;
		if (n_new_nodes <= 0)
			return res;

		// assert(mesh_.is_simplex(index.element));
		const Mesh3D *mesh3d = dynamic_cast<const Mesh3D *>(&mesh_);
		int start;

		if (connect_nodes_)
			start = face_offset_ + index.face * max_nodes_per_face_;
		else
		{
			if (mesh3d->is_boundary_face(index.face) || mesh3d->switch_element(index).element > index.element)
				start = face_offset_ + index.face * max_nodes_per_face_;
			else
				start = face_offset_ + index.face * max_nodes_per_face_ + max_nodes_per_face_ / 2;
		}

		const int start_node_id = primitive_to_node_[start];

#ifndef NDEBUG
		if (!connect_nodes_)
		{
			assert(start_node_id < 0);
		}
#endif

		if (start_node_id < 0 || !connect_nodes_)
		{
			int loc_index = 0;
			for (int i = 1; i <= n_new_nodes; ++i)
			{
				const int end = mesh3d->is_simplex(index.element) ? (n_new_nodes - i + 1) : n_new_nodes;
				for (int j = 1; j <= end; ++j)
				{
					const int primitive_id = start + loc_index;
					const auto [node, node_id] = mesh3d->face_node(index, n_new_nodes, i, j);

					primitive_to_node_[primitive_id] = n_nodes();

					in_ordered_vertices_.push_back(node_id);
					node_to_primitive_.push_back(primitive_id);
					node_to_primitive_gid_.push_back(index.face);

					nodes_.row(primitive_id) = node;

					res.push_back(primitive_to_node_[primitive_id]);
					assert(in_ordered_vertices_.size() == n_nodes());

					++loc_index;
				}
			}
		}
		else
		{
			if (n_new_nodes == 1)
			{
				res.push_back(start_node_id);
			}
			else
			{
				const int total_nodes = mesh3d->is_simplex(index.element) ? (n_new_nodes * (n_new_nodes + 1) / 2) : (n_new_nodes * n_new_nodes);
				for (int i = 1; i <= n_new_nodes; ++i)
				{
					const int end = mesh3d->is_simplex(index.element) ? (n_new_nodes - i + 1) : n_new_nodes;
					for (int j = 1; j <= end; ++j)
					{
						const auto [p, _] = mesh3d->face_node(index, n_new_nodes, i, j);

						bool found = false;
						for (int k = start; k < start + total_nodes; ++k)
						{
							const double dist = (nodes_.row(k) - p).norm();
							if (dist < 1e-10)
							{
								res.push_back(primitive_to_node_[k]);
								found = true;
								break;
							}
						}

						assert(found);
					}
				}
			}
		}

#ifndef NDEBUG
		if (mesh3d->is_simplex(index.element))
			assert(res.size() == size_t(n_new_nodes * (n_new_nodes + 1) / 2));
		else
			assert(res.size() == size_t(n_new_nodes * n_new_nodes));
#endif
		return res;
	}

	std::vector<int> MeshNodes::node_ids_from_cell(const Navigation3D::Index &index, const int n_new_nodes)
	{
		std::vector<int> res;
		const int start = cell_offset_ + index.element * max_nodes_per_cell_;
		const Mesh3D *mesh3d = dynamic_cast<const Mesh3D *>(&mesh_);

		if (mesh3d->is_simplex(index.element))
		{
			int loc_index = 0;
			for (int i = 1; i <= n_new_nodes; ++i)
			{
				const int endj = (n_new_nodes - i + 1);
				for (int j = 1; j <= endj; ++j)
				{
					const int endk = (n_new_nodes - i - j + 2);
					for (int k = 1; k <= endk; ++k)
					{
						const int primitive_id = start + loc_index;

						auto [node, node_id] = mesh3d->cell_node(index, n_new_nodes, i, j, k);

						primitive_to_node_[primitive_id] = n_nodes();

						in_ordered_vertices_.push_back(node_id);
						node_to_primitive_.push_back(primitive_id);
						node_to_primitive_gid_.push_back(index.element);

						nodes_.row(primitive_id) = node;
						res.push_back(primitive_to_node_[primitive_id]);
						assert(in_ordered_vertices_.size() == n_nodes());

						++loc_index;
					}
				}
			}
		}
		else
		{
			int loc_index = 0;
			for (int i = 1; i <= n_new_nodes; ++i)
			{
				for (int j = 1; j <= n_new_nodes; ++j)
				{
					for (int k = 1; k <= n_new_nodes; ++k)
					{
						const int primitive_id = start + loc_index;

						const auto [node, node_id] = mesh3d->cell_node(index, n_new_nodes, i, j, k);
						primitive_to_node_[primitive_id] = n_nodes();

						in_ordered_vertices_.push_back(node_id);
						node_to_primitive_.push_back(primitive_id);
						node_to_primitive_gid_.push_back(index.element);

						nodes_.row(primitive_id) = node;
						res.push_back(primitive_to_node_[primitive_id]);
						assert(in_ordered_vertices_.size() == n_nodes());

						++loc_index;
					}
				}
			}
		}

#ifndef NDEBUG
		if (res.size() == 1 && connect_nodes_)
		{
			const int idx = node_id_from_cell(index.element);
			assert(idx == res.front());
		}
#endif

#ifndef NDEBUG
		if (mesh3d->is_simplex(index.element))
		{
			int n_cell_nodes = 0;
			for (int pp = 0; pp <= n_new_nodes; ++pp)
				n_cell_nodes += (pp * (pp + 1) / 2);
			assert(res.size() == size_t(n_cell_nodes));
		}
		else
			assert(res.size() == size_t(n_new_nodes * n_new_nodes * n_new_nodes));
#endif

		return res;
	}

	int MeshNodes::node_id_from_vertex(int v)
	{
		return node_id_from_primitive(v);
	}

	int MeshNodes::node_id_from_edge(int e)
	{
		return node_id_from_primitive(edge_offset_ + e * max_nodes_per_edge_);
	}

	int MeshNodes::node_id_from_face(int f)
	{
		return node_id_from_primitive(face_offset_ + f * max_nodes_per_face_);
	}

	int MeshNodes::node_id_from_cell(int c)
	{
		return node_id_from_primitive(cell_offset_ + c * max_nodes_per_cell_);
	}

	////////////////////////////////////////////////////////////////////////////////

	int MeshNodes::vertex_from_node_id(int node_id) const
	{
		const int res = node_to_primitive_[node_id];
		assert(res >= 0 && res < edge_offset_);
		return res;
	}

	int MeshNodes::edge_from_node_id(int node_id) const
	{
		const int res = node_to_primitive_[node_id];
		assert(res >= edge_offset_ && res < face_offset_);
		return res - edge_offset_;
	}

	int MeshNodes::face_from_node_id(int node_id) const
	{
		const int res = node_to_primitive_[node_id];
		assert(res >= face_offset_ && res < cell_offset_);
		return res - face_offset_;
	}

	int MeshNodes::cell_from_node_id(int node_id) const
	{
		const int res = node_to_primitive_[node_id];
		assert(res >= cell_offset_ && res < n_nodes());
		return res - cell_offset_;
	}

	////////////////////////////////////////////////////////////////////////////////

	std::vector<int> MeshNodes::boundary_nodes() const
	{
		std::vector<int> res;
		res.reserve(n_nodes());
		for (size_t prim_id = 0; prim_id < primitive_to_node_.size(); ++prim_id)
		{
			int node_id = primitive_to_node_[prim_id];
			if (node_id >= 0 && is_boundary_[prim_id])
			{
				res.push_back(node_id);
			}
		}
		return res;
	}

	int MeshNodes::count_nonnegative_nodes(int start_i, int end_i) const
	{
		int count = 0;
		for (int i = start_i; i < end_i; i++)
			count += primitive_to_node_[i] >= 0;
		return count;
	}

} // namespace polyfem::mesh