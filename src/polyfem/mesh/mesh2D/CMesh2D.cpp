#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/Navigation.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh2D/Refinement.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <cassert>
#include <array>
#include <unordered_map>
#include <type_traits>

namespace polyfem
{
	using namespace utils;

	namespace mesh
	{
		void CMesh2D::remove_elements(const std::vector<bool> &keep)
		{
			assert(keep.size() == n_faces());

			std::unordered_map<std::pair<int, int>, int, utils::HashPair> old_edge_ids;
			std::unordered_map<std::pair<int, int>, EdgeNodes, utils::HashPair> old_edge_nodes;
			for (int e = 0; e < n_edges(); ++e)
			{
				const std::pair<int, int> key = std::minmax(edge_vertex(e, 0), edge_vertex(e, 1));
				if (has_boundary_ids())
					old_edge_ids[key] = boundary_ids_[e];
				if (e < edge_nodes_.size())
					old_edge_nodes[key] = edge_nodes_[e];
			}

			auto filter_nodes = [&keep](auto &nodes) {
				if (nodes.size() != keep.size())
					return;
				std::decay_t<decltype(nodes)> filtered;
				for (int i = 0; i < keep.size(); ++i)
					if (keep[i])
						filtered.push_back(nodes[i]);
				nodes = std::move(filtered);
			};
			filter_nodes(face_nodes_);
			filter_nodes(cell_nodes_);
			filter_element_data(keep);

			// All vertices are kept (as CMesh3D does), so vertex indices and the
			// vertex-keyed maps above stay valid.
			std::vector<bool> to_delete(mesh_.n_facets(), false);
			for (int f = 0; f < keep.size(); ++f)
				to_delete[f] = !keep[f];
			mesh_.delete_facets(to_delete);

			Navigation::prepare_mesh(mesh_);

			if (!old_edge_ids.empty())
			{
				boundary_ids_.resize(n_edges());
				for (int e = 0; e < n_edges(); ++e)
				{
					const std::pair<int, int> key = std::minmax(edge_vertex(e, 0), edge_vertex(e, 1));
					const auto it = old_edge_ids.find(key);
					boundary_ids_[e] = it == old_edge_ids.end() ? get_default_boundary_id(e) : it->second;
				}
			}
			else
				boundary_ids_.clear();

			edge_nodes_.clear();
			if (!old_edge_nodes.empty())
			{
				edge_nodes_.resize(n_edges());
				for (int e = 0; e < n_edges(); ++e)
				{
					const std::pair<int, int> key = std::minmax(edge_vertex(e, 0), edge_vertex(e, 1));
					const auto it = old_edge_nodes.find(key);
					if (it != old_edge_nodes.end())
						edge_nodes_[e] = it->second;
				}
			}

			in_ordered_edges_.resize(n_edges(), 2);
			for (int e = 0; e < n_edges(); ++e)
				in_ordered_edges_.row(e) << edge_vertex(e, 0), edge_vertex(e, 1);
			compute_elements_tag();
		}

		void CMesh2D::refine(const int n_refinement, const double t)
		{
			// return;
			if (n_refinement <= 0)
			{
				return;
			}

			clear_data_lost_by_refinement();

			bool all_simplicial = true;
			for (int e = 0; e < n_elements(); ++e)
			{
				all_simplicial &= is_simplex(e);
			}

			for (int i = 0; i < n_refinement; ++i)
			{
				PolygonMesh mesh = mesh_;

				mesh_.clear();

				// TODO add tags to the refinement
				if (all_simplicial)
				{
					refine_triangle_mesh(mesh, mesh_);
				}
				else if (t <= 0)
				{
					refine_polygonal_mesh(mesh, mesh_, Polygons::catmul_clark_split_func());
				}
				else
				{
					refine_polygonal_mesh(mesh, mesh_, Polygons::polar_split_func(t));
				}

				Navigation::prepare_mesh(mesh_);
			}

			compute_elements_tag();
			generate_edges(mesh_);

			in_ordered_vertices_ = Eigen::VectorXi::LinSpaced(mesh_.n_vertices(), 0, mesh_.n_vertices() - 1);
			assert(in_ordered_vertices_[0] == 0);
			assert(in_ordered_vertices_[1] == 1);
			assert(in_ordered_vertices_[2] == 2);
			assert(in_ordered_vertices_[in_ordered_vertices_.size() - 1] == mesh_.n_vertices() - 1);

			in_ordered_edges_.resize(mesh_.n_edges(), 2);

			for (int e = 0; e < mesh_.n_edges(); ++e)
			{
				for (int lv = 0; lv < 2; ++lv)
				{
					in_ordered_edges_(e, lv) = mesh_.edge_vertex(e, lv);
				}
				assert(in_ordered_edges_(e, 0) != in_ordered_edges_(e, 1));
			}
			assert(in_ordered_edges_.size() > 0);

			in_ordered_faces_.resize(0, 0);
		}

		bool CMesh2D::build_topology(const MeshData &data)
		{
			const Eigen::MatrixXd &V = data.vertices;
			const Eigen::MatrixXi &F = data.elements;
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			mesh_.clear();
			mesh_.create_vertices(V.rows());
			for (int i = 0; i < V.rows(); ++i)
			{
				PolygonMesh::Point &point = mesh_.point(i);
				point[0] = V(i, 0);
				point[1] = V(i, 1);
				point[2] = 0;
			}
			for (int i = 0; i < F.rows(); ++i)
			{
				std::vector<int> vertices;
				vertices.reserve(F.cols());
				for (int j = 0; j < F.cols() && F(i, j) >= 0; ++j)
					vertices.push_back(F(i, j));
				if (vertices.size() < 3)
					return false;
				mesh_.create_polygon(vertices);
			}

			orient_normals_2d(mesh_);
			Navigation::prepare_mesh(mesh_);

			compute_elements_tag();
			return true;
		}

		void CMesh2D::attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes)
		{
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			edge_nodes_.resize(n_edges());
			face_nodes_.resize(n_faces());

			orders_.setOnes(n_faces(), 1);

			assert(nodes.size() == n_faces());

			for (int f = 0; f < n_faces(); ++f)
			{
				auto index = get_index_from_face(f);

				const auto &nodes_ids = nodes[f];

				if (nodes_ids.size() == 3)
				{
					orders_(f) = 1;
					continue;
				}
				// P2
				else if (nodes_ids.size() == 6)
				{
					orders_(f) = 2;

					for (int le = 0; le < 3; ++le)
					{
						auto &n = edge_nodes_[index.edge];

						// nodes not aleardy created
						if (n.nodes.size() <= 0)
						{
							n.v1 = index.vertex;
							n.v2 = switch_vertex(index).vertex;

							int node_index = 0;
							if ((n.v1 == nodes_ids[0] && n.v2 == nodes_ids[1]) || (n.v2 == nodes_ids[0] && n.v1 == nodes_ids[1]))
								node_index = 3;
							else if ((n.v1 == nodes_ids[1] && n.v2 == nodes_ids[2]) || (n.v2 == nodes_ids[1] && n.v1 == nodes_ids[2]))
								node_index = 4;
							else
								node_index = 5;

							n.nodes.resize(1, 2);
							n.nodes << V(nodes_ids[node_index], 0), V(nodes_ids[node_index], 1);
							n.nodes_ids.push_back(nodes_ids[node_index]);
						}
						index = next_around_face(index);
					}
				}
				// P3
				else if (nodes_ids.size() == 10)
				{
					orders_(f) = 3;

					for (int le = 0; le < 3; ++le)
					{
						auto &n = edge_nodes_[index.edge];

						// nodes not aleardy created
						if (n.nodes.size() <= 0)
						{
							n.v1 = index.vertex;
							n.v2 = switch_vertex(index).vertex;

							int node_index1 = 0;
							int node_index2 = 0;
							if (n.v1 == nodes_ids[0] && n.v2 == nodes_ids[1])
							{
								node_index1 = 3;
								node_index2 = 4;
							}
							else if (n.v2 == nodes_ids[0] && n.v1 == nodes_ids[1])
							{
								node_index1 = 4;
								node_index2 = 3;
							}
							else if (n.v1 == nodes_ids[1] && n.v2 == nodes_ids[2])
							{
								node_index1 = 5;
								node_index2 = 6;
							}
							else if (n.v2 == nodes_ids[1] && n.v1 == nodes_ids[2])
							{
								node_index1 = 6;
								node_index2 = 5;
							}
							else if (n.v1 == nodes_ids[2] && n.v2 == nodes_ids[0])
							{
								node_index1 = 7;
								node_index2 = 8;
							}
							else
							{
								assert(n.v2 == nodes_ids[2] && n.v1 == nodes_ids[0]);
								node_index1 = 8;
								node_index2 = 7;
							}

							n.nodes.resize(2, 2);
							n.nodes.row(0) << V(nodes_ids[node_index1], 0), V(nodes_ids[node_index1], 1);
							n.nodes.row(1) << V(nodes_ids[node_index2], 0), V(nodes_ids[node_index2], 1);

							n.nodes_ids.push_back(nodes_ids[node_index1]);
							n.nodes_ids.push_back(nodes_ids[node_index2]);
						}
						index = next_around_face(index);
					}

					{
						auto &n = face_nodes_[f];
						n.v1 = mesh_.facet_vertex(f, 0);
						n.v2 = mesh_.facet_vertex(f, 1);
						n.v3 = mesh_.facet_vertex(f, 2);
						n.nodes.resize(1, 2);
						n.nodes << V(nodes_ids[9], 0), V(nodes_ids[9], 1);
						n.nodes_ids.push_back(nodes_ids[9]);
					}
				}
				// P4
				else if (nodes_ids.size() == 15)
				{
					orders_(f) = 4;
					assert(false);
					// unsupported P4 for geometry, need meshes for testing
				}
				// unsupported
				else
				{
					log_and_throw_error("Unsupported higher-order triangle with {} nodes (supported: 3, 6, 10).", nodes_ids.size());
				}
			}

			if (orders_.maxCoeff() == 1)
				orders_.resize(0, 0);
		}

		std::pair<RowVectorNd, int> CMesh2D::edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const
		{
			if (orders_.size() <= 0 || orders_(index.face) == 1 || edge_nodes_.empty() || edge_nodes_[index.edge].nodes.rows() != n_new_nodes)
			{
				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);

				const double t = i / (n_new_nodes + 1.0);

				return std::make_pair((1 - t) * v1 + t * v2, -1);
			}

			const auto &n = edge_nodes_[index.edge];
			if (n.v1 == index.vertex)
				return std::make_pair(n.nodes.row(i - 1), n.nodes_ids[i - 1]);
			else
			{
				assert(n.v2 == index.vertex);
				return std::make_pair(n.nodes.row(n.nodes.rows() - i), n.nodes_ids[n.nodes_ids.size() - i]);
			}
		}

		std::pair<RowVectorNd, int> CMesh2D::face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const
		{
			if (is_simplex(index.face))
			{
				if (orders_.size() <= 0 || orders_(index.face) == 1 || orders_(index.face) == 2 || face_nodes_.empty() || face_nodes_[index.face].nodes.rows() != n_new_nodes)
				{
					const auto v1 = point(index.vertex);
					const auto v2 = point(switch_vertex(index).vertex);
					const auto v3 = point(switch_vertex(switch_edge(index)).vertex);

					const double b2 = i / (n_new_nodes + 2.0);
					const double b3 = j / (n_new_nodes + 2.0);
					const double b1 = 1 - b3 - b2;
					assert(b3 < 1);
					assert(b3 > 0);

					return std::make_pair(b1 * v1 + b2 * v2 + b3 * v3, -1);
				}

				assert(orders_(index.face) == 3);
				// unsupported P4 for geometry
				const auto &n = face_nodes_[index.face];
				return std::make_pair(n.nodes.row(0), n.nodes_ids[0]);
			}
			else if (is_cube(index.face))
			{
				// supports only blilinear quads
				assert(orders_.size() <= 0 || orders_(index.face) == 1);

				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);
				const auto v3 = point(switch_vertex(switch_edge(switch_vertex(index))).vertex);
				const auto v4 = point(switch_vertex(switch_edge(index)).vertex);

				const double b1 = i / (n_new_nodes + 1.0);
				const double b2 = j / (n_new_nodes + 1.0);

				return std::make_pair(v1 * (1 - b1) * (1 - b2) + v2 * b1 * (1 - b2) + v3 * b1 * b2 + v4 * (1 - b1) * b2, -1);
			}

			assert(false);
			return std::make_pair(RowVectorNd(2, 1), -1);
		}

		void CMesh2D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
		{
			PolygonMesh::Point min_corner, max_corner;
			mesh_.bounding_box(min_corner, max_corner);
			min.resize(2);
			max.resize(2);

			min(0) = min_corner[0];
			min(1) = min_corner[1];

			max(0) = max_corner[0];
			max(1) = max_corner[1];
		}

		void CMesh2D::normalize()
		{

			PolygonMesh::Point min_corner, max_corner;
			mesh_.bounding_box(min_corner, max_corner);
			PolygonMesh::Point extent = max_corner - min_corner;
			double scaling = std::max(extent[0], std::max(extent[1], extent[2]));
			// const PolygonMesh::Point origin = 0.5 * (min_corner + max_corner);
			const PolygonMesh::Point origin = min_corner;
			for (int v = 0; v < mesh_.n_vertices(); ++v)
			{
				mesh_.point(v) = (mesh_.point(v) - origin) / scaling;
			}
			Eigen::RowVector2d shift;
			shift << origin[0], origin[1];
			for (auto &n : edge_nodes_)
			{
				if (n.nodes.size() > 0)
					n.nodes = (n.nodes.rowwise() - shift) / scaling;
			}
			for (auto &n : face_nodes_)
			{
				if (n.nodes.size() > 0)
					n.nodes = (n.nodes.rowwise() - shift) / scaling;
			}
			Eigen::Vector2d normalization_translation;
			normalization_translation << -origin[0] / scaling, -origin[1] / scaling;
			transform_higher_order_data(
				Eigen::Matrix2d::Identity() / scaling,
				normalization_translation);

			logger().debug("-- bbox before normalization:");
			logger().debug("   min   : {} {}", min_corner[0], min_corner[1]);
			logger().debug("   max   : {} {}", max_corner[0], max_corner[1]);
			logger().debug("   extent: {} {}", max_corner[0] - min_corner[0], max_corner[1] - min_corner[1]);
			mesh_.bounding_box(min_corner, max_corner);
			logger().debug("-- bbox after normalization:");
			logger().debug("   min   : {} {}", min_corner[0], min_corner[1]);
			logger().debug("   max   : {} {}", max_corner[0], max_corner[1]);
			logger().debug("   extent: {} {}", max_corner[0] - min_corner[0], max_corner[1] - min_corner[1]);

			Eigen::MatrixXd p0, p1, p;
			get_edges(p0, p1);
			p = p0 - p1;
			logger().debug("-- edge length after normalization:");
			logger().debug("   min: {}", p.rowwise().norm().minCoeff());
			logger().debug("   max: {}", p.rowwise().norm().maxCoeff());
			logger().debug("   avg: {}", p.rowwise().norm().mean());
		}

		double CMesh2D::edge_length(const int gid) const
		{
			const int v0 = mesh_.edge_vertex(gid, 0);
			const int v1 = mesh_.edge_vertex(gid, 1);

			return (point(v0) - point(v1)).norm();
		}

		void CMesh2D::set_point(const int global_index, const RowVectorNd &p)
		{
			mesh_.point(global_index)[0] = p(0);
			mesh_.point(global_index)[1] = p(1);
		}

		RowVectorNd CMesh2D::point(const int global_index) const
		{
			const PolygonMesh::Point &pt = mesh_.point(global_index);
			RowVectorNd p(2);
			p(0) = pt[0];
			p(1) = pt[1];
			return p;
		}

		bool CMesh2D::is_boundary_element(const int element_global_id) const
		{
			auto index = get_index_from_face(element_global_id);

			for (int i = 0; i < n_face_vertices(element_global_id); ++i)
			{
				if (is_boundary_edge(index.edge))
					return true;

				index = next_around_face(index);
			}

			return false;
		}

		// void CMesh2D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
		// {
		// 	ranges.clear();

		// 	std::vector<Eigen::MatrixXi> local_tris(mesh_.facets.nb());
		// 	std::vector<Eigen::MatrixXd> local_pts(mesh_.facets.nb());

		// 	int total_tris = 0;
		// 	int total_pts = 0;

		// 	ranges.push_back(0);

		// 	for (GEO::index_t f = 0; f < mesh_.facets.nb(); ++f)
		// 	{
		// 		const int n_vertices = mesh_.facets.nb_vertices(f);

		// 		Eigen::MatrixXd face_pts(n_vertices, 2);
		// 		// Eigen::MatrixXi edges(n_vertices,2);
		// 		local_tris[f].resize(n_vertices - 2, 3);

		// 		for (int i = 0; i < n_vertices; ++i)
		// 		{
		// 			const int vertex = mesh_.facets.vertex(f, i);
		// 			const double *pt = mesh_.vertices.point_ptr(vertex);
		// 			face_pts(i, 0) = pt[0];
		// 			face_pts(i, 1) = pt[1];

		// 			// edges(i, 0) = i;
		// 			// edges(i, 1) = (i+1) % n_vertices;
		// 		}

		// 		for (int i = 1; i < n_vertices - 1; ++i)
		// 		{
		// 			local_tris[f].row(i - 1) << 0, i, i + 1;
		// 		}

		// 		local_pts[f] = face_pts;

		// 		total_tris += local_tris[f].rows();
		// 		total_pts += local_pts[f].rows();

		// 		ranges.push_back(total_tris);

		// 		assert(local_pts[f].rows() == face_pts.rows());
		// 	}

		// 	tris.resize(total_tris, 3);
		// 	pts.resize(total_pts, 2);

		// 	int tri_index = 0;
		// 	int pts_index = 0;
		// 	for (std::size_t i = 0; i < local_tris.size(); ++i)
		// 	{
		// 		tris.block(tri_index, 0, local_tris[i].rows(), local_tris[i].cols()) = local_tris[i].array() + pts_index;
		// 		tri_index += local_tris[i].rows();

		// 		pts.block(pts_index, 0, local_pts[i].rows(), local_pts[i].cols()) = local_pts[i];
		// 		pts_index += local_pts[i].rows();
		// 	}
		// }

		void CMesh2D::compute_elements_tag()
		{
			elements_tag_.clear();
			compute_element_tags(mesh_, elements_tag_);
		}

		void CMesh2D::update_elements_tag()
		{
			compute_element_tags(mesh_, elements_tag_);
		}

		RowVectorNd CMesh2D::edge_barycenter(const int index) const
		{
			const int v0 = mesh_.edge_vertex(index, 0);
			const int v1 = mesh_.edge_vertex(index, 1);

			return 0.5 * (point(v0) + point(v1));
		}

		void CMesh2D::compute_body_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &)> &marker)
		{
			body_ids_.resize(n_elements());
			std::fill(body_ids_.begin(), body_ids_.end(), -1);

			for (int e = 0; e < n_elements(); ++e)
			{
				const auto bary = face_barycenter(e);
				body_ids_[e] = marker(e, element_vertices(e), bary);
			}
		}

		void CMesh2D::compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker)
		{
			boundary_ids_.resize(n_edges());

			for (int e = 0; e < n_edges(); ++e)
			{
				bool is_boundary = is_boundary_edge(e);
				const auto p = edge_barycenter(e);
				std::vector<int> vs = {edge_vertex(e, 0), edge_vertex(e, 1)};
				std::sort(vs.begin(), vs.end());
				boundary_ids_[e] = marker(e, vs, p, is_boundary);
			}
		}

		void CMesh2D::append(const Mesh &mesh)
		{
			assert(typeid(mesh) == typeid(CMesh2D));
			Mesh::append(mesh);

			const CMesh2D &mesh2d = dynamic_cast<const CMesh2D &>(mesh);

			const int n_v = n_vertices();
			const int n_f = n_faces();

			mesh_.create_vertices(mesh2d.n_vertices());
			for (int i = n_v; i < mesh_.n_vertices(); ++i)
			{
				set_point(i, mesh2d.point(i - n_v));
			}

			std::vector<int> indices;
			for (int i = 0; i < mesh2d.n_faces(); ++i)
			{
				indices.clear();
				for (int j = 0; j < mesh2d.mesh_.n_facet_vertices(i); ++j)
					indices.push_back(mesh2d.mesh_.facet_vertex(i, j) + n_v);

				mesh_.create_polygon(indices);
			}

			assert(n_vertices() == n_v + mesh2d.n_vertices());
			assert(n_faces() == n_f + mesh2d.n_faces());

			Navigation::prepare_mesh(mesh_);
		}

		std::unique_ptr<Mesh> CMesh2D::copy() const
		{
			Eigen::MatrixXd vertices(n_vertices(), 2);
			int max_face_size = 0;
			for (int i = 0; i < n_vertices(); ++i)
				vertices.row(i) = point(i);
			for (int i = 0; i < n_faces(); ++i)
				max_face_size = std::max(max_face_size, n_face_vertices(i));

			Eigen::MatrixXi faces = Eigen::MatrixXi::Constant(n_faces(), max_face_size, -1);
			for (int i = 0; i < n_faces(); ++i)
				for (int j = 0; j < n_face_vertices(i); ++j)
					faces(i, j) = face_vertex(i, j);

			std::unique_ptr<CMesh2D> copy_mesh = std::make_unique<CMesh2D>();
			copy_mesh->build_topology(MeshData(std::move(vertices), std::move(faces)));

			// Manually copy parent's data
			copy_mesh->elements_tag_ = this->elements_tag_;
			copy_mesh->node_ids_ = this->node_ids_;
			copy_mesh->boundary_ids_ = this->boundary_ids_;
			copy_mesh->body_ids_ = this->body_ids_;
			copy_mesh->geometry_ids_ = this->geometry_ids_;
			copy_mesh->orders_ = this->orders_;
			copy_mesh->is_rational_ = this->is_rational_;
			copy_mesh->edge_nodes_ = this->edge_nodes_;
			copy_mesh->face_nodes_ = this->face_nodes_;
			copy_mesh->cell_nodes_ = this->cell_nodes_;
			copy_mesh->cell_weights_ = this->cell_weights_;
			copy_mesh->has_explicit_polyhedral_topology_ = this->has_explicit_polyhedral_topology_;
			copy_mesh->higher_order_nodes_ = this->higher_order_nodes_;
			copy_mesh->higher_order_connectivity_ = this->higher_order_connectivity_;
			copy_mesh->in_ordered_vertices_ = this->in_ordered_vertices_;
			copy_mesh->in_ordered_edges_ = this->in_ordered_edges_;
			copy_mesh->in_ordered_faces_ = this->in_ordered_faces_;

			return copy_mesh;
		}
	} // namespace mesh
} // namespace polyfem
