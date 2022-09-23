#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/Navigation.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh2D/Refinement.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_repair.h>

#include <cassert>
#include <array>

namespace polyfem
{
	using namespace io;
	using namespace utils;

	namespace mesh
	{
		void CMesh2D::refine(const int n_refinement, const double t)
		{
			// return;
			if (n_refinement <= 0)
			{
				return;
			}

			orders_.resize(0, 0);

			bool all_simplicial = true;
			for (int e = 0; e < n_elements(); ++e)
			{
				all_simplicial &= is_simplex(e);
			}

			for (int i = 0; i < n_refinement; ++i)
			{
				GEO::Mesh mesh;
				mesh.copy(mesh_);

				c2e_.reset();
				boundary_vertices_.reset();
				boundary_edges_.reset();

				mesh_.clear(false, false);

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
				c2e_ = std::make_unique<GEO::Attribute<GEO::index_t>>(mesh_.facet_corners.attributes(), "edge_id");
				boundary_vertices_ = std::make_unique<GEO::Attribute<bool>>(mesh_.vertices.attributes(), "boundary_vertex");
				boundary_edges_ = std::make_unique<GEO::Attribute<bool>>(mesh_.edges.attributes(), "boundary_edge");
			}

			compute_elements_tag();
			generate_edges(mesh_);

			in_ordered_vertices_ = Eigen::VectorXi::LinSpaced(mesh_.vertices.nb(), 0, mesh_.vertices.nb() - 1);
			assert(in_ordered_vertices_[0] == 0);
			assert(in_ordered_vertices_[1] == 1);
			assert(in_ordered_vertices_[2] == 2);
			assert(in_ordered_vertices_[in_ordered_vertices_.size() - 1] == mesh_.vertices.nb() - 1);

			in_ordered_edges_.resize(mesh_.edges.nb(), 2);

			for (int e = 0; e < (int)mesh_.edges.nb(); ++e)
			{
				for (int lv = 0; lv < 2; ++lv)
				{
					in_ordered_edges_(e, lv) = mesh_.edges.vertex(e, lv);
				}
				assert(in_ordered_edges_(e, 0) != in_ordered_edges_(e, 1));
			}
			assert(in_ordered_edges_.size() > 0);

			in_ordered_faces_.resize(0, 0);
		}

		bool CMesh2D::load(const std::string &path)
		{
			// This method should be used for special loading, like hybrid in 3d

			// edge_nodes_.clear();
			// face_nodes_.clear();
			// cell_nodes_.clear();
			// order_ = 1;

			// c2e_.reset();
			// boundary_vertices_.reset();
			// boundary_edges_.reset();

			// mesh_.clear(false,false);

			// if (!StringUtils::endswith(path, "msh"))
			// {
			// 	Eigen::MatrixXd vertices;
			// 	Eigen::MatrixXi cells;
			// 	std::vector<std::vector<int>> elements;
			// 	std::vector<std::vector<double>> weights;

			// 	if(!MshReader::load(path, vertices, cells, elements, weights))
			// 		return false;

			// 	build_from_matrices(vertices, cells);
			// 	attach_higher_order_nodes(vertices, elements);
			// 	cell_weights_ = weights;
			// }
			// else
			// {
			// 	if(!mesh_load(path, mesh_))
			// 		return false;
			// }

			// orient_normals_2d(mesh_);
			// Navigation::prepare_mesh(mesh_);
			// c2e_ = std::make_unique<GEO::Attribute<GEO::index_t>>(mesh_.facet_corners.attributes(), "edge_id");
			// boundary_vertices_ = std::make_unique<GEO::Attribute<bool>>(mesh_.vertices.attributes(), "boundary_vertex");
			// boundary_edges_ = std::make_unique<GEO::Attribute<bool>>(mesh_.edges.attributes(), "boundary_edge");

			// compute_elements_tag();
			assert(false);
			return false;
		}

		bool CMesh2D::load(const GEO::Mesh &mesh)
		{
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			c2e_.reset();
			boundary_vertices_.reset();
			boundary_edges_.reset();

			mesh_.clear(false, false);
			mesh_.copy(mesh);

			orient_normals_2d(mesh_);
			Navigation::prepare_mesh(mesh_);
			c2e_ = std::make_unique<GEO::Attribute<GEO::index_t>>(mesh_.facet_corners.attributes(), "edge_id");
			boundary_vertices_ = std::make_unique<GEO::Attribute<bool>>(mesh_.vertices.attributes(), "boundary_vertex");
			boundary_edges_ = std::make_unique<GEO::Attribute<bool>>(mesh_.edges.attributes(), "boundary_edge");

			compute_elements_tag();
			return true;
		}

		bool CMesh2D::save(const std::string &path) const
		{
			if (!mesh_save(mesh_, path))
				return false;

			return true;
		}

		bool CMesh2D::build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			c2e_.reset();
			boundary_vertices_.reset();
			boundary_edges_.reset();

			mesh_.clear(false, false);
			to_geogram_mesh(V, F, mesh_);

			orient_normals_2d(mesh_);
			Navigation::prepare_mesh(mesh_);

			c2e_ = std::make_unique<GEO::Attribute<GEO::index_t>>(mesh_.facet_corners.attributes(), "edge_id");
			boundary_vertices_ = std::make_unique<GEO::Attribute<bool>>(mesh_.vertices.attributes(), "boundary_vertex");
			boundary_edges_ = std::make_unique<GEO::Attribute<bool>>(mesh_.edges.attributes(), "boundary_edge");

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

			orders_.resize(n_faces(), 1);

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
						}
						index = next_around_face(index);
					}

					{
						auto &n = face_nodes_[f];
						n.v1 = mesh_.facets.vertex(f, 0);
						n.v2 = mesh_.facets.vertex(f, 1);
						n.v3 = mesh_.facets.vertex(f, 2);
						n.nodes.resize(1, 2);
						n.nodes << V(nodes_ids[9], 0), V(nodes_ids[9], 1);
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
					assert(false);
				}
			}

			if (orders_.maxCoeff() == 1)
				orders_.resize(0, 0);
		}

		RowVectorNd CMesh2D::edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const
		{
			if (orders_.size() <= 0 || orders_(index.face) == 1 || edge_nodes_.empty() || edge_nodes_[index.edge].nodes.rows() != n_new_nodes)
			{
				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);

				const double t = i / (n_new_nodes + 1.0);

				return (1 - t) * v1 + t * v2;
			}

			const auto &n = edge_nodes_[index.edge];
			if (n.v1 == index.vertex)
				return n.nodes.row(i - 1);
			else
			{
				assert(n.v2 == index.vertex);
				return n.nodes.row(n.nodes.rows() - i);
			}
		}

		RowVectorNd CMesh2D::face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const
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

					return b1 * v1 + b2 * v2 + b3 * v3;
				}

				assert(orders_(index.face) == 3);
				// unsupported P4 for geometry
				const auto &n = face_nodes_[index.face];
				return n.nodes.row(0);
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

				return v1 * (1 - b1) * (1 - b2) + v2 * b1 * (1 - b2) + v3 * b1 * b2 + v4 * (1 - b1) * b2;
			}

			assert(false);
			return RowVectorNd(2, 1);
		}

		void CMesh2D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
		{
			GEO::vec3 min_corner, max_corner;
			GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);
			min.resize(2);
			max.resize(2);

			min(0) = min_corner.x;
			min(1) = min_corner.y;

			max(0) = max_corner.x;
			max(1) = max_corner.y;
		}

		void CMesh2D::normalize()
		{

			GEO::vec3 min_corner, max_corner;
			GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);
			GEO::vec3 extent = max_corner - min_corner;
			double scaling = std::max(extent[0], std::max(extent[1], extent[2]));
			// const GEO::vec3 origin = 0.5 * (min_corner + max_corner);
			const GEO::vec3 origin = min_corner;
			for (GEO::index_t v = 0; v < mesh_.vertices.nb(); ++v)
			{
				mesh_.vertices.point(v) = (mesh_.vertices.point(v) - origin) / scaling;
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

			logger().debug("-- bbox before normalization:");
			logger().debug("   min   : {} {}", min_corner[0], min_corner[1]);
			logger().debug("   max   : {} {}", max_corner[0], max_corner[1]);
			logger().debug("   extent: {} {}", max_corner[0] - min_corner[0], max_corner[1] - min_corner[1]);
			GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);
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
			const int v0 = mesh_.edges.vertex(gid, 0);
			const int v1 = mesh_.edges.vertex(gid, 1);

			return (point(v0) - point(v1)).norm();
		}

		void CMesh2D::set_point(const int global_index, const RowVectorNd &p)
		{
			mesh_.vertices.point(global_index).x = p(0);
			mesh_.vertices.point(global_index).y = p(1);
		}

		RowVectorNd CMesh2D::point(const int global_index) const
		{
			const double *ptr = mesh_.vertices.point_ptr(global_index);
			RowVectorNd p(2);
			p(0) = ptr[0];
			p(1) = ptr[1];
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

		void CMesh2D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
		{
			ranges.clear();

			std::vector<Eigen::MatrixXi> local_tris(mesh_.facets.nb());
			std::vector<Eigen::MatrixXd> local_pts(mesh_.facets.nb());

			int total_tris = 0;
			int total_pts = 0;

			ranges.push_back(0);

			for (GEO::index_t f = 0; f < mesh_.facets.nb(); ++f)
			{
				const int n_vertices = mesh_.facets.nb_vertices(f);

				Eigen::MatrixXd face_pts(n_vertices, 2);
				// Eigen::MatrixXi edges(n_vertices,2);
				local_tris[f].resize(n_vertices - 2, 3);

				for (int i = 0; i < n_vertices; ++i)
				{
					const int vertex = mesh_.facets.vertex(f, i);
					const double *pt = mesh_.vertices.point_ptr(vertex);
					face_pts(i, 0) = pt[0];
					face_pts(i, 1) = pt[1];

					// edges(i, 0) = i;
					// edges(i, 1) = (i+1) % n_vertices;
				}

				for (int i = 1; i < n_vertices - 1; ++i)
				{
					local_tris[f].row(i - 1) << 0, i, i + 1;
				}

				local_pts[f] = face_pts;

				total_tris += local_tris[f].rows();
				total_pts += local_pts[f].rows();

				ranges.push_back(total_tris);

				assert(local_pts[f].rows() == face_pts.rows());
			}

			tris.resize(total_tris, 3);
			pts.resize(total_pts, 2);

			int tri_index = 0;
			int pts_index = 0;
			for (std::size_t i = 0; i < local_tris.size(); ++i)
			{
				tris.block(tri_index, 0, local_tris[i].rows(), local_tris[i].cols()) = local_tris[i].array() + pts_index;
				tri_index += local_tris[i].rows();

				pts.block(pts_index, 0, local_pts[i].rows(), local_pts[i].cols()) = local_pts[i];
				pts_index += local_pts[i].rows();
			}
		}

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
			const int v0 = mesh_.edges.vertex(index, 0);
			const int v1 = mesh_.edges.vertex(index, 1);

			return 0.5 * (point(v0) + point(v1));
		}

		void CMesh2D::compute_body_ids(const std::function<int(const size_t, const RowVectorNd &)> &marker)
		{
			body_ids_.resize(n_elements());
			std::fill(body_ids_.begin(), body_ids_.end(), -1);

			for (int e = 0; e < n_elements(); ++e)
			{
				const auto bary = face_barycenter(e);
				body_ids_[e] = marker(e, bary);
			}
		}

		void CMesh2D::compute_boundary_ids(const double eps)
		{
			boundary_ids_.resize(n_edges());
			std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

			RowVectorNd min_corner, max_corner;
			bounding_box(min_corner, max_corner);

			// implement me properly
			for (int e = 0; e < n_edges(); ++e)
			{
				if (!is_boundary_edge(e))
					continue;

				const auto p = edge_barycenter(e);

				if (fabs(p(0) - min_corner[0]) < eps)
					boundary_ids_[e] = 1;
				else if (fabs(p(1) - min_corner[1]) < eps)
					boundary_ids_[e] = 2;
				else if (fabs(p(0) - max_corner[0]) < eps)
					boundary_ids_[e] = 3;
				else if (fabs(p(1) - max_corner[1]) < eps)
					boundary_ids_[e] = 4;

				else
					boundary_ids_[e] = 7;
			}
		}

		void CMesh2D::compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker)
		{
			boundary_ids_.resize(n_edges());
			std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

			// implement me properly
			for (int e = 0; e < n_edges(); ++e)
			{
				if (!is_boundary_edge(e))
					continue;

				const auto p = edge_barycenter(e);

				boundary_ids_[e] = marker(p);
			}
		}

		void CMesh2D::compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker)
		{
			boundary_ids_.resize(n_edges());

			for (int e = 0; e < n_edges(); ++e)
			{
				const bool is_boundary = is_boundary_edge(e);
				const auto p = edge_barycenter(e);
				boundary_ids_[e] = marker(p, is_boundary);
			}
		}

		void CMesh2D::compute_boundary_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker)
		{
			boundary_ids_.resize(n_edges());

			for (int e = 0; e < n_edges(); ++e)
			{
				const bool is_boundary = is_boundary_edge(e);
				const auto p = edge_barycenter(e);
				boundary_ids_[e] = marker(e, p, is_boundary);
			}
		}

		void CMesh2D::compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker)
		{
			boundary_ids_.resize(n_edges());

			for (int e = 0; e < n_edges(); ++e)
			{
				bool is_boundary = is_boundary_edge(e);
				std::vector<int> vs = {edge_vertex(e, 0), edge_vertex(e, 1)};
				std::sort(vs.begin(), vs.end());
				boundary_ids_[e] = marker(vs, is_boundary);
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

			mesh_.vertices.create_vertices(mesh2d.n_vertices());
			for (int i = n_v; i < (int)mesh_.vertices.nb(); ++i)
			{
				GEO::vec3 &p = mesh_.vertices.point(i);
				set_point(i, mesh2d.point(i - n_v));
			}

			std::vector<GEO::index_t> indices;
			for (int i = 0; i < mesh2d.n_faces(); ++i)
			{
				indices.clear();
				for (int j = 0; j < mesh2d.mesh_.facets.nb_vertices(i); ++j)
					indices.push_back(mesh2d.mesh_.facets.vertex(i, j) + n_v);

				mesh_.facets.create_polygon(indices.size(), &indices[0]);
			}

			assert(n_vertices() == n_v + mesh2d.n_vertices());
			assert(n_faces() == n_f + mesh2d.n_faces());

			c2e_.reset();
			boundary_vertices_.reset();
			boundary_edges_.reset();
			Navigation::prepare_mesh(mesh_);
			c2e_ = std::make_unique<GEO::Attribute<GEO::index_t>>(mesh_.facet_corners.attributes(), "edge_id");
			boundary_vertices_ = std::make_unique<GEO::Attribute<bool>>(mesh_.vertices.attributes(), "boundary_vertex");
			boundary_edges_ = std::make_unique<GEO::Attribute<bool>>(mesh_.edges.attributes(), "boundary_edge");
		}
	} // namespace mesh
} // namespace polyfem
