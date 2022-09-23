#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/mesh/mesh3D/Navigation3D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3DStorage.hpp>
#include <geogram/mesh/mesh.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <array>

namespace polyfem
{
	namespace mesh
	{
		class CMesh3D : public Mesh3D
		{
		public:
			CMesh3D() = default;
			virtual ~CMesh3D() = default;
			CMesh3D(CMesh3D &&) = default;
			CMesh3D &operator=(CMesh3D &&) = default;
			CMesh3D(const CMesh3D &) = default;
			CMesh3D &operator=(const CMesh3D &) = default;

			bool is_conforming() const override { return true; }

			void refine(const int n_refinement, const double t) override;

			int n_cells() const override { return int(mesh_.elements.size()); }
			int n_faces() const override { return int(mesh_.faces.size()); }
			int n_edges() const override { return int(mesh_.edges.size()); }
			int n_vertices() const override { return int(mesh_.points.cols()); }

			inline int n_face_vertices(const int f_id) const override { return mesh_.faces[f_id].vs.size(); }
			inline int n_cell_vertices(const int c_id) const override { return mesh_.elements[c_id].vs.size(); }
			inline int n_cell_edges(const int c_id) const override { return mesh_.elements[c_id].es.size(); }
			inline int n_cell_faces(const int c_id) const override { return mesh_.elements[c_id].fs.size(); }
			inline int cell_vertex(const int c_id, const int lv_id) const override { return mesh_.elements[c_id].vs[lv_id]; }
			inline int cell_face(const int c_id, const int lf_id) const override { return mesh_.elements[c_id].fs[lf_id]; }
			inline int cell_edge(const int c_id, const int le_id) const override { return mesh_.elements[c_id].es[le_id]; }
			inline int face_vertex(const int f_id, const int lv_id) const override { return mesh_.faces[f_id].vs[lv_id]; }
			inline int edge_vertex(const int e_id, const int lv_id) const override { return mesh_.edges[e_id].vs[lv_id]; }

			void elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const override;
			void barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const override;

			bool is_boundary_vertex(const int vertex_global_id) const override { return mesh_.vertices[vertex_global_id].boundary; }
			bool is_boundary_edge(const int edge_global_id) const override { return mesh_.edges[edge_global_id].boundary; }
			bool is_boundary_face(const int face_global_id) const override { return mesh_.faces[face_global_id].boundary; }
			bool is_boundary_element(const int element_global_id) const override;

			bool save(const std::string &path) const override;

			bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) override;

			void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) override;

			void normalize() override;

			double quad_area(const int gid) const override;

			void compute_elements_tag() override;

			RowVectorNd kernel(const int cell_id) const override;
			RowVectorNd point(const int vertex_id) const override;
			void set_point(const int global_index, const RowVectorNd &p) override;
			RowVectorNd edge_barycenter(const int e) const override;
			RowVectorNd face_barycenter(const int f) const override;
			RowVectorNd cell_barycenter(const int c) const override;

			void bounding_box(RowVectorNd &min, RowVectorNd &max) const override;

			// navigation wrapper
			Navigation3D::Index get_index_from_element(int hi, int lf, int lv) const override { return Navigation3D::get_index_from_element_face(mesh_, hi, lf, lv); }
			Navigation3D::Index get_index_from_element(int hi) const override { return Navigation3D::get_index_from_element_face(mesh_, hi); }

			Navigation3D::Index get_index_from_element_edge(int hi, int v0, int v1) const override { return Navigation3D::get_index_from_element_edge(mesh_, hi, v0, v1); }
			Navigation3D::Index get_index_from_element_face(int hi, int v0, int v1, int v2) const override { return Navigation3D::get_index_from_element_tri(mesh_, hi, v0, v1, v2); }

			inline std::vector<uint32_t> vertex_neighs(const int v_gid) const override { return mesh_.vertices[v_gid].neighbor_hs; }
			inline std::vector<uint32_t> edge_neighs(const int e_gid) const override { return mesh_.edges[e_gid].neighbor_hs; }

			// Navigation in a surface mesh
			Navigation3D::Index switch_vertex(Navigation3D::Index idx) const override { return Navigation3D::switch_vertex(mesh_, idx); }
			Navigation3D::Index switch_edge(Navigation3D::Index idx) const override { return Navigation3D::switch_edge(mesh_, idx); }
			Navigation3D::Index switch_face(Navigation3D::Index idx) const override { return Navigation3D::switch_face(mesh_, idx); }
			Navigation3D::Index switch_element(Navigation3D::Index idx) const override { return Navigation3D::switch_element(mesh_, idx); }

			// Iterate in a mesh
			inline Navigation3D::Index next_around_edge(Navigation3D::Index idx) const override { return Navigation3D::next_around_3Dedge(mesh_, idx); }
			inline Navigation3D::Index next_around_face(Navigation3D::Index idx) const override { return Navigation3D::next_around_2Dface(mesh_, idx); }

			void get_vertex_elements_neighs(const int v_id, std::vector<int> &ids) const override
			{
				ids.clear();
				ids.insert(ids.begin(), mesh_.vertices[v_id].neighbor_hs.begin(), mesh_.vertices[v_id].neighbor_hs.end());
			}
			void get_edge_elements_neighs(const int e_id, std::vector<int> &ids) const override
			{
				ids.clear();
				ids.insert(ids.begin(), mesh_.edges[e_id].neighbor_hs.begin(), mesh_.edges[e_id].neighbor_hs.end());
			}

			void compute_boundary_ids(const double eps) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker) override;
			void compute_body_ids(const std::function<int(const size_t, const RowVectorNd &)> &marker) override;
			void compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker) override;

			void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;

			// used for sweeping 2D mesh
			Mesh3DStorage &mesh_storge()
			{
				std::cerr << "never user this function" << std::endl;
				return mesh_;
			}
			static void geomesh_2_mesh_storage(const GEO::Mesh &gm, Mesh3DStorage &m);

			void append(const Mesh &mesh) override;

		protected:
			bool load(const std::string &path) override;
			bool load(const GEO::Mesh &M) override;

		private:
			Mesh3DStorage mesh_;
		};
	} // namespace mesh
} // namespace polyfem
