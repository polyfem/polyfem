#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include "Navigation3D.hpp"
#include "Mesh3DStorage.hpp"
#include <geogram/mesh/mesh.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <array>

namespace polyfem
{
	namespace mesh
	{
		class Mesh3D : public Mesh
		{
		public:
			Mesh3D() = default;
			virtual ~Mesh3D() = default;
			POLYFEM_DEFAULT_MOVE_COPY(Mesh3D)

			bool is_conforming() const override { return true; }

			void refine(const int n_refinement, const double t, std::vector<int> &parent_nodes) override;

			inline bool is_volume() const override { return true; }

			int n_cells() const override { return int(mesh_.elements.size()); }
			int n_faces() const override { return int(mesh_.faces.size()); }
			int n_edges() const override { return int(mesh_.edges.size()); }
			int n_vertices() const override { return int(mesh_.points.cols()); }

			inline int n_face_vertices(const int f_id) const override { return mesh_.faces[f_id].vs.size(); }
			inline int n_cell_vertices(const int c_id) const { return mesh_.elements[c_id].vs.size(); }
			inline int n_cell_faces(const int c_id) const { return mesh_.elements[c_id].fs.size(); }
			inline int cell_vertex(const int c_id, const int lv_id) const override { return mesh_.elements[c_id].vs[lv_id]; }
			inline int cell_face(const int c_id, const int lf_id) const { return mesh_.elements[c_id].fs[lf_id]; }
			inline int cell_edge(const int c_id, const int le_id) const { return mesh_.elements[c_id].es[le_id]; }
			inline int face_vertex(const int f_id, const int lv_id) const override { return mesh_.faces[f_id].vs[lv_id]; }
			inline int edge_vertex(const int e_id, const int lv_id) const override { return mesh_.edges[e_id].vs[lv_id]; }

			void elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const override;
			void barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const override;

			bool is_boundary_vertex(const int vertex_global_id) const override { return mesh_.vertices[vertex_global_id].boundary; }
			bool is_boundary_edge(const int edge_global_id) const override { return mesh_.edges[edge_global_id].boundary; }
			bool is_boundary_face(const int face_global_id) const override { return mesh_.faces[face_global_id].boundary; }
			bool is_boundary_element(const int element_global_id) const override;

			bool save(const std::string &path) const override;
			bool save(const std::vector<int> &fs, const int ringN, const std::string &path) const;
			bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) override;

			void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) override;
			RowVectorNd edge_node(const Navigation3D::Index &index, const int n_new_nodes, const int i) const;
			RowVectorNd face_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j) const;
			RowVectorNd cell_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j, const int k) const;

			void normalize() override;

			virtual double quad_area(const int gid) const override;
			virtual double tri_area(const int gid) const override;

			void compute_elements_tag() override;

			RowVectorNd kernel(const int cell_id) const;
			virtual RowVectorNd point(const int vertex_id) const override;
			virtual RowVectorNd edge_barycenter(const int e) const override;
			virtual RowVectorNd face_barycenter(const int f) const override;
			virtual RowVectorNd cell_barycenter(const int c) const override;

			virtual void bounding_box(RowVectorNd &min, RowVectorNd &max) const override;

			//navigation wrapper
			Navigation3D::Index get_index_from_element(int hi, int lf, int lv) const { return Navigation3D::get_index_from_element_face(mesh_, hi, lf, lv); }
			Navigation3D::Index get_index_from_element(int hi) const { return Navigation3D::get_index_from_element_face(mesh_, hi); }

			Navigation3D::Index get_index_from_element_edge(int hi, int v0, int v1) const { return Navigation3D::get_index_from_element_edge(mesh_, hi, v0, v1); }
			Navigation3D::Index get_index_from_element_face(int hi, int v0, int v1, int v2) const { return Navigation3D::get_index_from_element_tri(mesh_, hi, v0, v1, v2); }

			inline std::vector<uint32_t> vertex_neighs(const int v_gid) const { return mesh_.vertices[v_gid].neighbor_hs; }
			inline std::vector<uint32_t> edge_neighs(const int e_gid) const { return mesh_.edges[e_gid].neighbor_hs; }

			// Navigation in a surface mesh
			Navigation3D::Index switch_vertex(Navigation3D::Index idx) const { return Navigation3D::switch_vertex(mesh_, idx); }
			Navigation3D::Index switch_edge(Navigation3D::Index idx) const { return Navigation3D::switch_edge(mesh_, idx); }
			Navigation3D::Index switch_face(Navigation3D::Index idx) const { return Navigation3D::switch_face(mesh_, idx); }
			Navigation3D::Index switch_element(Navigation3D::Index idx) const { return Navigation3D::switch_element(mesh_, idx); }

			// Iterate in a mesh
			inline Navigation3D::Index next_around_edge(Navigation3D::Index idx) const { return Navigation3D::next_around_3Dedge(mesh_, idx); }
			inline Navigation3D::Index next_around_face(Navigation3D::Index idx) const { return Navigation3D::next_around_2Dface(mesh_, idx); }

			void to_face_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> &to_face) const;
			void to_vertex_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> &to_vertex) const;
			void to_edge_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> &to_edge) const;

			// Retrieves hex vertices numbered as follows:
			//   v7────v6
			//   ╱┆    ╱│
			// v4─┼──v5 │
			//  │v3┄┄┄┼v2
			//  │╱    │╱
			// v0────v1
			std::array<int, 8> get_ordered_vertices_from_hex(const int element_index) const;
			std::array<int, 4> get_ordered_vertices_from_tet(const int element_index) const;

			void get_vertex_elements_neighs(const int v_id, std::vector<int> &ids) const
			{
				ids.clear();
				ids.insert(ids.begin(), mesh_.vertices[v_id].neighbor_hs.begin(), mesh_.vertices[v_id].neighbor_hs.end());
			}
			void get_edge_elements_neighs(const int e_id, std::vector<int> &ids) const
			{
				ids.clear();
				ids.insert(ids.begin(), mesh_.edges[e_id].neighbor_hs.begin(), mesh_.edges[e_id].neighbor_hs.end());
			}

			void compute_boundary_ids(const double eps) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker) override;
			void compute_body_ids(const std::function<int(const RowVectorNd &)> &marker) override;

			void compute_element_barycenters(Eigen::MatrixXd &barycenters) const override { cell_barycenters(barycenters); }
			void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;
			void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;
			void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const override;

			//used for sweeping 2D mesh
			Mesh3DStorage &mesh_storge()
			{
				std::cerr << "never user this function" << std::endl;
				return mesh_;
			}
			static void geomesh_2_mesh_storage(const GEO::Mesh &gm, Mesh3DStorage &m);

		protected:
			bool load(const std::string &path) override;
			bool load(const GEO::Mesh &M) override;

		private:
			Mesh3DStorage mesh_;
		};
	} // namespace mesh
} // namespace polyfem
