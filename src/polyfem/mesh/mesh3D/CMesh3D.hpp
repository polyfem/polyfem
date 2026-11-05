#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/mesh/mesh3D/Navigation3D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3DStorage.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <array>

namespace polyfem
{
	namespace mesh
	{
		struct StandardCell;

		/// Conforming 3D mesh of tetrahedra, hexahedra, prisms, pyramids and
		/// general polyhedra, stored in a Mesh3DStorage.
		///
		/// Numbering: vertices and cells follow the input order; faces are
		/// created cell by cell (local facet order of CellConnectivity) and keep
		/// the vertex order of the lower-id cell; edges are sorted by (min, max)
		/// vertex id. cell_vertex() returns the input vertex order of a cell (the
		/// order of MeshData::elements); the topology uses the descriptor order.
		/// Hex meshes are reoriented so that neighboring hexes agree (needed by
		/// the spline bases), and every cell has a kernel point (a point from
		/// which the cell is star-shaped) used by the polyhedral quadrature and
		/// by the polyhedral refinement.
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

			int n_cells() const override { return int(mesh_.n_cells()); }
			int n_faces() const override { return int(mesh_.n_faces()); }
			int n_edges() const override { return int(mesh_.n_edges()); }
			int n_vertices() const override { return int(mesh_.points.cols()); }

			inline int n_face_vertices(const int f_id) const override { return mesh_.face_vs[f_id].size(); }
			inline int n_cell_vertices(const int c_id) const override { return mesh_.cell_vs[c_id].size(); }
			inline int n_cell_edges(const int c_id) const override { return mesh_.cell_es[c_id].size(); }
			inline int n_cell_faces(const int c_id) const override { return mesh_.cell_fs[c_id].size(); }
			inline int cell_vertex(const int c_id, const int lv_id) const override
			{
				return mesh_.cell_input_vs.row_size(c_id) == 0 ? mesh_.cell_vs[c_id][lv_id] : mesh_.cell_input_vs[c_id][lv_id];
			}
			inline int cell_face(const int c_id, const int lf_id) const override { return mesh_.cell_fs[c_id][lf_id]; }
			inline int cell_face_orientation(const int c_id, const int lf_id) const override { return mesh_.cell_fs_flag[c_id][lf_id]; }
			inline int cell_edge(const int c_id, const int le_id) const override { return mesh_.cell_es[c_id][le_id]; }
			inline int face_vertex(const int f_id, const int lv_id) const override { return mesh_.face_vs[f_id][lv_id]; }
			inline int edge_vertex(const int e_id, const int lv_id) const override { return mesh_.edge_vs[e_id][lv_id]; }

			void barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const override;

			bool is_boundary_vertex(const int vertex_global_id) const override { return mesh_.vertex_boundary[vertex_global_id]; }
			bool is_boundary_edge(const int edge_global_id) const override { return mesh_.edge_boundary[edge_global_id]; }
			bool is_boundary_face(const int face_global_id) const override { return mesh_.face_boundary[face_global_id]; }
			bool is_boundary_element(const int element_global_id) const override;

			bool build_topology(const MeshData &data) override;

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

			inline std::vector<uint32_t> vertex_neighs(const int v_gid) const override { return mesh_.vertex_hs[v_gid].to_vector(); }
			inline std::vector<uint32_t> edge_neighs(const int e_gid) const override { return mesh_.edge_hs[e_gid].to_vector(); }

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
				ids.insert(ids.begin(), mesh_.vertex_hs[v_id].begin(), mesh_.vertex_hs[v_id].end());
			}
			void get_edge_elements_neighs(const int e_id, std::vector<int> &ids) const override
			{
				ids.clear();
				ids.insert(ids.begin(), mesh_.edge_hs[e_id].begin(), mesh_.edge_hs[e_id].end());
			}

			void compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker) override;

			void compute_body_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &)> &marker) override;

			// void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;

			// used for sweeping 2D mesh
			Mesh3DStorage &mesh_storge()
			{
				std::cerr << "never user this function" << std::endl;
				return mesh_;
			}

			void append(const Mesh &mesh) override;

			/// Also moves the kernel points of the elements, used by polyhedral
			/// quadrature and refinement.
			void apply_affine_transformation(const MatrixNd &A, const VectorNd &b) override;

			std::unique_ptr<Mesh> copy() const override;

		protected:
			void remove_elements(const std::vector<bool> &keep) override;

		private:
			bool build_storage_from_cells(const Eigen::MatrixXd &V, const std::vector<StandardCell> &cells);
			Mesh3DStorage mesh_;
		};
	} // namespace mesh
} // namespace polyfem
