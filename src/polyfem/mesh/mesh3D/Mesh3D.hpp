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
			Mesh3D(Mesh3D &&) = default;
			Mesh3D &operator=(Mesh3D &&) = default;
			Mesh3D(const Mesh3D &) = default;
			Mesh3D &operator=(const Mesh3D &) = default;

			inline bool is_volume() const override { return true; }

			virtual int n_cell_edges(const int c_id) const = 0;
			virtual int n_cell_faces(const int c_id) const = 0;
			virtual int cell_face(const int c_id, const int lf_id) const = 0;
			virtual int cell_edge(const int c_id, const int le_id) const = 0;

			void elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const override;
			void barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const override;

			void compute_cell_jacobian(const int el_id, const Eigen::MatrixXd &reference_map, Eigen::MatrixXd &jacobian) const;

			virtual RowVectorNd kernel(const int cell_id) const = 0;

			double tri_area(const int gid) const override;

			RowVectorNd edge_node(const Navigation3D::Index &index, const int n_new_nodes, const int i) const;
			RowVectorNd face_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j) const;
			RowVectorNd cell_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j, const int k) const;

			void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;
			void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const override;

			// navigation wrapper
			virtual Navigation3D::Index get_index_from_element(int hi, int lf, int lv) const = 0;
			virtual Navigation3D::Index get_index_from_element(int hi) const = 0;

			virtual Navigation3D::Index get_index_from_element_edge(int hi, int v0, int v1) const = 0;
			virtual Navigation3D::Index get_index_from_element_face(int hi, int v0, int v1, int v2) const = 0;

			virtual std::vector<uint32_t> vertex_neighs(const int v_gid) const = 0;
			virtual std::vector<uint32_t> edge_neighs(const int e_gid) const = 0;

			// Navigation in a surface mesh
			virtual Navigation3D::Index switch_vertex(Navigation3D::Index idx) const = 0;
			virtual Navigation3D::Index switch_edge(Navigation3D::Index idx) const = 0;
			virtual Navigation3D::Index switch_face(Navigation3D::Index idx) const = 0;
			virtual Navigation3D::Index switch_element(Navigation3D::Index idx) const = 0;

			// Iterate in a mesh
			virtual Navigation3D::Index next_around_edge(Navigation3D::Index idx) const = 0;
			virtual Navigation3D::Index next_around_face(Navigation3D::Index idx) const = 0;

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
			virtual std::array<int, 4> get_ordered_vertices_from_tet(const int element_index) const;

			virtual void get_vertex_elements_neighs(const int v_id, std::vector<int> &ids) const = 0;
			virtual void get_edge_elements_neighs(const int e_id, std::vector<int> &ids) const = 0;

			void compute_element_barycenters(Eigen::MatrixXd &barycenters) const override { cell_barycenters(barycenters); }
		};
	} // namespace mesh
} // namespace polyfem
