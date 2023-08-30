#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/Navigation.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace polyfem
{
	namespace mesh
	{
		class Mesh2D : public Mesh
		{
		public:
			Mesh2D() = default;
			virtual ~Mesh2D() = default;
			Mesh2D(Mesh2D &&) = default;
			Mesh2D &operator=(Mesh2D &&) = default;
			Mesh2D(const Mesh2D &) = default;
			Mesh2D &operator=(const Mesh2D &) = default;

			bool is_volume() const override { return false; }

			int n_cells() const override { return 0; }
			inline int n_cell_vertices(const int c_id) const override { return n_face_vertices(c_id); }

			bool is_boundary_face(const int face_global_id) const override
			{
				assert(false);
				return false;
			}

			virtual RowVectorNd edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const = 0;
			virtual RowVectorNd face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const = 0;

			RowVectorNd cell_barycenter(const int index) const override
			{
				assert(false);
				return RowVectorNd(2);
			}

			void compute_face_jacobian(const int el_id, const Eigen::MatrixXd &reference_map, Eigen::MatrixXd &jacobian) const;

			// Navigation wrapper
			virtual Navigation::Index get_index_from_face(int f, int lv = 0) const = 0;

			// Navigation in a surface mesh
			virtual Navigation::Index switch_vertex(Navigation::Index idx) const = 0;
			virtual Navigation::Index switch_edge(Navigation::Index idx) const = 0;
			virtual Navigation::Index switch_face(Navigation::Index idx) const = 0;

			void barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const override;
			RowVectorNd face_barycenter(const int index) const override;
			void compute_element_barycenters(Eigen::MatrixXd &barycenters) const override { face_barycenters(barycenters); }
			void elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const override;

			// Iterate in a mesh
			inline Navigation::Index next_around_face(Navigation::Index idx) const { return switch_edge(switch_vertex(idx)); }
			inline Navigation::Index next_around_vertex(Navigation::Index idx) const { return switch_face(switch_edge(idx)); }

			void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;
			void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const override;
		};
	} // namespace mesh
} // namespace polyfem
