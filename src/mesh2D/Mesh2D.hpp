#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Mesh.hpp>
#include <polyfem/Navigation.hpp>
#include <polyfem/Types.hpp>

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace polyfem
{
	class Mesh2D : public Mesh
	{
	public:
		Mesh2D() = default;
		virtual ~Mesh2D() = default;
		POLYFEM_DEFAULT_MOVE_COPY(Mesh2D)

		bool is_volume() const override { return false; }

		int n_cells() const override { return 0; }

		virtual int n_face_vertices(const int f_id) const = 0;

		virtual int face_vertex(const int f_id, const int lv_id) const = 0;
		virtual int edge_vertex(const int e_id, const int lv_id) const = 0;

		bool is_boundary_face(const int face_global_id) const override
		{
			assert(false);
			return false;
		}

		virtual RowVectorNd edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const = 0;
		virtual RowVectorNd face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const = 0;

		virtual void set_point(const int global_index, const RowVectorNd &p) = 0;
		RowVectorNd cell_barycenter(const int index) const override
		{
			assert(false);
			return RowVectorNd(2);
		}

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
} // namespace polyfem
