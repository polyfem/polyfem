#ifndef MESH_HPP
#define MESH_HPP

#include "navigation.hpp"

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace poly_fem
{
	class Mesh
	{
	public:
		inline bool is_volume() const { return mesh_.cells.nb() > 0; }

		inline int n_elements() const { return is_volume() ? mesh_.cells.nb() : mesh_.facets.nb(); }
		inline int n_pts() const { return mesh_.vertices.nb(); }

		inline int n_element_vertices(const int element_index) const { return is_volume() ? mesh_.cells.nb_vertices(element_index) : mesh_.facets.nb_vertices(element_index);}
		inline int vertex_global_index(const int element_index, const int local_index) const { return is_volume() ? mesh_.cells.vertex(element_index, local_index) : mesh_.facets.vertex(element_index, local_index); }

		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const;

		void point(const int global_index, Eigen::MatrixXd &pt) const;

		bool load(const std::string &path);
		bool save(const std::string &path) const;

		int edge_node_id(const int edge_id) const;
		int vertex_node_id(const int vertex_id) const;

		inline int node_id_from_edge_index(const Navigation::Index &index) const
		{
			int id = switch_face(index).face;
			if(id >= 0) return id;

			id = edge_node_id(index.edge);
			assert(id >= 0);

			return id;
		}

		Eigen::MatrixXd node_from_edge_index(const Navigation::Index &index) const;
		Eigen::MatrixXd node_from_face(const int face_id) const;
		Eigen::MatrixXd node_from_vertex(const int &vertex_id) const;

		inline const GEO::Mesh  &mesh() const { return mesh_; }

		Navigation::Index get_index_from_face(int f, int lv = 0) const;

	// Navigation in a surface mesh
		Navigation::Index switch_vertex(Navigation::Index idx) const;
		Navigation::Index switch_edge(Navigation::Index idx) const;
		Navigation::Index switch_face(Navigation::Index idx) const;

		// Iterate in a mesh
		inline Navigation::Index next_around_face(Navigation::Index idx) const { return switch_edge(switch_vertex(idx)); }
		inline Navigation::Index next_around_edge(Navigation::Index idx) const { return switch_vertex(switch_face(idx)); }
		inline Navigation::Index next_around_vertex(Navigation::Index idx) const { return switch_face(switch_edge(idx)); }

		void create_boundary_nodes();

	private:
		GEO::Mesh mesh_;
	};
}

#endif //MESH_HPP
