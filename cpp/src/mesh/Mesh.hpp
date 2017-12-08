#ifndef MESH_HPP
#define MESH_HPP

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
	private:
		GEO::Mesh mesh_;
	};
}

#endif //MESH_HPP
