#ifndef MESH_HPP
#define MESH_HPP

#include "Navigation.hpp"

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace poly_fem
{
	enum ElementType {
		regular = 0,		//an interior hex, all its 12 edges are non-singular. an interior regular quad
		one_singular,		//an interior hex, one out of its 12 edges is singular. a quad with an irregular vertex
		multi_singular,		//an interior hex, more than one of its 12 edges is singular. does not apply for quad
		regular_boundary,	//regular boundary or attaching to a non regular that locally looks like a sliced grid
		boundary,			//either on boundary or attaching to a non regular
		non_regular			//polygon or polyhedron
	};

	class Mesh
	{
	public:
		virtual ~Mesh() { }

		virtual void refine(const int n_refiniment) = 0;

		virtual inline bool is_volume() const = 0;

		virtual inline int n_elements() const = 0;
		virtual inline int n_pts() const = 0;

		virtual inline int n_element_vertices(const int element_index) const = 0;
		virtual inline int vertex_global_index(const int element_index, const int local_index) const = 0;

		virtual double compute_mesh_size() const = 0;

		virtual void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const = 0;

		virtual void set_boundary_tags(std::vector<int> &tags) const = 0;

		virtual void point(const int global_index, Eigen::MatrixXd &pt) const = 0;

		virtual bool load(const std::string &path) = 0;
		virtual bool save(const std::string &path) const = 0;

		virtual void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const = 0;

		virtual void compute_element_tag(std::vector<ElementType> &ele_tag) const = 0;

		virtual void compute_barycenter(Eigen::MatrixXd &barycenters) const = 0;
	};
}

#endif //MESH_HPP
