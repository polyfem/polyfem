#ifndef MESH_HPP
#define MESH_HPP

#include "Navigation.hpp"

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace poly_fem
{
	// NOTE:
	// For the purpose of the tagging, elements (facets in 2D, cells in 3D) adjacent to a polytope are tagged as boundary,
	// and vertices incident to a polytope are also considered as boundary.
	enum class ElementType {
		RegularInteriorCube,        // Regular quad/hex inside a 3^n patch
		SimpleSingularInteriorCube, // Quad/hex incident to exactly 1 singular vertex (in 2D) or edge (in 3D)
		MultiSingularInteriorCube,  // Quad/Hex incident to more than 1 singular vertices (should not happen in 2D)
		RegularBoundaryCube,        // Boundary (internal or external) quad/hex, where all boundary vertices are incident to at most 2 quads/hexes
		SingularBoundaryCube,       // Boundary (internal or external) quad/hex that is not regular
		InteriorPolytope,           // Interior polytope
		BoundaryPolytope,           // Boundary polytope
		Undefined                   // For invalid configurations
	};

	class Mesh
	{
	public:
		virtual ~Mesh() { }

		virtual void refine(const int n_refiniment, const double t) = 0;

		virtual inline bool is_volume() const = 0;

		virtual inline int n_elements() const = 0;

		virtual inline int n_element_vertices(const int element_index) const = 0;

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
