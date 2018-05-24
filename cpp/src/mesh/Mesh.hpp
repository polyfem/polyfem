#ifndef MESH_HPP
#define MESH_HPP

#include "Navigation.hpp"
#include "Types.hpp"

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>
#include <memory>

namespace poly_fem
{
	// NOTE:
	// For the purpose of the tagging, elements (facets in 2D, cells in 3D) adjacent to a polytope
	// are tagged as boundary, and vertices incident to a polytope are also considered as boundary.
	enum class ElementType {
		Simplex, 					// Triangle/tet element
		RegularInteriorCube,        // Regular quad/hex inside a 3^n patch
		SimpleSingularInteriorCube, // Quad/hex incident to exactly 1 singular vertex (in 2D) or edge (in 3D)
		MultiSingularInteriorCube,  // Quad/Hex incident to more than 1 singular vertices (should not happen in 2D)
		RegularBoundaryCube,        // Boundary quad/hex, where all boundary vertices/edges are incident to at most 2 quads/hexes
		SimpleSingularBoundaryCube, // Quad incident to exactly 1 singular vertex (in 2D); hex incident to exactly 1 singular interior edge, 0 singular boundary edge, 1 boundary face (in 3D)
		MultiSingularBoundaryCube,  // Boundary hex that is not regular nor SimpleSingularBoundaryCube
		InterfaceCube,              // Quad/hex that is at the interface with a polytope (if a cube has both external boundary and and interface with a polytope, it is marked as interface)
		InteriorPolytope,           // Interior polytope
		BoundaryPolytope,           // Boundary polytope
		Undefined                   // For invalid configurations
	};

	class Mesh
	{
	public:
		static std::unique_ptr<Mesh> create(const std::string &path);
		static std::unique_ptr<Mesh> create(GEO::Mesh &M);

		virtual ~Mesh() = default;

		virtual void refine(const int n_refinement, const double t, std::vector<int> &parent_nodes) = 0;

		//Queries
		virtual bool is_volume() const = 0;
		int dimension() const { return (is_volume() ? 3 : 2); }

		int n_elements() const { return (is_volume() ? n_cells() : n_faces()); }

		virtual int n_cells() const = 0;
		virtual int n_faces() const = 0;
		virtual int n_edges() const = 0;
		virtual int n_vertices() const = 0;

		virtual bool is_boundary_vertex(const int vertex_global_id) const = 0;
		virtual bool is_boundary_edge(const int edge_global_id) const = 0;
		virtual bool is_boundary_face(const int face_global_id) const = 0;

		//IO
		virtual bool load(const std::string &path) = 0;
		virtual bool load(const GEO::Mesh &M) = 0;
		virtual bool save(const std::string &path) const = 0;

		virtual void normalize() = 0;

		//Tagging of the elements
		virtual void compute_elements_tag() = 0;
		virtual void update_elements_tag() { assert(false); }


		//d-1 primitive sizes
		virtual double edge_length(const int gid) const { assert(false);  return 0; }
		virtual double quad_area(const int gid) const { assert(false);  return 0; }
		virtual double tri_area(const int gid) const { assert(false);  return 0; }

		//Nodal access
		virtual RowVectorNd point(const int global_index) const = 0;
		virtual RowVectorNd edge_barycenter(const int e) const = 0;
		virtual RowVectorNd face_barycenter(const int f) const = 0;
		virtual RowVectorNd cell_barycenter(const int c) const = 0;
		void edge_barycenters(Eigen::MatrixXd &barycenters) const;
		void face_barycenters(Eigen::MatrixXd &barycenters) const;
		void cell_barycenters(Eigen::MatrixXd &barycenters) const;

		//Queries on the tags
		bool is_spline_compatible(const int el_id) const;
		bool is_cube(const int el_id) const;
		bool is_polytope(const int el_id) const;
		bool is_simplex(const int el_id) const;

		const std::vector<ElementType> &elements_tag() const { return elements_tag_; }


		//Boundary condition handling
		virtual void compute_boundary_ids() = 0;
		virtual void compute_boundary_ids(const std::function<int(const RowVectorNd&)> &marker) = 0;
		void set_tag(const int el, const ElementType type) { elements_tag_[el] = type; }
		inline int get_boundary_id(const int primitive) const { return boundary_ids_[primitive]; }

		//Visualization methods
		virtual void compute_element_barycenters(Eigen::MatrixXd &barycenters) const = 0;
		virtual void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const = 0;
		virtual void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const = 0;
		virtual void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const = 0;

	protected:
		std::vector<ElementType> elements_tag_;
		std::vector<int> boundary_ids_;
	};
}

#endif //MESH_HPP
