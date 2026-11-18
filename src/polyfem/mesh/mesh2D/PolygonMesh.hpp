#pragma once

#include <array>
#include <vector>

namespace polyfem::mesh
{
	/// @brief Minimal polygonal surface mesh used as the storage of CMesh2D.
	///
	/// Vertices carry 3D positions (z = 0 for planar meshes), facets are
	/// polygons stored as contiguous ranges of corners, and each corner knows
	/// the facet across the edge that starts at it. Edges and boundary flags are
	/// derived data filled by Navigation::prepare_mesh().
	///
	/// The connectivity semantics (facet gluing, facet flipping and deletion)
	/// and the point arithmetic follow geogram's GEO::Mesh, which this class
	/// replaces, so that meshes built with it are identical to the ones PolyFEM
	/// used to build.
	class PolygonMesh
	{
	public:
		/// 3D point. Each operator computes every coordinate with a single
		/// floating-point operation, like geogram's vec3.
		struct Point
		{
			std::array<double, 3> coords = {{0, 0, 0}};

			Point() = default;
			Point(const double x, const double y, const double z) : coords{{x, y, z}} {}

			double &operator[](const int i) { return coords[i]; }
			double operator[](const int i) const { return coords[i]; }

			Point &operator+=(const Point &v)
			{
				coords[0] += v[0];
				coords[1] += v[1];
				coords[2] += v[2];
				return *this;
			}
			Point operator+(const Point &v) const { return Point(coords[0] + v[0], coords[1] + v[1], coords[2] + v[2]); }
			Point operator-(const Point &v) const { return Point(coords[0] - v[0], coords[1] - v[1], coords[2] - v[2]); }
			Point operator/(const double s) const { return Point(coords[0] / s, coords[1] / s, coords[2] / s); }
			friend Point operator*(const double s, const Point &v) { return Point(s * v[0], s * v[1], s * v[2]); }
		};

		/// Removes all vertices, facets and edges.
		void clear();

		// -- Vertices ------------------------------------------------------------

		int n_vertices() const { return points_.size(); }
		const Point &point(const int v) const { return points_[v]; }
		Point &point(const int v) { return points_[v]; }

		/// Appends n vertices at the origin and returns the index of the first one.
		int create_vertices(int n);
		/// Appends a vertex and returns its index.
		int create_vertex(const Point &p);

		/// Bounding box of the vertices (min/max are initialized to the largest
		/// and lowest doubles, as geogram's get_bbox does).
		void bounding_box(Point &min, Point &max) const;

		// -- Facets --------------------------------------------------------------

		int n_facets() const { return int(facet_ptr_.size()) - 1; }
		int n_facet_vertices(const int f) const { return facet_ptr_[f + 1] - facet_ptr_[f]; }
		int facet_vertex(const int f, const int lv) const { return corner_vertex_[facet_ptr_[f] + lv]; }

		int n_corners() const { return corner_vertex_.size(); }
		int corners_begin(const int f) const { return facet_ptr_[f]; }
		int corners_end(const int f) const { return facet_ptr_[f + 1]; }
		int facet_corner(const int f, const int lv) const { return facet_ptr_[f] + lv; }
		int corner_vertex(const int c) const { return corner_vertex_[c]; }
		/// Facet across the edge from corner c to the next corner of its facet, or -1.
		int corner_adjacent_facet(const int c) const { return corner_adjacent_facet_[c]; }
		int next_corner_around_facet(const int f, const int c) const { return c + 1 == corners_end(f) ? corners_begin(f) : c + 1; }
		int prev_corner_around_facet(const int f, const int c) const { return c == corners_begin(f) ? corners_end(f) - 1 : c - 1; }

		/// Appends a polygon (not glued to its neighbors) and returns its index.
		int create_polygon(const std::vector<int> &vertices);

		/// Reverses the orientation of facet f (its vertex list is reversed).
		void flip_facet(int f);

		/// Removes all facets, keeping the vertices and the edges.
		void clear_facets();

		/// Removes the facets f with to_delete[f], keeping all vertices, and
		/// renumbers the remaining facets (and the corner adjacency) in order.
		void delete_facets(const std::vector<bool> &to_delete);

		/// Glues every corner without neighbor to the facet sharing its edge in
		/// opposite orientation (already glued corners are kept).
		void connect_facets();

		/// Assigns to each facet the index of its connected component (through
		/// the corner adjacency) and returns the number of components.
		int connected_components(std::vector<int> &component) const;

		// -- Edges (filled by Navigation::prepare_mesh) ---------------------------

		int n_edges() const { return edges_.size(); }
		int edge_vertex(const int e, const int lv) const { return edges_[e][lv]; }
		int create_edge(int v0, int v1);
		void clear_edges();

		/// Index of the edge starting at each corner.
		std::vector<int> corner_edge;
		/// Whether each edge is on the boundary.
		std::vector<bool> boundary_edge;
		/// Whether each vertex is on the boundary.
		std::vector<bool> boundary_vertex;

	private:
		std::vector<Point> points_;
		std::vector<int> facet_ptr_ = {0};
		std::vector<int> corner_vertex_;
		std::vector<int> corner_adjacent_facet_;
		std::vector<std::array<int, 2>> edges_;
	};
} // namespace polyfem::mesh
