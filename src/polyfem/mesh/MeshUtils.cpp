////////////////////////////////////////////////////////////////////////////////
#include "MeshUtils.hpp"

#include <polyfem/mesh/PointTriangleDistance.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/HashUtils.hpp>

#include <unordered_set>
#include <set>
#include <tuple>
#include <numeric>
#include <array>
#include <cmath>
#include <limits>

#include <igl/PI.h>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem::utils;

bool polyfem::mesh::is_planar(const Eigen::MatrixXd &V, const double tol)
{
	if (V.cols() == 2)
		return true;

	assert(V.cols() == 3);
	if (V.rows() == 0)
		return false;
	const double diff = V.col(2).maxCoeff() - V.col(2).minCoeff();

	return fabs(diff) < tol;
}

polyfem::mesh::PolygonMesh::Point polyfem::mesh::mesh_vertex(const PolygonMesh &M, int v)
{
	return M.point(v);
}

// -----------------------------------------------------------------------------

polyfem::mesh::PolygonMesh::Point polyfem::mesh::facet_barycenter(const PolygonMesh &M, int f)
{
	PolygonMesh::Point p(0, 0, 0);
	for (int lv = 0; lv < M.n_facet_vertices(f); ++lv)
	{
		p += polyfem::mesh::mesh_vertex(M, M.facet_vertex(f, lv));
	}
	return p / double(M.n_facet_vertices(f));
}

// -----------------------------------------------------------------------------

int polyfem::mesh::mesh_create_vertex(PolygonMesh &M, const PolygonMesh::Point &p)
{
	return M.create_vertex(p);
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::compute_element_tags(const PolygonMesh &M, std::vector<ElementType> &element_tags)
{
	using index_t = int;

	std::vector<ElementType> old_tags = element_tags;

	element_tags.resize(M.n_facets());

	// Step 0: Compute boundary vertices as true boundary + vertices incident to a polygon
	const std::vector<bool> &is_boundary_vertex = M.boundary_vertex;
	std::vector<bool> is_interface_vertex(M.n_vertices(), false);
	{
		for (index_t f = 0; f < M.n_facets(); ++f)
		{
			if (M.n_facet_vertices(f) != 4 || (!old_tags.empty() && old_tags[f] == ElementType::INTERIOR_POLYTOPE))
			{
				// Vertices incident to polygonal facets (triangles or > 4 vertices) are marked as interface
				for (index_t lv = 0; lv < M.n_facet_vertices(f); ++lv)
				{
					is_interface_vertex[M.facet_vertex(f, lv)] = true;
				}
			}
		}
	}

	// Step 1: Determine which vertices are regular or not
	//
	// Interior vertices are regular if they are incident to exactly 4 quads
	// Boundary vertices are regular if they are incident to at most 2 quads, and no other facets
	std::vector<int> degree(M.n_vertices(), 0);
	std::vector<bool> is_regular_vertex(M.n_vertices());
	for (index_t f = 0; f < M.n_facets(); ++f)
	{
		if (M.n_facet_vertices(f) == 4)
		{
			// Only count incident quads for the degree
			for (index_t lv = 0; lv < M.n_facet_vertices(f); ++lv)
			{
				index_t v = M.facet_vertex(f, lv);
				degree[v]++;
			}
		}
	}
	for (index_t v = 0; v < M.n_vertices(); ++v)
	{
		// assert(degree[v] > 0); // We assume there are no isolated vertices here
		if (is_boundary_vertex[v] || is_interface_vertex[v])
		{
			is_regular_vertex[v] = (degree[v] <= 2);
		}
		else
		{
			is_regular_vertex[v] = (degree[v] == 4);
		}
	}

	// Step 2: Iterate over the facets and determine the type
	for (index_t f = 0; f < M.n_facets(); ++f)
	{
		assert(M.n_facet_vertices(f) > 2);
		if (!old_tags.empty() && old_tags[f] == ElementType::INTERIOR_POLYTOPE)
			continue;

		if (M.n_facet_vertices(f) == 4)
		{
			// Quad facet

			// a) Determine if it is on the mesh boundary
			bool is_boundary_facet = false;
			bool is_interface_facet = false;
			for (index_t lv = 0; lv < M.n_facet_vertices(f); ++lv)
			{
				if (is_boundary_vertex[M.facet_vertex(f, lv)])
				{
					is_boundary_facet = true;
				}
				if (is_interface_vertex[M.facet_vertex(f, lv)])
				{
					is_interface_facet = true;
				}
			}

			// b) Determine if it is regular or not
			if (is_boundary_facet || is_interface_facet)
			{
				// A boundary quad is regular iff all its vertices are incident to at most 2 other quads
				// We assume that non-boundary vertices of a boundary quads are always regular
				bool is_singular = false;
				for (index_t lv = 0; lv < M.n_facet_vertices(f); ++lv)
				{
					index_t v = M.facet_vertex(f, lv);
					if (is_boundary_vertex[v] || is_interface_vertex[v])
					{
						if (!is_regular_vertex[v])
						{
							is_singular = true;
							break;
						}
					}
					else
					{
						if (!is_regular_vertex[v])
						{
							element_tags[f] = ElementType::UNDEFINED;
							break;
						}
					}
				}

				if (is_interface_facet)
				{
					element_tags[f] = ElementType::INTERFACE_CUBE;
				}
				else if (is_singular)
				{
					element_tags[f] = ElementType::SIMPLE_SINGULAR_BOUNDARY_CUBE;
				}
				else
				{
					element_tags[f] = ElementType::REGULAR_BOUNDARY_CUBE;
				}
			}
			else
			{
				// An interior quad is regular if all its vertices are singular
				int nb_singulars = 0;
				for (index_t lv = 0; lv < M.n_facet_vertices(f); ++lv)
				{
					if (!is_regular_vertex[M.facet_vertex(f, lv)])
					{
						++nb_singulars;
					}
				}

				if (nb_singulars == 0)
				{
					element_tags[f] = ElementType::REGULAR_INTERIOR_CUBE;
				}
				else if (nb_singulars == 1)
				{
					element_tags[f] = ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE;
				}
				else
				{
					element_tags[f] = ElementType::MULTI_SINGULAR_INTERIOR_CUBE;
				}
			}
		}
		else
		{
			// Polygonal facet

			// Note: In this function, we consider triangles as polygonal facets
			ElementType tag = ElementType::INTERIOR_POLYTOPE;
			const std::vector<bool> &boundary_vertices = M.boundary_vertex;
			for (index_t lv = 0; lv < M.n_facet_vertices(f); ++lv)
			{
				if (boundary_vertices[M.facet_vertex(f, lv)])
				{
					tag = ElementType::BOUNDARY_POLYTOPE;
					break;
				}
			}

			element_tags[f] = tag;
		}
	}

	// TODO what happens at the neighs?
	// Override for simplices
	for (index_t f = 0; f < M.n_facets(); ++f)
	{
		if (M.n_facet_vertices(f) == 3)
		{
			element_tags[f] = ElementType::SIMPLEX;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

namespace
{

	// Same expression shapes as geogram's det(vec2, vec2) and
	// Geom::triangle_signed_area(vec2, vec2, vec2).
	inline double det_2d(const double ax, const double ay, const double bx, const double by)
	{
		return ax * by - ay * bx;
	}

	inline double triangle_signed_area_2d(
		const polyfem::mesh::PolygonMesh::Point &p1,
		const polyfem::mesh::PolygonMesh::Point &p2,
		const polyfem::mesh::PolygonMesh::Point &p3)
	{
		return 0.5 * det_2d(p2[0] - p1[0], p2[1] - p1[1], p3[0] - p1[0], p3[1] - p1[1]);
	}

	// Signed area of a polygonal facet
	double signed_area(const polyfem::mesh::PolygonMesh &M, int f)
	{
		double result = 0;
		int v0 = M.corner_vertex(M.corners_begin(f));
		const polyfem::mesh::PolygonMesh::Point &p0 = M.point(v0);
		for (int c = M.corners_begin(f) + 1; c + 1 < M.corners_end(f); c++)
		{
			int v1 = M.corner_vertex(c);
			const polyfem::mesh::PolygonMesh::Point p1 = polyfem::mesh::mesh_vertex(M, v1);
			int v2 = M.corner_vertex(c + 1);
			const polyfem::mesh::PolygonMesh::Point p2 = polyfem::mesh::mesh_vertex(M, v2);
			result += triangle_signed_area_2d(p0, p1, p2);
		}
		return result;
	}

} // anonymous namespace

void polyfem::mesh::orient_normals_2d(PolygonMesh &M)
{
	std::vector<int> component;
	int nb_components = M.connected_components(component);
	std::vector<double> comp_signed_volume(nb_components, 0.0);
	for (int f = 0; f < M.n_facets(); ++f)
	{
		comp_signed_volume[component[f]] += signed_area(M, f);
	}
	for (int f = 0; f < M.n_facets(); ++f)
	{
		if (comp_signed_volume[component[f]] < 0.0)
		{
			M.flip_facet(f);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::reorder_mesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const Eigen::VectorXi &C, Eigen::VectorXi &R)
{
	assert(V.rows() == C.size());
	int num_colors = C.maxCoeff() + 1;
	Eigen::VectorXi count(num_colors);
	count.setZero();
	for (int i = 0; i < C.size(); ++i)
	{
		++count[C(i)];
	}
	R.resize(num_colors + 1);
	R(0) = 0;
	for (int c = 0; c < num_colors; ++c)
	{
		R(c + 1) = R(c) + count(c);
	}
	count.setZero();
	Eigen::VectorXi remap(C.size());
	for (int i = 0; i < C.size(); ++i)
	{
		remap[i] = R(C(i)) + count[C(i)];
		++count[C(i)];
	}
	// Remap vertices
	Eigen::MatrixXd NV(V.rows(), V.cols());
	for (int v = 0; v < V.rows(); ++v)
	{
		NV.row(remap(v)) = V.row(v);
	}
	V = NV;
	// Remap face indices
	for (int f = 0; f < F.rows(); ++f)
	{
		for (int lv = 0; lv < F.cols(); ++lv)
		{
			F(f, lv) = remap(F(f, lv));
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

namespace
{

	// Plain 3D vector whose arithmetic mirrors geogram's vec3 operation by
	// operation (same expression shapes), so the helpers below round exactly
	// like the geogram code they replace.
	struct Vec3
	{
		double x, y, z;
	};

	inline Vec3 vec3(const double *p) { return {p[0], p[1], p[2]}; }
	inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
	inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
	inline Vec3 operator*(double s, const Vec3 &v) { return {s * v.x, s * v.y, s * v.z}; }
	inline double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	inline double length2(const Vec3 &v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
	inline Vec3 cross(const Vec3 &a, const Vec3 &b)
	{
		return {
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x};
	}
	inline double distance2(const Vec3 &a, const Vec3 &b)
	{
		const double dx = a.x - b.x;
		const double dy = a.y - b.y;
		const double dz = a.z - b.z;
		return dx * dx + dy * dy + dz * dz;
	}

	inline double tetra_signed_volume(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &p4)
	{
		return dot(p2 - p1, cross(p3 - p1, p4 - p1)) / 6.0;
	}

	double point_segment_squared_distance(const Vec3 &point, const Vec3 &v0, const Vec3 &v1)
	{
		const double l2 = distance2(v0, v1);
		const double t = dot(point - v0, v1 - v0);
		if (t <= 0.0 || l2 == 0.0)
			return distance2(point, v0);
		else if (t > l2)
			return distance2(point, v1);
		const double lambda1 = t / l2;
		const double lambda0 = 1.0 - lambda1;
		const Vec3 closest_point = lambda0 * v0 + lambda1 * v1;
		return distance2(point, closest_point);
	}

	double cos_angle(const Vec3 &a, const Vec3 &b)
	{
		const double lab = std::sqrt(length2(a) * length2(b));
		double result = (lab > 1e-50) ? (dot(a, b) / lab) : 1.0;
		// Numerical precision problems may produce normalized dot products
		// slightly outside [-1, 1].
		if (result < -1.0)
			result = -1.0;
		else if (result > 1.0)
			result = 1.0;
		return result;
	}

	// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
	// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
	int orientation(
		double x1, double y1, double x2, double y2, double &twice_signed_area)
	{
		twice_signed_area = y1 * x2 - x1 * y2;
		if (twice_signed_area > 0)
			return 1;
		else if (twice_signed_area < 0)
			return -1;
		else if (y2 > y1)
			return 1;
		else if (y2 < y1)
			return -1;
		else if (x1 > x2)
			return 1;
		else if (x1 < x2)
			return -1;
		else
			return 0; // only true when x1==x2 and y1==y2
	}

	// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
	// if true is returned, the barycentric coordinates are set in a,b,c.
	//
	// Note: This function comes from SDFGen by Christopher Batty.
	// https://github.com/christopherbatty/SDFGen/blob/master/makelevelset3.cpp
	bool point_in_triangle_2d(
		double x0, double y0, double x1, double y1,
		double x2, double y2, double x3, double y3,
		double &a, double &b, double &c)
	{
		x1 -= x0;
		x2 -= x0;
		x3 -= x0;
		y1 -= y0;
		y2 -= y0;
		y3 -= y0;
		int signa = orientation(x2, y2, x3, y3, a);
		if (signa == 0)
			return false;
		int signb = orientation(x3, y3, x1, y1, b);
		if (signb != signa)
			return false;
		int signc = orientation(x1, y1, x2, y2, c);
		if (signc != signa)
			return false;
		double sum = a + b + c;
		assert(sum != 0); // if the SOS signs match and are nonzero, there's no way all of a, b, and c are zero.
		a /= sum;
		b /= sum;
		c /= sum;
		return true;
	}

	// -----------------------------------------------------------------------------

	// Sign of the (approximate) signed area of the triangle p0, p1, p2.
	// Originally taken from geogram/src/lib/geogram/delaunay/delaunay_2d.cpp
	inline int orient_2d_inexact(
		const double p0x, const double p0y,
		const double p1x, const double p1y,
		const double p2x, const double p2y)
	{
		double a11 = p1x - p0x;
		double a12 = p1y - p0y;

		double a21 = p2x - p0x;
		double a22 = p2y - p0y;

		double Delta = a11 * a22 - a12 * a21;

		return (Delta > 0) ? 1 : ((Delta < 0) ? -1 : 0);
	}

	// -----------------------------------------------------------------------------

	/**
	 * @brief      { Intersect a vertical ray with a triangle }
	 *
	 * @param[in]  V     { Mesh vertices }
	 * @param[in]  F     { Mesh triangles }
	 * @param[in]  f     { Index of the triangle to intersect }
	 * @param[in]  q     { Query point (only XY coordinates are used) }
	 * @param[out] z     { Intersection }
	 *
	 * @return     { {-1,0,1} depending on the sign of the intersection. }
	 */
	template <int X = 0, int Y = 1, int Z = 2>
	int intersect_ray_z(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, int f, const Eigen::RowVector3d &q, double &z)
	{
		const auto p1 = V.row(F(f, 0));
		const auto p2 = V.row(F(f, 1));
		const auto p3 = V.row(F(f, 2));

		double u, v, w;
		if (point_in_triangle_2d(
				q[X], q[Y], p1[X], p1[Y], p2[X], p2[Y], p3[X], p3[Y], u, v, w))
		{
			z = u * p1[Z] + v * p2[Z] + w * p3[Z];
			return orient_2d_inexact(p1[X], p1[Y], p2[X], p2[Y], p3[X], p3[Y]);
		}

		return 0;
	}

	// -----------------------------------------------------------------------------

	// Bounding volume hierarchy over the triangles of (V, F), used to cull
	// triangles in signed_squared_distances. Culling never changes a result:
	// distances are the exact minimum of the same per-triangle values (a box is
	// skipped only if it is farther than the current minimum by more than any
	// rounding error), and the ray test visits exactly the triangles whose
	// closed xy bounding box contains the query point, which are the triangles
	// geogram's AABB box query used to report.
	class TriangleBVH
	{
	public:
		TriangleBVH(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
			: V_(V), F_(F), boxes_(F.rows()), order_(F.rows())
		{
			std::vector<std::array<double, 3>> centroids(F.rows());
			for (int f = 0; f < F.rows(); ++f)
			{
				for (int d = 0; d < 3; ++d)
				{
					const double a = V(F(f, 0), d), b = V(F(f, 1), d), c = V(F(f, 2), d);
					boxes_[f].min[d] = std::min(a, std::min(b, c));
					boxes_[f].max[d] = std::max(a, std::max(b, c));
					centroids[f][d] = (a + b + c) / 3;
				}
			}
			std::iota(order_.begin(), order_.end(), 0);
			nodes_.reserve(2 * F.rows());
			if (F.rows() > 0)
				build(0, F.rows(), centroids);
		}

		/// Calls fn(f) for every triangle f whose closed xy bounding box contains (x, y).
		template <typename Fn>
		void for_each_in_xy_column(const double x, const double y, Fn &&fn) const
		{
			if (nodes_.empty())
				return;
			std::vector<int> stack = {0};
			while (!stack.empty())
			{
				const Node &node = nodes_[stack.back()];
				stack.pop_back();
				if (!contains_xy(node.box, x, y))
					continue;
				if (node.left < 0)
				{
					for (int i = node.begin; i < node.end; ++i)
						if (contains_xy(boxes_[order_[i]], x, y))
							fn(order_[i]);
				}
				else
				{
					stack.push_back(node.right);
					stack.push_back(node.left);
				}
			}
		}

		/// Minimum over all triangles of point_triangle_squared_distance(p, triangle).
		double min_squared_distance(const Eigen::RowVector3d &p, const double margin) const
		{
			double best = std::numeric_limits<double>::max();
			if (nodes_.empty())
				return best;
			std::vector<int> stack = {0};
			while (!stack.empty())
			{
				const Node &node = nodes_[stack.back()];
				stack.pop_back();
				if (box_squared_distance(node.box, p) > best + margin)
					continue;
				if (node.left < 0)
				{
					for (int i = node.begin; i < node.end; ++i)
					{
						const int f = order_[i];
						const Eigen::RowVector3d v0 = V_.row(F_(f, 0)), v1 = V_.row(F_(f, 1)), v2 = V_.row(F_(f, 2));
						const double d = polyfem::mesh::point_triangle_squared_distance(p.data(), v0.data(), v1.data(), v2.data());
						if (d < best)
							best = d;
					}
				}
				else
				{
					// Visit the closest child first.
					const double dl = box_squared_distance(nodes_[node.left].box, p);
					const double dr = box_squared_distance(nodes_[node.right].box, p);
					if (dl < dr)
					{
						stack.push_back(node.right);
						stack.push_back(node.left);
					}
					else
					{
						stack.push_back(node.left);
						stack.push_back(node.right);
					}
				}
			}
			return best;
		}

	private:
		struct Box
		{
			std::array<double, 3> min, max;
		};

		struct Node
		{
			Box box;
			int begin, end;
			int left = -1, right = -1;
		};

		static bool contains_xy(const Box &box, const double x, const double y)
		{
			return box.min[0] <= x && x <= box.max[0] && box.min[1] <= y && y <= box.max[1];
		}

		static double box_squared_distance(const Box &box, const Eigen::RowVector3d &p)
		{
			double result = 0;
			for (int d = 0; d < 3; ++d)
			{
				const double delta = std::max(0.0, std::max(box.min[d] - p[d], p[d] - box.max[d]));
				result += delta * delta;
			}
			return result;
		}

		int build(const int begin, const int end, const std::vector<std::array<double, 3>> &centroids)
		{
			const int id = nodes_.size();
			nodes_.emplace_back();
			Box box = boxes_[order_[begin]];
			for (int i = begin + 1; i < end; ++i)
			{
				for (int d = 0; d < 3; ++d)
				{
					box.min[d] = std::min(box.min[d], boxes_[order_[i]].min[d]);
					box.max[d] = std::max(box.max[d], boxes_[order_[i]].max[d]);
				}
			}
			nodes_[id].box = box;
			nodes_[id].begin = begin;
			nodes_[id].end = end;
			if (end - begin <= 4)
				return id;

			int axis = 0;
			for (int d = 1; d < 3; ++d)
				if (box.max[d] - box.min[d] > box.max[axis] - box.min[axis])
					axis = d;
			const int mid = begin + (end - begin) / 2;
			std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
							 [&](const int a, const int b) { return centroids[a][axis] < centroids[b][axis]; });
			const int left = build(begin, mid, centroids);
			const int right = build(mid, end, centroids);
			nodes_[id].left = left;
			nodes_[id].right = right;
			return id;
		}

		const Eigen::MatrixXd &V_;
		const Eigen::MatrixXi &F_;
		std::vector<Box> boxes_;
		std::vector<int> order_;
		std::vector<Node> nodes_;
	};

	// -----------------------------------------------------------------------------

	// Flips the sign of D for every point inside the surface (V, F). The inside
	// test casts a vertical ray from each point and counts the crossings below
	// it, testing the triangles whose closed xy bounding box contains the point.
	void compute_sign(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const TriangleBVH &bvh,
					  const Eigen::MatrixXd &P, Eigen::VectorXd &D)
	{
		assert(P.cols() == 3);
		assert(D.size() == P.rows());

		for (int k = 0; k < P.rows(); ++k)
		{
			const Eigen::RowVector3d center = P.row(k);

			std::vector<std::pair<double, int>> inter;
			bvh.for_each_in_xy_column(center[0], center[1], [&](const int f) {
				double z;
				if (int s = intersect_ray_z(V, F, f, center, z))
				{
					inter.emplace_back(z, s);
				}
			});
			std::sort(inter.begin(), inter.end());

			std::vector<double> reduced;
			for (int i = 0, s = 0; i < (int)inter.size(); ++i)
			{
				const int ds = inter[i].second;
				s += ds;
				if ((s == -1 && ds < 0) || (s == 0 && ds > 0))
				{
					reduced.push_back(inter[i].first);
				}
			}

			int num_before = 0;
			for (double z : reduced)
			{
				if (z < center[2])
				{
					++num_before;
				}
			}
			if (num_before % 2 == 1)
			{
				// Point is inside
				D(k) *= -1.0;
			}
		}
	}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::signed_squared_distances(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
											 const Eigen::MatrixXd &P, Eigen::VectorXd &D)
{
	assert(V.cols() == 3);
	assert(F.cols() == 3);
	assert(P.cols() == 3);
	const TriangleBVH bvh(V, F);

	// Pruning slack, far above the rounding error of the distances (which is
	// of the order of 1e-15 times the squared size of the scene).
	double scale2 = 0;
	if (V.rows() > 0 && P.rows() > 0)
	{
		const Eigen::RowVector3d lo = V.colwise().minCoeff().cwiseMin(P.colwise().minCoeff());
		const Eigen::RowVector3d hi = V.colwise().maxCoeff().cwiseMax(P.colwise().maxCoeff());
		scale2 = (hi - lo).squaredNorm();
	}
	const double margin = 1e-10 * scale2;

	D.resize(P.rows());
	for (int i = 0; i < P.rows(); ++i)
		D(i) = bvh.min_squared_distance(P.row(i), margin);
	compute_sign(V, F, bvh, P, D);
}

// -----------------------------------------------------------------------------

double polyfem::mesh::signed_volume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
	assert(F.cols() == 3);
	assert(V.cols() == 3);
	std::array<Eigen::RowVector3d, 4> t;
	t[3] = Eigen::RowVector3d::Zero(V.cols());
	double volume_total = 0;
	for (int f = 0; f < F.rows(); ++f)
	{
		for (int lv = 0; lv < F.cols(); ++lv)
		{
			t[lv] = V.row(F(f, lv));
		}
		double vol = tetra_signed_volume(vec3(t[0].data()), vec3(t[1].data()), vec3(t[2].data()), vec3(t[3].data()));
		volume_total += vol;
	}
	return -volume_total;
}

// -----------------------------------------------------------------------------

void polyfem::mesh::orient_closed_surface(const Eigen::MatrixXd &V, Eigen::MatrixXi &F, bool positive)
{
	if ((positive ? 1 : -1) * signed_volume(V, F) < 0)
	{
		for (int f = 0; f < F.rows(); ++f)
		{
			F.row(f) = F.row(f).reverse().eval();
		}
	}
}

// -----------------------------------------------------------------------------

namespace
{
	struct EdgeInterfacePrimitive
	{
		polyfem::mesh::Navigation::Index index;
		Eigen::Vector2d from;
		Eigen::Vector2d to;
	};

	struct FaceInterfacePrimitive
	{
		polyfem::mesh::Navigation3D::Index index;
		std::vector<Eigen::Vector3d> vertices;
	};

	bool same_point(const polyfem::RowVectorNd &a, const polyfem::RowVectorNd &b)
	{
		return a.size() == b.size() && (a - b).squaredNorm() <= 1e-24;
	}

	bool overlapping_segments(const EdgeInterfacePrimitive &first, const EdgeInterfacePrimitive &second)
	{
		const Eigen::Vector2d direction = first.to - first.from;
		const double length = direction.norm();
		const double second_length = (second.to - second.from).norm();
		const double scale = std::max({1.0, length, second_length});
		const double tolerance = 1e-12 * scale;
		if (length <= tolerance || second_length <= tolerance)
			return false;

		auto cross = [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
			return a.x() * b.y() - a.y() * b.x();
		};
		if (std::abs(cross(direction, second.from - first.from)) > tolerance * length
			|| std::abs(cross(direction, second.to - first.from)) > tolerance * length)
			return false;

		const Eigen::Vector2d tangent = direction / length;
		const double second_from = (second.from - first.from).dot(tangent);
		const double second_to = (second.to - first.from).dot(tangent);
		const double overlap_begin = std::max(0.0, std::min(second_from, second_to));
		const double overlap_end = std::min(length, std::max(second_from, second_to));
		return overlap_end - overlap_begin > tolerance;
	}

	Eigen::Vector3d face_normal(const FaceInterfacePrimitive &face)
	{
		Eigen::Vector3d normal = Eigen::Vector3d::Zero();
		for (int i = 0; i < face.vertices.size(); ++i)
			normal += face.vertices[i].cross(face.vertices[(i + 1) % face.vertices.size()]);
		return normal;
	}

	Eigen::Vector2d project_point(const Eigen::Vector3d &point, const int dropped_axis)
	{
		if (dropped_axis == 0)
			return {point.y(), point.z()};
		if (dropped_axis == 1)
			return {point.x(), point.z()};
		return {point.x(), point.y()};
	}

	bool point_in_convex_polygon(
		const Eigen::Vector2d &point,
		const std::vector<Eigen::Vector2d> &polygon,
		const double tolerance)
	{
		double sign = 0;
		for (int i = 0; i < polygon.size(); ++i)
		{
			const Eigen::Vector2d edge = polygon[(i + 1) % polygon.size()] - polygon[i];
			const Eigen::Vector2d offset = point - polygon[i];
			const double cross = edge.x() * offset.y() - edge.y() * offset.x();
			if (std::abs(cross) <= tolerance)
				continue;
			if (sign == 0)
				sign = cross;
			else if (sign * cross < 0)
				return false;
		}
		return true;
	}

	bool overlapping_faces(const FaceInterfacePrimitive &first, const FaceInterfacePrimitive &second)
	{
		const Eigen::Vector3d first_normal = face_normal(first);
		const Eigen::Vector3d second_normal = face_normal(second);
		const double first_area_scale = first_normal.norm();
		const double second_area_scale = second_normal.norm();
		const double coordinate_scale = std::max({1.0,
												  first.vertices.front().norm(),
												  second.vertices.front().norm()});
		const double tolerance = 1e-12 * coordinate_scale;
		if (first_area_scale <= tolerance * tolerance || second_area_scale <= tolerance * tolerance)
			return false;

		const Eigen::Vector3d unit_normal = first_normal / first_area_scale;
		if (unit_normal.cross(second_normal / second_area_scale).norm() > 1e-10)
			return false;
		for (const auto &point : second.vertices)
			if (std::abs(unit_normal.dot(point - first.vertices.front())) > tolerance)
				return false;

		Eigen::Index dropped_axis;
		unit_normal.cwiseAbs().maxCoeff(&dropped_axis);
		std::vector<Eigen::Vector2d> first_polygon, second_polygon;
		first_polygon.reserve(first.vertices.size());
		second_polygon.reserve(second.vertices.size());
		for (const auto &point : first.vertices)
			first_polygon.push_back(project_point(point, dropped_axis));
		for (const auto &point : second.vertices)
			second_polygon.push_back(project_point(point, dropped_axis));

		auto contains = [tolerance](const auto &container, const auto &contained) {
			return std::all_of(contained.begin(), contained.end(), [&](const Eigen::Vector2d &point) {
				return point_in_convex_polygon(point, container, tolerance);
			});
		};
		return contains(first_polygon, second_polygon) || contains(second_polygon, first_polygon);
	}
} // namespace

std::vector<std::pair<polyfem::mesh::Navigation::Index, polyfem::mesh::Navigation::Index>>
polyfem::mesh::compute_mesh_interface(const Mesh2D &first, const Mesh2D &second)
{
	using Index = Navigation::Index;
	std::vector<EdgeInterfacePrimitive> first_edges, second_edges;
	auto collect = [](const Mesh2D &mesh, auto &edges) {
		for (int f = 0; f < mesh.n_faces(); ++f)
		{
			auto index = mesh.get_index_from_face(f);
			for (int le = 0; le < mesh.n_face_vertices(f); ++le)
			{
				if (mesh.is_boundary_edge(index.edge))
				{
					const auto opposite = mesh.switch_vertex(index);
					edges.push_back({index,
									 mesh.point(index.vertex).head<2>(),
									 mesh.point(opposite.vertex).head<2>()});
				}
				index = mesh.next_around_face(index);
			}
		}
	};
	collect(first, first_edges);
	collect(second, second_edges);

	std::vector<std::pair<Index, Index>> result;
	std::set<std::tuple<int, int, int, int, int, int>> visited_pairs;
	for (const auto &lhs : first_edges)
	{
		for (const auto &rhs : second_edges)
		{
			if (!overlapping_segments(lhs, rhs))
				continue;
			Index second_index = rhs.index;
			if ((lhs.to - lhs.from).dot(rhs.to - rhs.from) > 0)
				second_index = second.switch_vertex(second_index);
			const auto key = std::make_tuple(
				lhs.index.face, lhs.index.edge, lhs.index.vertex,
				second_index.face, second_index.edge, second_index.vertex);
			if (visited_pairs.insert(key).second)
				result.emplace_back(lhs.index, second_index);
		}
	}
	return result;
}

std::vector<std::pair<polyfem::mesh::Navigation3D::Index, polyfem::mesh::Navigation3D::Index>>
polyfem::mesh::compute_mesh_interface(const Mesh3D &first, const Mesh3D &second)
{
	using Index = Navigation3D::Index;
	std::vector<FaceInterfacePrimitive> first_faces, second_faces;
	auto collect = [](const Mesh3D &mesh, auto &faces) {
		std::unordered_set<int> visited_faces;
		for (int c = 0; c < mesh.n_cells(); ++c)
		{
			for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
			{
				const int face = mesh.cell_face(c, lf);
				if (!mesh.is_boundary_face(face) || !visited_faces.insert(face).second)
					continue;
				auto index = mesh.get_index_from_element(c, lf, 0);
				std::vector<Eigen::Vector3d> vertices(mesh.n_face_vertices(face));
				for (int lv = 0; lv < vertices.size(); ++lv)
					vertices[lv] = mesh.point(mesh.face_vertex(face, lv)).head<3>();
				faces.push_back({index, std::move(vertices)});
			}
		}
	};
	collect(first, first_faces);
	collect(second, second_faces);

	std::vector<std::pair<Index, Index>> result;
	for (const auto &lhs : first_faces)
	{
		for (const auto &rhs : second_faces)
		{
			if (!overlapping_faces(lhs, rhs))
				continue;
			Index second_index = rhs.index;
			for (int lv = 0; lv < second.n_face_vertices(second_index.face); ++lv)
			{
				if (same_point(first.point(lhs.index.vertex), second.point(second_index.vertex)))
					break;
				second_index = second.next_around_face(second_index);
			}
			result.emplace_back(lhs.index, second_index);
		}
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////

namespace
{

	bool approx_aligned(const double *a_, const double *b_, const double *p_, const double *q_, double tol = 1e-6)
	{
		const Vec3 a = vec3(a_), b = vec3(b_), p = vec3(p_), q = vec3(q_);
		double da = std::sqrt(point_segment_squared_distance(a, p, q));
		double db = std::sqrt(point_segment_squared_distance(b, p, q));
		double cos_theta = cos_angle(b - a, p - q);
		return (da < tol && db < tol && std::abs(std::abs(cos_theta) - 1.0) < tol);
	}

} // anonymous namespace

// -----------------------------------------------------------------------------

void polyfem::mesh::extract_parent_edges(const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IE,
										 const Eigen::MatrixXd &BV, const Eigen::MatrixXi &BE, Eigen::MatrixXi &OE)
{
	assert(IV.cols() == 2 || IV.cols() == 3);
	assert(BV.cols() == 2 || BV.cols() == 3);
	typedef std::pair<int, int> Edge;
	std::vector<Edge> selected;
	for (int e1 = 0; e1 < IE.rows(); ++e1)
	{
		Eigen::RowVector3d a;
		a.setZero();
		a.head(IV.cols()) = IV.row(IE(e1, 0));
		Eigen::RowVector3d b;
		b.setZero();
		b.head(IV.cols()) = IV.row(IE(e1, 1));
		for (int e2 = 0; e2 < BE.rows(); ++e2)
		{
			Eigen::RowVector3d p;
			p.setZero();
			p.head(BV.cols()) = BV.row(BE(e2, 0));
			Eigen::RowVector3d q;
			q.setZero();
			q.head(BV.cols()) = BV.row(BE(e2, 1));
			if (approx_aligned(a.data(), b.data(), p.data(), q.data()))
			{
				selected.emplace_back(IE(e1, 0), IE(e1, 1));
				break;
			}
		}
	}

	OE.resize(selected.size(), 2);
	for (int e = 0; e < OE.rows(); ++e)
	{
		OE.row(e) << selected[e].first, selected[e].second;
	}
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::find_triangle_surface_from_tets(
	const Eigen::MatrixXi &tets,
	Eigen::MatrixXi &faces)
{
	std::unordered_set<Eigen::Vector3i, HashMatrix> tri_to_tet(4 * tets.rows());
	for (int i = 0; i < tets.rows(); i++)
	{
		tri_to_tet.emplace(tets(i, 0), tets(i, 2), tets(i, 1));
		tri_to_tet.emplace(tets(i, 0), tets(i, 3), tets(i, 2));
		tri_to_tet.emplace(tets(i, 0), tets(i, 1), tets(i, 3));
		tri_to_tet.emplace(tets(i, 1), tets(i, 2), tets(i, 3));
	}

	std::vector<Eigen::RowVector3i> faces_vector;
	for (const auto &tri : tri_to_tet)
	{
		// find dual triangle with reversed indices:
		bool is_surface_triangle =
			tri_to_tet.find(Eigen::Vector3i(tri[2], tri[1], tri[0])) == tri_to_tet.end()
			&& tri_to_tet.find(Eigen::Vector3i(tri[1], tri[0], tri[2])) == tri_to_tet.end()
			&& tri_to_tet.find(Eigen::Vector3i(tri[0], tri[2], tri[1])) == tri_to_tet.end();
		if (is_surface_triangle)
		{
			faces_vector.emplace_back(tri[0], tri[1], tri[2]);
		}
	}

	faces.resize(faces_vector.size(), 3);
	for (int i = 0; i < faces.rows(); i++)
	{
		faces.row(i) = faces_vector[i];
	}
}

void polyfem::mesh::extract_triangle_surface_from_tets(
	const Eigen::MatrixXd &vertices,
	const Eigen::MatrixXi &tets,
	Eigen::MatrixXd &surface_vertices,
	Eigen::MatrixXi &tris)
{
	Eigen::MatrixXi full_tris;
	find_triangle_surface_from_tets(tets, full_tris);

	std::unordered_map<int, int> full_to_surface;
	std::vector<size_t> surface_to_full;
	for (int i = 0; i < full_tris.rows(); i++)
	{
		for (int j = 0; j < full_tris.cols(); j++)
		{
			if (full_to_surface.find(full_tris(i, j)) == full_to_surface.end())
			{
				full_to_surface[full_tris(i, j)] = surface_to_full.size();
				surface_to_full.push_back(full_tris(i, j));
			}
		}
	}

	surface_vertices.resize(surface_to_full.size(), 3);
	for (int i = 0; i < surface_to_full.size(); i++)
	{
		surface_vertices.row(i) = vertices.row(surface_to_full[i]);
	}

	tris.resize(full_tris.rows(), full_tris.cols());
	for (int i = 0; i < tris.rows(); i++)
	{
		for (int j = 0; j < tris.cols(); j++)
		{
			tris(i, j) = full_to_surface[full_tris(i, j)];
		}
	}
}

int polyfem::mesh::count_faces(const int dim, const Eigen::MatrixXi &cells)
{
	std::unordered_set<std::vector<int>, HashVector> boundaries;

	auto insert = [&](std::vector<int> v) {
		std::sort(v.begin(), v.end());
		boundaries.insert(v);
	};

	for (int i = 0; i < cells.rows(); i++)
	{
		const auto &cell = cells.row(i);
		if (cells.cols() == 3) // triangle
		{
			insert({{cell(0), cell(1)}});
			insert({{cell(1), cell(2)}});
			insert({{cell(2), cell(0)}});
		}
		else if (cells.cols() == 4 && dim == 2) // quadralateral
		{
			insert({{cell(0), cell(1)}});
			insert({{cell(1), cell(2)}});
			insert({{cell(2), cell(3)}});
			insert({{cell(3), cell(0)}});
		}
		else if (cells.cols() == 4 && dim == 3) // tetrahedron
		{
			insert({{cell(0), cell(2), cell(1)}});
			insert({{cell(0), cell(3), cell(2)}});
			insert({{cell(0), cell(1), cell(3)}});
			insert({{cell(1), cell(2), cell(3)}});
		}
		else if (cells.cols() == 8) // hexahedron
		{
			insert({{cell(0), cell(1), cell(2), cell(3)}});
			insert({{cell(1), cell(5), cell(6), cell(2)}});
			insert({{cell(5), cell(4), cell(7), cell(6)}});
			insert({{cell(0), cell(4), cell(7), cell(3)}});
			insert({{cell(0), cell(4), cell(5), cell(1)}});
			insert({{cell(2), cell(6), cell(7), cell(3)}});
		}
		else
		{
			throw std::runtime_error("count_boundary_elements not implemented for polygons");
		}
	}

	return boundaries.size();
}

void polyfem::mesh::generate_edges(PolygonMesh &M)
{
	typedef std::pair<int, int> Edge;

	if (M.n_edges() > 0)
		return;

	M.connect_facets();

	// Compute a list of all the edges
	std::vector<std::pair<Edge, int>> e2c; // edge to corner id
	for (int f = 0; f < M.n_facets(); ++f)
	{
		for (int c = M.corners_begin(f); c < M.corners_end(f); ++c)
		{
			int v = M.corner_vertex(c);
			int c2 = M.next_corner_around_facet(f, c);
			int v2 = M.corner_vertex(c2);
			e2c.emplace_back(std::make_pair(std::min(v, v2), std::max(v, v2)), c);
		}
	}
	std::sort(e2c.begin(), e2c.end());

	// Assign unique id to edges
	M.clear_edges();
	Edge prev_e(-1, -1);
	for (const auto &kv : e2c)
	{
		Edge e = kv.first;
		if (e != prev_e)
		{
			M.create_edge(e.first, e.second);
			prev_e = e;
		}
	}
}
