#pragma once

#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

namespace polyfem::utils
{
	/// @brief Compute the signed area of a 2D triangle defined by three points.
	/// @param a First point of the triangle.
	/// @param b Second point of the triangle.
	/// @param c Third point of the triangle.
	/// @return The signed area of the triangle.
	double triangle_area_2D(
		const Eigen::Vector2d &a,
		const Eigen::Vector2d &b,
		const Eigen::Vector2d &c);

	/// @brief Compute the area of a 3D triangle defined by three points.
	/// @param a First point of the triangle.
	/// @param b Second point of the triangle.
	/// @param c Third point of the triangle.
	/// @return The signed area of the triangle.
	double triangle_area_3D(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c);

	/// @brief Compute the signed area of a triangle defined by three points.
	/// @param V The vertices of the triangle as rows of a matrix.
	/// @return The signed area of the triangle.
	double triangle_area(const Eigen::MatrixXd V);

	/// @brief Compute the signed volume of a tetrahedron defined by four points.
	/// @param a First point of the tetrahedron.
	/// @param b Second point of the tetrahedron.
	/// @param c Third point of the tetrahedron.
	/// @param d Fourth point of the tetrahedron.
	/// @return The signed volume of the tetrahedron.
	double tetrahedron_volume(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c,
		const Eigen::Vector3d &d);

	/// @brief Compute the signed volume of a tetrahedron defined by four points.
	/// @param V The vertices of the terahedron as rows of a matrix.
	/// @return The signed volume of the tetrahedron.
	double tetrahedron_volume(const Eigen::MatrixXd V);

	/// @brief Compute the gradient of the signed area of a 2D triangle defined by three points.
	/// @param ax First point's x coordinate.
	/// @param ay First point's y coordinate.
	/// @param bx Second point's x coordinate.
	/// @param by Second point's y coordinate.
	/// @param cx Third point's x coordinate.
	/// @param cy Third point's y coordinate.
	/// @param g Output gradient with respect to the three points.
	void triangle_area_2D_gradient(double ax, double ay, double bx, double by, double cx, double cy, double g[6]);

	/// @brief Compute the Hessian of the signed area of a 2D triangle defined by three points.
	/// @param ax First point's x coordinate.
	/// @param ay First point's y coordinate.
	/// @param bx Second point's x coordinate.
	/// @param by Second point's y coordinate.
	/// @param cx Third point's x coordinate.
	/// @param cy Third point's y coordinate.
	/// @param g Output flattened Hessian with respect to the three points.
	void triangle_area_2D_hessian(double ax, double ay, double bx, double by, double cx, double cy, double H[36]);

	/// @brief Compute the gradient of the signed volume of a tetrahedron defined by four points.
	/// @param ax First point's x coordinate.
	/// @param ay First point's y coordinate.
	/// @param ay First point's z coordinate.
	/// @param bx Second point's x coordinate.
	/// @param by Second point's y coordinate.
	/// @param by Second point's z coordinate.
	/// @param cx Third point's x coordinate.
	/// @param cy Third point's y coordinate.
	/// @param cy Third point's z coordinate.
	/// @param dx Fourth point's x coordinate.
	/// @param dy Fourth point's y coordinate.
	/// @param dy Fourth point's z coordinate.
	/// @param g Output gradient with respect to the four points.
	void tetrahedron_volume_gradient(double ax, double ay, double az, double bx, double by, double bz, double cx, double cy, double cz, double dx, double dy, double dz, double g[12]);

	/// @brief Compute the gradient of the signed area of a 2D triangle defined by three points.
	/// @param ax First point's x coordinate.
	/// @param ay First point's y coordinate.
	/// @param ay First point's z coordinate.
	/// @param bx Second point's x coordinate.
	/// @param by Second point's y coordinate.
	/// @param by Second point's z coordinate.
	/// @param cx Third point's x coordinate.
	/// @param cy Third point's y coordinate.
	/// @param cy Third point's z coordinate.
	/// @param dx Fourth point's x coordinate.
	/// @param dy Fourth point's y coordinate.
	/// @param dy Fourth point's z coordinate.
	/// @param g Output flattened Hessian with respect to the four points.
	void tetrahedron_volume_hessian(double ax, double ay, double az, double bx, double by, double bz, double cx, double cy, double cz, double dx, double dy, double dz, double H[144]);

	/// @brief Reorder the vertices of a triangle so they are in clockwise order.
	/// @param triangle The vertices of the triangle as rows of a matrix.
	/// @return The vertices of the triangle in clockwise order as rows of a matrix.
	Eigen::MatrixXd triangle_to_clockwise_order(const Eigen::MatrixXd &triangle);

	/// @brief Convert a convex polygon to a list of triangles fanned around the first vertex.
	/// @param convex_polygon The vertices of the convex polygon as rows of a matrix.
	/// @return The list of triangles as matrices of rows.
	std::vector<Eigen::MatrixXd> triangle_fan(const Eigen::MatrixXd &convex_polygon);

	/// @brief Compute barycentric coordinates for point p with respect to a simplex.
	/// @param p Query point.
	/// @param V Verties of the simplex as rows of a matrix.
	/// @return The barycentric coordinates for point p with respect to the simplex.
	Eigen::VectorXd barycentric_coordinates(
		const Eigen::VectorXd &p,
		const Eigen::MatrixXd &V);

	/// @brief Determine if a 2D triangle intersects a 2D disk.
	/// @param t0 Triangle first vertex.
	/// @param t1 Triangle second vertex.
	/// @param t2 Triangle third vertex.
	/// @param center Center of the disk.
	/// @param radius Radius of the disk.
	/// @return True if the triangle intersects the disk, false otherwise.
	bool triangle_intersects_disk(
		const Eigen::Vector2d &t0,
		const Eigen::Vector2d &t1,
		const Eigen::Vector2d &t2,
		const Eigen::Vector2d &center,
		const double radius);

	/// @brief Determine if a 3D tetrahedron intersects a 3D ball.
	/// @param t0 Tetrahedron first vertex.
	/// @param t1 Tetrahedron second vertex.
	/// @param t2 Tetrahedron third vertex.
	/// @param t3 Tetrahedron fourth vertex.
	/// @param center Center of the ball.
	/// @param radius Radius of the ball.
	/// @return True if the tetrahedron intersects the ball, false otherwise.
	bool tetrahedron_intersects_ball(
		const Eigen::Vector3d &t0,
		const Eigen::Vector3d &t1,
		const Eigen::Vector3d &t2,
		const Eigen::Vector3d &t3,
		const Eigen::Vector3d &center,
		const double radius);

	/// @brief Determine if two edges are collinear.
	/// @param ea0 First vertex of the first edge.
	/// @param ea1 Second vertex of the first edge.
	/// @param eb0 First vertex of the second edge.
	/// @param eb1 Second vertex of the second edge.
	/// @param tol Tolerance for collinearity.
	/// @return True if the edges are collinear, false otherwise.
	bool are_edges_collinear(
		const VectorNd &ea0,
		const VectorNd &ea1,
		const VectorNd &eb0,
		const VectorNd &eb1,
		const double tol = 1e-10);

	/// @brief Determine if two triangles are coplanar.
	/// @param t00 First vertex of the first triangle.
	/// @param t01 Second vertex of the first triangle.
	/// @param t02 Third vertex of the first triangle.
	/// @param t10 First vertex of the second triangle.
	/// @param t11 Second vertex of the second triangle.
	/// @param t12 Third vertex of the second triangle.
	/// @param tol Tolerance for coplanarity.
	/// @return True if the triangles are coplanar, false otherwise.
	bool are_triangles_coplanar(
		const Eigen::Vector3d &t00,
		const Eigen::Vector3d &t01,
		const Eigen::Vector3d &t02,
		const Eigen::Vector3d &t10,
		const Eigen::Vector3d &t11,
		const Eigen::Vector3d &t12,
		const double tol = 1e-10);

	/// @brief Determine if two axis-aligned bounding boxes intersect.
	/// @param aabb0_min Minimum corner of the first AABB.
	/// @param aabb0_max Maximum corner of the first AABB.
	/// @param aabb1_min Minimum corner of the second AABB.
	/// @param aabb1_max Maximum corner of the second AABB.
	/// @return True if the AABBs intersect, false otherwise.
	bool are_aabbs_intersecting(
		const VectorNd &aabb0_min,
		const VectorNd &aabb0_max,
		const VectorNd &aabb1_min,
		const VectorNd &aabb1_max);
} // namespace polyfem::utils