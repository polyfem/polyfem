#pragma once

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
} // namespace polyfem::utils