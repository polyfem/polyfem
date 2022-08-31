#include "GeometryUtils.hpp"

#include <Eigen/Geometry>

namespace polyfem::utils
{
	double triangle_area_2D(
		const Eigen::Vector2d &a,
		const Eigen::Vector2d &b,
		const Eigen::Vector2d &c)
	{
		return ((b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y())) / 2.0;
	}

	double triangle_area_3D(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c)
	{
		return (b - a).cross(c - a).norm() / 2.0;
	}

	double triangle_area(const Eigen::MatrixXd V)
	{
		assert(V.rows() == 3);
		if (V.cols() == 2)
		{
			return triangle_area_2D(V.row(0), V.row(1), V.row(2));
		}
		else
		{
			return triangle_area_3D(V.row(0), V.row(1), V.row(2));
		}
	}

	double tetrahedron_volume(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c,
		const Eigen::Vector3d &d)
	{
		return (b - a).cross(c - a).dot(d - a) / 6.0;
	}

	double tetrahedron_volume(const Eigen::MatrixXd V)
	{
		assert(V.rows() == 4 && V.cols() == 3);
		return tetrahedron_volume(V.row(0), V.row(1), V.row(2), V.row(3));
	}

	Eigen::MatrixXd triangle_to_clockwise_order(const Eigen::MatrixXd &triangle)
	{
		// Only works for 2D triangles.
		assert(triangle.rows() == 3 && triangle.cols() == 2);

		if (triangle_area(triangle) <= 0)
			return triangle; // triangle aleady in clockwise order.

		Eigen::Matrix<double, 3, 2> triangle_clockwise;
		triangle_clockwise.row(0) = triangle.row(2);
		triangle_clockwise.row(1) = triangle.row(1);
		triangle_clockwise.row(2) = triangle.row(0);

		return triangle_clockwise;
	}

	std::vector<Eigen::MatrixXd> triangle_fan(const Eigen::MatrixXd &convex_polygon)
	{
		assert(convex_polygon.rows() >= 3);
		std::vector<Eigen::MatrixXd> triangles;
		for (int i = 1; i < convex_polygon.rows() - 1; ++i)
		{
			triangles.emplace_back(3, convex_polygon.cols());
			triangles.back().row(0) = convex_polygon.row(0);
			triangles.back().row(1) = convex_polygon.row(i);
			triangles.back().row(2) = convex_polygon.row(i + 1);
		}
		return triangles;
	}

	Eigen::VectorXd barycentric_coordinates(
		const Eigen::VectorXd &p,
		const Eigen::MatrixXd &V)
	{
		Eigen::MatrixXd A(p.size() + 1, p.size() + 1);
		A.topRows(p.size()) = V.transpose();
		A.bottomRows(1).setOnes();

		Eigen::VectorXd rhs(p.size() + 1);
		rhs.topRows(p.size()) = p;
		rhs.bottomRows(1).setOnes();

		// TODO: Can we use better than LU?
		const Eigen::VectorXd bc = A.partialPivLu().solve(rhs);
		assert((A * bc - rhs).norm() / rhs.norm() < 1e-12);

		return bc;
	}
} // namespace polyfem::utils