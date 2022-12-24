#include "GeometryUtils.hpp"

#include <Eigen/Geometry>

#include <igl/barycentric_coordinates.h>

#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

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

	bool tiangle_intersects_disk(
		const Eigen::Vector2d &t0,
		const Eigen::Vector2d &t1,
		const Eigen::Vector2d &t2,
		const Eigen::Vector2d &center,
		const double radius)
	{
		Eigen::RowVector3d bc;
		igl::barycentric_coordinates(
			center.transpose(), t0.transpose(), t1.transpose(), t2.transpose(),
			bc);

		if (bc.minCoeff() >= 0)
		{
			assert(bc.maxCoeff() <= 1); // bc.sum() == 1
			return true;
		}

		const std::array<std::array<const Eigen::Vector2d *, 2>, 3> edges = {{
			{{&t0, &t1}},
			{{&t1, &t2}},
			{{&t2, &t0}},
		}};

		for (const auto &e : edges)
			if (ipc::point_edge_distance(center, *e[0], *e[1]) <= radius)
				return true;

		return false;
	}

	bool tetrahedron_intersects_ball(
		const Eigen::Vector3d &t0,
		const Eigen::Vector3d &t1,
		const Eigen::Vector3d &t2,
		const Eigen::Vector3d &t3,
		const Eigen::Vector3d &center,
		const double radius)
	{
		Eigen::RowVector4d bc;
		igl::barycentric_coordinates(
			center.transpose(), t0.transpose(), t1.transpose(),
			t2.transpose(), t3.transpose(), bc);

		if (bc.minCoeff() >= 0)
		{
			assert(bc.maxCoeff() <= 1); // bc.sum() == 1
			return true;
		}

		const std::array<std::array<const Eigen::Vector3d *, 3>, 4> faces = {{
			{{&t0, &t1, &t2}},
			{{&t0, &t1, &t3}},
			{{&t0, &t2, &t3}},
			{{&t1, &t2, &t3}},
		}};

		for (const auto &f : faces)
			if (ipc::point_triangle_distance(center, *f[0], *f[1], *f[2]) <= radius)
				return true;

		return false;
	}
} // namespace polyfem::utils