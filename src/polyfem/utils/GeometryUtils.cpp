#include "GeometryUtils.hpp"

#include <polyfem/utils/Logger.hpp>

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

	bool triangle_intersects_disk(
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

		if (bc.minCoeff() >= 0) // point is inside triangle
		{
			assert(bc.maxCoeff() <= 1 + 1e-12); // bc.sum() == 1
			return true;
		}

		const std::array<std::array<const Eigen::Vector2d *, 2>, 3> edges = {{
			{{&t0, &t1}},
			{{&t1, &t2}},
			{{&t2, &t0}},
		}};

		const double radius_sqr = radius * radius;

		for (const auto &e : edges)
			if (ipc::point_edge_distance(center, *e[0], *e[1]) <= radius_sqr)
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
			assert(bc.maxCoeff() <= 1 + 1e-12); // bc.sum() == 1
			return true;
		}

		const std::array<std::array<const Eigen::Vector3d *, 3>, 4> faces = {{
			{{&t0, &t1, &t2}},
			{{&t0, &t1, &t3}},
			{{&t0, &t2, &t3}},
			{{&t1, &t2, &t3}},
		}};

		const double radius_sqr = radius * radius;

		for (const auto &f : faces)
			if (ipc::point_triangle_distance(center, *f[0], *f[1], *f[2]) <= radius_sqr)
				return true;

		return false;
	}

	bool are_edges_collinear(
		const Eigen::VectorXd &ea0,
		const Eigen::VectorXd &ea1,
		const Eigen::VectorXd &eb0,
		const Eigen::VectorXd &eb1,
		const double tol)
	{
		assert((ea0 - ea1).norm() != 0 && (eb1 - eb0).norm() != 0);
		const Eigen::VectorXd ea = (ea1 - ea0).normalized();
		const Eigen::VectorXd eb = (eb1 - eb0).normalized();
		return abs(ea.dot(eb)) > 1 - tol;
	}

	bool are_triangles_coplanar(
		const Eigen::Vector3d &t00,
		const Eigen::Vector3d &t01,
		const Eigen::Vector3d &t02,
		const Eigen::Vector3d &t10,
		const Eigen::Vector3d &t11,
		const Eigen::Vector3d &t12,
		const double tol)
	{
		const Eigen::Vector3d n0 = (t01 - t00).cross(t02 - t00).normalized();
		const Eigen::Vector3d n1 = (t11 - t10).cross(t12 - t10).normalized();
		return abs(n0.dot(n1)) > 1 - tol;
	}

	// =========================================================================
	// The following automatically generated using SymPy
	// =========================================================================

	void triangle_area_2D_gradient(double ax, double ay, double bx, double by, double cx, double cy, double g[6])
	{
		const auto t0 = -cy;
		const auto t1 = -cx;
		g[0] = (1.0 / 2.0) * by + (1.0 / 2.0) * t0;
		g[1] = -1.0 / 2.0 * bx - 1.0 / 2.0 * t1;
		g[2] = -1.0 / 2.0 * ay - 1.0 / 2.0 * t0;
		g[3] = (1.0 / 2.0) * ax + (1.0 / 2.0) * t1;
		g[4] = (1.0 / 2.0) * (ay - by);
		g[5] = (1.0 / 2.0) * (-ax + bx);
	}

	void triangle_area_2D_hessian(double ax, double ay, double bx, double by, double cx, double cy, double H[36])
	{
		H[0] = 0;
		H[1] = 0;
		H[2] = 0;
		H[3] = 1.0 / 2.0;
		H[4] = 0;
		H[5] = -1.0 / 2.0;
		H[6] = 0;
		H[7] = 0;
		H[8] = -1.0 / 2.0;
		H[9] = 0;
		H[10] = 1.0 / 2.0;
		H[11] = 0;
		H[12] = 0;
		H[13] = -1.0 / 2.0;
		H[14] = 0;
		H[15] = 0;
		H[16] = 0;
		H[17] = 1.0 / 2.0;
		H[18] = 1.0 / 2.0;
		H[19] = 0;
		H[20] = 0;
		H[21] = 0;
		H[22] = -1.0 / 2.0;
		H[23] = 0;
		H[24] = 0;
		H[25] = 1.0 / 2.0;
		H[26] = 0;
		H[27] = -1.0 / 2.0;
		H[28] = 0;
		H[29] = 0;
		H[30] = -1.0 / 2.0;
		H[31] = 0;
		H[32] = 1.0 / 2.0;
		H[33] = 0;
		H[34] = 0;
		H[35] = 0;
	}

	void tetrahedron_volume_gradient(double ax, double ay, double az, double bx, double by, double bz, double cx, double cy, double cz, double dx, double dy, double dz, double g[12])
	{
		const auto t0 = az - dz;
		const auto t1 = -cy;
		const auto t2 = by + t1;
		const auto t3 = ay - dy;
		const auto t4 = -cz;
		const auto t5 = bz + t4;
		const auto t6 = ay - by;
		const auto t7 = az + t4;
		const auto t8 = ay + t1;
		const auto t9 = az - bz;
		const auto t10 = t6 * t7 - t8 * t9;
		const auto t11 = -cx;
		const auto t12 = bx + t11;
		const auto t13 = ax - dx;
		const auto t14 = ax - bx;
		const auto t15 = ax + t11;
		const auto t16 = t14 * t7 - t15 * t9;
		const auto t17 = t14 * t8 - t15 * t6;
		g[0] = -1.0 / 6.0 * t0 * t2 - 1.0 / 6.0 * t10 + (1.0 / 6.0) * t3 * t5;
		g[1] = (1.0 / 6.0) * t0 * t12 - 1.0 / 6.0 * t13 * t5 + (1.0 / 6.0) * t16;
		g[2] = -1.0 / 6.0 * t12 * t3 + (1.0 / 6.0) * t13 * t2 - 1.0 / 6.0 * t17;
		g[3] = (1.0 / 6.0) * t0 * t8 - 1.0 / 6.0 * t3 * t7;
		g[4] = -1.0 / 6.0 * t0 * t15 + (1.0 / 6.0) * t13 * t7;
		g[5] = -1.0 / 6.0 * t13 * t8 + (1.0 / 6.0) * t15 * t3;
		g[6] = -1.0 / 6.0 * t0 * t6 + (1.0 / 6.0) * t3 * t9;
		g[7] = (1.0 / 6.0) * t0 * t14 - 1.0 / 6.0 * t13 * t9;
		g[8] = (1.0 / 6.0) * t13 * t6 - 1.0 / 6.0 * t14 * t3;
		g[9] = (1.0 / 6.0) * t10;
		g[10] = -1.0 / 6.0 * t16;
		g[11] = (1.0 / 6.0) * t17;
	}

	void tetrahedron_volume_hessian(double ax, double ay, double az, double bx, double by, double bz, double cx, double cy, double cz, double dx, double dy, double dz, double H[144])
	{
		const auto t0 = -dz;
		const auto t1 = cz + t0;
		const auto t2 = -1.0 / 6.0 * t1;
		const auto t3 = -dy;
		const auto t4 = cy + t3;
		const auto t5 = (1.0 / 6.0) * t4;
		const auto t6 = bz + t0;
		const auto t7 = (1.0 / 6.0) * t6;
		const auto t8 = by + t3;
		const auto t9 = -1.0 / 6.0 * t8;
		const auto t10 = -cz;
		const auto t11 = bz + t10;
		const auto t12 = -1.0 / 6.0 * t11;
		const auto t13 = -cy;
		const auto t14 = by + t13;
		const auto t15 = (1.0 / 6.0) * t14;
		const auto t16 = (1.0 / 6.0) * t1;
		const auto t17 = -dx;
		const auto t18 = cx + t17;
		const auto t19 = -1.0 / 6.0 * t18;
		const auto t20 = -1.0 / 6.0 * t6;
		const auto t21 = bx + t17;
		const auto t22 = (1.0 / 6.0) * t21;
		const auto t23 = (1.0 / 6.0) * t11;
		const auto t24 = -cx;
		const auto t25 = bx + t24;
		const auto t26 = -1.0 / 6.0 * t25;
		const auto t27 = -1.0 / 6.0 * t4;
		const auto t28 = (1.0 / 6.0) * t18;
		const auto t29 = (1.0 / 6.0) * t8;
		const auto t30 = -1.0 / 6.0 * t21;
		const auto t31 = -1.0 / 6.0 * t14;
		const auto t32 = (1.0 / 6.0) * t25;
		const auto t33 = az + t0;
		const auto t34 = -1.0 / 6.0 * t33;
		const auto t35 = ay + t3;
		const auto t36 = (1.0 / 6.0) * t35;
		const auto t37 = az + t10;
		const auto t38 = (1.0 / 6.0) * t37;
		const auto t39 = ay + t13;
		const auto t40 = -1.0 / 6.0 * t39;
		const auto t41 = (1.0 / 6.0) * t33;
		const auto t42 = ax + t17;
		const auto t43 = -1.0 / 6.0 * t42;
		const auto t44 = -1.0 / 6.0 * t37;
		const auto t45 = ax + t24;
		const auto t46 = (1.0 / 6.0) * t45;
		const auto t47 = -1.0 / 6.0 * t35;
		const auto t48 = (1.0 / 6.0) * t42;
		const auto t49 = (1.0 / 6.0) * t39;
		const auto t50 = -1.0 / 6.0 * t45;
		const auto t51 = az - bz;
		const auto t52 = -1.0 / 6.0 * t51;
		const auto t53 = ay - by;
		const auto t54 = (1.0 / 6.0) * t53;
		const auto t55 = (1.0 / 6.0) * t51;
		const auto t56 = ax - bx;
		const auto t57 = -1.0 / 6.0 * t56;
		const auto t58 = -1.0 / 6.0 * t53;
		const auto t59 = (1.0 / 6.0) * t56;
		H[0] = 0;
		H[1] = 0;
		H[2] = 0;
		H[3] = 0;
		H[4] = t2;
		H[5] = t5;
		H[6] = 0;
		H[7] = t7;
		H[8] = t9;
		H[9] = 0;
		H[10] = t12;
		H[11] = t15;
		H[12] = 0;
		H[13] = 0;
		H[14] = 0;
		H[15] = t16;
		H[16] = 0;
		H[17] = t19;
		H[18] = t20;
		H[19] = 0;
		H[20] = t22;
		H[21] = t23;
		H[22] = 0;
		H[23] = t26;
		H[24] = 0;
		H[25] = 0;
		H[26] = 0;
		H[27] = t27;
		H[28] = t28;
		H[29] = 0;
		H[30] = t29;
		H[31] = t30;
		H[32] = 0;
		H[33] = t31;
		H[34] = t32;
		H[35] = 0;
		H[36] = 0;
		H[37] = t16;
		H[38] = t27;
		H[39] = 0;
		H[40] = 0;
		H[41] = 0;
		H[42] = 0;
		H[43] = t34;
		H[44] = t36;
		H[45] = 0;
		H[46] = t38;
		H[47] = t40;
		H[48] = t2;
		H[49] = 0;
		H[50] = t28;
		H[51] = 0;
		H[52] = 0;
		H[53] = 0;
		H[54] = t41;
		H[55] = 0;
		H[56] = t43;
		H[57] = t44;
		H[58] = 0;
		H[59] = t46;
		H[60] = t5;
		H[61] = t19;
		H[62] = 0;
		H[63] = 0;
		H[64] = 0;
		H[65] = 0;
		H[66] = t47;
		H[67] = t48;
		H[68] = 0;
		H[69] = t49;
		H[70] = t50;
		H[71] = 0;
		H[72] = 0;
		H[73] = t20;
		H[74] = t29;
		H[75] = 0;
		H[76] = t41;
		H[77] = t47;
		H[78] = 0;
		H[79] = 0;
		H[80] = 0;
		H[81] = 0;
		H[82] = t52;
		H[83] = t54;
		H[84] = t7;
		H[85] = 0;
		H[86] = t30;
		H[87] = t34;
		H[88] = 0;
		H[89] = t48;
		H[90] = 0;
		H[91] = 0;
		H[92] = 0;
		H[93] = t55;
		H[94] = 0;
		H[95] = t57;
		H[96] = t9;
		H[97] = t22;
		H[98] = 0;
		H[99] = t36;
		H[100] = t43;
		H[101] = 0;
		H[102] = 0;
		H[103] = 0;
		H[104] = 0;
		H[105] = t58;
		H[106] = t59;
		H[107] = 0;
		H[108] = 0;
		H[109] = t23;
		H[110] = t31;
		H[111] = 0;
		H[112] = t44;
		H[113] = t49;
		H[114] = 0;
		H[115] = t55;
		H[116] = t58;
		H[117] = 0;
		H[118] = 0;
		H[119] = 0;
		H[120] = t12;
		H[121] = 0;
		H[122] = t32;
		H[123] = t38;
		H[124] = 0;
		H[125] = t50;
		H[126] = t52;
		H[127] = 0;
		H[128] = t59;
		H[129] = 0;
		H[130] = 0;
		H[131] = 0;
		H[132] = t15;
		H[133] = t26;
		H[134] = 0;
		H[135] = t40;
		H[136] = t46;
		H[137] = 0;
		H[138] = t54;
		H[139] = t57;
		H[140] = 0;
		H[141] = 0;
		H[142] = 0;
		H[143] = 0;
	}
} // namespace polyfem::utils