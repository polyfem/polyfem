#ifdef POLYFEM_WITH_MMG

#include <polyfem/mesh/MMGRemesh.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <vector>

using namespace polyfem::mesh;

namespace
{
	/// Total area of a 2D triangle mesh, requiring positive orientation.
	double checked_area(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
	{
		REQUIRE(V.cols() == 2);
		double area = 0;
		for (int f = 0; f < F.rows(); ++f)
		{
			const Eigen::Vector2d a = V.row(F(f, 1)) - V.row(F(f, 0));
			const Eigen::Vector2d b = V.row(F(f, 2)) - V.row(F(f, 0));
			const double signed_area = 0.5 * (a.x() * b.y() - a.y() * b.x());
			CHECK(signed_area > 0);
			area += signed_area;
		}
		return area;
	}

	/// Total volume of a tet mesh, requiring positive orientation.
	double checked_volume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T)
	{
		REQUIRE(V.cols() == 3);
		double volume = 0;
		for (int t = 0; t < T.rows(); ++t)
		{
			Eigen::Matrix3d J;
			for (int k = 0; k < 3; ++k)
				J.col(k) = (V.row(T(t, k + 1)) - V.row(T(t, 0))).transpose();
			const double signed_volume = J.determinant() / 6;
			CHECK(signed_volume > 0);
			volume += signed_volume;
		}
		return volume;
	}

	void unit_square(Eigen::MatrixXd &V, Eigen::MatrixXi &F)
	{
		V.resize(4, 2);
		V << 0, 0, 1, 0, 1, 1, 0, 1;
		F.resize(2, 3);
		F << 0, 1, 2, 0, 2, 3;
	}

	void unit_cube(Eigen::MatrixXd &V, Eigen::MatrixXi &T)
	{
		V.resize(8, 3);
		V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
			0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
		T.resize(6, 4);
		T << 0, 1, 2, 6,
			0, 2, 3, 6,
			0, 3, 7, 6,
			0, 7, 4, 6,
			0, 4, 5, 6,
			0, 5, 1, 6;
	}
} // namespace

TEST_CASE("MMG remeshes a 2D mesh", "[mmg]")
{
	Eigen::MatrixXd V, OV;
	Eigen::MatrixXi F, OF;
	unit_square(V, F);

	MmgOptions opt;
	opt.hmin = 0.05;
	opt.hmax = 0.2;
	SECTION("sizes from hmin/hmax")
	{
		REQUIRE(remesh_2d(V, F, OV, OF, opt));
	}
	SECTION("uniform size")
	{
		opt.hsiz = 0.2;
		REQUIRE(remesh_2d(V, F, OV, OF, opt));
	}
	SECTION("optimization with anisotropy")
	{
		// MMG does not allow hsiz with optim.
		opt.optim = true;
		opt.enable_anisotropy = true;
		REQUIRE(remesh_2d(V, F, OV, OF, opt));
	}
	SECTION("pinned vertices")
	{
		const std::vector<int> pinned = {0, 1, 2, 3};
		REQUIRE(remesh_2d(V, F, OV, OF, opt, &pinned));
		for (int v = 0; v < 4; ++v)
		{
			bool found = false;
			for (int w = 0; w < OV.rows(); ++w)
				found |= (OV.row(w) - V.row(v)).norm() < 1e-12;
			CHECK(found);
		}
	}
	CHECK(OF.rows() > F.rows());
	CHECK(checked_area(OV, OF) == Catch::Approx(1.0));
}

TEST_CASE("MMG remeshes a 2D mesh following a size field", "[mmg]")
{
	Eigen::MatrixXd V, OV;
	Eigen::MatrixXi F, OF;
	unit_square(V, F);
	// Smaller elements on the left.
	Eigen::VectorXd S(4);
	S << 0.05, 0.3, 0.3, 0.05;
	remesh_adaptive_2d(V, F, S, OV, OF);
	REQUIRE(OF.rows() > F.rows());
	CHECK(checked_area(OV, OF) == Catch::Approx(1.0));

	int left = 0, right = 0;
	for (int f = 0; f < OF.rows(); ++f)
		(OV(OF(f, 0), 0) + OV(OF(f, 1), 0) + OV(OF(f, 2), 0) < 1.5 ? left : right)++;
	CHECK(left > right);
}

TEST_CASE("MMG remeshes a tet mesh", "[mmg]")
{
	Eigen::MatrixXd V, OV;
	Eigen::MatrixXi T, OF, OT;
	unit_cube(V, T);

	MmgOptions opt;
	opt.hmin = 0.1;
	opt.hmax = 0.4;
	SECTION("sizes from hmin/hmax")
	{
		REQUIRE(remesh_3d(V, T, OV, OF, OT, opt));
	}
	SECTION("uniform size")
	{
		opt.hsiz = 0.4;
		REQUIRE(remesh_3d(V, T, OV, OF, OT, opt));
	}
	SECTION("optimization with anisotropy")
	{
		// MMG does not allow hsiz with optim.
		opt.optim = true;
		opt.enable_anisotropy = true;
		REQUIRE(remesh_3d(V, T, OV, OF, OT, opt));
	}
	SECTION("pinned vertices")
	{
		std::vector<int> pinned(8);
		for (int v = 0; v < 8; ++v)
			pinned[v] = v;
		REQUIRE(remesh_3d(V, T, OV, OF, OT, opt, &pinned));
	}
	CHECK(OT.rows() > T.rows());
	CHECK(OF.rows() > 0);
	CHECK(checked_volume(OV, OT) == Catch::Approx(1.0));
}

TEST_CASE("MMG remeshes a tet mesh following a size field", "[mmg]")
{
	Eigen::MatrixXd V, OV;
	Eigen::MatrixXi T, OF, OT;
	unit_cube(V, T);
	Eigen::VectorXd S = Eigen::VectorXd::Constant(8, 0.25);
	remesh_adaptive_3d(V, T, S, OV, OF, OT);
	REQUIRE(OT.rows() > T.rows());
	CHECK(checked_volume(OV, OT) == Catch::Approx(1.0));
}

#endif
