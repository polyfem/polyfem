#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <polyfem/utils/GeometryUtils.hpp>

TEST_CASE("Triangle area", "[geometry]")
{
	using namespace polyfem::utils;

	Eigen::Matrix3d V;
	V << 0, 0, 0,
		1, 0, 0,
		0, 1, 0;

	CHECK(triangle_area(V.leftCols<2>()) == Catch::Approx(0.5));
	CHECK(triangle_area(V) == Catch::Approx(0.5));

	Eigen::Matrix3d V_flipped = V;
	V_flipped.row(1) = V.row(2);
	V_flipped.row(2) = V.row(1);

	CHECK(triangle_area(V_flipped.leftCols<2>()) == Catch::Approx(-0.5));
	CHECK(triangle_area(V_flipped) == Catch::Approx(0.5));
}

TEST_CASE("Tetrahedron volume", "[geometry]")
{
	using namespace polyfem::utils;

	Eigen::Matrix<double, 4, 3> V;
	V << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;

	CHECK(tetrahedron_volume(V) == Catch::Approx(1 / 6.));

	Eigen::Matrix<double, 4, 3> V_flipped = V;
	V_flipped.row(2) = V.row(3);
	V_flipped.row(3) = V.row(2);

	CHECK(tetrahedron_volume(V_flipped) == Catch::Approx(-1 / 6.));
}
