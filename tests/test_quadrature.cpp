////////////////////////////////////////////////////////////////////////////////
#include <polyfem/quadrature/LineQuadrature.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::quadrature;

const double pi = 3.14159265358979323846264338327950288419717;

////////////////////////////////////////////////////////////////////////////////

namespace
{

	// double p01_exact() {
	// 	return pi * pi / 6.0;
	// }

	// Eigen::VectorXd p01_fun(const Eigen::MatrixXd &x) {
	// 	Eigen::VectorXd f = 1.0 / (1.0 - x.row(0).array() * x.row(1).array());
	// 	return f;
	// }

	// Eigen::AlignedBox2d p01_lim() {
	// 	Eigen::AlignedBox2d box;
	// 	box.min().setConstant(0.0);
	// 	box.max().setConstant(1.0);
	// 	return box;
	// }

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

TEST_CASE("weights", "[quadrature]")
{
	// Segment
	for (int order = 1; order <= 64; ++order)
	{
		LineQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Approx(1.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Triangle
	for (int order = 1; order < 16; ++order)
	{
		TriQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Approx(0.5).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Tetrahedron
	for (int order = 1; order < 16; ++order)
	{
		TetQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Approx(1.0 / 6.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}
}

//TEST_CASE("triangle", "[quadrature]") {
//	for (int order = 1; order < 10; ++order) {
//		Quadrature quadr;
//		TriQuadrature tri;
//		tri.get_quadrature(order, quadr);
//	}
//
//	// REQUIRE(poly_fem::determinant(mat) == Approx(mat.determinant()).margin(1e-12));
//}
