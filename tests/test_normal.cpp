////////////////////////////////////////////////////////////////////////////////

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::mesh;

TEST_CASE("triangle", "[normal]")
{
	Eigen::MatrixXd n;
	utils::BoundarySampler::normal_for_tri_edge(0, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(-1).margin(1e-13));

	static const double sqrt2 = 1. / sqrt(2.);
	utils::BoundarySampler::normal_for_tri_edge(1, n);
	REQUIRE(n(0) == Catch::Approx(sqrt2).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(sqrt2).margin(1e-13));

	utils::BoundarySampler::normal_for_tri_edge(2, n);
	REQUIRE(n(0) == Catch::Approx(-1).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
}

TEST_CASE("quad", "[normal]")
{
	Eigen::MatrixXd n;
	utils::BoundarySampler::normal_for_quad_edge(0, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(-1).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_edge(1, n);
	REQUIRE(n(0) == Catch::Approx(1).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_edge(2, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(1).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_edge(3, n);
	REQUIRE(n(0) == Catch::Approx(-1).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
}

TEST_CASE("tet", "[normal]")
{
	Eigen::MatrixXd n;
	utils::BoundarySampler::normal_for_tri_face(0, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(-1).margin(1e-13));

	utils::BoundarySampler::normal_for_tri_face(1, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(-1).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(0).margin(1e-13));

	static const double sqrt3 = 1. / sqrt(3.);
	utils::BoundarySampler::normal_for_tri_face(2, n);
	REQUIRE(n(0) == Catch::Approx(sqrt3).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(sqrt3).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(sqrt3).margin(1e-13));

	utils::BoundarySampler::normal_for_tri_face(3, n);
	REQUIRE(n(0) == Catch::Approx(-1).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(0).margin(1e-13));
}

TEST_CASE("hex", "[normal]")
{
	Eigen::MatrixXd n;
	utils::BoundarySampler::normal_for_quad_face(0, n);
	REQUIRE(n(0) == Catch::Approx(-1).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(0).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_face(1, n);
	REQUIRE(n(0) == Catch::Approx(1).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(0).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_face(2, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(-1).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(0).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_face(3, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(1).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(0).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_face(4, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(-1).margin(1e-13));

	utils::BoundarySampler::normal_for_quad_face(5, n);
	REQUIRE(n(0) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(1) == Catch::Approx(0).margin(1e-13));
	REQUIRE(n(2) == Catch::Approx(1).margin(1e-13));
}
