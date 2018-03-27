////////////////////////////////////////////////////////////////////////////////
#include "InterpolatedFunction.hpp"
#include "Bessel.hpp"

#include <Eigen/Dense>

#include <catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

TEST_CASE("interpolated_fun_2d", "[utils]") {
    Eigen::MatrixXd pts(3, 2); pts <<
    0, 0,
    3, 0,
    0, 3;

    Eigen::MatrixXi tri(1,3); tri << 0, 1, 2;
    Eigen::MatrixXd fun(3,4); fun.setRandom();

    Eigen::MatrixXd pt(1,2); pt << 1, 1;

    InterpolatedFunction2d i_fun(fun, pts, tri);
    const auto res = i_fun.interpolate(pt);

    REQUIRE((fun.colwise().mean() - res).norm() < 1e-10);
}


TEST_CASE("bessel", "[utils]") {
    REQUIRE(bessy0(0.1)    == Approx(-1.534238651350367).margin(1e-8));
    REQUIRE(bessy0(1.)     == Approx(0.088256964215677).margin(1e-8));
    REQUIRE(bessy0(10.)    == Approx(0.055671167283599).margin(1e-8));
    REQUIRE(bessy0(100.)   == Approx(-0.077244313365083).margin(1e-8));

    REQUIRE(bessy1(0.1)    == Approx(-6.458951094702027).margin(1e-8));
    REQUIRE(bessy1(1.)     == Approx(-0.781212821300289).margin(1e-8));
    REQUIRE(bessy1(10.)    == Approx(0.249015424206954).margin(1e-8));
    REQUIRE(bessy1(100.)   == Approx(-0.020372312002760).margin(1e-8));
}
