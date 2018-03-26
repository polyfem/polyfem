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
    REQUIRE(fabs(-1.534238651350367 - bessy0(0.1))  < 1e-8);
    REQUIRE(fabs(0.088256964215677 - bessy0(1.))    < 1e-8);
    REQUIRE(fabs(0.055671167283599 - bessy0(10.))   < 1e-8);
    REQUIRE(fabs(-0.077244313365083 - bessy0(100.)) < 1e-8);

    REQUIRE(fabs(-6.458951094702027 - bessy1(0.1))  < 1e-8);
    REQUIRE(fabs(-0.781212821300289 - bessy1(1.))    < 1e-8);
    REQUIRE(fabs(0.249015424206954 - bessy1(10.))   < 1e-8);
    REQUIRE(fabs(-0.020372312002760 - bessy1(100.)) < 1e-8);
}
