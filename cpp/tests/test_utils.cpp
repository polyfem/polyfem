////////////////////////////////////////////////////////////////////////////////
#include "InterpolatedFunction.hpp"

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
