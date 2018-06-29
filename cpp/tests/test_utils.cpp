////////////////////////////////////////////////////////////////////////////////
#include <polyfem/InterpolatedFunction.hpp>
#include <polyfem/Bessel.hpp>
#include <polyfem/ExpressionValue.hpp>


#include <Eigen/Dense>

#include <catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

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


TEST_CASE("expression", "[utils]") {
    json jexpr = {{"value", "x^2+sqrt(x*y)+sin(z)*x"}};
    json jexpr2d = {{"value", "x^2+sqrt(x*y)"}};
    json jval  = {{"value", 1}};

    ExpressionValue expr;   expr.init(jexpr["value"]);
    ExpressionValue expr2d; expr2d.init(jexpr2d["value"]);
    ExpressionValue val;    val.init(jval["value"]);

    REQUIRE(expr(2, 3, 4)   == Approx(2.*2.+sqrt(2.*3.)+sin(4.)*2.).margin(1e-10));
    REQUIRE(expr2d(2, 3)    == Approx(2.*2.+sqrt(2.*3.)).margin(1e-10));
    REQUIRE(val(2, 3, 4)    == Approx(1).margin(1e-16));
}
