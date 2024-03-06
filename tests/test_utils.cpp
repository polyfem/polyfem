////////////////////////////////////////////////////////////////////////////////
#include <polyfem/utils/InterpolatedFunction.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>
#include <polyfem/utils/Bessel.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#ifdef POLYFEM_WITH_REMESHING
#include <wmtk/TriMesh.h>
#endif

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::mesh;
using namespace polyfem::io;
using namespace polyfem::utils;

TEST_CASE("interpolated_fun_2d", "[utils]")
{
	Eigen::MatrixXd pts(3, 2);
	pts << 0, 0,
		3, 0,
		0, 3;

	Eigen::MatrixXi tri(1, 3);
	tri << 0, 1, 2;
	Eigen::MatrixXd fun(3, 4);
	fun.setRandom();

	Eigen::MatrixXd pt(1, 2);
	pt << 1, 1;

	InterpolatedFunction2d i_fun(fun, pts, tri);
	const auto res = i_fun.interpolate(pt);

	REQUIRE((fun.colwise().mean() - res).norm() == Catch::Approx(0).margin(1e-10));
}

TEST_CASE("rbf_interpolate", "[utils]")
{
#ifndef POLYFEM_OPENCL
	Eigen::MatrixXd in_pts(10, 3);
	in_pts.col(0) << 0.73142708, 0.15639157, 0.06799852, 0.61980247, 0.70461343, 0.96155237, 0.18068249, 0.09782913, 0.36740639, 0.26763186;
	in_pts.col(1) << 0.94896831, 0.37164925, 0.86693048, 0.87339727, 0.18393119, 0.19822407, 0.54455402, 0.98657281, 0.541773, 0.19644425;
	in_pts.col(2) << 0.41745562, 0.98444505, 0.15567433, 0.09762302, 0.69628704, 0.05620348, 0.34966505, 0.60069814, 0.79617982, 0.4012071;

	Eigen::MatrixXd fun(10, 1);
	fun << 0.14689884, 0.83814805, 0.67897605, 0.61621774, 0.12150901, 0.20614193, 0.27911847, 0.62222035, 0.98755679, 0.40910887;

	const double eps = 0.4052917899199118;

	RBFInterpolation rbf_fun(fun, in_pts, "multiquadric", eps);

	Eigen::MatrixXd out_pts(20, 3);
	const Eigen::MatrixXd t = VectorNd::LinSpaced(20, 0, 1);
	out_pts.col(0) = t;
	out_pts.col(1) = t;
	out_pts.col(2) = t;

	const auto actual = rbf_fun.interpolate(out_pts);

	Eigen::MatrixXd expected(20, 1);
	expected << 0.516894154053909, 0.476058965730433, 0.435748570197800, 0.397220357669309, 0.362961970820606, 0.337549413480876, 0.327784652734946, 0.339982127556460, 0.374173324760153, 0.420108170328222, 0.460205192638323, 0.478147060651189, 0.467001601598498, 0.431400523024562, 0.383076112995277, 0.334107457785502, 0.292629952299549, 0.262190022964549, 0.243125356328612, 0.234186308102585;

	for (int i = 0; i < 20; ++i)
		REQUIRE(actual(i) == Catch::Approx(expected(i)).margin(1e-10));
#endif
}

TEST_CASE("bessel", "[utils]")
{
	REQUIRE(bessy0(0.1) == Catch::Approx(-1.534238651350367).margin(1e-8));
	REQUIRE(bessy0(1.) == Catch::Approx(0.088256964215677).margin(1e-8));
	REQUIRE(bessy0(10.) == Catch::Approx(0.055671167283599).margin(1e-8));
	REQUIRE(bessy0(100.) == Catch::Approx(-0.077244313365083).margin(1e-8));

	REQUIRE(bessy1(0.1) == Catch::Approx(-6.458951094702027).margin(1e-8));
	REQUIRE(bessy1(1.) == Catch::Approx(-0.781212821300289).margin(1e-8));
	REQUIRE(bessy1(10.) == Catch::Approx(0.249015424206954).margin(1e-8));
	REQUIRE(bessy1(100.) == Catch::Approx(-0.020372312002760).margin(1e-8));
}

TEST_CASE("expression", "[utils]")
{
	json jexpr = {{"value", "x^2+sqrt(x*y)+sin(z)*x"}};
	json jexpr2d = {{"value", "x^2+sqrt(x*y)"}};
	json jval = {{"value", 1}};

	utils::ExpressionValue expr;
	expr.init(jexpr["value"]);
	utils::ExpressionValue expr2d;
	expr2d.init(jexpr2d["value"]);
	utils::ExpressionValue val;
	val.init(jval["value"]);

	expr.set_unit_type("");
	expr2d.set_unit_type("");
	val.set_unit_type("");

	REQUIRE(expr(2, 3, 4) == Catch::Approx(2. * 2. + sqrt(2. * 3.) + sin(4.) * 2.).margin(1e-10));
	REQUIRE(expr2d(2, 3) == Catch::Approx(2. * 2. + sqrt(2. * 3.)).margin(1e-10));
	REQUIRE(val(2, 3, 4) == Catch::Approx(1).margin(1e-16));
}

TEST_CASE("mshreader", "[utils]")
{
	const std::string path = POLYFEM_DATA_DIR;
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi cells;
	const auto mesh = Mesh::create(path + "/circle2.msh");
	REQUIRE(mesh);
}

TEST_CASE("inverse", "[utils]")
{
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat = Eigen::MatrixXd::Random(1, 1);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat2 = Eigen::MatrixXd::Random(2, 2);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat3 = Eigen::MatrixXd::Random(3, 3);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat_inv = mat.inverse();
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat2_inv = mat2.inverse();
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat3_inv = mat3.inverse();

	REQUIRE(((utils::inverse(mat) - mat_inv)).norm() == Catch::Approx(0).margin(1e-12));
	REQUIRE(((utils::inverse(mat2) - mat2_inv)).norm() == Catch::Approx(0).margin(1e-12));
	REQUIRE(((utils::inverse(mat3) - mat3_inv)).norm() == Catch::Approx(0).margin(1e-12));
}

#ifdef POLYFEM_WITH_REMESHING
TEST_CASE("wmtk_instatiation", "[utils]")
{
	wmtk::TriMesh mesh;
}
#endif