////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <polyfem/solver/forms/parametrization/SplineParametrizations.hpp>

#include <iostream>
#include <fstream>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;
using namespace polysolve;

#if defined(__linux__)

TEST_CASE("SplineParametrization-Debug", "[parametrization]")
{
	Eigen::MatrixXd control_points(4, 2);
	control_points << 0, -1,
		0., -0.33333333,
		0., 0.33333333,
		0, 1;

	Eigen::VectorXd knots(8);
	knots << 0,
		0,
		0,
		0,
		1,
		1,
		1,
		1;

	Eigen::VectorXd V(20);
	V << 0., -1.,
		0., -0.77777778,
		0., -0.55555556,
		0., -0.33333333,
		0., -0.11111111,
		0., 0.11111111,
		0., 0.33333333,
		0., 0.55555556,
		0., 0.77777778,
		0., 1.;

	BSplineParametrization1DTo2D parametrization(control_points, knots, 10, false);
	Eigen::VectorXd x = parametrization.inverse_eval(V);

	Eigen::MatrixXd dydx(control_points.size(), V.size());
	double eps = 1e-7;
	for (int i = 0; i < control_points.size(); ++i)
	{
		Eigen::VectorXd x_ = x;
		x_(i) += eps;
		auto y_plus = parametrization.eval(x_);
		x_(i) -= 2 * eps;
		auto y_minus = parametrization.eval(x_);
		auto fd = (y_plus - y_minus) / (2 * eps);
		dydx.row(i) = fd;
	}

	for (int i = 0; i < V.size(); ++i)
	{
		Eigen::VectorXd grad_y;
		grad_y.setZero(V.size());
		grad_y(i) = 1;

		Eigen::VectorXd grad_x;
		grad_x = parametrization.apply_jacobian(grad_y, x);

		REQUIRE((grad_x - (dydx * grad_y)).norm() < 1e-8);
	}
}

TEST_CASE("SplineParametrizationExcludeEnds-Debug", "[parametrization]")
{
	Eigen::MatrixXd control_points(4, 2);
	control_points << 0, -1,
		0., -0.33333333,
		0., 0.33333333,
		0, 1;

	Eigen::VectorXd knots(8);
	knots << 0,
		0,
		0,
		0,
		1,
		1,
		1,
		1;

	Eigen::VectorXd V(20);
	V << 0., -1.,
		0., -0.77777778,
		0., -0.55555556,
		0., -0.33333333,
		0., -0.11111111,
		0., 0.11111111,
		0., 0.33333333,
		0., 0.55555556,
		0., 0.77777778,
		0., 1.;

	BSplineParametrization1DTo2D parametrization(control_points, knots, 10, true);
	Eigen::VectorXd x = parametrization.inverse_eval(V);

	Eigen::MatrixXd dydx(control_points.size() - 2 * 2, V.size());
	double eps = 1e-7;
	for (int i = 0; i < control_points.size() - 2 * 2; ++i)
	{
		Eigen::VectorXd x_ = x;
		x_(i) += eps;
		auto y_plus = parametrization.eval(x_);
		x_(i) -= 2 * eps;
		auto y_minus = parametrization.eval(x_);
		auto fd = (y_plus - y_minus) / (2 * eps);
		dydx.row(i) = fd;
	}

	for (int i = 0; i < V.size(); ++i)
	{
		Eigen::VectorXd grad_y;
		grad_y.setZero(V.size());
		grad_y(i) = 1;

		Eigen::VectorXd grad_x;
		grad_x = parametrization.apply_jacobian(grad_y, x);

		REQUIRE((grad_x - (dydx * grad_y)).norm() < 1e-8);
	}
}

#endif