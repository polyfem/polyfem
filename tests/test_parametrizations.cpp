////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <polyfem/solver/forms/parametrization/SplineParametrizations.hpp>

#include <iostream>
#include <fstream>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;
using namespace polysolve;

#if defined(__linux__)

void verify_apply_jacobian(Parametrization &parametrization, const Eigen::VectorXd &y, bool print_grads = false)
{
	Eigen::VectorXd x = parametrization.inverse_eval(y);

	Eigen::MatrixXd dydx(x.size(), y.size());
	double eps = 1e-7;
	for (int i = 0; i < x.size(); ++i)
	{
		Eigen::VectorXd x_ = x;
		x_(i) += eps;
		auto y_plus = parametrization.eval(x_);
		x_(i) -= 2 * eps;
		auto y_minus = parametrization.eval(x_);
		auto fd = (y_plus - y_minus) / (2 * eps);
		dydx.row(i) = fd;
	}

	for (int i = 0; i < y.size(); ++i)
	{
		Eigen::VectorXd grad_y;
		grad_y.setZero(y.size());
		grad_y(i) = 1;

		Eigen::VectorXd grad_x;
		grad_x = parametrization.apply_jacobian(grad_y, x);

		if (print_grads)
			std::cout << std::setprecision(16) << grad_x.norm() << std::endl;
		REQUIRE((grad_x - (dydx * grad_y)).norm() < 1e-8);
	}
}

TEST_CASE("SplineParametrization", "[parametrization]")
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
	verify_apply_jacobian(parametrization, V);
}

TEST_CASE("SplineParametrizationExcludeEnds", "[parametrization]")
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
	verify_apply_jacobian(parametrization, V);
}

TEST_CASE("BoundedBiharmonicWeights", "[parametrization]")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector<std::vector<int>> e;
	std::vector<std::vector<double>> w;
	std::vector<int> ids;
	polyfem::io::MshReader::load("../cube_dense.msh", V, F, e, w, ids);
	V.conservativeResize(V.rows(), 3);
	V.col(2) = Eigen::VectorXd::Zero(V.rows());

	BoundedBiharmonicWeights2Dto3D parametrization(5, V, F);
	verify_apply_jacobian(parametrization, utils::flatten(V));

	// Eigen::MatrixXd bbw_weights = parametrization.get_bbw_weights();
	// for (int i = 0; i < bbw_weights.cols(); ++i)
	// {
	// 	auto V_ = V;
	// 	V_.col(2) += bbw_weights.col(i);
	// 	polyfem::io::OBJWriter::write(fmt::format("bbw_weights_{}.obj", i), V_, F);
	// }
}

#endif