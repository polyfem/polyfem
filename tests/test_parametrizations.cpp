////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/io/OBJReader.hpp>

#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/solver/forms/parametrization/SplineParametrizations.hpp>

#include <iostream>
#include <fstream>
#include <catch2/catch_all.hpp>
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

TEST_CASE("bbw-test", "[parametrization]")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi E, F;
	const std::string mesh_path = POLYFEM_DATA_DIR + std::string("/contact/meshes/2D/simple/circle/circle140.obj");
	io::OBJReader::read(mesh_path, V, E, F);
	V.conservativeResizeLike(Eigen::MatrixXd::Zero(V.rows(), 3));

	BoundedBiharmonicWeights2Dto3D lbs_with_bbw(5, V.rows(), V, F);

	Eigen::VectorXd y = utils::flatten(V);
	verify_apply_jacobian(lbs_with_bbw, y);
}

#endif
