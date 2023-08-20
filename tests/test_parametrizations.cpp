////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

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

#endif
