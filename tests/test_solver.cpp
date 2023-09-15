////////////////////////////////////////////////////////////////////////////////

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <cppoptlib/meta.h>
#include <cppoptlib/problem.h>
#include <cppoptlib/solver/bfgssolver.h>
#include <unsupported/Eigen/SparseExtra>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;

class Rosenbrock : public cppoptlib::Problem<double>
{
public:
	double value(const Eigen::VectorXd &x) override
	{
		const double t1 = (1 - x[0]);
		const double t2 = (x[1] - x[0] * x[0]);
		return t1 * t1 + 100 * t2 * t2;
	}
	void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override
	{
		grad[0] = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
		grad[1] = 200 * (x[1] - x[0] * x[0]);
	}
};

TEST_CASE("solver", "[solver]")
{
	Rosenbrock f;
	cppoptlib::BfgsSolver<Rosenbrock> solver;
	Eigen::VectorXd x(2);
	x << -1, 2;
	solver.minimize(f, x);
	std::cout << "argmin      " << x.transpose() << std::endl;
	std::cout << "f in argmin " << f(x) << std::endl;
	REQUIRE(f(x) < 1e-10);
}
