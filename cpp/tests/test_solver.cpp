////////////////////////////////////////////////////////////////////////////////

#include "TriQuadrature.hpp"
#include "FEBasis2d.hpp"
#include "auto_bases.hpp"

#include <catch.hpp>
#include <iostream>
#include <cppoptlib/meta.h>
#include <cppoptlib/problem.h>
#include <cppoptlib/solver/bfgssolver.h>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

class Rosenbrock : public cppoptlib::Problem<double> {
public:
    double value(const Eigen::VectorXd &x) {
        const double t1 = (1 - x[0]);
        const double t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }
    void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    }
};

TEST_CASE("solver", "[solver]") {
    Rosenbrock f;
    cppoptlib::BfgsSolver<Rosenbrock> solver;
    Eigen::VectorXd x(2); x << -1, 2;
    solver.minimize(f, x);
    std::cout << "argmin      " << x.transpose() << std::endl;
    std::cout << "f in argmin " << f(x) << std::endl;
    REQUIRE(f(x) < 1e-10);
}

TEST_CASE("bases", "[solver]") {
    TriQuadrature rule;
    Quadrature quad;
    rule.get_quadrature(12, quad);

    Eigen::MatrixXd expected, val;
    for(int i = 0; i < 3; ++i){
        poly_fem::FEBasis2d::linear_tri_basis_value(i, quad.points, expected);
        poly_fem::autogen::p_basis_value(1, (3-i)%3, quad.points, val);

        for(int j = 0; j < val.size(); ++j)
            REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

        poly_fem::FEBasis2d::linear_tri_basis_grad(i, quad.points, expected);
        poly_fem::autogen::p_grad_basis_value(1, (3-i)%3, quad.points, val);

        for(int j = 0; j < val.size(); ++j)
            REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
    }


    // poly_fem::FEBasis2d::quadr_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
    // poly_fem::FEBasis2d::quadr_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

}