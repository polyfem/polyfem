////////////////////////////////////////////////////////////////////////////////

#include "TriQuadrature.hpp"
#include "FEBasis2d.hpp"
#include "FEMSolver.hpp"

#include <catch.hpp>
#include <iostream>
#include <cppoptlib/meta.h>
#include <cppoptlib/problem.h>
#include <cppoptlib/solver/bfgssolver.h>
#include <unsupported/Eigen/SparseExtra>
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


#ifdef USE_HYPRE
TEST_CASE("hypre", "[solver]") {
    const std::string path = DATA_DIR;
    std::cout<<path<<std::endl;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solver = LinearSolver::create("Hypre", "");
        // solver->setParameters(params);
    Eigen::VectorXd b(A.rows()); b.setRandom();
    Eigen::VectorXd x(b.size());
    
    solver->analyzePattern(A);
    solver->factorize(A);
    solver->solve(b, x);

    // solver->getInfo(solver_info);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A*x-b).norm();
    REQUIRE(err < 1e-7);
}
#endif