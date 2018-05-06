////////////////////////////////////////////////////////////////////////////////

#include "TriQuadrature.hpp"
#include "FEBasis2d.hpp"
#include "auto_bases.hpp"

#include <catch.hpp>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;


TEST_CASE("P1", "[bases]") {
    TriQuadrature rule;
    Quadrature quad;
    rule.get_quadrature(12, quad);

    Eigen::MatrixXd expected, val;
    for(int i = 0; i < 3; ++i){
        poly_fem::FEBasis2d::linear_tri_basis_value(i, quad.points, expected);
        poly_fem::autogen::p_basis_value(1, i, quad.points, val);

        for(int j = 0; j < val.size(); ++j)
            REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

        poly_fem::FEBasis2d::linear_tri_basis_grad(i, quad.points, expected);
        poly_fem::autogen::p_grad_basis_value(1, i, quad.points, val);

        for(int j = 0; j < val.size(); ++j)
            REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
    }

}

TEST_CASE("P2", "[bases]") {
    TriQuadrature rule;
    Quadrature quad;
    rule.get_quadrature(12, quad);

    Eigen::MatrixXd expected, val;
    for(int i = 0; i < 6; ++i){
        poly_fem::FEBasis2d::quadr_tri_basis_value(i, quad.points, expected);
        poly_fem::autogen::p_basis_value(2, i, quad.points, val);

        for(int j = 0; j < val.size(); ++j)
            REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

        poly_fem::FEBasis2d::quadr_tri_basis_grad(i, quad.points, expected);
        poly_fem::autogen::p_grad_basis_value(2, i, quad.points, val);

        for(int j = 0; j < val.size(); ++j)
            REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
    }
}