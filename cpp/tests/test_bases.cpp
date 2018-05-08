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


TEST_CASE("P3", "[bases]") {
    Eigen::MatrixXd pts(10, 2);
    pts <<
    0, 0,
    1, 0,
    0, 1,
    1./3., 0,
    2./3., 0,
    2./3., 1./3.,
    1./3., 2./3.,
    0, 2./3.,
    0, 1./3.,
    1./3., 1./3.;

    Eigen::MatrixXd val;

    for(int i = 0; i < pts.rows(); ++i){
        poly_fem::autogen::p_basis_value(3, i, pts, val);

        // std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

        for(int j = 0; j < val.size(); ++j){
            if(i == j)
                REQUIRE(val(j) == Approx(1).margin(1e-10));
            else
                REQUIRE(val(j) == Approx(0).margin(1e-10));
        }
    }
}

TEST_CASE("P4", "[bases]") {
    Eigen::MatrixXd pts;
    poly_fem::autogen::p_nodes(4, pts);

    Eigen::MatrixXd val;
    for(int i = 0; i < pts.rows(); ++i){
        poly_fem::autogen::p_basis_value(4, i, pts, val);

        // std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

        for(int j = 0; j < val.size(); ++j){
            if(i == j)
                REQUIRE(val(j) == Approx(1).margin(1e-10));
            else
                REQUIRE(val(j) == Approx(0).margin(1e-10));
        }
    }
}
