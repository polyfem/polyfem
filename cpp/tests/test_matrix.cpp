////////////////////////////////////////////////////////////////////////////////
#include "MatrixUtils.hpp"

#include <Eigen/Dense>

#include <catch.hpp>
////////////////////////////////////////////////////////////////////////////////
TEST_CASE("determinant2", "[matrix]") {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat(2,2);
    mat.setRandom();

    REQUIRE(fabs(mat.determinant() - poly_fem::determinant(mat))<1e-10);
}

TEST_CASE("determinant3", "[matrix]") {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat(3,3);
    mat.setRandom();

    REQUIRE(fabs(mat.determinant() - poly_fem::determinant(mat))<1e-10);
}
