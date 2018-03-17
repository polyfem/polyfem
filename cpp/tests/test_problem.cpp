////////////////////////////////////////////////////////////////////////////////
#include "Problem.hpp"
#include "AssemblerUtils.hpp"
#include "Common.hpp"

#include <catch.hpp>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

const double k = 0.2;
const double lambda = 0.375, mu = 0.375;

template<typename T>
json get_params(const T &pts)
{
    return {
        {"k", k},
        {"size", pts.cols()},
        {"lambda", lambda},
        {"mu", mu},
        {"elasticity_tensor", {}}
    };
}


TEST_CASE("franke 2d", "[problem]") {
    Eigen::MatrixXd pts(400, 2);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("Franke");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();

    Eigen::MatrixXd fx;

    //fun
    {

        auto cx2 = (9*x-2) * (9*x-2);
        auto cy2 = (9*y-2) * (9*y-2);

        auto cx1 = (9*x+1) * (9*x+1);
        auto cx7 = (9*x-7) * (9*x-7);

        auto cy3 = (9*y-3) * (9*y-3);
        auto cx4 = (9*x-4) * (9*x-4);

        auto cy7 = (9*y-7) * (9*y-7);

        Eigen::MatrixXd val = (3./4.)*exp(-(1./4.)*cx2-(1./4.)*cy2)+(3./4.)*exp(-(1./49.)*cx1-(9./10.)*y-1./10.)+(1./2.)*exp(-(1./4.)*cx7-(1./4.)*cy3)-(1./5.)*exp(-cx4-cy7);
        fx = val;

        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }


    //grad
    {
        Eigen::MatrixXd gradX = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) * x + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) - 0.243e3 / 0.98e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.59e2 / 0.490e3 - 0.9e1 / 0.10e2 * y) * x - 0.27e2 / 0.98e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.59e2 / 0.490e3 - 0.9e1 / 0.10e2 * y) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) * x + 0.63e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2) * x - 0.72e2 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2);
        Eigen::MatrixXd gradY = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.59e2 / 0.490e3 - 0.9e1 / 0.10e2 * y) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2) * y - 0.126e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2);

        probl->exact_grad(pts, other);

        Eigen::MatrixXd diff = (other.col(0) - gradX);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        diff = (other.col(1) - gradY);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }


    //rhs
    {
        auto cx2 = (9*x-2) * (9*x-2);
        auto cy2 = (9*y-2) * (9*y-2);

        auto cx1 = (9*x+1) * (9*x+1);
        auto cx7 = (9*x-7) * (9*x-7);

        auto cy3 = (9*y-3) * (9*y-3);
        auto cx4 = (9*x-4) * (9*x-4);

        auto cy7 = (9*y-7) * (9*y-7);

        auto s1 = (-40.5 * x+9) * (-40.5 * x + 9);
        auto s2 = (-162./49. * x - 18./49.) * (-162./49. * x - 18./49.);
        auto s3 = (-40.5 * x + 31.5) * (-40.5 * x + 31.5);
        auto s4 = (-162. * x + 72) * (-162 * x + 72);

        auto s5 = (-40.5 * y + 9) * (-40.5 * y + 9);
        auto s6 = (-40.5 * y + 13.5) * (-40.5 * y + 13.5);
        auto s7 = (-162 * y + 126) * (-162 * y + 126);

        Eigen::MatrixXd rhs = 243./4. * (-0.25 * cx2 - 0.25 * cy2).exp() -   0.75 * s1 * (-0.25 * cx2 - 0.25 *cy2).exp() +
        36693./19600. * (-1./49. * cx1 - 0.9 * y - 0.1).exp()  - 0.75 * s2 * (- 1./49 * cx1 - 0.9 * y - 0.1).exp() +
        40.5 * (-0.25 * cx7 - 0.25 * cy3).exp()   - 0.5 * s3 * (-0.25 * cx7 - 0.25 * cy3).exp() -
        324./5.  * (-cx4-cy7).exp() + 0.2 * s4 * (-cx4-cy7).exp() -
        0.75 * s5 * (-0.25 * cx2 - 0.25 *cy2).exp()  - 0.5 * s6 * (-0.25 * cx7 - 0.25 * cy3).exp() +
        0.2 * s7 * (-cx4-cy7).exp();
        rhs*=-1;

        probl->rhs("Laplacian", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        rhs += k * fx;
        probl->rhs("Helmholtz", pts, other);
        diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("franke 3d", "[problem]") {
    Eigen::MatrixXd pts(400, 3);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("Franke");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();
    auto z = pts.col(2).array();

    Eigen::MatrixXd fx;

    ///fun
    {
        auto cx2 = (9*x-2) * (9*x-2);
        auto cy2 = (9*y-2) * (9*y-2);
        auto cz2 = (9*z-2) * (9*z-2);

        auto cx1 = (9*x+1) * (9*x+1);
        auto cx7 = (9*x-7) * (9*x-7);

        auto cy3 = (9*y-3) * (9*y-3);
        auto cx4 = (9*x-4) * (9*x-4);
        auto cy7 = (9*y-7) * (9*y-7);

        auto cz5 = (9*y-5) * (9*y-5);

        Eigen::MatrixXd val =
        3./4. * exp( -1./4.*cx2 - 1./4.*cy2 - 1./4.*cz2) +
        3./4. * exp(-1./49. * cx1 - 9./10.*y - 1./10. -  9./10.*z - 1./10.) +
        1./2. * exp(-1./4. * cx7 - 1./4. * cy3 - 1./4. * cz5) -
        1./5. * exp(- cx4 - cy7 - cz5);
        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        fx = val;
    }


    //grad
    {

        Eigen::MatrixXd gradX = (-59535 * x + 13230) * exp(-0.81e2 / 0.4e1 * x * x + (9 * x) - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) / 0.1960e4 + (-39690 * x + 30870) * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) / 0.1960e4 + (-4860 * x - 540) * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) / 0.1960e4 + 0.162e3 / 0.5e1 * exp(-(81 * x * x) - 0.162e3 * y * y + (72 * x) + 0.216e3 * y - 0.90e2) * (x - 0.4e1 / 0.9e1);
        Eigen::MatrixXd gradY = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) - 0.81e2 / 0.2e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) * y + 0.18e2 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) + 0.324e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.162e3 * y * y + 0.72e2 * x + 0.216e3 * y - 0.90e2) * y - 0.216e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.162e3 * y * y + 0.72e2 * x + 0.216e3 * y - 0.90e2);
        Eigen::MatrixXd gradZ = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * z + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z);

        probl->exact_grad(pts, other);

        Eigen::MatrixXd diff = (other.col(0) - gradX);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        diff = (other.col(1) - gradY);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        diff = (other.col(2) - gradZ);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }


    //rhs
    {
        Eigen::MatrixXd rhs =
        (1181472075 * x * x + 1181472075 * y * y + 1181472075 * z * z - 525098700 * x - 525098700 * y - 525098700 * z + 87516450) / 960400. *
        exp(-81./4. * x * x + 9 * x - 3 - 81./4. * y * y + 9 * y - 81./4. * z * z + 9 * z) +

        (787648050 * x * x + 3150592200 * y * y - 1225230300 * x - 2800526400 * y + 1040473350) / 960400. *
        exp(-81./4. * x * x + 63./2. * x - 83./4. - 81./2. * y * y + 36 * y) +

        (7873200 * x * x + 1749600 * x - 1117314) / 960400. *
        exp(-81./49. * x * x - 18./49. * x - 54./245. - 9./10. * y - 9./10. * z) -

        26244./ 5. * (x * x + 4 * y * y - 8./9. * x - 16./3. * y + 317./162.) *
        exp(-81 * x * x - 162 * y * y + 72 * x + 216 * y - 90);
        // rhs*=-1;

        probl->rhs("Laplacian", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        rhs += k*fx;
        probl->rhs("Helmholtz", pts, other);
        diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("linear", "[problem]") {
    Eigen::MatrixXd pts(400, 2);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("Linear");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();

    {
        Eigen::MatrixXd val = x;
        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    {

        Eigen::MatrixXd gradX = x;
        Eigen::MatrixXd gradY = x;
        gradX.setOnes();
        gradY.setZero();

        probl->exact_grad(pts, other);

        Eigen::MatrixXd diff = (other.col(0) - gradX);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        diff = (other.col(1) - gradY);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

    }



    {
        Eigen::MatrixXd rhs = x;
        rhs.setZero();

        probl->rhs("Laplacian", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    {
        Eigen::MatrixXd rhs = k*x;

        probl->rhs("Helmholtz", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("quadratic", "[problem]") {
    Eigen::MatrixXd pts(400, 2);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("Quadratic");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();

    {
        Eigen::MatrixXd val = 5*x*x;
        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    {

        Eigen::MatrixXd gradX = 10 * x;
        Eigen::MatrixXd gradY = x;
        gradY.setZero();

        probl->exact_grad(pts, other);

        Eigen::MatrixXd diff = (other.col(0) - gradX);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        diff = (other.col(1) - gradY);
        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

    }



    {
        Eigen::MatrixXd rhs = x;
        rhs.setConstant(10);

        probl->rhs("Laplacian", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    {
        Eigen::MatrixXd rhs = k*5*x*x + 10;

        probl->rhs("Helmholtz", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("zero bc 2d", "[problem]") {
    Eigen::MatrixXd pts(400, 2);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("Zero_BC");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();

    {
        Eigen::MatrixXd val = (1 - x)  * x * x * y * (1-y) *(1-y);
        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }


    {
        Eigen::MatrixXd rhs = -4 * x * y * (1 - y) * (1 - y) + 2 * (1 - x) * y * (1 - y) *(1 - y) - 4 * (1 - x) * x * x * (1 - y) + 2 * (1 - x) * x * x * y;

        probl->rhs("Laplacian", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("zero bc 3d", "[problem]") {
    Eigen::MatrixXd pts(40, 3);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("Zero_BC");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();
    auto z = pts.col(2).array();

    {
        Eigen::MatrixXd val = (1 - x)  * x * x * y * (1-y) *(1-y) * z * (1 - z);
        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    {
        Eigen::MatrixXd rhs =  (0.2e1 * pow(x, 0.3e1) - 0.2e1 * x * x +  (6 * z * z - 6 * z) * x -  (2 * z * z) +  (2 * z)) * pow(y, 0.3e1) + (-0.4e1 * pow(x, 0.3e1) + 0.4e1 * x * x +  (-12 * z * z + 12 * z) * x +  (4 * z * z) -  (4 * z)) * y * y + ( (6 * z * z - 6 * z + 2) * pow(x, 0.3e1) +  (-6 * z * z + 6 * z - 2) * x * x +  (6 * z * z - 6 * z) * x -  (2 * z * z) +  (2 * z)) * y - 0.4e1 * x * x *  z *  (z - 1) * (x - 0.1e1);
        probl->rhs("Laplacian", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("elasticity 2d", "[problem]") {
    Eigen::MatrixXd pts(400, 2);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("ElasticExact");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();

    //fun
    {
        Eigen::MatrixXd val(pts.rows(), pts.cols());
        val.col(0) = (y*y*y + x*x + x*y)/10.;
        val.col(1) = (3*x*x*x*x + x*y*y + x)/10.;

        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }


    //rhs
    {
        Eigen::MatrixXd rhs(pts.rows(), pts.cols());
        rhs.col(0) = 2./5.*mu + lambda*(1./5+1./5.*y) + 4./5.*mu*y;
        rhs.col(1) = 2*mu*(9./5.*x*x + 1./20.) + 2./5.*mu*x + lambda*(1./10.+1./5.*x);

        probl->rhs("LinearElasticity", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        probl->rhs("HookeLinearElasticity", pts, other);
        diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    //rhs
    {
        Eigen::MatrixXd rhs(pts.rows(), pts.cols());
        rhs.col(0) =  (432 * lambda * y + 1008 * lambda + 2016 * mu) * pow(x, 0.6e1) / 0.1000e4 +  ((864 * mu + 432 * lambda) * y + 8688 * mu + 4320 * lambda) * pow(x, 0.5e1) / 0.1000e4 +  ((24 * lambda + 168 * mu) * y + 240 * mu) * pow(x, 0.4e1) / 0.1000e4 +  ((144 * lambda + 288 * mu) *  pow( y,  3) + (192 * mu + 96 * lambda) * y * y + (76 * lambda + 8 * mu) * y + 792 * mu + 96 * lambda) * pow(x, 0.3e1) / 0.1000e4 +  ((120 * mu + 60 * lambda) *  pow( y,  3) + (1812 * mu + 372 * lambda) * y * y + (114 * mu + 57 * lambda) * y + 798 * mu + 397 * lambda) * x * x / 0.1000e4 +  ((120 * mu + 60 * lambda) *  pow( y,  3) + (250 * lambda + 480 * mu) * y * y + (316 * mu + 216 * lambda) * y + 300 * mu + 140 * lambda) * x / 0.1000e4 +  ((90 * lambda + 168 * mu) *  pow( y,  5)) / 0.1000e4 +  ((20 * mu + 10 * lambda) *  pow( y,  4)) / 0.1000e4 +  ((24 * mu + 21 * lambda) *  pow( y,  3)) / 0.1000e4 +  ((290 * mu + 145 * lambda) * y * y) / 0.1000e4 +  ((920 * mu + 263 * lambda) * y) / 0.1000e4 + 0.211e3 / 0.500e3 *  mu + 0.201e3 / 0.1000e4 *  lambda;
        rhs.col(1) =  (15552 * mu + 7776 * lambda) * pow(x, 0.8e1) / 0.1000e4 +  (144 * lambda + 288 * mu) * pow(x, 0.7e1) / 0.1000e4 + 0.324e3 / 0.125e3 * ( mu +  lambda / 0.2e1) *  (y * y + 1) * pow(x, 0.5e1) +  ((192 * lambda + 384 * mu) * y * y + 144 * mu * y + 300 * mu + 174 * lambda) * pow(x, 0.4e1) / 0.1000e4 +  ((156 * lambda + 132 * mu) * y * y + (3312 * mu + 1296 * lambda) * y + 1922 * mu + 965 * lambda) * pow(x, 0.3e1) / 0.1000e4 +  ((108 * mu + 216 * lambda) *  pow( y,  4) + (144 * lambda + 288 * mu) * y * y + (976 * mu + 488 * lambda) * y + 3708 * mu + 94 * lambda) * x * x / 0.1000e4 +  ((54 * lambda + 108 * mu) *  pow( y,  4) + 48 * mu *  pow( y,  3) + (35 * mu + 18 * lambda) * y * y + (100 * lambda + 192 * mu) * y + 451 * mu + 226 * lambda) * x / 0.1000e4 +  ((3 * lambda + 21 * mu) *  pow( y,  4)) / 0.1000e4 +  ((624 * mu + 222 * lambda) *  pow( y,  3)) / 0.1000e4 +  ((109 * mu + 23 * lambda) * y * y) / 0.1000e4 +  ((154 * mu + 52 * lambda) * y) / 0.1000e4 + 0.7e1 / 0.50e2 *  mu + 0.3e1 / 0.25e2 *  lambda;

        probl->rhs("SaintVenant", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}


TEST_CASE("elasticity 3d", "[problem]") {
    Eigen::MatrixXd pts(400, 3);
    Eigen::MatrixXd other;
    pts.setRandom();

    const auto &probl = ProblemFactory::factory().get_problem("ElasticExact");
    const json params = get_params(pts);

    auto &assembler = AssemblerUtils::instance();
    assembler.set_parameters(params);

    auto x = pts.col(0).array();
    auto y = pts.col(1).array();
    auto z = pts.col(2).array();

    ///fun
    {
        Eigen::MatrixXd val(pts.rows(), pts.cols());
        val.col(0) = (x*y + x*x + y*y*y + 6*z)/10.;
        val.col(1) = (z*x - z*z*z + x*y*y + 3*x*x*x*x)/10.;
        val.col(2) = (x*y*z + z*z*y*y - 2*x)/10.;

        probl->exact(pts, other);
        Eigen::MatrixXd diff = (other - val);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }


    //rhs
    {
        Eigen::MatrixXd rhs(pts.rows(), pts.cols());
        rhs.col(0) = 2./5.*mu + lambda * (1./5. + 3./10.*y) + 9./10.*mu*y;
        rhs.col(1) = 2*mu* (9./5. * x * x + 1./20.) + 2./5. * mu * x + lambda * (1./10. + 3./10. * x + 2./5. * y * z) + 2 * mu * (1./5. * y * z + 1./20. * x - 3./10.*z);
        rhs.col(2) = 1./5. * mu * z * z + 2./5. * mu * y * y + 1./5. * lambda * y * y;

        probl->rhs("LinearElasticity", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

        probl->rhs("HookeLinearElasticity", pts, other);
        diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }

    //rhs
    {
        Eigen::MatrixXd rhs(pts.rows(), pts.cols());
        rhs.col(0) = (0.864e3 * pow(x, 0.5e1) + 0.24e2 * pow(x, 0.4e1) - 0.72e2 * z * pow(x, 0.3e1) + ( (72 * y * y) + 0.72e2 * z) * x * x + ( (7 * y * y) + z * z +  (12 * y) + 0.2e1 * z + 0.10e2) * x + 0.4e1 * z *   pow( y,  3) + (-0.6e1 * z + 0.5e1) *  (y * y) + (0.4e1 * pow(z, 0.3e1) + 0.94e2) *  y - 0.13e2 * z * z + 0.40e2) * mu / 0.100e3 + 0.108e3 / 0.25e2 * lambda * (pow(x, 0.5e1) + ( (y * y) / 0.12e2 + z / 0.12e2) * x * x + (0.5e1 / 0.432e3 *  y *  y + z * z / 0.432e3 + 0.1e1 / 0.72e2) * x + z *   pow( y,  3) / 0.216e3 +  (y * y) / 0.144e3 + (pow(z, 0.3e1) / 0.216e3 + 0.2e1 / 0.27e2) *  y - z * z / 0.144e3 + 0.5e1 / 0.108e3);
        rhs.col(1) = (8 * lambda * z * z + 12 * mu * z * z + 20 * lambda + 38 * mu) * pow(y, 0.3e1) / 0.100e3 +  (6 * lambda * z * x + 8 * mu * z * x + 6 * mu) * y * y / 0.100e3 + ( ((8 *  pow( z,  4) + z * z + (-12 * x + 42) * z + 96 *  pow( x,  3) + 9 * x * x + 12 * x + 1) * mu) + 0.24e2 *  lambda * (  pow( z,  4) / 0.6e1 +   pow( x,  3) + 0.5e1 / 0.24e2 *  x *  x +  (z * z) / 0.24e2 +  x / 0.4e1 + 0.7e1 / 0.4e1 *  z + 0.1e1 / 0.24e2)) * y / 0.100e3 +  ((4 *  pow( z,  3) * x + 360 * x * x + 54 * x - 62 * z + 10) * mu) / 0.100e3 +  (lambda * ( pow( z,  3) * x + 16 * x - z + 5)) / 0.50e2;
        rhs.col(2) = (8 * lambda * y * y + 12 * mu * y * y + 18 * lambda + 36 * mu) * pow(z, 0.3e1) / 0.100e3 + ((-108 * x * x + 8 * x * y - 6 * x + 20) * mu + 6 * lambda * x * y) * z * z / 0.100e3 + ((8 *  pow(y, 4) + x * x + y * y - 12 * x + 1) * mu + lambda * (4 *  pow(y, 4) + x * x + y * y - 6 * x + 1)) * z / 0.100e3 + ((4 *  pow(y, 3) * x + 48 *  pow(x, 3) + 2 * x * x + 41 * y * y + 34 * y + 12) * mu) / 0.100e3 + 0.3e1 / 0.25e2 * (( pow(y, 3) * x) / 0.6e1 +  pow(x, 3) + 0.7e1 / 0.4e1 * y * y - y / 0.6e1) * lambda;

        probl->rhs("SaintVenant", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}