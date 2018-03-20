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

        //rhs
    {
        Eigen::MatrixXd rhs(pts.rows(), pts.cols());
        rhs.col(0) =  (-0.8640e4 * (pow(x, 0.6e1) * y - pow(x, 0.5e1) / 0.18e2 - 0.5e1 / 0.36e2 * pow(x, 0.4e1) * y - 0.5e1 / 0.18e2 * pow(x, 0.4e1) + y * pow(x, 0.3e1) / 0.6e1 - 0.25e2 / 0.36e2 * pow(x, 0.3e1) - 0.89e2 / 0.72e2 * x * x * y * y - x * x / 0.216e3 + 0.5e1 / 0.216e3 * x * y * y + 0.29e2 / 0.216e3 * x * y - 0.5e1 / 0.216e3 * x + pow(y, 0.5e1) / 0.72e2 + pow(y, 0.3e1) / 0.48e2 + 0.103e3 / 0.432e3 * y + 0.5e1 / 0.24e2) * lambda * log(-0.3e1 * pow(y, 0.4e1) + (-0.36e2 * pow(x, 0.3e1) + x - 0.3e1) * y * y + (0.4e1 * x * x + 0.20e2 * x + 0.10e2) * y - 0.12e2 * pow(x, 0.4e1) + 0.19e2 * x + 0.100e3) + 0.17280e5 * (pow(x, 0.6e1) * y - pow(x, 0.5e1) / 0.18e2 - 0.5e1 / 0.36e2 * pow(x, 0.4e1) * y - 0.5e1 / 0.18e2 * pow(x, 0.4e1) + y * pow(x, 0.3e1) / 0.6e1 - 0.25e2 / 0.36e2 * pow(x, 0.3e1) - 0.89e2 / 0.72e2 * x * x * y * y - x * x / 0.216e3 + 0.5e1 / 0.216e3 * x * y * y + 0.29e2 / 0.216e3 * x * y - 0.5e1 / 0.216e3 * x + pow(y, 0.5e1) / 0.72e2 + pow(y, 0.3e1) / 0.48e2 + 0.103e3 / 0.432e3 * y + 0.5e1 / 0.24e2) * lambda * (log(0.2e1) + log(0.5e1)) + 0.144e3 / 0.5e1 * mu * (0.3e1 * y + 0.1e1) * pow(x, 0.8e1) + 0.864e3 / 0.5e1 * mu * (0.3e1 * y + 0.1e1) * y * y * pow(x, 0.7e1) + 0.48e2 / 0.5e1 * y * (0.81e2 * mu * pow(y, 0.4e1) + 0.27e2 * mu * pow(y, 0.3e1) - 0.6e1 * mu * y + 0.900e3 * lambda + 0.898e3 * mu) * pow(x, 0.6e1) - 0.24e2 / 0.5e1 * (0.36e2 * mu * pow(y, 0.4e1) + 0.15e2 * mu * pow(y, 0.3e1) + 0.61e2 * mu * y * y + 0.77e2 * mu * y + 0.100e3 * lambda + 0.119e3 * mu) * pow(x, 0.5e1) - 0.16e2 * (0.54e2 * mu * pow(y, 0.4e1) + 0.66e2 * mu * pow(y, 0.3e1) + 0.25e2 * mu * y * y + 0.75e2 * lambda * y + 0.168e3 * mu * y + 0.150e3 * lambda + 0.180e3 * mu) * pow(x, 0.4e1) + 0.8e1 / 0.5e1 * (0.81e2 * mu * pow(y, 0.7e1) + 0.27e2 * mu * pow(y, 0.6e1) + 0.81e2 * mu * pow(y, 0.5e1) - 0.240e3 * mu * pow(y, 0.4e1) - 0.2729e4 * mu * pow(y, 0.3e1) - 0.823e3 * mu * y * y + 0.900e3 * lambda * y + 0.919e3 * mu * y - 0.3750e4 * lambda - 0.3750e4 * mu) * pow(x, 0.3e1) - (0.72e2 * mu * pow(y, 0.6e1) + 0.21e2 * mu * pow(y, 0.5e1) - 0.49e2 * mu * pow(y, 0.4e1) - 0.1570e4 * mu * pow(y, 0.3e1) + 0.53400e5 * lambda * y * y + 0.48202e5 * mu * y * y - 0.2643e4 * mu * y + 0.200e3 * lambda - 0.161e3 * mu) * x * x / 0.5e1 + 0.2e1 / 0.5e1 * (-0.9e1 * mu * pow(y, 0.7e1) - 0.183e3 * mu * pow(y, 0.6e1) - 0.240e3 * mu * pow(y, 0.5e1) - 0.210e3 * mu * pow(y, 0.4e1) + 0.679e3 * mu * pow(y, 0.3e1) + 0.500e3 * lambda * y * y + 0.7313e4 * mu * y * y + 0.2900e4 * lambda * y + 0.10790e5 * mu * y - 0.500e3 * lambda + 0.1400e4 * mu) * x + 0.27e2 / 0.5e1 * mu * pow(y, 0.9e1) + 0.9e1 / 0.5e1 * mu * pow(y, 0.8e1) + 0.54e2 / 0.5e1 * mu * pow(y, 0.7e1) - 0.162e3 / 0.5e1 * mu * pow(y, 0.6e1) + 0.3e1 / 0.5e1 * (-0.411e3 * mu + 0.200e3 * lambda) * pow(y, 0.5e1) - 0.771e3 / 0.5e1 * mu * pow(y, 0.4e1) + 0.12e2 * (-0.11e2 * mu + 0.15e2 * lambda) * pow(y, 0.3e1) + 0.1100e4 * mu * y * y + 0.20e2 * (0.423e3 * mu + 0.103e3 * lambda) * y + 0.3800e4 * mu + 0.1800e4 * lambda) * pow(0.36e2 * y * y * pow(x, 0.3e1) + 0.12e2 * pow(x, 0.4e1) + 0.3e1 * pow(y, 0.4e1) - 0.4e1 * x * x * y - x * y * y - 0.20e2 * x * y + 0.3e1 * y * y - 0.19e2 * x - 0.10e2 * y - 0.100e3, -0.2e1);
        rhs.col(1) =  (0.1440e4 * (pow(x, 0.4e1) * y - pow(x, 0.4e1) / 0.3e1 - 0.5e1 / 0.4e1 * y * y * pow(x, 0.3e1) + 0.5e1 * y * pow(x, 0.3e1) - pow(x, 0.3e1) / 0.18e2 - 0.9e1 / 0.4e1 * x * x * pow(y, 0.4e1) - 0.5e1 / 0.9e1 * x * x + x * pow(y, 0.3e1) / 0.3e1 - x * y * y / 0.144e3 - x * y / 0.18e2 - 0.67e2 / 0.48e2 * x + 0.5e1 / 0.48e2 * pow(y, 0.4e1) + 0.5e1 / 0.4e1 * pow(y, 0.3e1) + 0.7e1 / 0.16e2 * y * y + 0.25e2 / 0.72e2 * y - 0.25e2 / 0.36e2) * lambda * log(-0.3e1 * pow(y, 0.4e1) + (-0.36e2 * pow(x, 0.3e1) + x - 0.3e1) * y * y + (0.4e1 * x * x + 0.20e2 * x + 0.10e2) * y - 0.12e2 * pow(x, 0.4e1) + 0.19e2 * x + 0.100e3) - 0.2880e4 * (pow(x, 0.4e1) * y - pow(x, 0.4e1) / 0.3e1 - 0.5e1 / 0.4e1 * y * y * pow(x, 0.3e1) + 0.5e1 * y * pow(x, 0.3e1) - pow(x, 0.3e1) / 0.18e2 - 0.9e1 / 0.4e1 * x * x * pow(y, 0.4e1) - 0.5e1 / 0.9e1 * x * x + x * pow(y, 0.3e1) / 0.3e1 - x * y * y / 0.144e3 - x * y / 0.18e2 - 0.67e2 / 0.48e2 * x + 0.5e1 / 0.48e2 * pow(y, 0.4e1) + 0.5e1 / 0.4e1 * pow(y, 0.3e1) + 0.7e1 / 0.16e2 * y * y + 0.25e2 / 0.72e2 * y - 0.25e2 / 0.36e2) * lambda * (log(0.2e1) + log(0.5e1)) + 0.2592e4 / 0.5e1 * mu * pow(x, 0.10e2) + 0.144e3 / 0.5e1 * mu * (0.108e3 * y * y + 0.1e1) * pow(x, 0.9e1) + 0.864e3 / 0.5e1 * mu * (0.27e2 * pow(y, 0.3e1) + y - 0.2e1) * y * pow(x, 0.8e1) + 0.48e2 / 0.5e1 * mu * (0.27e2 * pow(y, 0.4e1) - 0.108e3 * pow(y, 0.3e1) - 0.9e1 * y * y - 0.182e3 * y - 0.171e3) * pow(x, 0.7e1) - 0.24e2 / 0.5e1 * (0.1092e4 * pow(y, 0.3e1) + 0.961e3 * y * y + 0.200e3 * y + 0.1819e4) * mu * pow(x, 0.6e1) + 0.16e2 / 0.5e1 * (0.243e3 * pow(y, 0.6e1) + 0.243e3 * pow(y, 0.4e1) - 0.891e3 * pow(y, 0.3e1) - 0.8000e4 * y * y + 0.156e3 * y - 0.150e3) * mu * pow(x, 0.5e1) - 0.2e1 / 0.5e1 * (-0.108e3 * mu * pow(y, 0.6e1) + 0.216e3 * mu * pow(y, 0.5e1) - 0.117e3 * mu * pow(y, 0.4e1) + 0.212e3 * mu * pow(y, 0.3e1) - 0.1142e4 * mu * y * y + 0.3600e4 * lambda * y - 0.10516e5 * mu * y - 0.1200e4 * lambda - 0.4449e4 * mu) * pow(x, 0.4e1) + (-0.108e3 * mu * pow(y, 0.6e1) - 0.2184e4 * mu * pow(y, 0.5e1) - 0.2159e4 * mu * pow(y, 0.4e1) - 0.1784e4 * mu * pow(y, 0.3e1) + 0.9000e4 * lambda * y * y + 0.18266e5 * mu * y * y - 0.36000e5 * lambda * y + 0.44400e5 * mu * y + 0.400e3 * lambda + 0.69161e5 * mu) * pow(x, 0.3e1) / 0.5e1 + 0.2e1 / 0.5e1 * (0.81e2 * mu * pow(y, 0.8e1) + 0.159e3 * mu * pow(y, 0.6e1) - 0.600e3 * mu * pow(y, 0.5e1) + 0.8100e4 * lambda * pow(y, 0.4e1) + 0.2721e4 * mu * pow(y, 0.4e1) - 0.590e3 * mu * pow(y, 0.3e1) - 0.4257e4 * mu * y * y + 0.20190e5 * mu * y + 0.2000e4 * lambda + 0.93900e5 * mu) * x * x - (-0.9e1 * mu * pow(y, 0.8e1) - 0.18e2 * mu * pow(y, 0.6e1) + 0.60e2 * mu * pow(y, 0.5e1) + 0.591e3 * mu * pow(y, 0.4e1) + 0.2400e4 * lambda * pow(y, 0.3e1) + 0.2460e4 * mu * pow(y, 0.3e1) - 0.50e2 * lambda * y * y + 0.450e3 * mu * y * y - 0.400e3 * lambda * y - 0.2400e4 * mu * y - 0.10050e5 * lambda - 0.20050e5 * mu) * x / 0.5e1 - 0.10e2 * (0.15e2 * pow(y, 0.4e1) + 0.180e3 * pow(y, 0.3e1) + 0.63e2 * y * y + 0.50e2 * y - 0.100e3) * (mu + lambda)) * pow(0.36e2 * y * y * pow(x, 0.3e1) + 0.12e2 * pow(x, 0.4e1) + 0.3e1 * pow(y, 0.4e1) - 0.4e1 * x * x * y - x * y * y - 0.20e2 * x * y + 0.3e1 * y * y - 0.19e2 * x - 0.10e2 * y - 0.100e3, -0.2e1);

        probl->rhs("NeoHookean", pts, other);
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
        rhs.col(0) = ((36 * lambda + 36 * mu) * z * z + 168 * mu + 90 * lambda) * pow(y, 0.5e1) / 0.1000e4 + (0.24e2 * ( mu + 0.5e1 / 0.4e1 *  lambda) *  z * x +  (4 * lambda * z * z) +  ((52 * mu + 26 * lambda) * z) +  (20 * mu) +  (10 * lambda)) * pow(y, 0.4e1) / 0.1000e4 + ( (288 * mu + 144 * lambda) * pow(x, 0.3e1) +  (123 * mu + 66 * lambda) * x * x +  ((8 * lambda + 12 * mu) * z * z + (-28 * mu + 8 * lambda) * z + 145 * mu + 73 * lambda) * x +  ((48 * mu + 24 * lambda) *  pow( z,  4)) +  ((3 * mu + 6 * lambda) * z * z) +  ((272 * lambda + 160 * mu) * z) +  (14 * mu) +  (9 * lambda)) * pow(y, 0.3e1) / 0.1000e4 + ( (192 * mu + 96 * lambda) * pow(x, 0.3e1) +  ((6 * lambda + 8 * mu) * z + 1814 * mu + 375 * lambda) * x * x +  ((36 * mu + 18 * lambda) *  pow( z,  3) - 12 * mu * z + 512 * mu + 350 * lambda) * x +  (4 * lambda *  pow( z,  4)) +  ((80 * mu + 50 * lambda) *  pow( z,  3)) +  ((2 * mu + lambda) * z * z) +  ((-236 * mu + 30 * lambda) * z) +  (502 * mu) +  (279 * lambda)) * y * y / 0.1000e4 + (0.432e3 *  lambda * pow(x, 0.6e1) +  (864 * mu + 432 * lambda) * pow(x, 0.5e1) +  (24 * lambda + 168 * mu) * pow(x, 0.4e1) +  ((-72 * mu + 72 * lambda) * z + 9 * mu + 5 * lambda) * pow(x, 0.3e1) +  ((6 * mu + 3 * lambda) * z * z + (60 * mu + 36 * lambda) * z + 42 * mu + 24 * lambda) * x * x +  ((8 * mu + 4 * lambda) *  pow( z,  4) + (8 * mu + 8 * lambda) *  pow( z,  3) + (20 * lambda + 52 * mu) * z * z + (48 * mu + 42 * lambda) * z + 329 * mu + 255 * lambda) * x +  (27 * lambda *  pow( z,  4)) +  ((40 * mu + 20 * lambda) *  pow( z,  3)) -  (21 * mu * z * z) +  ((-8 * mu - 4 * lambda) * z) +  (1224 * mu) +  (468 * lambda)) * y / 0.1000e4 +  (2016 * mu + 1008 * lambda) * pow(x, 0.6e1) / 0.1000e4 +  (8688 * mu + 4320 * lambda) * pow(x, 0.5e1) / 0.1000e4 +  (-144 * mu * z + 240 * mu) * pow(x, 0.4e1) / 0.1000e4 +  ((-528 * mu + 96 * lambda) * z + 888 * mu + 72 * lambda) * pow(x, 0.3e1) / 0.1000e4 +  ((4 * mu + 2 * lambda) *  pow( z,  3) + (-646 * mu + 3 * lambda) * z * z + (366 * lambda + 730 * mu) * z + 98 * mu + 50 * lambda) * x * x / 0.1000e4 +  ((-52 * mu - 2 * lambda) * z * z + (-118 * mu - 38 * lambda) * z + 290 * mu + 150 * lambda) * x / 0.1000e4 + 0.9e1 / 0.1000e4 *  lambda *   pow( z,  4) +  ((216 * mu + 108 * lambda) *  pow( z,  3)) / 0.1000e4 +  ((-8 * mu - 29 * lambda) * z * z) / 0.1000e4 +  ((26 * mu + 6 * lambda) * z) / 0.1000e4 + 0.12e2 / 0.25e2 *  mu + 0.6e1 / 0.25e2 *  lambda;
        rhs.col(1) = (15552 * mu + 7776 * lambda) * pow(x, 0.8e1) / 0.1000e4 +  (288 * mu + 144 * lambda) * pow(x, 0.7e1) / 0.1000e4 - 0.108e3 / 0.125e3 * ( mu +  lambda / 0.2e1) * z * pow(x, 0.6e1) + 0.324e3 / 0.125e3 * ( mu +  lambda / 0.2e1) * (y * y + z) * pow(x, 0.5e1) + ( (12 * mu + 30 * lambda) * z * z +  (48 * mu + 24 * lambda) * z +  (396 * mu + 222 * lambda) * y * y + 0.144e3 *  mu * y +  (336 * mu) +  (192 * lambda)) * pow(x, 0.4e1) / 0.1000e4 + (0.48e2 * y *  (mu + 2 * lambda) * pow(z, 0.3e1) +  (-550 * mu - 251 * lambda) * z * z + ( (48 * mu + 96 * lambda) * pow(y, 0.3e1) +  (-144 * mu - 72 * lambda) * y * y +  mu +  lambda) * z +  (86 * mu + 159 * lambda) * y * y +  (3432 * mu + 1776 * lambda) * y +  (1924 * mu) +  (966 * lambda)) * pow(x, 0.3e1) / 0.1000e4 + ((0.72e2 *  lambda * y * y +  (162 * lambda) +  (324 * mu)) * pow(z, 0.4e1) + ( (16 * mu + 8 * lambda) * y -  (3 * mu) -  (6 * lambda)) * pow(z, 0.3e1) + (0.72e2 *  lambda * pow(y, 0.4e1) +  (36 * mu + 18 * lambda) * y * y +  (6 * lambda + 8 * mu) * y +  (96 * mu) +  (48 * lambda)) * z * z + ( (16 * lambda + 16 * mu) * pow(y, 0.3e1) +  (813 * lambda + 186 * mu) * y * y +  (-144 * mu - 72 * lambda) * y -  (18 * mu) -  (24 * lambda)) * z +  (108 * mu + 216 * lambda) * pow(y, 0.4e1) +  (4 * mu + 2 * lambda) * pow(y, 0.3e1) +  (72 * mu + 36 * lambda) * y * y +  (1006 * mu + 538 * lambda) * y +  (3744 * mu) +  (760 * lambda)) * x * x / 0.1000e4 + (( (24 * mu + 12 * lambda) * y * y +  (-30 * lambda - 24 * mu) * y +  (18 * mu) +  (9 * lambda)) * pow(z, 0.4e1) + ( (8 * lambda + 12 * mu) * y * y +  (149 * mu) +  (75 * lambda)) * pow(z, 0.3e1) + ( (20 * lambda + 24 * mu) * pow(y, 0.4e1) +  (-36 * mu - 18 * lambda) * pow(y, 0.3e1) +  (4 * lambda + 7 * mu) * y * y +  (22 * mu) +  lambda) * z * z + ( (8 * mu + 4 * lambda) * pow(y, 0.4e1) +  (184 * mu + 174 * lambda) * y * y +  (-360 * mu - 200 * lambda) * y +  (13 * mu) -  (113 * lambda)) * z +  (109 * mu + 55 * lambda) * pow(y, 0.4e1) + 0.48e2 *  mu * pow(y, 0.3e1) +  (59 * mu + 30 * lambda) * y * y +  (210 * mu + 98 * lambda) * y +  (576 * mu) +  (360 * lambda)) * x / 0.1000e4 + ( (-36 * lambda - 36 * mu) * y * y -  (162 * mu) -  (81 * lambda)) * pow(z, 0.5e1) / 0.1000e4 + ( (42 * lambda + 84 * mu) * y -  (60 * mu)) * pow(z, 0.4e1) / 0.1000e4 + ( (-48 * mu - 24 * lambda) * pow(y, 0.4e1) +  (12 * mu + 2 * lambda) * pow(y, 0.3e1) +  (-6 * lambda - 3 * mu) * y * y -  (22 * mu) -  (9 * lambda)) * pow(z, 0.3e1) / 0.1000e4 + ( (82 * lambda + 128 * mu) * pow(y, 0.3e1) +  (-404 * mu - 192 * lambda) * y * y +  (28 * lambda - 92 * mu) * y -  (36 * mu)) * z * z / 0.1000e4 + ( (4 * mu + 2 * lambda) * pow(y, 0.5e1) +  (-30 * lambda - 6 * mu) * pow(y, 0.4e1) - 0.3e1 *  mu * y * y +  (574 * mu + 392 * lambda) * y -  (796 * mu) -  (120 * lambda)) * z / 0.1000e4 +  (17 * mu + 3 * lambda) * pow(y, 0.4e1) / 0.1000e4 +  (634 * mu + 232 * lambda) * pow(y, 0.3e1) / 0.1000e4 +  (100 * mu + 20 * lambda) * y * y / 0.1000e4 +  (22 * mu + 10 * lambda) * y / 0.1000e4 + 0.9e1 / 0.50e2 *  mu +  lambda / 0.10e2;
        rhs.col(2) = ((24 * mu + 12 * lambda) * z * z + 10 * lambda) * pow(y, 0.6e1) / 0.1000e4 + 0.3e1 / 0.125e3 * ( mu +  lambda / 0.2e1) *  z * x * pow(y, 0.5e1) + ( ((40 * lambda + 80 * mu) *  pow( z,  4)) +  ((55 * lambda + 104 * mu) * z * z) +  ((242 * mu + 124 * lambda) * z) + 0.24e2 *  lambda * pow(x, 0.3e1) +  (7 * lambda + 6 * mu) * x * x + 0.6e1 *  lambda * x +  lambda) * pow(y, 0.4e1) / 0.1000e4 + (0.96e2 * ( mu +  lambda / 0.2e1) * x *   pow( z,  3) + ( (72 * mu + 36 * lambda) * x * x +  (52 * mu + 26 * lambda) * x +  (207 * mu) -  (9 * lambda)) *  z +  (105 * lambda + 123 * mu) * x +  (20 * lambda)) * pow(y, 0.3e1) / 0.1000e4 + ( ((24 * mu + 12 * lambda) *  pow( z,  6)) +  ((50 * lambda + 100 * mu) *  pow( z,  4)) + (-0.216e3 *  mu * x * x - 0.84e2 *  mu * x +  (206 * lambda) +  (404 * mu)) *   pow( z,  3) + ( (192 * mu + 72 * lambda) * pow(x, 0.3e1) +  (60 * mu + 30 * lambda) * x * x +  (4 * mu) +  (6 * lambda)) *  (z * z) + ( (48 * lambda + 96 * mu) * pow(x, 0.3e1) + 0.20e2 *  mu * x * x + 0.30e2 *  mu * x +  (240 * mu) +  (42 * lambda)) *  z + 0.144e3 *  lambda * pow(x, 0.6e1) +  (-66 * lambda - 142 * mu) * x * x +  (52 * mu + 28 * lambda) * x +  (484 * mu) +  (244 * lambda)) * y * y / 0.1000e4 + (0.24e2 * ( mu +  lambda / 0.2e1) * x *   pow( z,  5) + ( (40 * mu + 20 * lambda) * x -  (399 * mu) -  (15 * lambda)) *   pow( z,  3) + (-0.252e3 *  mu * pow(x, 0.3e1) +  (762 * mu + 36 * lambda) * x * x +  (456 * mu + 230 * lambda) * x +  (20 * mu) +  (40 * lambda)) *  (z * z) + ( (864 * mu + 432 * lambda) * pow(x, 0.5e1) +  (24 * lambda + 168 * mu) * pow(x, 0.4e1) +  (12 * mu + 6 * lambda) * pow(x, 0.3e1) +  (144 * mu + 8 * lambda) * x +  (40 * mu) +  (20 * lambda)) *  z +  (72 * mu + 12 * lambda) * pow(x, 0.4e1) + 0.6e1 *  mu * pow(x, 0.3e1) + 0.12e2 *  mu * x +  (232 * mu) -  (84 * lambda)) * y / 0.1000e4 + 0.9e1 / 0.1000e4 *   pow( z,  6) *  lambda + ( (6 * mu + 3 * lambda) * x * x - 0.6e1 *  lambda * x +  lambda) *   pow( z,  4) / 0.1000e4 + (0.24e2 *  lambda * pow(x, 0.3e1) +  (180 * lambda) +  (360 * mu)) *   pow( z,  3) / 0.1000e4 + (0.144e3 *  lambda * pow(x, 0.6e1) +  (-1078 * mu + 6 * lambda) * x * x +  (-188 * mu + 36 * lambda) * x +  (46 * lambda) +  (246 * mu)) *  (z * z) / 0.1000e4 + (0.744e3 *  mu * pow(x, 0.3e1) +  (-56 * mu - 30 * lambda) * x * x +  (-94 * mu - 50 * lambda) * x +  (10 * lambda) +  (10 * mu)) *  z / 0.1000e4 +  (-1728 * mu - 864 * lambda) * pow(x, 0.5e1) / 0.1000e4 - 0.6e1 / 0.125e3 *  mu * pow(x, 0.4e1) +  (480 * mu + 120 * lambda) * pow(x, 0.3e1) / 0.1000e4 + 0.13e2 / 0.250e3 *  mu * x * x +  (-20 * mu - 12 * lambda) * x / 0.1000e4 +  mu / 0.25e2 -  lambda / 0.25e2;

        probl->rhs("SaintVenant", pts, other);
        Eigen::MatrixXd diff = (other - rhs);

        REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
    }
}