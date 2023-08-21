////////////////////////////////////////////////////////////////////////////////
#include <polyfem/assembler/Problem.hpp>
#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/Helmholtz.hpp>
#include <polyfem/assembler/LinearElasticity.hpp>
#include <polyfem/assembler/HookeLinearElasticity.hpp>
#include <polyfem/assembler/SaintVenantElasticity.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>

#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/Common.hpp>

#include <catch2/catch_test_macros.hpp>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;

const double k = 0.2;
const double lambda = 0.375, mu = 0.375;

json get_params()
{
	return {
		{"k", k},
		{"lambda", lambda},
		{"mu", mu}};
}

TEST_CASE("franke 2d", "[problem]")
{
	Eigen::MatrixXd pts(400, 2);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("Franke");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();

	Eigen::MatrixXd fx;

	// fun
	{

		auto cx2 = (9 * x - 2) * (9 * x - 2);
		auto cy2 = (9 * y - 2) * (9 * y - 2);

		auto cx1 = (9 * x + 1) * (9 * x + 1);
		auto cx7 = (9 * x - 7) * (9 * x - 7);

		auto cy3 = (9 * y - 3) * (9 * y - 3);
		auto cx4 = (9 * x - 4) * (9 * x - 4);

		auto cy7 = (9 * y - 7) * (9 * y - 7);

		Eigen::MatrixXd val = (3. / 4.) * exp(-(1. / 4.) * cx2 - (1. / 4.) * cy2) + (3. / 4.) * exp(-(1. / 49.) * cx1 - (9. / 10.) * y - 1. / 10.) + (1. / 2.) * exp(-(1. / 4.) * cx7 - (1. / 4.) * cy3) - (1. / 5.) * exp(-cx4 - cy7);
		fx = val;

		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	// grad
	{
		Eigen::MatrixXd gradX = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) * x + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) - 0.243e3 / 0.98e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.59e2 / 0.490e3 - 0.9e1 / 0.10e2 * y) * x - 0.27e2 / 0.98e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.59e2 / 0.490e3 - 0.9e1 / 0.10e2 * y) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) * x + 0.63e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2) * x - 0.72e2 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2);
		Eigen::MatrixXd gradY = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.59e2 / 0.490e3 - 0.9e1 / 0.10e2 * y) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.29e2 / 0.2e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2) * y - 0.126e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y + 0.72e2 * x + 0.126e3 * y - 0.65e2);

		probl->exact_grad(pts, 1, other);

		Eigen::MatrixXd diff = (other.col(0) - gradX);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		diff = (other.col(1) - gradY);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	// rhs
	{
		auto cx2 = (9 * x - 2) * (9 * x - 2);
		auto cy2 = (9 * y - 2) * (9 * y - 2);

		auto cx1 = (9 * x + 1) * (9 * x + 1);
		auto cx7 = (9 * x - 7) * (9 * x - 7);

		auto cy3 = (9 * y - 3) * (9 * y - 3);
		auto cx4 = (9 * x - 4) * (9 * x - 4);

		auto cy7 = (9 * y - 7) * (9 * y - 7);

		auto s1 = (-40.5 * x + 9) * (-40.5 * x + 9);
		auto s2 = (-162. / 49. * x - 18. / 49.) * (-162. / 49. * x - 18. / 49.);
		auto s3 = (-40.5 * x + 31.5) * (-40.5 * x + 31.5);
		auto s4 = (-162. * x + 72) * (-162 * x + 72);

		auto s5 = (-40.5 * y + 9) * (-40.5 * y + 9);
		auto s6 = (-40.5 * y + 13.5) * (-40.5 * y + 13.5);
		auto s7 = (-162 * y + 126) * (-162 * y + 126);

		Eigen::MatrixXd rhs = 243. / 4. * (-0.25 * cx2 - 0.25 * cy2).exp() - 0.75 * s1 * (-0.25 * cx2 - 0.25 * cy2).exp() + 36693. / 19600. * (-1. / 49. * cx1 - 0.9 * y - 0.1).exp() - 0.75 * s2 * (-1. / 49 * cx1 - 0.9 * y - 0.1).exp() + 40.5 * (-0.25 * cx7 - 0.25 * cy3).exp() - 0.5 * s3 * (-0.25 * cx7 - 0.25 * cy3).exp() - 324. / 5. * (-cx4 - cy7).exp() + 0.2 * s4 * (-cx4 - cy7).exp() - 0.75 * s5 * (-0.25 * cx2 - 0.25 * cy2).exp() - 0.5 * s6 * (-0.25 * cx7 - 0.25 * cy3).exp() + 0.2 * s7 * (-cx4 - cy7).exp();
		rhs *= -1;

		Units units;

		Helmholtz helmholtz;
		helmholtz.set_size(2);
		helmholtz.add_multimaterial(0, params, units);

		Laplacian laplacian;
		laplacian.set_size(2);
		laplacian.add_multimaterial(0, params, units);

		probl->rhs(laplacian, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		rhs += k * k * fx;
		probl->rhs(helmholtz, pts, 1, other);
		diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("franke 3d", "[problem]")
{
	Eigen::MatrixXd pts(400, 3);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("Franke");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();
	auto z = pts.col(2).array();

	Eigen::MatrixXd fx;

	/// fun
	{
		auto cx2 = (9 * x - 2) * (9 * x - 2);
		auto cy2 = (9 * y - 2) * (9 * y - 2);
		auto cz2 = (9 * z - 2) * (9 * z - 2);

		auto cx1 = (9 * x + 1) * (9 * x + 1);
		auto cx7 = (9 * x - 7) * (9 * x - 7);

		auto cy3 = (9 * y - 3) * (9 * y - 3);
		auto cx4 = (9 * x - 4) * (9 * x - 4);
		auto cy7 = (9 * y - 7) * (9 * y - 7);

		auto cz5 = (9 * z - 5) * (9 * z - 5);

		Eigen::MatrixXd val =
			3. / 4. * exp(-1. / 4. * cx2 - 1. / 4. * cy2 - 1. / 4. * cz2) + 3. / 4. * exp(-1. / 49. * cx1 - 9. / 10. * y - 1. / 10. - 9. / 10. * z - 1. / 10.) + 1. / 2. * exp(-1. / 4. * cx7 - 1. / 4. * cy3 - 1. / 4. * cz5) - 1. / 5. * exp(-cx4 - cy7 - cz5);
		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		fx = val;
	}

	// grad
	{
		Eigen::MatrixXd gradX = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * x + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.243e3 / 0.98e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) * x - 0.27e2 / 0.98e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) * x + 0.63e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y - 0.81e2 * z * z + 0.72e2 * x + 0.126e3 * y + 0.90e2 * z - 0.90e2) * x - 0.72e2 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y - 0.81e2 * z * z + 0.72e2 * x + 0.126e3 * y + 0.90e2 * z - 0.90e2);
		Eigen::MatrixXd gradY = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y - 0.81e2 * z * z + 0.72e2 * x + 0.126e3 * y + 0.90e2 * z - 0.90e2) * y - 0.126e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y - 0.81e2 * z * z + 0.72e2 * x + 0.126e3 * y + 0.90e2 * z - 0.90e2);
		Eigen::MatrixXd gradZ = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * z + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) - 0.81e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) * z + 0.45e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) + 0.162e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.81e2 * y * y - 0.81e2 * z * z + 0.72e2 * x + 0.126e3 * y + 0.90e2 * z - 0.90e2) * z - 0.18e2 * exp(-0.81e2 * x * x - 0.81e2 * y * y - 0.81e2 * z * z + 0.72e2 * x + 0.126e3 * y + 0.90e2 * z - 0.90e2);

		probl->exact_grad(pts, 1, other);

		Eigen::MatrixXd diff = (other.col(0) - gradX);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		diff = (other.col(1) - gradY);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		diff = (other.col(2) - gradZ);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	// rhs
	{
		Eigen::MatrixXd rhs = (787648050 * x * x + 787648050 * y * y + 787648050 * z * z - 1225230300 * x - 525098700 * y - 875164500 * z + 748751850) * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.4e1 * y * y + 0.27e2 / 0.2e1 * y - 0.81e2 / 0.4e1 * z * z + 0.45e2 / 0.2e1 * z) / 0.960400e6 + (1181472075 * x * x + 1181472075 * y * y + 1181472075 * z * z - 525098700 * x - 525098700 * y - 525098700 * z + 87516450) * exp(-0.81e2 / 0.4e1 * x * x + (9 * x) - 0.3e1 - 0.81e2 / 0.4e1 * y * y + (9 * y) - 0.81e2 / 0.4e1 * z * z + (9 * z)) / 0.960400e6 + (-5040947520 * x * x - 5040947520 * y * y - 5040947520 * z * z + 4480842240 * x + 7841473920 * y + 5601052800 * z - 5507701920) * exp((-81 * x * x - 81 * y * y - 81 * z * z + 72 * x + 126 * y + 90 * z - 90)) / 0.960400e6 + 0.19683e5 / 0.2401e4 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) * ((x * x) + 0.2e1 / 0.9e1 * x - 0.2299e4 / 0.16200e5);
		// rhs*=-1;

		Units units;

		Helmholtz helmholtz;
		helmholtz.set_size(3);
		helmholtz.add_multimaterial(0, params, units);

		Laplacian laplacian;
		laplacian.set_size(3);
		laplacian.add_multimaterial(0, params, units);

		probl->rhs(laplacian, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		rhs += k * k * fx;
		probl->rhs(helmholtz, pts, 1, other);
		diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("linear", "[problem]")
{
	Eigen::MatrixXd pts(400, 2);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("Linear");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();

	{
		Eigen::MatrixXd val = x;
		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	{

		Eigen::MatrixXd gradX = x;
		Eigen::MatrixXd gradY = x;
		gradX.setOnes();
		gradY.setZero();

		probl->exact_grad(pts, 1, other);

		Eigen::MatrixXd diff = (other.col(0) - gradX);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		diff = (other.col(1) - gradY);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	Units units;

	Helmholtz helmholtz;
	helmholtz.set_size(2);
	helmholtz.add_multimaterial(0, params, units);

	Laplacian laplacian;
	laplacian.set_size(2);
	laplacian.add_multimaterial(0, params, units);

	{
		Eigen::MatrixXd rhs = x;
		rhs.setZero();

		probl->rhs(laplacian, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	{
		Eigen::MatrixXd rhs = k * k * x;

		probl->rhs(helmholtz, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("quadratic", "[problem]")
{
	Eigen::MatrixXd pts(400, 2);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("Quadratic");

	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();

	{
		Eigen::MatrixXd val = x * x;
		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	{

		Eigen::MatrixXd gradX = 2 * x;
		Eigen::MatrixXd gradY = x;
		gradY.setZero();

		probl->exact_grad(pts, 1, other);

		Eigen::MatrixXd diff = (other.col(0) - gradX);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		diff = (other.col(1) - gradY);
		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	Units units;

	Helmholtz helmholtz;
	helmholtz.set_size(2);
	helmholtz.add_multimaterial(0, params, units);

	Laplacian laplacian;
	laplacian.set_size(2);
	laplacian.add_multimaterial(0, params, units);

	{
		Eigen::MatrixXd rhs = x;
		rhs.setConstant(2);

		probl->rhs(laplacian, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	{
		Eigen::MatrixXd rhs = k * k * x * x + 2;

		probl->rhs(helmholtz, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("zero bc 2d", "[problem]")
{
	Eigen::MatrixXd pts(400, 2);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("Zero_BC");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();

	{
		Eigen::MatrixXd val = (1 - x) * x * x * y * (1 - y) * (1 - y);
		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	Units units;

	Laplacian laplacian;
	laplacian.set_size(2);
	laplacian.add_multimaterial(0, params, units);

	{
		Eigen::MatrixXd rhs = -4 * x * y * (1 - y) * (1 - y) + 2 * (1 - x) * y * (1 - y) * (1 - y) - 4 * (1 - x) * x * x * (1 - y) + 2 * (1 - x) * x * x * y;

		probl->rhs(laplacian, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("zero bc 3d", "[problem]")
{
	Eigen::MatrixXd pts(40, 3);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("Zero_BC");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();
	auto z = pts.col(2).array();

	{
		Eigen::MatrixXd val = (1 - x) * x * x * y * (1 - y) * (1 - y) * z * (1 - z);
		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
	Units units;

	Laplacian laplacian;
	laplacian.set_size(3);
	laplacian.add_multimaterial(0, params, units);

	{
		Eigen::MatrixXd rhs = (0.2e1 * pow(x, 0.3e1) - 0.2e1 * x * x + (6 * z * z - 6 * z) * x - (2 * z * z) + (2 * z)) * pow(y, 0.3e1) + (-0.4e1 * pow(x, 0.3e1) + 0.4e1 * x * x + (-12 * z * z + 12 * z) * x + (4 * z * z) - (4 * z)) * y * y + ((6 * z * z - 6 * z + 2) * pow(x, 0.3e1) + (-6 * z * z + 6 * z - 2) * x * x + (6 * z * z - 6 * z) * x - (2 * z * z) + (2 * z)) * y - 0.4e1 * x * x * z * (z - 1) * (x - 0.1e1);
		probl->rhs(laplacian, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("elasticity 2d", "[problem]")
{
	Eigen::MatrixXd pts(400, 2);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("ElasticExact");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();

	// fun
	{
		Eigen::MatrixXd val(pts.rows(), pts.cols());
		val.col(0) = (y * y * y + x * x + x * y) / 50.;
		val.col(1) = (3 * x * x * x * x + x * y * y + x) / 50.;

		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
	Units units;

	LinearElasticity le;
	le.set_size(2);
	le.add_multimaterial(0, params, units);

	HookeLinearElasticity ho;
	ho.set_size(2);
	ho.add_multimaterial(0, params, units);

	SaintVenantElasticity sv;
	sv.set_size(2);
	sv.add_multimaterial(0, params, units);

	NeoHookeanElasticity nh;
	nh.set_size(2);
	nh.add_multimaterial(0, params, units);

	// rhs
	{
		Eigen::MatrixXd rhs(pts.rows(), pts.cols());
		rhs.col(0) = (1. / 25. * (y + 1)) * lambda + (1. / 25. * (4. * y + 2)) * mu;
		rhs.col(1) = (1. / 50. * (36 * x * x + 4 * x + 1)) * mu + (1. / 25.) * x * lambda + (1. / 50.) * lambda;

		probl->rhs(le, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		probl->rhs(ho, pts, 1, other);
		diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	// rhs
	{
		Eigen::MatrixXd rhs(pts.rows(), pts.cols());
		rhs.col(0) = (432 * lambda * y + 1008 * lambda + 2016 * mu) * pow(x, 0.6e1) / 0.125000e6 + ((864 * mu + 432 * lambda) * y + 43248 * mu + 21600 * lambda) * pow(x, 0.5e1) / 0.125000e6 + ((24 * lambda + 168 * mu) * y + 1200 * mu) * pow(x, 0.4e1) / 0.125000e6 + ((144 * lambda + 288 * mu) * pow(y, 3) + (192 * mu + 96 * lambda) * y * y + (76 * lambda + 8 * mu) * y + 3192 * mu + 96 * lambda) * pow(x, 0.3e1) / 0.125000e6 + ((120 * mu + 60 * lambda) * pow(y, 3) + (9012 * mu + 1812 * lambda) * y * y + (114 * mu + 57 * lambda) * y + 3838 * mu + 1917 * lambda) * x * x / 0.125000e6 + ((120 * mu + 60 * lambda) * pow(y, 3) + (1130 * lambda + 2160 * mu) * y * y + (1436 * mu + 1016 * lambda) * y + 1500 * mu + 700 * lambda) * x / 0.125000e6 + ((90 * lambda + 168 * mu) * pow(y, 5)) / 0.125000e6 + ((20 * mu + 10 * lambda) * pow(y, 4)) / 0.125000e6 + ((24 * mu + 21 * lambda) * pow(y, 3)) / 0.125000e6 + ((1410 * mu + 705 * lambda) * y * y) / 0.125000e6 + ((20600 * mu + 5303 * lambda) * y) / 0.125000e6 + 0.5051e4 / 0.62500e5 * mu + 0.5001e4 / 0.125000e6 * lambda;
		rhs.col(1) = (15552 * mu + 7776 * lambda) * pow(x, 0.8e1) / 0.125000e6 + (144 * lambda + 288 * mu) * pow(x, 0.7e1) / 0.125000e6 + 0.324e3 / 0.15625e5 * (y * y + 1) * (mu + lambda / 0.2e1) * pow(x, 0.5e1) + ((192 * lambda + 384 * mu) * y * y + 144 * mu * y + 300 * mu + 174 * lambda) * pow(x, 0.4e1) / 0.125000e6 + ((156 * lambda + 132 * mu) * y * y + (15792 * mu + 6096 * lambda) * y + 9602 * mu + 4805 * lambda) * pow(x, 0.3e1) / 0.125000e6 + ((108 * mu + 216 * lambda) * pow(y, 4) + (144 * lambda + 288 * mu) * y * y + (4816 * mu + 2408 * lambda) * y + 90108 * mu + 254 * lambda) * x * x / 0.125000e6 + ((54 * lambda + 108 * mu) * pow(y, 4) + 48 * mu * pow(y, 3) + (35 * mu + 18 * lambda) * y * y + (500 * lambda + 912 * mu) * y + 10211 * mu + 5106 * lambda) * x / 0.125000e6 + ((3 * lambda + 21 * mu) * pow(y, 4)) / 0.125000e6 + ((3104 * mu + 1102 * lambda) * pow(y, 3)) / 0.125000e6 + ((509 * mu + 103 * lambda) * y * y) / 0.125000e6 + ((754 * mu + 252 * lambda) * y) / 0.125000e6 + 0.27e2 / 0.1250e4 * mu + 0.13e2 / 0.625e3 * lambda;

		probl->rhs(sv, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	// rhs
	{
		Eigen::MatrixXd rhs(pts.rows(), pts.cols());
		rhs.col(0) = (-0.43200e5 * lambda * (pow(x, 0.6e1) * y - pow(x, 0.5e1) / 0.18e2 - 0.5e1 / 0.36e2 * pow(x, 0.4e1) * y - 0.25e2 / 0.18e2 * pow(x, 0.4e1) + pow(x, 0.3e1) * y / 0.6e1 - 0.125e3 / 0.36e2 * pow(x, 0.3e1) - 0.449e3 / 0.72e2 * x * x * y * y - x * x / 0.216e3 + 0.25e2 / 0.216e3 * x * y * y + 0.149e3 / 0.216e3 * x * y - 0.25e2 / 0.216e3 * x + pow(y, 0.5e1) / 0.72e2 + pow(y, 0.3e1) / 0.48e2 + 0.2503e4 / 0.432e3 * y + 0.1225e4 / 0.216e3) * log(-0.3e1 * pow(y, 0.4e1) + (-0.36e2 * pow(x, 0.3e1) + x - 0.3e1) * y * y + (0.4e1 * x * x + 0.100e3 * x + 0.50e2) * y - 0.12e2 * pow(x, 0.4e1) + 0.99e2 * x + 0.2500e4) + 0.86400e5 * lambda * (pow(x, 0.6e1) * y - pow(x, 0.5e1) / 0.18e2 - 0.5e1 / 0.36e2 * pow(x, 0.4e1) * y - 0.25e2 / 0.18e2 * pow(x, 0.4e1) + pow(x, 0.3e1) * y / 0.6e1 - 0.125e3 / 0.36e2 * pow(x, 0.3e1) - 0.449e3 / 0.72e2 * x * x * y * y - x * x / 0.216e3 + 0.25e2 / 0.216e3 * x * y * y + 0.149e3 / 0.216e3 * x * y - 0.25e2 / 0.216e3 * x + pow(y, 0.5e1) / 0.72e2 + pow(y, 0.3e1) / 0.48e2 + 0.2503e4 / 0.432e3 * y + 0.1225e4 / 0.216e3) * (log(0.2e1) + 0.2e1 * log(0.5e1)) + 0.144e3 / 0.25e2 * mu * (0.3e1 * y + 0.1e1) * pow(x, 0.8e1) + 0.864e3 / 0.25e2 * y * y * mu * (0.3e1 * y + 0.1e1) * pow(x, 0.7e1) + 0.48e2 / 0.25e2 * y * (0.81e2 * mu * pow(y, 0.4e1) + 0.27e2 * mu * pow(y, 0.3e1) - 0.6e1 * mu * y + 0.22500e5 * lambda + 0.22498e5 * mu) * pow(x, 0.6e1) - 0.24e2 / 0.25e2 * (0.36e2 * mu * pow(y, 0.4e1) + 0.15e2 * mu * pow(y, 0.3e1) + 0.301e3 * mu * y * y + 0.397e3 * mu * y + 0.2500e4 * lambda + 0.2599e4 * mu) * pow(x, 0.5e1) - 0.16e2 / 0.5e1 * (0.270e3 * mu * pow(y, 0.4e1) + 0.354e3 * mu * pow(y, 0.3e1) + 0.133e3 * mu * y * y + 0.1875e4 * lambda * y + 0.4140e4 * mu * y + 0.18750e5 * lambda + 0.19500e5 * mu) * pow(x, 0.4e1) + 0.8e1 / 0.25e2 * (0.81e2 * mu * pow(y, 0.7e1) + 0.27e2 * mu * pow(y, 0.6e1) + 0.81e2 * mu * pow(y, 0.5e1) - 0.1320e4 * mu * pow(y, 0.4e1) - 0.67649e5 * mu * pow(y, 0.3e1) - 0.22103e5 * mu * y * y + 0.22500e5 * lambda * y + 0.22599e5 * mu * y - 0.468750e6 * lambda - 0.468750e6 * mu) * pow(x, 0.3e1) - (0.72e2 * mu * pow(y, 0.6e1) + 0.21e2 * mu * pow(y, 0.5e1) - 0.529e3 * mu * pow(y, 0.4e1) - 0.31970e5 * mu * pow(y, 0.3e1) + 0.6735000e7 * lambda * y * y + 0.6605002e7 * mu * y * y - 0.69203e5 * mu * y + 0.5000e4 * lambda - 0.4801e4 * mu) * x * x / 0.25e2 + 0.2e1 / 0.25e2 * (-0.9e1 * mu * pow(y, 0.7e1) - 0.903e3 * mu * pow(y, 0.6e1) - 0.1200e4 * mu * pow(y, 0.5e1) - 0.1050e4 * mu * pow(y, 0.4e1) + 0.21359e5 * mu * pow(y, 0.3e1) + 0.62500e5 * lambda * y * y + 0.834553e6 * mu * y * y + 0.372500e6 * lambda * y + 0.1369950e7 * mu * y - 0.62500e5 * lambda + 0.185000e6 * mu) * x + 0.27e2 / 0.25e2 * mu * pow(y, 0.9e1) + 0.9e1 / 0.25e2 * mu * pow(y, 0.8e1) + 0.54e2 / 0.25e2 * mu * pow(y, 0.7e1) - 0.882e3 / 0.25e2 * mu * pow(y, 0.6e1) + 0.3e1 / 0.25e2 * (-0.10091e5 * mu + 0.5000e4 * lambda) * pow(y, 0.5e1) - 0.15891e5 / 0.25e2 * mu * pow(y, 0.4e1) + 0.36e2 * (-0.17e2 * mu + 0.25e2 * lambda) * pow(y, 0.3e1) + 0.29500e5 * mu * y * y + 0.100e3 * (0.10103e5 * mu + 0.2503e4 * lambda) * y + 0.495000e6 * mu + 0.245000e6 * lambda) * pow(0.36e2 * pow(x, 0.3e1) * y * y + 0.12e2 * pow(x, 0.4e1) + 0.3e1 * pow(y, 0.4e1) - 0.4e1 * x * x * y - x * y * y - 0.100e3 * x * y + 0.3e1 * y * y - 0.99e2 * x - 0.50e2 * y - 0.2500e4, -0.2e1);
		rhs.col(1) = (0.7200e4 * (pow(x, 0.4e1) * y - pow(x, 0.4e1) / 0.3e1 - 0.5e1 / 0.4e1 * pow(x, 0.3e1) * y * y + 0.25e2 * pow(x, 0.3e1) * y - pow(x, 0.3e1) / 0.18e2 - 0.9e1 / 0.4e1 * x * x * pow(y, 0.4e1) - 0.25e2 / 0.9e1 * x * x + x * pow(y, 0.3e1) / 0.3e1 - x * y * y / 0.144e3 - 0.11e2 / 0.18e2 * x * y - 0.1667e4 / 0.48e2 * x + 0.5e1 / 0.48e2 * pow(y, 0.4e1) + 0.25e2 / 0.4e1 * pow(y, 0.3e1) + 0.101e3 / 0.48e2 * y * y + 0.125e3 / 0.72e2 * y - 0.625e3 / 0.36e2) * lambda * log(-0.3e1 * pow(y, 0.4e1) + (-0.36e2 * pow(x, 0.3e1) + x - 0.3e1) * y * y + (0.4e1 * x * x + 0.100e3 * x + 0.50e2) * y - 0.12e2 * pow(x, 0.4e1) + 0.99e2 * x + 0.2500e4) - 0.14400e5 * (pow(x, 0.4e1) * y - pow(x, 0.4e1) / 0.3e1 - 0.5e1 / 0.4e1 * pow(x, 0.3e1) * y * y + 0.25e2 * pow(x, 0.3e1) * y - pow(x, 0.3e1) / 0.18e2 - 0.9e1 / 0.4e1 * x * x * pow(y, 0.4e1) - 0.25e2 / 0.9e1 * x * x + x * pow(y, 0.3e1) / 0.3e1 - x * y * y / 0.144e3 - 0.11e2 / 0.18e2 * x * y - 0.1667e4 / 0.48e2 * x + 0.5e1 / 0.48e2 * pow(y, 0.4e1) + 0.25e2 / 0.4e1 * pow(y, 0.3e1) + 0.101e3 / 0.48e2 * y * y + 0.125e3 / 0.72e2 * y - 0.625e3 / 0.36e2) * lambda * (log(0.2e1) + 0.2e1 * log(0.5e1)) + 0.2592e4 / 0.25e2 * mu * pow(x, 0.10e2) + 0.144e3 / 0.25e2 * mu * (0.108e3 * y * y + 0.1e1) * pow(x, 0.9e1) + 0.864e3 / 0.25e2 * (0.27e2 * pow(y, 0.3e1) + y - 0.2e1) * y * mu * pow(x, 0.8e1) + 0.48e2 / 0.25e2 * (0.27e2 * pow(y, 0.4e1) - 0.108e3 * pow(y, 0.3e1) - 0.9e1 * y * y - 0.902e3 * y - 0.891e3) * mu * pow(x, 0.7e1) - 0.24e2 / 0.25e2 * (0.5412e4 * pow(y, 0.3e1) + 0.5281e4 * y * y + 0.1000e4 * y + 0.45099e5) * mu * pow(x, 0.6e1) + 0.16e2 / 0.25e2 * (0.243e3 * pow(y, 0.6e1) + 0.243e3 * pow(y, 0.4e1) - 0.4491e4 * pow(y, 0.3e1) - 0.202040e6 * y * y + 0.816e3 * y - 0.3750e4) * mu * pow(x, 0.5e1) - 0.2e1 / 0.25e2 * (-0.108e3 * mu * pow(y, 0.6e1) + 0.216e3 * mu * pow(y, 0.5e1) - 0.117e3 * mu * pow(y, 0.4e1) + 0.212e3 * mu * pow(y, 0.3e1) - 0.5782e4 * mu * y * y + 0.90000e5 * lambda * y - 0.268596e6 * mu * y - 0.30000e5 * lambda - 0.118209e6 * mu) * pow(x, 0.4e1) + (-0.108e3 * mu * pow(y, 0.6e1) - 0.10824e5 * mu * pow(y, 0.5e1) - 0.10799e5 * mu * pow(y, 0.4e1) - 0.8824e4 * mu * pow(y, 0.3e1) + 0.225000e6 * lambda * y * y + 0.494906e6 * mu * y * y - 0.4500000e7 * lambda * y + 0.4718000e7 * mu * y + 0.10000e5 * lambda + 0.8929801e7 * mu) * pow(x, 0.3e1) / 0.25e2 + 0.2e1 / 0.25e2 * (0.81e2 * mu * pow(y, 0.8e1) + 0.159e3 * mu * pow(y, 0.6e1) - 0.3000e4 * mu * pow(y, 0.5e1) + 0.202500e6 * lambda * pow(y, 0.4e1) + 0.67281e5 * mu * pow(y, 0.4e1) - 0.2950e4 * mu * pow(y, 0.3e1) - 0.105297e6 * mu * y * y + 0.2504950e7 * mu * y + 0.250000e6 * lambda + 0.56747500e8 * mu) * x * x - (-0.9e1 * mu * pow(y, 0.8e1) - 0.18e2 * mu * pow(y, 0.6e1) + 0.300e3 * mu * pow(y, 0.5e1) + 0.14991e5 * mu * pow(y, 0.4e1) + 0.60000e5 * lambda * pow(y, 0.3e1) + 0.60300e5 * mu * pow(y, 0.3e1) - 0.1250e4 * lambda * y * y + 0.11250e5 * mu * y * y - 0.110000e6 * lambda * y - 0.360000e6 * mu * y - 0.6251250e7 * lambda - 0.12501250e8 * mu) * x / 0.25e2 - 0.50e2 * (0.15e2 * pow(y, 0.4e1) + 0.900e3 * pow(y, 0.3e1) + 0.303e3 * y * y + 0.250e3 * y - 0.2500e4) * (mu + lambda)) * pow(0.36e2 * pow(x, 0.3e1) * y * y + 0.12e2 * pow(x, 0.4e1) + 0.3e1 * pow(y, 0.4e1) - 0.4e1 * x * x * y - x * y * y - 0.100e3 * x * y + 0.3e1 * y * y - 0.99e2 * x - 0.50e2 * y - 0.2500e4, -0.2e1);

		probl->rhs(nh, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}

TEST_CASE("elasticity 3d", "[problem]")
{
	Eigen::MatrixXd pts(400, 3);
	Eigen::MatrixXd other;
	pts.setRandom();

	const auto &probl = ProblemFactory::factory().get_problem("ElasticExact");
	const json params = get_params();

	auto x = pts.col(0).array();
	auto y = pts.col(1).array();
	auto z = pts.col(2).array();

	/// fun
	{
		Eigen::MatrixXd val(pts.rows(), pts.cols());
		val.col(0) = (x * y + x * x + y * y * y + 6 * z) / 80.;
		val.col(1) = (z * x - z * z * z + x * y * y + 3 * x * x * x * x) / 80.;
		val.col(2) = (x * y * z + z * z * y * y - 2 * x) / 80.;

		probl->exact(pts, 1, other);
		Eigen::MatrixXd diff = (other - val);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	Units units;

	LinearElasticity le;
	le.set_size(3);
	le.add_multimaterial(0, params, units);

	HookeLinearElasticity ho;
	ho.set_size(3);
	ho.add_multimaterial(0, params, units);

	SaintVenantElasticity sv;
	sv.set_size(3);
	sv.add_multimaterial(0, params, units);

	// rhs
	{
		Eigen::MatrixXd rhs(pts.rows(), pts.cols());
		rhs.col(0) = (1. / 80 * (3 * y + 2)) * lambda + (1. / 80) * mu * (9 * y + 4);
		rhs.col(1) = (1. / 80 * (36 * x * x + 5 * x + 1 + (4 * y - 6) * z)) * mu + (3. / 80. * ((4. / 3.) * y * z + x + 1. / 3.)) * lambda;
		rhs.col(2) = (1. / 40. * (2 * y * y + z * z)) * mu + (1. / 40.) * y * y * lambda;

		probl->rhs(le, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);

		probl->rhs(ho, pts, 1, other);
		diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}

	// rhs
	{
		Eigen::MatrixXd rhs(pts.rows(), pts.cols());
		rhs.col(0) = ((36 * lambda + 36 * mu) * z * z + 168 * mu + 90 * lambda) * pow(y, 0.5e1) / 0.512000e6 + (0.24e2 * (mu + 0.5e1 / 0.4e1 * lambda) * z * x + (4 * lambda * z * z) + ((52 * mu + 26 * lambda) * z) + (20 * mu) + (10 * lambda)) * pow(y, 0.4e1) / 0.512000e6 + ((144 * lambda + 288 * mu) * pow(x, 0.3e1) + (123 * mu + 66 * lambda) * x * x + ((8 * lambda + 12 * mu) * z * z + (-28 * mu + 8 * lambda) * z + 145 * mu + 73 * lambda) * x + ((48 * mu + 24 * lambda) * pow(z, 4)) + ((3 * mu + 6 * lambda) * z * z) + ((2092 * lambda + 1280 * mu) * z) + (14 * mu) + (9 * lambda)) * pow(y, 0.3e1) / 0.512000e6 + ((192 * mu + 96 * lambda) * pow(x, 0.3e1) + ((6 * lambda + 8 * mu) * z + 14414 * mu + 2895 * lambda) * x * x + ((36 * mu + 18 * lambda) * pow(z, 3) - 12 * mu * z + 3732 * mu + 2590 * lambda) * x + (4 * lambda * pow(z, 4)) + ((80 * mu + 50 * lambda) * pow(z, 3)) + ((2 * mu + lambda) * z * z) + ((310 * lambda - 1916 * mu) * z) + (3932 * mu) + (2169 * lambda)) * y * y / 0.512000e6 + (0.432e3 * lambda * pow(x, 0.6e1) + (864 * mu + 432 * lambda) * pow(x, 0.5e1) + (24 * lambda + 168 * mu) * pow(x, 0.4e1) + ((-72 * mu + 72 * lambda) * z + 9 * mu + 5 * lambda) * pow(x, 0.3e1) + ((6 * mu + 3 * lambda) * z * z + (60 * mu + 36 * lambda) * z + 42 * mu + 24 * lambda) * x * x + ((8 * mu + 4 * lambda) * pow(z, 4) + (8 * lambda + 8 * mu) * pow(z, 3) + (20 * lambda + 52 * mu) * z * z + (322 * lambda + 328 * mu) * z + 2429 * mu + 1935 * lambda) * x + (27 * lambda * pow(z, 4)) + ((320 * mu + 160 * lambda) * pow(z, 3)) - (21 * mu * z * z) + ((-8 * mu - 4 * lambda) * z) + (58764 * mu) + (19788 * lambda)) * y / 0.512000e6 + (2016 * mu + 1008 * lambda) * pow(x, 0.6e1) / 0.512000e6 + (69168 * mu + 34560 * lambda) * pow(x, 0.5e1) / 0.512000e6 + (-144 * mu * z + 1920 * mu) * pow(x, 0.4e1) / 0.512000e6 + ((-5568 * mu + 96 * lambda) * z + 5088 * mu + 72 * lambda) * pow(x, 0.3e1) / 0.512000e6 + ((4 * mu + 2 * lambda) * pow(z, 3) + (-646 * mu + 3 * lambda) * z * z + (2886 * lambda + 5770 * mu) * z + 448 * mu + 260 * lambda) * x * x / 0.512000e6 + ((18 * mu + 68 * lambda) * z * z + (-398 * mu - 38 * lambda) * z + 2320 * mu + 1200 * lambda) * x / 0.512000e6 + 0.9e1 / 0.512000e6 * lambda * pow(z, 4) + ((216 * mu + 108 * lambda) * pow(z, 3)) / 0.512000e6 + ((-78 * mu - 239 * lambda) * z * z) / 0.512000e6 + ((166 * mu + 6 * lambda) * z) / 0.512000e6 + 0.321e3 / 0.6400e4 * mu + 0.321e3 / 0.12800e5 * lambda;
		rhs.col(1) = (15552 * mu + 7776 * lambda) * pow(x, 0.8e1) / 0.512000e6 + (144 * lambda + 288 * mu) * pow(x, 0.7e1) / 0.512000e6 - 0.27e2 / 0.16000e5 * (mu + lambda / 0.2e1) * z * pow(x, 0.6e1) + 0.81e2 / 0.16000e5 * (mu + lambda / 0.2e1) * (y * y + z) * pow(x, 0.5e1) + ((12 * mu + 30 * lambda) * z * z + (48 * mu + 24 * lambda) * z + (396 * mu + 222 * lambda) * y * y + 0.144e3 * mu * y + (336 * mu) + (192 * lambda)) * pow(x, 0.4e1) / 0.512000e6 + (0.48e2 * y * (mu + 2 * lambda) * pow(z, 0.3e1) + (-550 * mu - 251 * lambda) * z * z + ((48 * mu + 96 * lambda) * pow(y, 0.3e1) + (-144 * mu - 72 * lambda) * y * y + mu + lambda) * z + (86 * mu + 159 * lambda) * y * y + (26112 * mu + 13536 * lambda) * y + (15364 * mu) + (7686 * lambda)) * pow(x, 0.3e1) / 0.512000e6 + ((0.72e2 * lambda * y * y + (162 * lambda) + (324 * mu)) * pow(z, 0.4e1) + ((16 * mu + 8 * lambda) * y - (3 * mu) - (6 * lambda)) * pow(z, 0.3e1) + (0.72e2 * lambda * pow(y, 0.4e1) + (36 * mu + 18 * lambda) * y * y + (6 * lambda + 8 * mu) * y + (96 * mu) + (48 * lambda)) * z * z + ((16 * lambda + 16 * mu) * pow(y, 0.3e1) + (5853 * lambda + 186 * mu) * y * y + (-144 * mu - 72 * lambda) * y - (18 * mu) - (24 * lambda)) * z + (108 * mu + 216 * lambda) * pow(y, 0.4e1) + (4 * mu + 2 * lambda) * pow(y, 0.3e1) + (72 * mu + 36 * lambda) * y * y + (7936 * mu + 4248 * lambda) * y + (230544 * mu) + (1040 * lambda)) * x * x / 0.512000e6 + (((24 * mu + 12 * lambda) * y * y + (-24 * mu - 30 * lambda) * y + (18 * mu) + (9 * lambda)) * pow(z, 0.4e1) + ((8 * lambda + 12 * mu) * y * y + (429 * mu) + (215 * lambda)) * pow(z, 0.3e1) + ((20 * lambda + 24 * mu) * pow(y, 0.4e1) + (-36 * mu - 18 * lambda) * pow(y, 0.3e1) + (4 * lambda + 7 * mu) * y * y + (162 * mu) + lambda) * z * z + ((8 * mu + 4 * lambda) * pow(y, 0.4e1) + (1434 * lambda + 1304 * mu) * y * y + (-1460 * lambda - 2880 * mu) * y + (13 * mu) - (953 * lambda)) * z + (109 * mu + 55 * lambda) * pow(y, 0.4e1) + 0.48e2 * mu * pow(y, 0.3e1) + (339 * mu + 170 * lambda) * y * y + (1470 * mu + 798 * lambda) * y + (32356 * mu) + (19400 * lambda)) * x / 0.512000e6 + ((-36 * mu - 36 * lambda) * y * y - (162 * mu) - (81 * lambda)) * pow(z, 0.5e1) / 0.512000e6 + ((644 * mu + 322 * lambda) * y - (480 * mu)) * pow(z, 0.4e1) / 0.512000e6 + ((-48 * mu - 24 * lambda) * pow(y, 0.4e1) + (12 * mu + 2 * lambda) * pow(y, 0.3e1) + (-3 * mu - 6 * lambda) * y * y - (22 * mu) - (9 * lambda)) * pow(z, 0.3e1) / 0.512000e6 + ((642 * lambda + 968 * mu) * pow(y, 0.3e1) + (-2924 * mu - 1452 * lambda) * y * y + (98 * lambda - 22 * mu) * y - (36 * mu)) * z * z / 0.512000e6 + ((4 * mu + 2 * lambda) * pow(y, 0.5e1) + (-30 * lambda - 6 * mu) * pow(y, 0.4e1) - 0.3e1 * mu * y * y + (25522 * lambda + 26964 * mu) * y - (38456 * mu) - (120 * lambda)) * z / 0.512000e6 + (17 * mu + 3 * lambda) * pow(y, 0.4e1) / 0.512000e6 + (5044 * mu + 1842 * lambda) * pow(y, 0.3e1) / 0.512000e6 + (800 * mu + 160 * lambda) * y * y / 0.512000e6 + (92 * mu + 80 * lambda) * y / 0.512000e6 + 0.11e2 / 0.800e3 * mu + lambda / 0.80e2;
		rhs.col(2) = ((24 * mu + 12 * lambda) * z * z + 10 * lambda) * pow(y, 0.6e1) / 0.512000e6 + 0.3e1 / 0.64000e5 * (mu + lambda / 0.2e1) * z * x * pow(y, 0.5e1) + (((80 * mu + 40 * lambda) * pow(z, 4)) + ((55 * lambda + 104 * mu) * z * z) + ((1922 * mu + 964 * lambda) * z) + 0.24e2 * lambda * pow(x, 0.3e1) + (7 * lambda + 6 * mu) * x * x + 0.6e1 * lambda * x + lambda) * pow(y, 0.4e1) / 0.512000e6 + (0.96e2 * (mu + lambda / 0.2e1) * x * pow(z, 3) + ((72 * mu + 36 * lambda) * x * x + (52 * mu + 26 * lambda) * x + (207 * mu) - (9 * lambda)) * z + (805 * lambda + 963 * mu) * x + (160 * lambda)) * pow(y, 0.3e1) / 0.512000e6 + (((24 * mu + 12 * lambda) * pow(z, 6)) + ((50 * lambda + 100 * mu) * pow(z, 4)) + (-0.216e3 * mu * x * x - 0.84e2 * mu * x + (1606 * lambda) + (3204 * mu)) * pow(z, 3) + ((192 * mu + 72 * lambda) * pow(x, 0.3e1) + (60 * mu + 30 * lambda) * x * x + (4 * mu) + (6 * lambda)) * (z * z) + ((48 * lambda + 96 * mu) * pow(x, 0.3e1) + 0.20e2 * mu * x * x + 0.30e2 * mu * x + (1640 * mu) + (322 * lambda)) * z + 0.144e3 * lambda * pow(x, 0.6e1) + (-66 * lambda - 142 * mu) * x * x + (52 * mu + 308 * lambda) * x + (25754 * mu) + (12914 * lambda)) * y * y / 0.512000e6 + (0.24e2 * (mu + lambda / 0.2e1) * x * pow(z, 5) + ((40 * mu + 20 * lambda) * x - (2919 * mu) - (15 * lambda)) * pow(z, 3) + (-0.252e3 * mu * pow(x, 0.3e1) + (5802 * mu + 36 * lambda) * x * x + (3536 * mu + 1770 * lambda) * x + (160 * mu) + (320 * lambda)) * (z * z) + ((864 * mu + 432 * lambda) * pow(x, 0.5e1) + (24 * lambda + 168 * mu) * pow(x, 0.4e1) + (12 * mu + 6 * lambda) * pow(x, 0.3e1) + (704 * mu + 8 * lambda) * x + (320 * mu) + (160 * lambda)) * z + (72 * mu + 12 * lambda) * pow(x, 0.4e1) + 0.6e1 * mu * pow(x, 0.3e1) + 0.12e2 * mu * x + (1912 * mu) - (644 * lambda)) * y / 0.512000e6 + 0.9e1 / 0.512000e6 * pow(z, 6) * lambda + ((6 * mu + 3 * lambda) * x * x - 0.6e1 * lambda * x + lambda) * pow(z, 4) / 0.512000e6 + (0.24e2 * lambda * pow(x, 0.3e1) + (1440 * lambda) + (2880 * mu)) * pow(z, 3) / 0.512000e6 + (0.144e3 * lambda * pow(x, 0.6e1) + (-8638 * mu + 6 * lambda) * x * x + (-1448 * mu + 316 * lambda) * x + (46 * lambda) + (12986 * mu)) * (z * z) / 0.512000e6 + (0.4944e4 * mu * pow(x, 0.3e1) + (504 * mu + 250 * lambda) * x * x + (-724 * mu - 400 * lambda) * x + (80 * lambda) + (80 * mu)) * z / 0.512000e6 + (-1728 * mu - 864 * lambda) * pow(x, 0.5e1) / 0.512000e6 - 0.3e1 / 0.32000e5 * mu * pow(x, 0.4e1) + (3840 * mu + 960 * lambda) * pow(x, 0.3e1) / 0.512000e6 + 0.83e2 / 0.128000e6 * mu * x * x + (-20 * mu - 12 * lambda) * x / 0.512000e6 + mu / 0.1600e4 - lambda / 0.1600e4;

		probl->rhs(sv, pts, 1, other);
		Eigen::MatrixXd diff = (other - rhs);

		REQUIRE(diff.array().abs().maxCoeff() < 1e-10);
	}
}