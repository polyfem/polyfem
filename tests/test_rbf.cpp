////////////////////////////////////////////////////////////////////////////////
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>
#include <polyfem/basis/function/RBFWithLinear.hpp>
#include <polyfem/basis/function/RBFWithQuadratic.hpp>
#include <polyfem/basis/function/RBFWithQuadraticLagrange.hpp>
#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::io;
using namespace polyfem::utils;

TEST_CASE("harmonic RBF fit is stable under sample reordering", "[rbf_test]")
{
	const int variant = GENERATE(0, 1, 2, 3);
	CAPTURE(variant);
	const double pi = std::acos(-1.0);
	Eigen::MatrixXd centers(64, 2), samples(256, 2);
	for (int i = 0; i < centers.rows(); ++i)
	{
		const double angle = 2 * pi * i / centers.rows();
		centers.row(i) << 2 * std::cos(angle), 2 * std::sin(angle);
	}
	Eigen::MatrixXd rhs(samples.rows(), 1);
	for (int i = 0; i < samples.rows(); ++i)
	{
		const double angle = 2 * pi * i / samples.rows();
		samples.row(i) << std::cos(angle), std::sin(angle);
		rhs(i) = std::exp(samples(i, 0)) * std::cos(samples(i, 1));
	}
	Eigen::MatrixXd reversed_samples = samples.colwise().reverse().eval();
	Eigen::MatrixXd reversed_rhs = rhs.colwise().reverse().eval();

	quadrature::Quadrature quadr;
	quadrature::QuadQuadrature().get_quadrature(20, quadr);
	quadr.points.array() -= 0.5;
	Eigen::MatrixXd integrals(1, 2);
	integrals(0, 0) = (quadr.weights.array() * quadr.points.col(0).array().exp() * quadr.points.col(1).array().cos()).sum();
	integrals(0, 1) = -(quadr.weights.array() * quadr.points.col(0).array().exp() * quadr.points.col(1).array().sin()).sum();

	Eigen::MatrixXd query(26, 2);
	for (int i = 0; i < 9; ++i)
		query.row(i) << 0.4 * (i % 3 - 1), 0.4 * (i / 3 - 1);
	for (int i = 0; i < 17; ++i)
	{
		const double angle = 2 * pi * (i + 0.3) / 17;
		query.row(i + 9) << 0.97 * std::cos(angle), 0.97 * std::sin(angle);
	}

	auto check_fit = [&](auto &fit, auto &reversed_fit) {
		Eigen::MatrixXd values, reversed_values, gradient, reversed_gradient;
		fit.basis(0, query, values);
		reversed_fit.basis(0, query, reversed_values);
		fit.grad(0, query, gradient);
		reversed_fit.grad(0, query, reversed_gradient);
		const Eigen::ArrayXd expected = query.col(0).array().exp() * query.col(1).array().cos();
		CHECK((values.col(0).array() - expected).abs().maxCoeff() < 1e-9);
		CHECK((gradient.col(0).array() - expected).abs().maxCoeff() < 1e-8);
		CHECK((gradient.col(1).array() + query.col(0).array().exp() * query.col(1).array().sin()).abs().maxCoeff() < 1e-8);
		CHECK((values - reversed_values).cwiseAbs().maxCoeff() < 1e-10);
		CHECK((gradient - reversed_gradient).cwiseAbs().maxCoeff() < 1e-9);
		if (variant > 0)
		{
			fit.basis(0, quadr.points, values);
			fit.grad(0, quadr.points, gradient);
			const Eigen::ArrayXd weights = quadr.weights;
			Eigen::MatrixXd moments(1, integrals.cols());
			moments(0, 0) = (weights * gradient.col(0).array()).sum();
			moments(0, 1) = (weights * gradient.col(1).array()).sum();
			if (variant >= 2)
			{
				const Eigen::ArrayXd x = quadr.points.col(0), y = quadr.points.col(1);
				moments(0, 2) = (weights * (y * gradient.col(0).array() + x * gradient.col(1).array())).sum();
				moments(0, 3) = (weights * (2 * x * gradient.col(0).array() + 2 * values.col(0).array())).sum();
				moments(0, 4) = (weights * (2 * y * gradient.col(1).array() + 2 * values.col(0).array())).sum();
			}
			CHECK((moments - integrals).cwiseAbs().maxCoeff() < 1e-10);
		}
	};
	if (variant < 2)
	{
		basis::RBFWithLinear fit(centers, samples, integrals, quadr, rhs, variant == 1);
		basis::RBFWithLinear reversed_fit(centers, reversed_samples, integrals, quadr, reversed_rhs, variant == 1);
		check_fit(fit, reversed_fit);
		return;
	}

	// Match the five consistency moments of the analytic field.
	const Eigen::ArrayXd x = quadr.points.col(0), y = quadr.points.col(1), weights = quadr.weights;
	const Eigen::ArrayXd f = x.exp() * y.cos(), dx = f, dy = -x.exp() * y.sin();
	integrals.conservativeResize(1, 5);
	integrals(0, 2) = (weights * (y * dx + x * dy)).sum();
	integrals(0, 3) = (weights * (2 * x * dx + 2 * f)).sum();
	integrals(0, 4) = (weights * (2 * y * dy + 2 * f)).sum();
	assembler::Laplacian laplacian;
	laplacian.set_size(2);
	if (variant == 2)
	{
		basis::RBFWithQuadratic fit(laplacian, centers, samples, integrals, quadr, rhs);
		basis::RBFWithQuadratic reversed_fit(laplacian, centers, reversed_samples, integrals, quadr, reversed_rhs);
		check_fit(fit, reversed_fit);
	}
	else
	{
		basis::RBFWithQuadraticLagrange fit(laplacian, centers, samples, integrals, quadr, rhs);
		basis::RBFWithQuadraticLagrange reversed_fit(laplacian, centers, reversed_samples, integrals, quadr, reversed_rhs);
		check_fit(fit, reversed_fit);
	}
}

TEST_CASE("interpolation", "[rbf_test]")
{
	const std::string path = POLYFEM_DATA_DIR;
	Eigen::MatrixXd disp, pts;
	read_matrix(path + "/disp.txt", disp);
	read_matrix(path + "/pts.txt", pts);

	RBFInterpolation rbf(disp, pts, "thin_plate", 1e-2);

	const auto vals = rbf.interpolate(pts);

	for (int i = 0; i < pts.rows(); ++i)
	{
		for (int j = 0; j < pts.cols(); ++j)
		{
			REQUIRE(disp(i, j) == Catch::Approx(vals(i, j)).margin(1e-9));
		}
	}
}
