#include <polyfem/time_integrator/ImplicitEuler.hpp>
#include <polyfem/time_integrator/ImplicitNewmark.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polyfem/utils/Logger.hpp>

#include <finitediff.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

using namespace polyfem;
using namespace polyfem::time_integrator;

TEST_CASE("time integrator", "[time_integrator]")
{
	const double dt = GENERATE(0.1, 0.01, 0.001);
	const int n = 10;
	Eigen::VectorXd x_prev = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd v_prev = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd a_prev = Eigen::VectorXd::Zero(n);

	std::shared_ptr<ImplicitTimeIntegrator> time_integrator;
	json params;

	SECTION("Implicit Euler")
	{
		time_integrator = std::make_shared<ImplicitEuler>();
		params = R"({})"_json;
	}
	SECTION("Implicit Newmark")
	{
		time_integrator = std::make_shared<ImplicitNewmark>();
		params = R"({
	        "gamma": 0.5,
	        "beta": 0.25
	    })"_json;
	}
	SECTION("BDF")
	{
		time_integrator = std::make_shared<ImplicitNewmark>();
		params = R"({
	        "steps": 2
	    })"_json;
	}

	time_integrator->init(x_prev, v_prev, a_prev, dt);

	CHECK(time_integrator->dt() == dt);

	const auto f = [&time_integrator](const Eigen::VectorXd &x) -> double {
		return 0.5 * time_integrator->compute_velocity(x).squaredNorm();
	};
	const auto gradf = [&time_integrator](const Eigen::VectorXd &x) -> Eigen::VectorXd {
		return time_integrator->dv_dx() * time_integrator->compute_velocity(x);
	};

	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
	const int n_rand = 10;
	for (int rand = 0; rand < n_rand; ++rand)
	{
		// Test gradient with finite differences
		{
			const Eigen::VectorXd grad = gradf(x);

			Eigen::VectorXd fgrad;
			fd::finite_gradient(x, f, fgrad);

			if (!fd::compare_gradient(grad, fgrad))
			{
				logger().trace("Gradient mismatch");
				logger().trace("Gradient: {}", grad.transpose());
				logger().trace("Finite gradient: {}", fgrad.transpose());
			}

			CHECK(fd::compare_gradient(grad, fgrad));
		}

		// Test hessian with finite differences
		{
			const Eigen::MatrixXd hess =
				std::pow(time_integrator->dv_dx(), 2) * Eigen::MatrixXd::Identity(n, n);

			Eigen::MatrixXd fhess;
			fd::finite_jacobian(x, gradf, fhess);

			if (!fd::compare_hessian(hess, fhess))
			{
				logger().trace("Hessian mismatch");
				logger().trace("Hessian:\n{}", hess);
				logger().trace("Finite hessian:\n{}", fhess);
			}

			CHECK(fd::compare_hessian(hess, fhess));
		}

		x.setRandom();
		x /= 100;
	}
}

TEST_CASE("first order implicit Euler", "[time_integrator]")
{
	const double dt = 0.1;
	const int n = 10;
	Eigen::VectorXd x_prev = Eigen::VectorXd::LinSpaced(n, 1, n);
	Eigen::VectorXd v_prev = Eigen::VectorXd::Ones(n);
	Eigen::VectorXd a_prev = Eigen::VectorXd::Ones(n);

	ImplicitEuler time_integrator(ImplicitTimeIntegrator::DynamicOrder::First);
	time_integrator.init(x_prev, v_prev, a_prev, dt);

	CHECK((time_integrator.x_tilde() - x_prev).norm() < 1e-12);
	CHECK(time_integrator.acceleration_scaling() == dt);

	const Eigen::VectorXd x = 2 * x_prev;
	CHECK((time_integrator.compute_velocity(x) - (x - x_prev) / dt).norm() < 1e-12);
	CHECK(time_integrator.compute_acceleration(time_integrator.compute_velocity(x)).norm() < 1e-12);

	time_integrator.update_quantities(x);
	CHECK((time_integrator.x_prev() - x).norm() < 1e-12);
	CHECK(time_integrator.a_prev().norm() < 1e-12);
}

TEST_CASE("first order BDF", "[time_integrator]")
{
	const double dt = 0.1;
	const int n = 10;
	Eigen::MatrixXd x_prevs(n, 2);
	x_prevs.col(0) = Eigen::VectorXd::LinSpaced(n, 1, n);
	x_prevs.col(1) = Eigen::VectorXd::LinSpaced(n, 2, 2 * n);
	const Eigen::MatrixXd v_prevs = Eigen::MatrixXd::Ones(n, 2);
	const Eigen::MatrixXd a_prevs = Eigen::MatrixXd::Ones(n, 2);

	BDF time_integrator(2, ImplicitTimeIntegrator::DynamicOrder::First);
	time_integrator.init(x_prevs, v_prevs, a_prevs, dt);

	const Eigen::VectorXd x_tilde = 4.0 / 3.0 * x_prevs.col(0) - 1.0 / 3.0 * x_prevs.col(1);
	CHECK((time_integrator.x_tilde() - x_tilde).norm() < 1e-12);
	CHECK(time_integrator.acceleration_scaling() == 2.0 / 3.0 * dt);

	const Eigen::VectorXd x = 2 * x_prevs.col(0);
	CHECK((time_integrator.compute_velocity(x) - (x - x_tilde) / (2.0 / 3.0 * dt)).norm() < 1e-12);
	CHECK(time_integrator.compute_acceleration(time_integrator.compute_velocity(x)).norm() < 1e-12);

	time_integrator.update_quantities(x);
	CHECK((time_integrator.x_prev() - x).norm() < 1e-12);
	CHECK(time_integrator.a_prev().norm() < 1e-12);
}

TEST_CASE("time integrator factories", "[time_integrator]")
{
	const Eigen::VectorXd x_prev = Eigen::VectorXd::Zero(1);
	const Eigen::VectorXd v_prev = Eigen::VectorXd::Zero(1);
	const Eigen::VectorXd a_prev = Eigen::VectorXd::Zero(1);
	const double dt = 0.1;

	const auto default_integrator =
		ImplicitTimeIntegrator::construct_time_integrator("ImplicitEuler");
	CHECK(std::dynamic_pointer_cast<ImplicitEuler>(default_integrator) != nullptr);

	const auto first_order_bdf = ImplicitTimeIntegrator::construct_bdf_integrator(
		R"({"type": "ImplicitEuler"})"_json,
		ImplicitTimeIntegrator::DynamicOrder::First);
	first_order_bdf->init(x_prev, v_prev, a_prev, dt);
	CHECK(first_order_bdf->steps() == 1);
	CHECK(first_order_bdf->acceleration_scaling() == dt);

	const auto configured_bdf = ImplicitTimeIntegrator::construct_bdf_integrator(
		R"({"type": "BDF", "steps": 2})"_json);
	Eigen::MatrixXd x_prevs = Eigen::MatrixXd::Zero(1, 2);
	configured_bdf->init(x_prevs, x_prevs, x_prevs, dt);
	CHECK(configured_bdf->steps() == 2);
}
