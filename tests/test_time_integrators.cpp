#include <polyfem/time_integrator/ImplicitEuler.hpp>
#include <polyfem/time_integrator/ImplicitNewmark.hpp>
#include <polyfem/time_integrator/BDF.hpp>

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
				std::cout << "Gradient mismatch" << std::endl;
				std::cout << "Gradient: " << grad.transpose() << std::endl;
				std::cout << "Finite gradient: " << fgrad.transpose() << std::endl;
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
				std::cout << "Hessian mismatch" << std::endl;
				std::cout << "Hessian:\n"
						  << hess << std::endl;
				std::cout << "Finite hessian:\n"
						  << fhess << std::endl;
			}

			CHECK(fd::compare_hessian(hess, fhess));
		}

		x.setRandom();
		x /= 100;
	}
}