///////////////////////////////////////////////////////////////////////////////
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

#include <polyfem/State.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/AdjointTools.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/solver/AdjointNLProblem.hpp>

#include <polyfem/solver/forms/adjoint_forms/SpatialIntegralForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/CompositeForms.hpp>

#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
// #include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <paraviewo/VTUWriter.hpp>
#include <finitediff.hpp>

#include <catch2/catch_all.hpp>
#include <math.h>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;

namespace
{

	bool load_json(const std::string &json_file, json &out)
	{
		std::ifstream file(json_file);

		if (!file.is_open())
			return false;

		file >> out;

		out["root_path"] = json_file;

		return true;
	}

	std::shared_ptr<State> create_state_and_solve(const json &args)
	{
		std::shared_ptr<State> state = AdjointOptUtils::create_state(args, solver::CacheLevel::Derivatives, -1);
		Eigen::MatrixXd sol, pressure;
		state->solve_problem(sol, pressure);

		return state;
	}

	std::shared_ptr<State> create_state_and_solve(const json &args, Eigen::MatrixXd &sol)
	{
		std::shared_ptr<State> state = AdjointOptUtils::create_state(args, solver::CacheLevel::Derivatives, -1);
		Eigen::MatrixXd pressure;
		state->solve_problem(sol, pressure);

		return state;
	}

	void sample_field(const State &state, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> field, Eigen::MatrixXd &discrete_field, const int order = 1)
	{
		Eigen::MatrixXd tmp;
		tmp.setZero(1, state.mesh->dimension());
		tmp = field(tmp);
		const int actual_dim = tmp.cols();

		if (order >= 1)
		{
			Eigen::MatrixXd V;
			state.get_vertices(V);

			discrete_field = utils::flatten(field(V));
		}
		else if (order == 0)
		{
			Eigen::MatrixXd centers;
			if (state.mesh->is_volume())
				state.mesh->cell_barycenters(centers);
			else
				state.mesh->face_barycenters(centers);

			discrete_field = utils::flatten(field(centers));
		}
	}

	void verify_adjoint(AdjointNLProblem& problem, const Eigen::VectorXd &x, const Eigen::MatrixXd &theta, const double dt, const double tol)
	{
		problem.solution_changed(x);
		problem.save_to_file(0, x);
		double functional_val = problem.value(x);

		Eigen::VectorXd one_form;
		problem.gradient(x, one_form);
		double derivative = (one_form.array() * theta.array()).sum();

		problem.solution_changed(x + theta * dt);
		double next_functional_val = problem.value(x + theta * dt);

		problem.solution_changed(x - theta * dt);
		double former_functional_val = problem.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
		std::cout << std::setprecision(12) << "relative error: " << abs((finite_difference - derivative) / derivative) << "\n";
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
	}

	void verify_adjoint(AdjointNLProblem& problem, const Eigen::VectorXd &x, const double dt, const double tol)
	{
		problem.solution_changed(x);
		problem.save_to_file(0, x);
		double functional_val = problem.value(x);

		Eigen::VectorXd one_form;
		problem.gradient(x, one_form);
		Eigen::VectorXd theta = one_form;
		double derivative = (one_form.array() * theta.array()).sum();

		problem.solution_changed(x + theta * dt);
		double next_functional_val = problem.value(x + theta * dt);

		problem.solution_changed(x - theta * dt);
		double former_functional_val = problem.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
		std::cout << std::setprecision(12) << "relative error: " << abs((finite_difference - derivative) / derivative) << "\n";
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
	}
} // namespace


TEST_CASE("homogenize-stress-periodic", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "homogenize-stress-periodic.json", in_args);
	auto state_ptr = AdjointOptUtils::create_state(in_args, solver::CacheLevel::Derivatives, -1);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-periodic-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	VariableToSimulationGroup var2sim;
	var2sim.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = std::dynamic_pointer_cast<SumCompositeForm>(AdjointOptUtils::create_form(opt_args["functionals"], var2sim, states));

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::VectorXd x = var2sim[0]->inverse_eval();

	Eigen::VectorXd theta;
	theta.setRandom(x.size());

	nl_problem->solution_changed(x);
	
	verify_adjoint(*nl_problem, x, theta, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-5);
}

TEST_CASE("homogenize-stress", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "homogenize-stress.json", in_args);
	auto state_ptr = AdjointOptUtils::create_state(in_args, solver::CacheLevel::Derivatives, -1);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	VariableToSimulationGroup var2sim;
	var2sim.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = std::dynamic_pointer_cast<SumCompositeForm>(AdjointOptUtils::create_form(opt_args["functionals"], var2sim, states));

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	nl_problem->solution_changed(x);
	Eigen::VectorXd one_form;
	nl_problem->gradient(x, one_form);

	Eigen::VectorXd theta;
	theta.setRandom(x.size());

	const double eps = 1e-5;
	Eigen::VectorXd min = V.colwise().minCoeff();
	Eigen::VectorXd max = V.colwise().maxCoeff();
	for (int i = 0; i < V.rows(); i++)
	{
		auto vert = state.mesh->point(i);
		if (vert(0) < min(0) + eps || vert(0) > max(0) - eps || vert(1) < min(1) + eps || vert(1) > max(1) - eps)
		{
			for (int d = 0; d < 2; d++)
				theta(i * 2 + d) = 0;
		}
	}
	nl_problem->solution_changed(x);
	verify_adjoint(*nl_problem, x, theta, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-4);
}
