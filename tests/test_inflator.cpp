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
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/io/VTUWriter.hpp>
#include <finitediff.hpp>

#include <catch2/catch.hpp>
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

	void verify_adjoint(std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, AdjointForm &obj, State &state, const Eigen::VectorXd &x, const double dt, const double tol, bool print_grad = false)
	{
		obj.solution_changed(x);
		double functional_val = obj.value(x);

		state.solve_adjoint_cached(obj.compute_adjoint_rhs(x, state));
		Eigen::VectorXd one_form;
		obj.first_derivative(x, one_form);
		Eigen::VectorXd theta = one_form.normalized();
		double derivative = (one_form.array() * theta.array()).sum();

		for (auto &v2s : variable_to_simulations)
			v2s->update(x + theta * dt);
		state.build_basis();
		solve_pde(state);
		obj.solution_changed(x + theta * dt);
		double next_functional_val = obj.value(x + theta * dt);

		for (auto &v2s : variable_to_simulations)
			v2s->update(x - theta * dt);
		state.build_basis();
		solve_pde(state);
		obj.solution_changed(x - theta * dt);
		double former_functional_val = obj.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

		REQUIRE(derivative == Approx(finite_difference).epsilon(tol));
	}

	void verify_adjoint(std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, AdjointForm &obj, State &state, const Eigen::VectorXd &x, const Eigen::MatrixXd &theta, const double dt, const double tol, bool print_grad = false)
	{
		obj.solution_changed(x);
		double functional_val = obj.value(x);

		state.solve_adjoint_cached(obj.compute_adjoint_rhs(x, state));
		Eigen::VectorXd one_form;
		obj.first_derivative(x, one_form);
		double derivative = (one_form.array() * theta.array()).sum();

		for (auto &v2s : variable_to_simulations)
			v2s->update(x + theta * dt);
		state.build_basis();
		solve_pde(state);
		obj.solution_changed(x + theta * dt);
		double next_functional_val = obj.value(x + theta * dt);

		for (auto &v2s : variable_to_simulations)
			v2s->update(x - theta * dt);
		state.build_basis();
		solve_pde(state);
		obj.solution_changed(x - theta * dt);
		double former_functional_val = obj.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
		std::cout << std::setprecision(12) << "relative error: " << abs((finite_difference - derivative) / derivative) << "\n";
		REQUIRE(derivative == Approx(finite_difference).epsilon(tol));
	}

	void verify_adjoint_expensive(std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, AdjointForm &obj, State &state, const Eigen::VectorXd &x, const double dt)
	{
		obj.solution_changed(x);
		double functional_val = obj.value(x);

		state.solve_adjoint_cached(obj.compute_adjoint_rhs(x, state));
		Eigen::VectorXd analytic;
		obj.first_derivative(x, analytic);

		std::cout << std::setprecision(12) << "derivative: " << analytic.transpose() << "\n";

		Eigen::VectorXd fd;
		fd.setZero(x.size());
		for (int d = 0; d < x.size(); d++)
		{
			Eigen::VectorXd theta;
			theta.setZero(x.size());
			theta(d) = 1;

			for (auto &v2s : variable_to_simulations)
				v2s->update(x + theta * dt);
			state.build_basis();
			solve_pde(state);
			obj.solution_changed(x + theta * dt);
			double next_functional_val = obj.value(x + theta * dt);

			for (auto &v2s : variable_to_simulations)
				v2s->update(x - theta * dt);
			state.build_basis();
			solve_pde(state);
			obj.solution_changed(x - theta * dt);
			double former_functional_val = obj.value(x - theta * dt);

			fd(d) = (next_functional_val - former_functional_val) / dt / 2;
		}

		std::cout << "fd: " << fd.transpose() << "\n";
	}

} // namespace

TEST_CASE("isosurface-inflator-periodic", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/isosurface-inflator-periodic");
	// chdir(path.c_str());
	auto work_path = std::filesystem::current_path();
	std::filesystem::current_path(path);
	json in_args;
	load_json("state.json", in_args);

	json opt_args;
	load_json("opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::shared_ptr<State> state_ptr = create_state(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::VectorXd x;
	nlohmann::adl_serializer<Eigen::VectorXd>::from_json(opt_args["parameters"][0]["initial"], x);

	for (auto &v2s : variable_to_simulations)
		v2s->update(x);
	state.build_basis();
	solve_pde(state);

	verify_adjoint(variable_to_simulations, *obj, state, x, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-3);
	// verify_adjoint_expensive(variable_to_simulations, *obj, state, x, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>());

	std::filesystem::current_path(work_path);
}

TEST_CASE("isosurface-inflator", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/isosurface-inflator");
	// chdir(path.c_str());
	auto work_path = std::filesystem::current_path();
	std::filesystem::current_path(path);
	json in_args;
	load_json("state.json", in_args);

	json opt_args;
	load_json("opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::shared_ptr<State> state_ptr = create_state(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::VectorXd x;
	nlohmann::adl_serializer<Eigen::VectorXd>::from_json(opt_args["parameters"][0]["initial"], x);

	for (auto &v2s : variable_to_simulations)
		v2s->update(x);
	state.build_basis();
	solve_pde(state);

	verify_adjoint(variable_to_simulations, *obj, state, x, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-3);
	// verify_adjoint_expensive(variable_to_simulations, *obj, state, x, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>());

	std::filesystem::current_path(work_path);
}
