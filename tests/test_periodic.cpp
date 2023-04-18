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

	std::shared_ptr<State> create_state_and_solve(const json &args)
	{
		std::shared_ptr<State> state = create_state(args);
		Eigen::MatrixXd sol, pressure;
		state->solve_problem(sol, pressure);

		return state;
	}

	std::shared_ptr<State> create_state_and_solve(const json &args, Eigen::MatrixXd &sol)
	{
		std::shared_ptr<State> state = create_state(args);
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


TEST_CASE("homogenize-stress-periodic", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "homogenize-stress-periodic.json", in_args);
	auto state_ptr = create_state(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-periodic-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = std::dynamic_pointer_cast<SumCompositeForm>(create_form(opt_args["functionals"], variable_to_simulations, states));

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();

	// nl_problem->solution_changed(x);
	// Eigen::VectorXd grad;
	// nl_problem->gradient(x, grad);

	// Eigen::VectorXd fgrad;
	// fgrad.setZero(x.size());
	// const double eps = opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>();
	// for (int i = x.size() - 2; i < x.size(); i++)
	// {
	// 	Eigen::VectorXd y = x;

	// 	y(i) = x(i) + eps;
	// 	nl_problem->solution_changed(y);
	// 	const double a = obj->value(y);

	// 	y(i) = x(i) - eps;
	// 	nl_problem->solution_changed(y);
	// 	const double b = obj->value(y);

	// 	fgrad(i) = (a - b) / (2 * eps);
	// }

	// std::cout << "fgrad " << fgrad.tail(2).transpose() << "\n";
	// std::cout << "grad " << grad.tail(2).transpose() << "\n";

	// io::VTUWriter writer;
	// writer.add_field("grad", utils::unflatten(periodic_grad_to_full(one_form), 2));
	// writer.add_field("fgrad", utils::unflatten(periodic_grad_to_full(fgrad), 2));
	
	// auto ids = state.primitive_to_node();
	// Eigen::VectorXd ids_ = Eigen::Map<Eigen::VectorXi>(ids.data(), ids.size()).cast<double>();
	// writer.add_field("ids", ids_);

	// Eigen::MatrixXi F;
	// state.get_elements(F);
	// writer.write_mesh("debug.vtu", V, F);

	Eigen::VectorXd theta;
	theta.setRandom(x.size());
	nl_problem->solution_changed(x);
	
	// verify_adjoint_expensive(variable_to_simulations, *obj, state, x, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>());
	verify_adjoint(variable_to_simulations, *obj, state, x, theta, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-4);
}

TEST_CASE("homogenize-stress", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "homogenize-stress.json", in_args);
	auto state_ptr = create_state(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = std::dynamic_pointer_cast<SumCompositeForm>(create_form(opt_args["functionals"], variable_to_simulations, states));

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	nl_problem->solution_changed(x);
	Eigen::VectorXd one_form;
	nl_problem->gradient(x, one_form);

	// if (opt_args["solver"]["nonlinear"]["debug_fd"])
	// {
	// 	Eigen::VectorXd fgrad;
	// 	fgrad.setZero(x.size());
	// 	fd::finite_gradient(
	// 		x, [&](const Eigen::VectorXd &y) -> double 
	// 		{
	// 			nl_problem->solution_changed(y);
	// 			return obj->value(y);
	// 		}, fgrad, fd::AccuracyOrder::SECOND, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>());

	// 	io::VTUWriter writer;
	// 	writer.add_field("grad", utils::unflatten(one_form, 2));
	// 	writer.add_field("fgrad", utils::unflatten(fgrad, 2));
	// 	writer.add_field("disp", utils::unflatten(state.diff_cached.u(0), 2)(state.primitive_to_node(), Eigen::all));
		
	// 	auto ids = state.primitive_to_node();
	// 	Eigen::VectorXd ids_ = Eigen::Map<Eigen::VectorXi>(ids.data(), ids.size()).cast<double>();
	// 	writer.add_field("ids", ids_);

	// 	Eigen::MatrixXi F;
	// 	state.get_elements(F);
	// 	writer.write_mesh("debug.vtu", V, F);
	// }

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
	verify_adjoint(variable_to_simulations, *obj, state, x, theta, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-4);
}

// TEST_CASE("force-shape", "[adjoint_method]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
// 	json in_args;
// 	load_json(path + "linear_elasticity-surface.json", in_args);
// 	in_args["boundary_conditions"]["rhs"] = {0, 0};
// 	auto state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	json opt_args;
// 	load_json(path + "linear_elasticity-surface-opt.json", opt_args);
// 	opt_args = apply_opt_json_spec(opt_args, false);

// 	std::vector<std::shared_ptr<State>> states({state_ptr});

// 	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
// 	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

// 	Eigen::MatrixXd V;
// 	state.get_vertices(V);
// 	Eigen::VectorXd x = utils::flatten(V);

// 	const int dim = V.cols();
// 	const double dt = opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>();
// 	auto nl_problem = state.solve_data.nl_problem;
// 	Eigen::VectorXd weights, full_sol, reduced_sol;
// 	{
// 		in_args["boundary_conditions"]["rhs"] = {3, 5};
// 		auto state_aux_ptr = create_state_and_solve(in_args);

// 		full_sol = state_aux_ptr->diff_cached.u(0);
// 		reduced_sol = nl_problem->full_to_reduced(full_sol);
// 		weights.setRandom(nl_problem->reduced_size());
// 	}

// 	nl_problem->solution_changed(reduced_sol);

// 	Eigen::VectorXd force_grad;
// 	{
// 		AdjointTools::dJ_shape_static_adjoint_term(state, full_sol, nl_problem->reduced_to_full(weights), force_grad);
// 		force_grad *= -1;
// 	}
	
// 	Eigen::VectorXd fgrad;
// 	fd::finite_gradient(
// 		x, [&](const Eigen::VectorXd &y) -> double 
// 		{
// 			for (auto &v2s : variable_to_simulations)
// 				v2s->update(y);
// 			state.build_basis();
// 			Eigen::VectorXd force_;
// 			nl_problem->solution_changed(reduced_sol);
// 			nl_problem->gradient(reduced_sol, force_);
// 			return force_.dot(weights);
// 		}, fgrad, fd::AccuracyOrder::SECOND, dt);

// 	io::VTUWriter writer;
// 	writer.add_field("grad", utils::unflatten(force_grad, 2));
// 	writer.add_field("fgrad", utils::unflatten(fgrad, 2));

// 	Eigen::MatrixXi F;
// 	state.get_elements(F);
// 	writer.write_mesh("debug.vtu", V, F);
// }

// TEST_CASE("periodic-force-periodic-shape", "[adjoint_method]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
// 	json in_args;
// 	load_json(path + "homogenize-stress-periodic.json", in_args);
// 	auto state_ptr = create_state(in_args);
// 	State &state = *state_ptr;

// 	json opt_args;
// 	load_json(path + "homogenize-stress-periodic-opt.json", opt_args);
// 	opt_args = apply_opt_json_spec(opt_args, false);

// 	std::vector<std::shared_ptr<State>> states({state_ptr});

// 	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
// 	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

// 	auto obj = std::dynamic_pointer_cast<SumCompositeForm>(create_form(opt_args["functionals"], variable_to_simulations, states));

// 	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

// 	Eigen::MatrixXd V;
// 	state.get_vertices(V);
// 	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();

// 	nl_problem->solution_changed(x);

// 	auto periodic_grad_to_full = [&](const Eigen::VectorXd &y) -> Eigen::VectorXd {
// 		Eigen::VectorXd z;
// 		z.setZero(V.size());
// 		for (int i = 0; i < V.rows(); i++)
// 			z.segment(i * V.cols(), V.cols()) = y.segment(state.periodic_mesh_map->full_to_periodic(i) * V.cols(), V.cols()).array();

// 		return z;
// 	};

// 	const int dim = V.cols();
// 	const double dt = opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>();
// 	std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state.solve_data.nl_problem);
	
// 	Eigen::VectorXd weights, reduced_sol;
// 	reduced_sol = homo_problem->full_to_reduced(state.diff_cached.u(0), state.diff_cached.disp_grad());
// 	weights.setRandom(homo_problem->macro_reduced_size() + homo_problem->reduced_size());
	
// 	homo_problem->solution_changed(reduced_sol);
// 	// state.solve_data.elastic_form->set_weight(0);
// 	// state.solve_data.periodic_contact_form->set_weight(1);

// 	/* Cheap verification */
// 	Eigen::VectorXd grad;
// 	AdjointTools::dJ_periodic_shape_adjoint_term(state, state.diff_cached.u(0), weights, grad);
// 	grad *= -1;

// 	Eigen::VectorXd shape_perturb;
// 	shape_perturb.setRandom(x.size());
		
// 	Eigen::VectorXd fgrad;
// 	fd::finite_gradient(
// 		Eigen::VectorXd::Zero(1), [&](const Eigen::VectorXd &y) -> double
// 		{
// 			for (auto &v2s : variable_to_simulations)
// 				v2s->update(x + shape_perturb * y(0));
// 			state.build_basis();
// 			Eigen::VectorXd force;
// 			homo_problem->solution_changed(reduced_sol);
// 			homo_problem->gradient(reduced_sol, force);
// 			return force.dot(weights);
// 		}, fgrad, fd::AccuracyOrder::SECOND, dt);
	
// 	std::cout << "error " << (grad.dot(shape_perturb) - fgrad(0)) << " norm " << grad.dot(shape_perturb) << "\n";

// 	/* Expensive verification */
// 	// Eigen::MatrixXd grad;
// 	// grad.setZero(weights.size(), x.size());
// 	// for (int i = 0; i < weights.size(); i++)
// 	// {
// 	// 	weights.setZero();
// 	// 	weights(i) = 1;
// 	// 	Eigen::VectorXd force_grad;
// 	// 	{
// 	// 		AdjointTools::dJ_periodic_shape_adjoint_term(state, state.diff_cached.u(0), weights, force_grad);
// 	// 		force_grad *= -1;
// 	// 	}
// 	// 	grad.row(i) = force_grad;
// 	// }
		
// 	// Eigen::MatrixXd fgrad;
// 	// fd::finite_jacobian(
// 	// 	x, [&](const Eigen::VectorXd &y) -> Eigen::VectorXd 
// 	// 	{
// 	// 		for (auto &v2s : variable_to_simulations)
// 	// 			v2s->update(y);
// 	// 		state.build_basis();
// 	// 		Eigen::VectorXd force;
// 	// 		homo_problem->solution_changed(reduced_sol);
// 	// 		homo_problem->gradient(reduced_sol, force);
// 	// 		return force;
// 	// 	}, fgrad, fd::AccuracyOrder::SECOND, dt);

// 	// if ((grad - fgrad).norm() < 1e-5 * grad.norm())
// 	// 	logger().warn("error {}, norm {}", (grad - fgrad).norm(), grad.norm());
// 	// else
// 	// {
// 	// 	logger().error("error {}, norm {}", (grad - fgrad).norm(), grad.norm());
// 	// 	logger().error("Large error, save to mat...");
// 	// 	Eigen::saveMarket(grad.sparseView().eval(), "grad.mat");
// 	// 	Eigen::saveMarket(fgrad.sparseView().eval(), "fgrad.mat");

// 	// 	grad = grad.leftCols(grad.cols() - 2).eval();
// 	// 	fgrad = fgrad.leftCols(fgrad.cols() - 2).eval();
// 	// 	if ((grad - fgrad).norm() > 1e-5 * grad.norm())
// 	// 	{
// 	// 		logger().error("Large error on periodic vertices, save to vtu...");
// 	// 		for (int i = 0; i < grad.rows(); i++)
// 	// 		{
// 	// 			io::VTUWriter writer;
// 	// 			writer.add_field("grad", utils::unflatten(periodic_grad_to_full(grad.row(i)), 2));
// 	// 			writer.add_field("fgrad", utils::unflatten(periodic_grad_to_full(fgrad.row(i)), 2));

// 	// 			auto ids = state.primitive_to_node();
// 	// 			Eigen::VectorXd ids_ = Eigen::Map<Eigen::VectorXi>(ids.data(), ids.size()).cast<double>();
// 	// 			writer.add_field("ids", ids_);

// 	// 			for (int i = 0; i < ids_.size(); i++)
// 	// 				ids_(i) = (double)state.bases_to_periodic_map(ids[i] * 2) / 2;
// 	// 			writer.add_field("periodic_ids", ids_);

// 	// 			Eigen::MatrixXi F;
// 	// 			state.get_elements(F);
// 	// 			writer.write_mesh("debug" + std::to_string(i) + ".vtu", V, F);
// 	// 		}
// 	// 	}
// 	// }
// }
