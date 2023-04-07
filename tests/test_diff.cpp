///////////////////////////////////////////////////////////////////////////////
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <jse/jse.h>

#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/solver/forms/adjoint_forms/SpatialIntegralForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/CompositeForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/TransientForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/SmoothingForms.hpp>

#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>
#include <polyfem/solver/forms/parametrization/NodeCompositeParametrizations.hpp>

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

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

		if (one_form.size() == state.ndof())
		{
			state.args["output"]["paraview"]["file_name"] = "debug.vtu";
			state.export_data(utils::flatten(utils::unflatten(one_form, state.mesh->dimension())(state.node_to_primitive(), Eigen::all)), Eigen::MatrixXd());
		}

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

TEST_CASE("laplacian", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "laplacian.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "laplacian-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();

	// Eigen::MatrixXd velocity_discrete(x.size(), 1);
	// velocity_discrete.setRandom();
	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = vel(i) * cos(vel(i));
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(state, velocity, velocity_discrete);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 3e-5);
}

TEST_CASE("boundary-smoothing", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "laplacian.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "laplacian-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	BoundarySmoothingForm obj(variable_to_simulations, state, true, 3);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setRandom(x.size(), 1);

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-6, 1e-6);
}

TEST_CASE("linear_elasticity-surface-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "linear_elasticity-surface-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "linear_elasticity-surface-3d-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	PositionForm obj(variable_to_simulations, state, opt_args["functionals"][0]);
	obj.set_integral_type(SpatialIntegralType::SURFACE);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-7, 1e-5);
}

TEST_CASE("linear_elasticity-surface", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "linear_elasticity-surface.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "linear_elasticity-surface-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	PositionForm obj(variable_to_simulations, state, opt_args["functionals"][0]);
	obj.set_integral_type(SpatialIntegralType::SURFACE);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("topology-compliance", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "topology-compliance.json", in_args);

	json opt_args;
	load_json(path + "topology-compliance-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<Parametrization>> map_list = {std::make_shared<PowerMap>(5), std::make_shared<AppendConstantMap>(state.bases.size(), state.args["materials"]["nu"]), std::make_shared<ENu2LambdaMu>(state.mesh->is_volume())};
	CompositeParametrization composite_map(map_list);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ElasticVariableToSimulation>(state_ptr, composite_map));

	ComplianceForm obj(variable_to_simulations, state, opt_args["functionals"][0]);

	Eigen::MatrixXd theta(state.bases.size(), 1);
	for (int e = 0; e < state.bases.size(); e++)
		theta(e) = (rand() % 1000) / 1000.0;

	Eigen::VectorXd y(state.bases.size() * 2);
	y << state.assembler.lame_params().lambda_mat_, state.assembler.lame_params().mu_mat_;

	Eigen::VectorXd x = composite_map.inverse_eval(y);

	for (auto &v2s : variable_to_simulations)
		v2s->update(x);
	state.build_basis();
	solve_pde(state);

	verify_adjoint(variable_to_simulations, obj, state, x, theta, 1e-4, 1e-6);
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

TEST_CASE("neohookean-stress-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "neohookean-stress-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "neohookean-stress-3d-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 1e-5);
}

TEST_CASE("shape-neumann-nodes", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-neumann-nodes.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-neumann-nodes-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, VariableToBoundaryNodes({}, *state_ptr, 2)));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 1000) / 1000.0;
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;

	Eigen::VectorXd x;
	int opt_bnodes = 0;
	int dim;
	{
		const auto &mesh = state.mesh;
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();
		dim = mesh->dimension();

		std::set<int> node_ids;
		std::set<int> total_bnode_ids;
		for (const auto &lb : state.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

				if (boundary_id == 2)
					for (long n = 0; n < nodes.size(); ++n)
						node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
			}
		}
		opt_bnodes = node_ids.size();
	}
	x.resize(opt_bnodes * dim);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_parametrization().get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 1e-3);
}

TEST_CASE("neumann-shape-derivative", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-neumann-nodes.json", in_args);
	Eigen::MatrixXd sol;
	auto state_ptr = create_state_and_solve(in_args, sol);
	State &state = *state_ptr;

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 1000) / 1000.0;
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, VariableToBoundaryNodes({}, *state_ptr, 2)));

	Eigen::VectorXd x;
	int opt_bnodes = 0;
	int dim;
	{
		const auto &mesh = state.mesh;
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();
		dim = mesh->dimension();

		std::set<int> node_ids;
		std::set<int> total_bnode_ids;
		for (const auto &lb : state.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

				if (boundary_id == 2)
					for (long n = 0; n < nodes.size(); ++n)
						node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
			}
		}
		opt_bnodes = node_ids.size();
	}
	x.resize(opt_bnodes * dim);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_parametrization().get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	double t0 = 0;
	double dt = 0.025;
	double timesteps = 10;

	// auto form = state.solve_data.body_form;
	double eps = 1e-7;

	std::vector<Eigen::MatrixXd> hess_vec;
	for (int i = 1; i <= timesteps; ++i)
	{
		Eigen::MatrixXd hess(state.n_bases * state.mesh->dimension(), x.size());

		for (int k = 0; k < state.n_bases * state.mesh->dimension(); ++k)
		{
			Eigen::VectorXd indicator = Eigen::VectorXd::Zero(state.n_bases * state.mesh->dimension());
			indicator(k) = 1;
			Eigen::VectorXd term;
			state.solve_data.body_form->force_shape_derivative(state.n_geom_bases, t0 + i * dt, state.diff_cached.u(i), indicator, term);
			hess.row(k) = variable_to_simulations[0]->get_parametrization().apply_jacobian(term, x);
		}

		hess_vec.push_back(hess);
	}

	std::vector<Eigen::MatrixXd> hess_fd_vec;
	for (int i = 1; i <= timesteps; ++i)
	{
		Eigen::MatrixXd hess_fd(state.n_bases * state.mesh->dimension(), x.size());

		for (int k = 0; k < x.size(); ++k)
		{

			Eigen::VectorXd h_plus, h_minus;
			{
				Eigen::VectorXd y = x;
				y(k) += eps;
				variable_to_simulations[0]->update(y);
				state.build_basis();
				state.assemble_rhs();
				state.assemble_stiffness_mat();
				auto form = std::make_shared<BodyForm>(state.n_bases * dim, state.n_pressure_bases, state.boundary_nodes, state.local_boundary,
													   state.local_neumann_boundary, state.n_boundary_samples(), state.rhs, *state.solve_data.rhs_assembler,
													   state.assembler.density(), /*apply_DBC=*/true, /*is_formulation_mixed=*/false,
													   state.problem->is_time_dependent());
				form->update_quantities(t0 + i * dt, state.diff_cached.u(i - 1));

				Eigen::VectorXd term;
				form->first_derivative(state.diff_cached.u(i), term);
				h_plus = term;
			}

			{
				Eigen::VectorXd y = x;
				y(k) -= eps;
				variable_to_simulations[0]->update(y);
				state.build_basis();
				state.assemble_rhs();
				state.assemble_stiffness_mat();
				auto form = std::make_shared<BodyForm>(state.n_bases * dim, state.n_pressure_bases, state.boundary_nodes, state.local_boundary,
													   state.local_neumann_boundary, state.n_boundary_samples(), state.rhs, *state.solve_data.rhs_assembler,
													   state.assembler.density(), /*apply_DBC=*/true, /*is_formulation_mixed=*/false,
													   state.problem->is_time_dependent());
				form->update_quantities(t0 + i * dt, state.diff_cached.u(i - 1));

				Eigen::VectorXd term;
				form->first_derivative(state.diff_cached.u(i), term);
				h_minus = term;
			}

			hess_fd.col(k) = (h_plus - h_minus) / (2 * eps);
		}

		hess_fd_vec.push_back(hess_fd);
	}

	for (int i = 1; i <= timesteps; ++i)
	{
		std::cout << "comparison" << std::endl;
		std::cout << "norm of difference " << (hess_fd_vec[i - 1] - hess_vec[i - 1]).norm() << std::endl;
	}
}

TEST_CASE("shape-pressure-neumann-nodes", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-pressure-neumann-nodes.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-pressure-neumann-nodes-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, VariableToBoundaryNodes({}, *state_ptr, 2)));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 1000) / 1000.0;
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;

	Eigen::VectorXd x;
	int opt_bnodes = 0;
	int dim;
	{
		const auto &mesh = state.mesh;
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();
		dim = mesh->dimension();

		std::set<int> node_ids;
		std::set<int> total_bnode_ids;
		for (const auto &lb : state.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

				if (boundary_id == 2)
					for (long n = 0; n < nodes.size(); ++n)
						node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
			}
		}
		opt_bnodes = node_ids.size();
	}
	x.resize(opt_bnodes * dim);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_parametrization().get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-8, 1e-3);
}

TEST_CASE("homogenize-stress", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "homogenize-stress.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(x.size(), 1);
	const double eps = 1e-5;
	Eigen::VectorXd min = V.colwise().minCoeff();
	Eigen::VectorXd max = V.colwise().maxCoeff();
	for (int i = 0; i < V.rows(); i++)
		for (int d = 0; d < 2; d++)
		{
			auto vert = state.mesh->point(i);
			if (state.mesh->is_boundary_vertex(i) && vert(0) > min(0) + eps && vert(0) < max(0) - eps && vert(1) > min(1) + eps && vert(1) < max(1) - eps)
				velocity_discrete(i * 2 + d) = (rand() % 10000) / 1.0e4;
		}

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>(), 1e-5);
}

TEST_CASE("periodic-contact-force", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "homogenize-stress.json", in_args);
	in_args["contact"]["periodic"] = true;
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::VectorXd theta;
	theta.setZero(x.size());
	const double eps = 1e-5;
	Eigen::VectorXd min = V.colwise().minCoeff();
	Eigen::VectorXd max = V.colwise().maxCoeff();
	for (int i = 0; i < V.rows(); i++)
		for (int d = 0; d < 2; d++)
		{
			auto vert = state.mesh->point(i);
			if (state.mesh->is_boundary_vertex(i) && vert(0) > min(0) + eps && vert(0) < max(0) - eps && vert(1) > min(1) + eps && vert(1) < max(1) - eps)
				theta(i * 2 + d) = (rand() % 10000) / 1.0e4;
		}

	const double dt = opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>();
	Eigen::VectorXd weights, extended_sol;
	{
		const int dim = V.cols();
		std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state.solve_data.nl_problem);
		extended_sol.setZero(state.diff_cached.u(0).size() + dim * dim);
		extended_sol.head(state.diff_cached.u(0).size()) = state.diff_cached.u(0) - io::Evaluator::generate_linear_field(state.n_bases, state.mesh_nodes, state.diff_cached.disp_grad());
		extended_sol.tail(dim * dim) = utils::flatten(state.diff_cached.disp_grad());

		weights.setRandom(extended_sol.size());
	}

	Eigen::VectorXd force;
	state.solve_data.periodic_contact_form->first_derivative(extended_sol, force);
	double functional_val = force.dot(weights);

	Eigen::VectorXd force_grad;
	state.solve_data.periodic_contact_form->force_shape_derivative(state.solve_data.periodic_contact_form->get_constraint_set(), extended_sol, weights, force_grad);
	force_grad = -utils::flatten(utils::unflatten(state.down_sampling_mat * force_grad, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));

	for (auto &v2s : variable_to_simulations)
		v2s->update(x + theta * dt);
	state.build_basis();
	state.solve_data.periodic_contact_form->solution_changed(extended_sol);
	state.solve_data.periodic_contact_form->first_derivative(extended_sol, force);
	double next_functional_val = force.dot(weights);

	for (auto &v2s : variable_to_simulations)
		v2s->update(x - theta * dt);
	state.build_basis();
	state.solve_data.periodic_contact_form->solution_changed(extended_sol);
	state.solve_data.periodic_contact_form->first_derivative(extended_sol, force);
	double former_functional_val = force.dot(weights);

	double derivative = force_grad.dot(theta);
	double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-6));
}

// TEST_CASE("contact-force", "[adjoint_method]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
// 	json in_args;
// 	load_json(path + "homogenize-stress.json", in_args);
// 	in_args["contact"]["periodic"] = false;
// 	auto state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	json opt_args;
// 	load_json(path + "homogenize-stress-opt.json", opt_args);
// 	opt_args = apply_opt_json_spec(opt_args, false);

// 	std::vector<std::shared_ptr<State>> states({state_ptr});

// 	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
// 	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

// 	Eigen::MatrixXd V;
// 	state.get_vertices(V);
// 	Eigen::VectorXd x = utils::flatten(V);

// 	Eigen::VectorXd theta;
// 	theta.setZero(x.size());
// 	const double eps = 1e-5;
// 	Eigen::VectorXd min = V.colwise().minCoeff();
// 	Eigen::VectorXd max = V.colwise().maxCoeff();
// 	for (int i = 0; i < V.rows(); i++)
// 		for (int d = 0; d < 2; d++)
// 		{
// 			auto vert = state.mesh->point(i);
// 			if (state.mesh->is_boundary_vertex(i) && vert(0) > min(0) + eps && vert(0) < max(0) - eps && vert(1) > min(1) + eps && vert(1) < max(1) - eps)
// 				theta(i * 2 + d) = (rand() % 10000) / 1.0e4;
// 		}

// 	const double dt = opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>();
// 	Eigen::VectorXd weights, sol;
// 	{
// 		std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state.solve_data.nl_problem);
// 		sol = state.diff_cached.u(0) + io::Evaluator::generate_linear_field(state.n_bases, state.mesh_nodes, state.diff_cached.disp_grad());
// 		weights.setRandom(sol.size());
// 	}

// 	auto form = state.solve_data.contact_form;

// 	Eigen::VectorXd force;
// 	form->first_derivative(sol, force);
// 	double functional_val = force.dot(weights);

// 	Eigen::VectorXd force_grad;
// 	form->force_shape_derivative(state.diff_cached.contact_set(0), sol, weights, force_grad);
// 	force_grad = -utils::flatten(utils::unflatten(state.down_sampling_mat * force_grad, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));

// 	for (auto &v2s : variable_to_simulations)
// 		v2s->update(x + theta * dt);
// 	state.build_basis();
// 	form->solution_changed(sol);
// 	form->first_derivative(sol, force);
// 	double next_functional_val = force.dot(weights);

// 	for (auto &v2s : variable_to_simulations)
// 		v2s->update(x - theta * dt);
// 	state.build_basis();
// 	form->solution_changed(sol);
// 	form->first_derivative(sol, force);
// 	double former_functional_val = force.dot(weights);

// 	double derivative = force_grad.dot(theta);
// 	double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
// 	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
// 	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

// 	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-2));
// }

TEST_CASE("elastic-force", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "homogenize-stress.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "homogenize-stress-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::VectorXd theta;
	theta.setZero(x.size());
	const double eps = 1e-5;
	Eigen::VectorXd min = V.colwise().minCoeff();
	Eigen::VectorXd max = V.colwise().maxCoeff();
	for (int i = 0; i < V.rows(); i++)
		for (int d = 0; d < 2; d++)
		{
			auto vert = state.mesh->point(i);
			if (state.mesh->is_boundary_vertex(i) && vert(0) > min(0) + eps && vert(0) < max(0) - eps && vert(1) > min(1) + eps && vert(1) < max(1) - eps)
				theta(i * 2 + d) = (rand() % 10000) / 1.0e4;
		}

	const double dt = opt_args["solver"]["nonlinear"]["debug_fd_eps"].get<double>();
	Eigen::VectorXd weights, sol;
	{
		std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state.solve_data.nl_problem);
		sol = state.diff_cached.u(0) + io::Evaluator::generate_linear_field(state.n_bases, state.mesh_nodes, state.diff_cached.disp_grad());
		weights.setRandom(sol.size());
	}

	auto form = state.solve_data.elastic_form;

	Eigen::VectorXd force;
	form->first_derivative(sol, force);
	double functional_val = force.dot(weights);

	Eigen::VectorXd force_grad;
	form->force_shape_derivative(x.size(), sol, sol, weights, force_grad);
	force_grad = -utils::flatten(utils::unflatten(force_grad, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));

	for (auto &v2s : variable_to_simulations)
		v2s->update(x + theta * dt);
	state.build_basis();
	// form->solution_changed(sol);
	form->first_derivative(sol, force);
	double next_functional_val = force.dot(weights);

	for (auto &v2s : variable_to_simulations)
		v2s->update(x - theta * dt);
	state.build_basis();
	// form->solution_changed(sol);
	form->first_derivative(sol, force);
	double former_functional_val = force.dot(weights);

	double derivative = force_grad.dot(theta);
	double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-6));
}

TEST_CASE("shape-contact", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-contact.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-contact-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-8, 1e-5);
}

TEST_CASE("node-trajectory", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "node-trajectory.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "node-trajectory-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ElasticVariableToSimulation>(state_ptr, CompositeParametrization()));

	Eigen::MatrixXd targets(state.n_bases, state.mesh->dimension());
	std::vector<int> actives;
	for (int i = 0; i < targets.size(); i++)
		targets(i) = (rand() % 10) / 10.;
	for (int i = 0; i < targets.rows(); i++)
		actives.push_back(i);

	NodeTargetForm obj(state, variable_to_simulations, actives, targets);

	Eigen::VectorXd x(state.mesh->n_elements() * 2);
	x << state.assembler.lame_params().lambda_mat_, state.assembler.lame_params().mu_mat_;

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-5, 1e-5);
}

TEST_CASE("damping-transient", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "damping-transient.json", in_args);
	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "damping-transient-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::string root_path = "";
	if (utils::is_param_valid(opt_args, "root_path"))
		root_path = opt_args["root_path"].get<std::string>();

	// compute reference solution
	json in_args_ref;
	load_json(path + "damping-transient-target.json", in_args_ref);
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<DampingCoeffientVariableToSimulation>(state_ptr, CompositeParametrization()));

	auto obj_aux = std::make_shared<TargetForm>(variable_to_simulations, state, opt_args["functionals"][0]);

	std::vector<int> tmp_ids = opt_args["functionals"][0]["reference_cached_body_ids"];
	std::set<int> reference_cached_body_ids = std::set(tmp_ids.begin(), tmp_ids.end());
	obj_aux->set_reference(state_reference, reference_cached_body_ids);

	TransientForm obj(variable_to_simulations, state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], {}, obj_aux);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(2);

	Eigen::VectorXd x(2);
	x << state.args["materials"]["psi"], state.args["materials"]["phi"];

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-5, 1e-6);
}

TEST_CASE("material-transient", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "material-transient.json", in_args);

	json opt_args;
	load_json(path + "material-transient-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["materials"]["E"] = 1e5;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ElasticVariableToSimulation>(state_ptr, CompositeParametrization()));

	auto obj_aux = std::make_shared<TargetForm>(variable_to_simulations, state, opt_args["functionals"][0]);

	std::vector<int> tmp_ids = opt_args["functionals"][0]["reference_cached_body_ids"];
	obj_aux->set_reference(state_reference, {1, 3});

	TransientForm obj(variable_to_simulations, state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], {}, obj_aux);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(state.bases.size() * 2);
	velocity_discrete *= 1e3;

	Eigen::VectorXd x(velocity_discrete.size());
	x << state.assembler.lame_params().lambda_mat_, state.assembler.lame_params().mu_mat_;

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-5, 1e-4);
}

TEST_CASE("shape-transient-friction", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-transient-friction.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-transient-friction-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
	for (int i = 0; i < state.n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}
	velocity_discrete.normalize();

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("shape-transient-friction-sdf", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-transient-friction-sdf.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "shape-transient-friction-sdf-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	Eigen::MatrixXd control_points;
	Eigen::VectorXd knots;
	double delta;
	control_points.setZero(4, 2);
	control_points << 1, 0.4,
		0.66666667, 0.73333333,
		0.43333333, 1,
		0.1, 1;
	knots.setZero(8);
	knots << 0,
		0,
		0,
		0,
		1,
		1,
		1,
		1;
	delta = 0.05;

	std::shared_ptr<SDFTargetForm> obj_aux = std::make_shared<SDFTargetForm>(variable_to_simulations, state, opt_args["functionals"][0]);
	obj_aux->set_bspline_target(control_points, knots, delta);
	TransientForm obj(variable_to_simulations, state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], {}, obj_aux);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
	for (int i = 0; i < state.n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}

	velocity_discrete.normalize();

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("initial-contact", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "initial-contact.json", in_args);
	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "initial-contact-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::string root_path = "";
	if (utils::is_param_valid(opt_args, "root_path"))
		root_path = opt_args["root_path"].get<std::string>();

	// compute reference solution
	json in_args_ref;
	load_json(path + "initial-contact-target.json", in_args_ref);
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<InitialConditionVariableToSimulation>(state_ptr, CompositeParametrization()));

	auto obj_aux = std::make_shared<TargetForm>(variable_to_simulations, state, opt_args["functionals"][0]);

	std::vector<int> tmp_ids = opt_args["functionals"][0]["reference_cached_body_ids"];
	std::set<int> reference_cached_body_ids = std::set(tmp_ids.begin(), tmp_ids.end());
	obj_aux->set_reference(state_reference, reference_cached_body_ids);

	TransientForm obj(variable_to_simulations, state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], {}, obj_aux);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.ndof() * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(state.ndof() + i * 2 + 0) = -2.;
		velocity_discrete(state.ndof() + i * 2 + 1) = -1.;
	}

	Eigen::VectorXd x(velocity_discrete.size());
	x << state.initial_sol_update, state.initial_vel_update;

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-5, 1e-5);
}

// TEST_CASE("barycenter", "[adjoint_method]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
// 	json in_args;
// 	load_json(path + "barycenter.json", in_args);

// 	json opt_args;
// 	load_json(path + "barycenter-opt.json", opt_args);
// 	opt_args = apply_opt_json_spec(opt_args, false);

// 	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	json shape_arg = opt_args["parameters"][1];

// 	Eigen::MatrixXd centers;
// 	{
// 		auto in_args_ref = in_args;
// 		in_args_ref["initial_conditions"]["velocity"][0]["value"][0] = 4;
// 		in_args_ref["initial_conditions"]["velocity"][0]["value"][1] = -1;
// 		std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);
// 		std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
// 		std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, shape_arg);
// 		BarycenterTargetObjective func_aux(*state_reference, shape_param, opt_args["functionals"][0], Eigen::MatrixXd::Zero(state_reference->diff_cached.size(), state_reference->mesh->dimension()));
// 		centers.setZero(state_reference->diff_cached.size(), state_reference->mesh->dimension());
// 		for (int t = 0; t < state_reference->diff_cached.size(); t++)
// 		{
// 			func_aux.set_time_step(t);
// 			centers.row(t) = func_aux.get_barycenter();
// 		}
// 	}

// 	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
// 	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, shape_arg);
// 	std::shared_ptr<InitialConditionParameter> initial_param = std::make_shared<InitialConditionParameter>(states_ptr, opt_args["parameters"][0]);

// 	std::shared_ptr<StaticObjective> func_aux = std::make_shared<BarycenterTargetObjective>(state, shape_param, opt_args["functionals"][0], centers);
// 	TransientObjective func(state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], func_aux);

// 	Eigen::MatrixXd velocity_discrete;
// 	velocity_discrete.setZero(state.ndof() * 2, 1);
// 	for (int i = 0; i < state.n_bases; i++)
// 	{
// 		velocity_discrete(state.ndof() + i * 2 + 0) = -2.;
// 		velocity_discrete(state.ndof() + i * 2 + 1) = -1.;
// 	}

// 	verify_adjoint(func, state, initial_param, "initial", velocity_discrete, 1e-6, 1e-5);
// }

// TEST_CASE("dirichlet-sdf", "[adjoint_method]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
// 	json in_args;
// 	load_json(path + "dirichlet-sdf.json", in_args);

// 	json opt_args;
// 	load_json(path + "dirichlet-sdf-opt.json", opt_args);
// 	opt_args = apply_opt_json_spec(opt_args, false);

// 	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	Eigen::MatrixXd control_points, delta;
// 	Eigen::VectorXd knots;
// 	control_points.setZero(4, 2);
// 	control_points << -2.5, -0.1,
// 		-1, -1,
// 		1, 1,
// 		2.5, -0.1;
// 	knots.setZero(8);
// 	knots << 0,
// 		0,
// 		0,
// 		0,
// 		1,
// 		1,
// 		1,
// 		1;
// 	delta.setZero(1, 2);
// 	delta << 0.5, 0.5;

// 	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
// 	std::shared_ptr<ControlParameter> control_param = std::make_shared<ControlParameter>(states_ptr, opt_args["parameters"][0]);
// 	auto sdf_aux = std::make_shared<SDFTargetObjective>(state, nullptr, opt_args["functionals"][0]);
// 	sdf_aux->set_bspline_target(control_points, knots, delta(0));
// 	std::shared_ptr<StaticObjective> func_aux = sdf_aux;
// 	json functional_args = opt_args["functionals"][0];

// 	std::string transient_integral_type;
// 	if (opt_args["functionals"][0]["transient_integral_type"] == "steps")
// 	{
// 		auto steps = opt_args["functionals"][0]["steps"].get<std::vector<int>>();
// 		transient_integral_type = "[";
// 		for (auto s : steps)
// 			transient_integral_type += std::to_string(s) + ",";
// 		transient_integral_type.pop_back();
// 		transient_integral_type += "]";
// 	}
// 	else
// 		transient_integral_type = opt_args["functionals"][0]["transient_integral_type"];
// 	TransientObjective func(state.args["time"]["time_steps"], state.args["time"]["dt"], transient_integral_type, func_aux);

// 	int time_steps = state.args["time"]["time_steps"].get<int>();

// 	int dirichlet_dof = 3;
// 	Eigen::MatrixXd velocity_discrete;
// 	velocity_discrete.setZero(time_steps * 3 * dirichlet_dof, 1);
// 	for (int j = 0; j < time_steps; ++j)
// 		for (int k = 0; k < dirichlet_dof; ++k)
// 		{
// 			for (int i = 0; i < 3; ++i)
// 			{
// 				double random_val = (rand() % 200) / 100. - 1.;
// 				velocity_discrete(j * 3 * dirichlet_dof + i * dirichlet_dof + k) = random_val;
// 			}
// 		}

// 	auto initial_guess = control_param->initial_guess();
// 	auto perturb_fn = [&initial_guess](std::shared_ptr<Parameter> param, std::shared_ptr<State> &state, const Eigen::MatrixXd &dx) {
// 		initial_guess += dx;
// 		param->pre_solve(initial_guess);
// 	};

// 	verify_adjoint_dirichlet(func, state_ptr, control_param, velocity_discrete, perturb_fn, 1e-4, 1e-3);

// 	json temp_args = in_args;
// 	auto perturb_fn_json = [&temp_args, time_steps](std::shared_ptr<Parameter> param, std::shared_ptr<State> &state, const Eigen::MatrixXd &dx) {
// 		for (int t = 0; t < time_steps; ++t)
// 			for (int k = 0; k < 2; ++k)
// 				for (int i = 0; i < 3; ++i)
// 					temp_args["boundary_conditions"]["dirichlet_boundary"][i]["value"][k][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][i]["value"][k][t].get<double>() + dx(t * 3 * 3 + i * 3 + k);
// 		state->init(temp_args, false);
// 		state->args["optimization"]["enabled"] = true;
// 	};

// 	verify_adjoint_dirichlet(func, state_ptr, control_param, velocity_discrete, perturb_fn_json, 1e-7, 1e-5);
// }

// TEST_CASE("dirichlet-ref", "[adjoint_method]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
// 	json in_args;
// 	load_json(path + "dirichlet-ref.json", in_args);

// 	json opt_args;
// 	load_json(path + "dirichlet-ref-opt.json", opt_args);
// 	opt_args = apply_opt_json_spec(opt_args, false);

// 	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	int time_steps = state.args["time"]["time_steps"].get<int>();

// 	json ref_args = in_args;
// 	for (int t = 0; t < time_steps; ++t)
// 	{
// 		ref_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][0][t] = ref_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][0][t].get<double>() - 0.5 * t;
// 		ref_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][0][t] = ref_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][0][t].get<double>() + 0.5 * t;
// 	}
// 	std::shared_ptr<State> state_ref = create_state_and_solve(ref_args);

// 	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
// 	std::shared_ptr<ControlParameter> control_param = std::make_shared<ControlParameter>(states_ptr, opt_args["parameters"][0]);
// 	std::shared_ptr<TargetObjective> func_aux = std::make_shared<TargetObjective>(state, nullptr, opt_args["functionals"][0]);
// 	func_aux->set_reference(state_ref, {2});
// 	TransientObjective func(state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], func_aux);

// 	int dirichlet_dof = 3;
// 	Eigen::MatrixXd velocity_discrete;
// 	velocity_discrete.setZero(time_steps * 3 * dirichlet_dof, 1);
// 	for (int j = 0; j < time_steps; ++j)
// 		for (int k = 0; k < dirichlet_dof; ++k)
// 		{
// 			for (int i = 0; i < 3; ++i)
// 			{
// 				double random_val = (rand() % 200) / 100. - 1.;
// 				velocity_discrete(j * 3 * dirichlet_dof + i * dirichlet_dof + k) = random_val;
// 			}
// 		}

// 	auto initial_guess = control_param->initial_guess();
// 	auto perturb_fn = [&initial_guess](std::shared_ptr<Parameter> param, std::shared_ptr<State> &state, const Eigen::MatrixXd &dx) {
// 		initial_guess += dx;
// 		param->pre_solve(initial_guess);
// 	};

// 	verify_adjoint_dirichlet(func, state_ptr, control_param, velocity_discrete, perturb_fn, 1e-4, 1e-3);

// 	json temp_args = in_args;
// 	auto perturb_fn_json = [&temp_args, time_steps](std::shared_ptr<Parameter> param, std::shared_ptr<State> &state, const Eigen::MatrixXd &dx) {
// 		for (int t = 0; t < time_steps; ++t)
// 			for (int k = 0; k < 2; ++k)
// 				for (int i = 0; i < 3; ++i)
// 					temp_args["boundary_conditions"]["dirichlet_boundary"][i]["value"][k][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][i]["value"][k][t].get<double>() + dx(t * 3 * 3 + i * 3 + k);
// 		state->init(temp_args, false);
// 		state->args["optimization"]["enabled"] = true;
// 	};

// 	verify_adjoint_dirichlet(func, state_ptr, control_param, velocity_discrete, perturb_fn_json, 1e-7, 1e-5);
// }