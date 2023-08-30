///////////////////////////////////////////////////////////////////////////////
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

#include <polyfem/State.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/AdjointTools.hpp>

#include <polyfem/solver/forms/adjoint_forms/SmoothingForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/TargetForms.hpp>
#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>
#include <polyfem/solver/forms/parametrization/NodeCompositeParametrizations.hpp>

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
		std::shared_ptr<State> state = AdjointOptUtils::create_state(args);
		Eigen::MatrixXd sol, pressure;
		state->solve_problem(sol, pressure);

		return state;
	}

	std::shared_ptr<State> create_state_and_solve(const json &args, Eigen::MatrixXd &sol)
	{
		std::shared_ptr<State> state = AdjointOptUtils::create_state(args);
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
		AdjointOptUtils::solve_pde(state);
		obj.solution_changed(x + theta * dt);
		double next_functional_val = obj.value(x + theta * dt);

		for (auto &v2s : variable_to_simulations)
			v2s->update(x - theta * dt);
		state.build_basis();
		AdjointOptUtils::solve_pde(state);
		obj.solution_changed(x - theta * dt);
		double former_functional_val = obj.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
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
		AdjointOptUtils::solve_pde(state);
		obj.solution_changed(x + theta * dt);
		double next_functional_val = obj.value(x + theta * dt);

		for (auto &v2s : variable_to_simulations)
			v2s->update(x - theta * dt);
		state.build_basis();
		AdjointOptUtils::solve_pde(state);
		obj.solution_changed(x - theta * dt);
		double former_functional_val = obj.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
		std::cout << std::setprecision(12) << "relative error: " << abs((finite_difference - derivative) / derivative) << "\n";
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
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
			AdjointOptUtils::solve_pde(state);
			obj.solution_changed(x + theta * dt);
			double next_functional_val = obj.value(x + theta * dt);

			for (auto &v2s : variable_to_simulations)
				v2s->update(x - theta * dt);
			state.build_basis();
			AdjointOptUtils::solve_pde(state);
			obj.solution_changed(x - theta * dt);
			double former_functional_val = obj.value(x - theta * dt);

			fd(d) = (next_functional_val - former_functional_val) / dt / 2;
		}

		std::cout << "fd: " << fd.transpose() << "\n";
	}

} // namespace

TEST_CASE("laplacian", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "laplacian.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "laplacian-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

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

TEST_CASE("boundary-smoothing", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "laplacian.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "laplacian-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	BoundarySmoothingForm obj(variable_to_simulations, state, true, 3);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setRandom(x.size(), 1);

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("linear_elasticity-surface-3d", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "linear_elasticity-surface-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "linear_elasticity-surface-3d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	PositionForm obj(variable_to_simulations, state, opt_args["functionals"][0]);
	obj.set_integral_type(SpatialIntegralType::surface);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-7, 1e-5);
}

TEST_CASE("linear_elasticity-surface", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "linear_elasticity-surface.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "linear_elasticity-surface-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states({state_ptr});

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	PositionForm obj(variable_to_simulations, state, opt_args["functionals"][0]);
	obj.set_integral_type(SpatialIntegralType::surface);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("topology-compliance", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "topology-compliance.json", in_args);

	json opt_args;
	load_json(path + "topology-compliance-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<Parametrization>> map_list = {std::make_shared<PowerMap>(5), std::make_shared<InsertConstantMap>(state.bases.size(), state.args["materials"]["nu"]), std::make_shared<ENu2LambdaMu>(state.mesh->is_volume())};
	CompositeParametrization composite_map(map_list);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ElasticVariableToSimulation>(state_ptr, composite_map));

	std::vector<std::shared_ptr<State>> states({state_ptr});
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd theta(state.bases.size(), 1);
	for (int e = 0; e < state.bases.size(); e++)
		theta(e) = (rand() % 1000) / 1000.0;

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();

	for (auto &v2s : variable_to_simulations)
		v2s->update(x);
	state.build_basis();
	AdjointOptUtils::solve_pde(state);

	verify_adjoint(variable_to_simulations, *obj, state, x, theta, 1e-4, 1e-2);
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsdiff = "[test_adjoint]";
#else
std::string tagsdiff = "[.][test_adjoint]";
#endif

TEST_CASE("neohookean-stress-3d", tagsdiff)
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "neohookean-stress-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "neohookean-stress-3d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 1e-5);
}

TEST_CASE("shape-neumann-nodes", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-neumann-nodes.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-neumann-nodes-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));
	{
		VariableToBoundaryNodes variable_to_node(*state_ptr, 2);
		variable_to_simulations[0]->set_output_indexing(variable_to_node.get_output_indexing());
	}

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

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
	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 1e-3);
}

// TEST_CASE("neumann-shape-derivative", "[test_adjoint]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
// 	json in_args;
// 	load_json(path + "shape-pressure-neumann-nodes.json", in_args);
// 	Eigen::MatrixXd sol;
// 	auto state_ptr = create_state_and_solve(in_args, sol);
// 	State &state = *state_ptr;

// 	auto velocity = [](const Eigen::MatrixXd &position) {
// 		auto vel = position;
// 		for (int i = 0; i < vel.size(); i++)
// 		{
// 			vel(i) = (rand() % 1000) / 1000.0;
// 		}
// 		return vel;
// 	};
// 	Eigen::MatrixXd velocity_discrete;

// 	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
// 	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));
// 	{
// 		VariableToBoundaryNodes variable_to_node(*state_ptr, 2);
// 		variable_to_simulations[0]->set_output_indexing(variable_to_node.get_output_indexing());
// 	}

// 	Eigen::VectorXd x;
// 	int opt_bnodes = 0;
// 	int dim;
// 	{
// 		const auto &mesh = state.mesh;
// 		const auto &bases = state.bases;
// 		const auto &gbases = state.geom_bases();
// 		dim = mesh->dimension();

// 		std::set<int> node_ids;
// 		std::set<int> total_bnode_ids;
// 		for (const auto &lb : state.total_local_boundary)
// 		{
// 			const int e = lb.element_id();
// 			for (int i = 0; i < lb.size(); ++i)
// 			{
// 				const int primitive_global_id = lb.global_primitive_id(i);
// 				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
// 				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

// 				if (boundary_id == 2)
// 					for (long n = 0; n < nodes.size(); ++n)
// 						node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
// 			}
// 		}
// 		opt_bnodes = node_ids.size();
// 	}
// 	x.resize(opt_bnodes * dim);

// 	Eigen::MatrixXd V;
// 	state.get_vertices(V);
// 	Eigen::VectorXd V_flat = utils::flatten(V);
// 	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
// 	for (int i = 0; i < b_idx.size(); ++i)
// 		x(i) = V_flat(b_idx(i));
// 	velocity_discrete = velocity(x);

// 	double t0 = 0;
// 	double dt = 0.025;
// 	double timesteps = 10;

// 	double eps = 1e-8;

// 	std::vector<Eigen::MatrixXd> u;
// 	for (int i = 0; i <= timesteps; ++i)
// 		u.push_back(state.diff_cached.u(i));

// 	std::vector<Eigen::MatrixXd> hess_vec;
// 	for (int i = 1; i <= timesteps; ++i)
// 	{
// 		Eigen::MatrixXd hess(state.n_bases * state.mesh->dimension(), x.size());

// 		for (int k = 0; k < state.n_bases * state.mesh->dimension(); ++k)
// 		{
// 			Eigen::VectorXd indicator = Eigen::VectorXd::Zero(state.n_bases * state.mesh->dimension());
// 			indicator(k) = 1;
// 			Eigen::VectorXd term;
// 			state.solve_data.body_form->force_shape_derivative(state.n_geom_bases, t0 + i * dt, u[i], indicator, term);
// 			hess.row(k) = variable_to_simulations[0]->apply_parametrization_jacobian(term, x);
// 		}

// 		hess_vec.push_back(hess);
// 	}

// 	std::vector<Eigen::MatrixXd> hess_fd_vec;
// 	for (int i = 1; i <= timesteps; ++i)
// 	{
// 		Eigen::MatrixXd hess_fd(state.n_bases * state.mesh->dimension(), x.size());

// 		for (int k = 0; k < x.size(); ++k)
// 		{

// 			Eigen::VectorXd h_plus, h_minus;
// 			{
// 				Eigen::VectorXd y = x;
// 				y(k) += eps;
// 				variable_to_simulations[0]->update(y);
// 				state.build_basis();
// 				state.assemble_rhs();
// 				state.assemble_stiffness_mat();
// 				auto form = std::make_shared<BodyForm>(state.n_bases * dim, state.n_pressure_bases, state.boundary_nodes, state.local_boundary,
// 													   state.local_neumann_boundary, state.n_boundary_samples(), state.rhs, *state.solve_data.rhs_assembler,
// 													   state.assembler.density(), /*apply_DBC=*/true, /*is_formulation_mixed=*/false,
// 													   state.problem->is_time_dependent());
// 				form->update_quantities(t0 + i * dt, u[i - 1]);

// 				form->first_derivative(u[i], h_plus);
// 				h_plus /= form->weight();
// 			}

// 			{
// 				Eigen::VectorXd y = x;
// 				y(k) -= eps;
// 				variable_to_simulations[0]->update(y);
// 				state.build_basis();
// 				state.assemble_rhs();
// 				state.assemble_stiffness_mat();
// 				auto form = std::make_shared<BodyForm>(state.n_bases * dim, state.n_pressure_bases, state.boundary_nodes, state.local_boundary,
// 													   state.local_neumann_boundary, state.n_boundary_samples(), state.rhs, *state.solve_data.rhs_assembler,
// 													   state.assembler.density(), /*apply_DBC=*/true, /*is_formulation_mixed=*/false,
// 													   state.problem->is_time_dependent());
// 				form->update_quantities(t0 + i * dt, u[i - 1]);

// 				form->first_derivative(u[i], h_minus);
// 				h_minus /= form->weight();
// 			}

// 			hess_fd.col(k) = (h_plus - h_minus) / (2 * eps);
// 		}

// 		hess_fd_vec.push_back(hess_fd);
// 	}

// 	for (int i = 1; i <= timesteps; ++i)
// 	{
// 		std::cout << "comparison" << std::endl;
// 		std::cout << "norm of difference " << (hess_fd_vec[i - 1] - hess_vec[i - 1]).norm() << std::endl;
// 		std::cout << "norm of derivative " << hess_vec[i - 1].norm() << std::endl;
// 		// std::cout << hess_vec[i - 1] << std::endl;
// 		std::cout << "norm of fd " << hess_fd_vec[i - 1].norm() << std::endl;
// 		// std::cout << hess_fd_vec[i - 1] << std::endl;
// 	}
// }

// TEST_CASE("neumann-u-derivative", "[test_adjoint]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
// 	json in_args;
// 	load_json(path + "shape-pressure-neumann-nodes.json", in_args);
// 	Eigen::MatrixXd sol;
// 	auto state_ptr = create_state_and_solve(in_args, sol);
// 	State &state = *state_ptr;

// 	double t0 = 0;
// 	double dt = 0.025;
// 	double timesteps = 10;

// 	double eps = 1e-7;

// 	state.solve_data.body_form->set_project_to_psd(false);

// 	std::vector<Eigen::MatrixXd> hess_vec;
// 	for (int i = 1; i <= timesteps; ++i)
// 	{
// 		StiffnessMatrix hess;
// 		state.solve_data.body_form->hessian_wrt_u_prev(state.diff_cached.u(i - 1), t0 + i * dt, hess);

// 		std::cout << "solution norm: " << state.diff_cached.u(i).norm() << std::endl;

// 		std::cout << "hessian norm: " << hess.norm() << std::endl;
// 		// std::cout << hess << std::endl;

// 		Eigen::MatrixXd hess_fd(hess.rows(), hess.cols());
// 		for (int j = 0; j < state.diff_cached.u(i).size(); ++j)
// 		{
// 			Eigen::MatrixXd u_prev = state.diff_cached.u(i - 1);
// 			Eigen::MatrixXd u = state.diff_cached.u(i);

// 			Eigen::VectorXd h_plus, h_minus;
// 			u_prev(j) += eps;
// 			state.solve_data.body_form->update_quantities(t0 + i * dt, u_prev);
// 			state.solve_data.body_form->first_derivative(u, h_plus);
// 			h_plus /= state.solve_data.body_form->weight();
// 			u_prev(j) -= 2 * eps;
// 			state.solve_data.body_form->update_quantities(t0 + i * dt, u_prev);
// 			state.solve_data.body_form->first_derivative(u, h_minus);
// 			h_minus /= state.solve_data.body_form->weight();
// 			Eigen::VectorXd fd = (h_plus - h_minus) / (2 * eps);

// 			hess_fd.col(j) = fd;
// 		}

// 		std::cout << "fd norm: " << hess_fd.norm() << std::endl;
// 		// std::cout << hess_fd.sparseView(1, 1e-15) << std::endl;

// 		std::cout << "difference is " << (hess_fd.sparseView() - hess).norm() << std::endl;
// 	}
// }

TEST_CASE("shape-pressure-neumann-nodes", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-pressure-neumann-nodes.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-pressure-neumann-nodes-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));
	{
		VariableToBoundaryNodes variable_to_node(*state_ptr, 2);
		variable_to_simulations[0]->set_output_indexing(variable_to_node.get_output_indexing());
	}

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

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
	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-8, 1e-3);
}

// TEST_CASE("shape-contact-force-norm", "[test_adjoint]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
// 	json in_args;
// 	load_json(path + "shape-contact-force-norm.json", in_args);
// 	auto state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	std::vector<std::shared_ptr<State>> states({state_ptr});

// 	json opt_args;
// 	load_json(path + "shape-contact-force-norm-opt.json", opt_args);
// 	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

// 	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
// 	variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));
// 	{
// 		VariableToBoundaryNodesExclusive variable_to_node(*state_ptr, {1, 2});
// 		variable_to_simulations[0]->set_output_indexing(variable_to_node.get_output_indexing());
// 	}

// 	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

// 	srand(100);
// 	auto velocity = [](const Eigen::MatrixXd &position) {
// 		auto vel = position;
// 		for (int i = 0; i < vel.size(); i++)
// 		{
// 			vel(i) = (rand() % 1000) / 1000.0;
// 		}
// 		return vel;
// 	};
// 	Eigen::MatrixXd velocity_discrete;

// 	Eigen::VectorXd x;
// 	int opt_bnodes = 0;
// 	int dim;
// 	{
// 		const auto &mesh = state.mesh;
// 		const auto &bases = state.bases;
// 		const auto &gbases = state.geom_bases();
// 		dim = mesh->dimension();

// 		std::set<int> node_ids;
// 		std::set<int> total_bnode_ids;
// 		for (const auto &lb : state.total_local_boundary)
// 		{
// 			const int e = lb.element_id();
// 			for (int i = 0; i < lb.size(); ++i)
// 			{
// 				const int primitive_global_id = lb.global_primitive_id(i);
// 				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
// 				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

// 				if (boundary_id == 1 || boundary_id == 2)
// 					for (long n = 0; n < nodes.size(); ++n)
// 						node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);

// 				for (long n = 0; n < nodes.size(); ++n)
// 					total_bnode_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
// 			}
// 		}
// 		opt_bnodes = total_bnode_ids.size() - node_ids.size();
// 	}
// 	x.resize(opt_bnodes * dim);

// 	Eigen::MatrixXd V;
// 	state.get_vertices(V);
// 	Eigen::VectorXd V_flat = utils::flatten(V);
// 	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
// 	for (int i = 0; i < b_idx.size(); ++i)
// 		x(i) = V_flat(b_idx(i));
// 	velocity_discrete = velocity(x);

// 	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 1e-3);
// }

TEST_CASE("shape-contact", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-contact.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-contact-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(variable_to_simulations, *obj, state, x, 1e-8, 1e-5);
}

TEST_CASE("node-trajectory", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "node-trajectory.json", in_args);
	auto state_ptr = AdjointOptUtils::create_state(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "node-trajectory-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ElasticVariableToSimulation>(state_ptr, CompositeParametrization()));

	Eigen::MatrixXd targets(state.n_bases, state.mesh->dimension());
	std::vector<int> actives;
	for (int i = 0; i < targets.size(); i++)
		targets(i) = (rand() % 10) / 10.;
	for (int i = 0; i < targets.rows(); i++)
		actives.push_back(i);

	NodeTargetForm obj(state, variable_to_simulations, actives, targets);

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();
	variable_to_simulations[0]->update(x);
	AdjointOptUtils::solve_pde(state);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(variable_to_simulations, obj, state, x, velocity_discrete, 1e-5, 1e-4);
}

TEST_CASE("damping-transient", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "damping-transient.json", in_args);
	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "damping-transient-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::string root_path = "";
	if (utils::is_param_valid(opt_args, "root_path"))
		root_path = opt_args["root_path"].get<std::string>();

	// compute reference solution
	json in_args_ref;
	load_json(path + "damping-transient-target.json", in_args_ref);
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<DampingCoeffientVariableToSimulation>(state_ptr, CompositeParametrization()));

	std::vector<std::shared_ptr<State>> states = {state_ptr, state_reference};
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(2);

	Eigen::VectorXd x(2);
	x << state.args["materials"]["psi"], state.args["materials"]["phi"];

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, opt_args["solver"]["nonlinear"]["debug_fd_eps"], 1e-4);
}

TEST_CASE("material-transient", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "material-transient.json", in_args);

	json opt_args;
	load_json(path + "material-transient-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	// compute reference solution
	auto in_args_ref = in_args;
	in_args_ref["materials"]["E"] = 1e5;
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::shared_ptr<State> state_ptr = AdjointOptUtils::create_state(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<ElasticVariableToSimulation>(state_ptr, CompositeParametrization()));

	std::vector<std::shared_ptr<State>> states = {state_ptr, state_reference};
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(state.bases.size() * 2);
	velocity_discrete *= 1e3;

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();
	variable_to_simulations[0]->update(x);
	AdjointOptUtils::solve_pde(state);

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, opt_args["solver"]["nonlinear"]["debug_fd_eps"], 1e-4);
}

TEST_CASE("shape-transient-friction", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-transient-friction.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-transient-friction-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

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

// TEST_CASE("shape-transient-friction-sdf", "[test_adjoint]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
// 	json in_args;
// 	load_json(path + "shape-transient-friction-sdf.json", in_args);
// 	auto state_ptr = create_state_and_solve(in_args);
// 	State &state = *state_ptr;

// 	json opt_args;
// 	load_json(path + "shape-transient-friction-sdf-opt.json", opt_args);
// 	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

// 	std::vector<std::shared_ptr<State>> states({state_ptr});

// 	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
// 	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

// 	// Eigen::MatrixXd control_points;
// 	// Eigen::VectorXd knots;
// 	// double delta;
// 	// control_points.setZero(4, 2);
// 	// control_points << 1, 0.4,
// 	// 	0.66666667, 0.73333333,
// 	// 	0.43333333, 1,
// 	// 	0.1, 1;
// 	// knots.setZero(8);
// 	// knots << 0,
// 	// 	0,
// 	// 	0,
// 	// 	0,
// 	// 	1,
// 	// 	1,
// 	// 	1,
// 	// 	1;
// 	// delta = 0.05;

// 	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

// 	Eigen::MatrixXd velocity_discrete;
// 	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
// 	for (int i = 0; i < state.n_geom_bases; ++i)
// 	{
// 		velocity_discrete(i * 2 + 0) = rand() % 1000;
// 		velocity_discrete(i * 2 + 1) = rand() % 1000;
// 	}

// 	velocity_discrete.normalize();

// 	Eigen::MatrixXd V;
// 	state.get_vertices(V);
// 	Eigen::VectorXd x = utils::flatten(V);

// 	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-7, 1e-5);
// }

TEST_CASE("initial-contact", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "initial-contact.json", in_args);
	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "initial-contact-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::string root_path = "";
	if (utils::is_param_valid(opt_args, "root_path"))
		root_path = opt_args["root_path"].get<std::string>();

	// compute reference solution
	json in_args_ref;
	load_json(path + "initial-contact-target.json", in_args_ref);
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<InitialConditionVariableToSimulation>(state_ptr, CompositeParametrization()));

	std::vector<std::shared_ptr<State>> states({state_ptr, state_reference});
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setRandom(state.ndof() * 2, 1);
	// for (int i = 0; i < state.n_bases; i++)
	// {
	// 	velocity_discrete(state.ndof() + i * 2 + 0) = -2.;
	// 	velocity_discrete(state.ndof() + i * 2 + 1) = -1.;
	// }

	Eigen::VectorXd x(velocity_discrete.size());
	x << state.initial_sol_update, state.initial_vel_update;

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-5, 1e-5);
}

TEST_CASE("barycenter", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "barycenter.json", in_args);
	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "barycenter-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	std::string root_path = "";
	if (utils::is_param_valid(opt_args, "root_path"))
		root_path = opt_args["root_path"].get<std::string>();

	// compute reference solution
	json in_args_ref;
	load_json(path + "barycenter-target.json", in_args_ref);
	std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	variable_to_simulations.push_back(std::make_shared<InitialConditionVariableToSimulation>(state_ptr, CompositeParametrization()));

	std::vector<std::shared_ptr<State>> states({state_ptr, state_reference});
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.ndof() * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(state.ndof() + i * 2 + 0) = -2.;
		velocity_discrete(state.ndof() + i * 2 + 1) = -1.;
	}

	Eigen::VectorXd x(velocity_discrete.size());
	x << state.initial_sol_update, state.initial_vel_update;

	verify_adjoint(variable_to_simulations, *obj, state, x, velocity_discrete, 1e-6, 1e-5);
}

// TEST_CASE("dirichlet-sdf", "[test_adjoint]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
// 	json in_args;
// 	load_json(path + "dirichlet-sdf.json", in_args);

// 	json opt_args;
// 	load_json(path + "dirichlet-sdf-opt.json", opt_args);
// 	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

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
// 		state->optimization_enabled = true;
// 	};

// 	verify_adjoint_dirichlet(func, state_ptr, control_param, velocity_discrete, perturb_fn_json, 1e-7, 1e-5);
// }

// TEST_CASE("dirichlet-ref", "[test_adjoint]")
// {
// 	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
// 	json in_args;
// 	load_json(path + "dirichlet-ref.json", in_args);

// 	json opt_args;
// 	load_json(path + "dirichlet-ref-opt.json", opt_args);
// 	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

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
// 		state->optimization_enabled = true;
// 	};

// 	verify_adjoint_dirichlet(func, state_ptr, control_param, velocity_discrete, perturb_fn_json, 1e-7, 1e-5);
// }
