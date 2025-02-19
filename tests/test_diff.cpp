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
#include <polyfem/solver/forms/parametrization/NodeCompositeParametrizations.hpp>
#include <polyfem/solver/AdjointNLProblem.hpp>

#include <catch2/catch_all.hpp>
#include <math.h>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;

namespace
{
	std::string append_root_path(const std::string &path)
	{
		return POLYFEM_DATA_DIR + std::string("/differentiable/input/") + path;
	}

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

	void verify_adjoint(AdjointNLProblem &problem, const Eigen::VectorXd &x, const Eigen::MatrixXd &theta, const double dt, const double tol)
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
		double back_finite_difference = (functional_val - former_functional_val) / dt;
		double front_finite_difference = (next_functional_val - functional_val) / dt;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(16) << "forward fd " << front_finite_difference << " backward fd " << back_finite_difference << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
		std::cout << std::setprecision(12) << "relative error: " << abs((finite_difference - derivative) / derivative) << "\n";
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
	}

	std::tuple<std::shared_ptr<AdjointForm>, VariableToSimulationGroup, std::vector<std::shared_ptr<State>>> prepare_test(json &opt_args)
	{
		opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);
		for (auto &arg : opt_args["states"])
			arg["path"] = append_root_path(arg["path"]);

		std::vector<std::shared_ptr<State>> states = AdjointOptUtils::create_states(opt_args["states"], solver::CacheLevel::Derivatives, 16);

		/* DOF */
		int ndof = 0;
		std::vector<int> variable_sizes;
		for (const auto &arg : opt_args["parameters"])
		{
			int size = AdjointOptUtils::compute_variable_size(arg, states);
			ndof += size;
			variable_sizes.push_back(size);
		}

		/* variable to simulations */
		VariableToSimulationGroup var2sim;
		var2sim.init(opt_args["variable_to_simulation"], states, variable_sizes);

		/* forms */
		std::shared_ptr<AdjointForm> obj = AdjointOptUtils::create_form(
			opt_args["functionals"], var2sim, states);

		return {obj, var2sim, states};
	}
} // namespace

TEST_CASE("laplacian", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("laplacian-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	Eigen::VectorXd x = var2sim[0]->inverse_eval();
	for (auto &v2s : var2sim)
		v2s->update(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = vel(i) * cos(vel(i));
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(*states[0], velocity, velocity_discrete);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-8);
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

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = std::make_shared<PositionForm>(variable_to_simulations, state, opt_args["functionals"][0]);
	obj->set_integral_type(SpatialIntegralType::Surface);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-5);
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

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {}));

	auto obj = std::make_shared<PositionForm>(variable_to_simulations, state, opt_args["functionals"][0]);
	obj->set_integral_type(SpatialIntegralType::Surface);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-6, 1e-5);
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

	CompositeParametrization composite_map({std::make_shared<PowerMap>(5),
											std::make_shared<InsertConstantMap>(state.bases.size(), state.args["materials"]["nu"]),
											std::make_shared<ENu2LambdaMu>(state.mesh->is_volume())});

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(std::make_unique<ElasticVariableToSimulation>(state_ptr, composite_map));

	std::vector<std::shared_ptr<State>> states({state_ptr});
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::MatrixXd theta(state.bases.size(), 1);
	for (int e = 0; e < state.bases.size(); e++)
		theta(e) = (rand() % 1000) / 1000.0;

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();

	verify_adjoint(*nl_problem, x, theta, 1e-2, 1e-4);
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsdiff = "[test_adjoint]";
#else
std::string tagsdiff = "[.][test_adjoint]";
#endif

TEST_CASE("neohookean-stress-3d", tagsdiff)
{
	json opt_args;
	load_json(append_root_path("neohookean-stress-3d-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd V;
	states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-5);
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

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(std::make_unique<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));

	json composite_map_args = R"({
		"composite_map_type": "boundary",
		"surface_selection": [2]
	})"_json;
	variable_to_simulations[0]->set_output_indexing(composite_map_args);

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

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

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-3);
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

// 	VariableToSimulationGroup variable_to_simulations;
// 	variable_to_simulations.push_back(std::make_unique<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));
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
// 													   state.assembler.density(), /*is_formulation_mixed=*/false,
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
// 													   state.assembler.density(), /*is_formulation_mixed=*/false,
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

TEST_CASE("shape-pressure-nodes-2d", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-pressure-nodes-2d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-pressure-nodes-2d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {2}));

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
	x.resize(6);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
	std::cout << "b_idx " << b_idx.size() << std::endl;
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-3);
}

TEST_CASE("static-control-pressure-nodes-3d", "[.][test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "static-control-pressure-nodes-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "static-control-pressure-nodes-3d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	{
		std::vector<int> pressure_boundaries = {2};
		auto v2s = std::make_shared<PressureVariableToSimulation>(state_ptr, CompositeParametrization());
		v2s->set_pressure_boundaries(pressure_boundaries);
		variable_to_simulations.push_back(v2s);
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

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();
	velocity_discrete = velocity(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-3, 1e-3);
}

TEST_CASE("control-pressure-walker-2d", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "walker.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "walker-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	int ndof = 0;
	std::vector<int> variable_sizes;
	for (const auto &arg : opt_args["parameters"])
	{
		int size = solver::AdjointOptUtils::compute_variable_size(arg, states);
		ndof += size;
		variable_sizes.push_back(size);
	}

	VariableToSimulationGroup variable_to_simulations;
	for (const auto &var : opt_args["variable_to_simulation"])
		variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(var, states, variable_sizes));

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

	Eigen::VectorXd x = AdjointOptUtils::inverse_evaluation(opt_args["parameters"], ndof, variable_sizes, variable_to_simulations);
	velocity_discrete = velocity(x);

	std::cout << "x: " << x << std::endl;

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-4, 1e-3);
}

TEST_CASE("shape-walker-2d", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "walker-shape.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "walker-shape-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	int ndof = 0;
	std::vector<int> variable_sizes;
	for (const auto &arg : opt_args["parameters"])
	{
		int size = solver::AdjointOptUtils::compute_variable_size(arg, states);
		ndof += size;
		variable_sizes.push_back(size);
	}

	VariableToSimulationGroup variable_to_simulations;
	for (const auto &var : opt_args["variable_to_simulation"])
		variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(var, states, variable_sizes));

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

	Eigen::VectorXd x = AdjointOptUtils::inverse_evaluation(opt_args["parameters"], ndof, variable_sizes, variable_to_simulations);
	velocity_discrete = velocity(x);

	std::cout << "x: " << x << std::endl;

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-3);
}

TEST_CASE("shape-contact-force-norm", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-contact-force-norm.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-contact-force-norm-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(std::make_unique<ShapeVariableToSimulation>(state_ptr, CompositeParametrization()));

	json composite_map_args = R"({
		"composite_map_type": "boundary_excluding_surface",
		"surface_selection": [1, 2]
	})"_json;
	variable_to_simulations[0]->set_output_indexing(composite_map_args);

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	srand(100);
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

				if (boundary_id == 1 || boundary_id == 2)
					for (long n = 0; n < nodes.size(); ++n)
						node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);

				for (long n = 0; n < nodes.size(); ++n)
					total_bnode_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
			}
		}
		opt_bnodes = total_bnode_ids.size() - node_ids.size();
	}
	x.resize(opt_bnodes * dim);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-3);
}

TEST_CASE("shape-contact-force-norm-3d", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-contact-force-norm-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-contact-force-norm-3d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {42}));

	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	srand(100);
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
	x.resize(42);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-6, 1e-3);
}

TEST_CASE("shape-contact", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("shape-contact-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd V;
	states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	nl_problem->solution_changed(x);
	Eigen::VectorXd one_form;
	nl_problem->gradient(x, one_form);

	verify_adjoint(*nl_problem, x, one_form.normalized(), 1e-7, 1e-5);
}

TEST_CASE("node-trajectory", "[test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "node-trajectory.json", in_args);
	auto state_ptr = AdjointOptUtils::create_state(in_args, solver::CacheLevel::Derivatives, -1);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "node-trajectory-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(std::make_unique<ElasticVariableToSimulation>(state_ptr, CompositeParametrization()));

	Eigen::MatrixXd targets(state.n_bases, state.mesh->dimension());
	std::vector<int> actives;
	for (int i = 0; i < targets.size(); i++)
		targets(i) = (rand() % 10) / 10.;
	for (int i = 0; i < targets.rows(); i++)
		actives.push_back(i);

	auto obj = std::make_shared<NodeTargetForm>(state, variable_to_simulations, actives, targets);

	std::vector<std::shared_ptr<State>> states({state_ptr});
	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();

	Eigen::MatrixXd velocity_discrete(x.size(), 1);
	velocity_discrete.setRandom();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-5, 1e-4);
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

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(std::make_unique<DampingCoeffientVariableToSimulation>(state_ptr, CompositeParametrization()));

	std::vector<std::shared_ptr<State>> states = {state_ptr, state_reference};
	auto obj = AdjointOptUtils::create_form(opt_args["functionals"], variable_to_simulations, states);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(2);

	Eigen::VectorXd x(2);
	x << state.args["materials"]["psi"], state.args["materials"]["phi"];

	verify_adjoint(*nl_problem, x, velocity_discrete, opt_args["solver"]["nonlinear"]["debug_fd_eps"], 1e-4);
}

TEST_CASE("material-transient", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("material-transient-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(states[0]->bases.size() * 2);
	velocity_discrete *= 1e3;

	Eigen::VectorXd x = var2sim[0]->inverse_eval();

	verify_adjoint(*nl_problem, x, velocity_discrete, opt_args["solver"]["nonlinear"]["debug_fd_eps"], 1e-4);
}

TEST_CASE("shape-transient-friction", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("shape-transient-friction-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(states[0]->n_geom_bases * 2, 1);
	for (int i = 0; i < velocity_discrete.size(); ++i)
		velocity_discrete(i) = rand() % 1000;
	velocity_discrete.normalize();

	Eigen::MatrixXd V;
	states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("shape-transient-friction-sdf", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("shape-transient-friction-sdf-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(states[0]->n_geom_bases * 2, 1);
	for (int i = 0; i < velocity_discrete.size(); ++i)
		velocity_discrete(i) = rand() % 1000;
	velocity_discrete.normalize();

	Eigen::MatrixXd V;
	states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-5);
}

TEST_CASE("3d-shape-mesh-target", "[.][test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("3d-shape-mesh-target-opt.json"), opt_args);
	opt_args["functionals"][0]["mesh_path"] = POLYFEM_DATA_DIR + std::string("/") + opt_args["functionals"][0]["mesh_path"].get<std::string>();
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::VectorXd x(opt_args["parameters"][0]["number"].get<int>());
	Eigen::MatrixXd velocity_discrete;

	Eigen::MatrixXd V;
	states[0]->get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = var2sim[0]->get_output_indexing(x);
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete.setZero(x.size(), 1);
	for (int i = 0; i < velocity_discrete.size(); ++i)
		velocity_discrete(i) = rand() % 1000;
	velocity_discrete.normalize();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-5);
}

TEST_CASE("initial-contact", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("initial-contact-opt.json"), opt_args);
	json new_obj = R"({
            "type": "min-dist-target",
            "state": 0,
            "target": [0.05, 0.2],
            "volume_selection": [1],
            "steps": [2, 4]
        })"_json;
	opt_args["functionals"].push_back(new_obj);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setRandom(states[0]->ndof() * 2, 1);

	Eigen::VectorXd x = var2sim[0]->inverse_eval();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-5, 1e-5);
}

TEST_CASE("friction-contact", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("initial-contact-opt.json"), opt_args);
	opt_args["variable_to_simulation"][0]["type"] = "friction";
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::VectorXd velocity_discrete(1);
	velocity_discrete << 1.;

	Eigen::VectorXd x(1);
	x << 0.2;

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-4, 1e-6);
}

TEST_CASE("barycenter", "[test_adjoint]")
{
	json opt_args;
	load_json(append_root_path("barycenter-opt.json"), opt_args);
	auto [obj, var2sim, states] = prepare_test(opt_args);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(states[0]->ndof() * 2, 1);
	for (int i = 0; i < states[0]->n_bases; i++)
	{
		velocity_discrete(states[0]->ndof() + i * 2 + 0) = -2.;
		velocity_discrete(states[0]->ndof() + i * 2 + 1) = -1.;
	}

	Eigen::VectorXd x = var2sim[0]->inverse_eval();

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-6, 1e-5);
}

// TEST_CASE("dirichlet-sdf", "[test_adjoint]")
// {
// 	json opt_args;
// 	load_json(append_root_path("dirichlet-sdf-opt.json"), opt_args);
// 	auto [obj, var2sim, states] = prepare_test(opt_args);

// 	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, var2sim, states, opt_args);

// 	int dirichlet_dof = 3;
// 	Eigen::MatrixXd velocity_discrete;
// 	velocity_discrete.setRandom(states[0]->args["time"]["time_steps"].get<int>() * states[0]->mesh->dimension() * dirichlet_dof, 1);

// 	Eigen::VectorXd x(60);
// 	x.segment(0, 20) = var2sim[0]->inverse_eval();
// 	x.segment(20, 20) = var2sim[1]->inverse_eval();
// 	x.segment(40, 20) = var2sim[2]->inverse_eval();

// 	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-5, 1e-5);
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

TEST_CASE("shape-pressure-nodes-3d", "[.][test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "shape-pressure-nodes-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "shape-pressure-nodes-3d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	variable_to_simulations.push_back(AdjointOptUtils::create_variable_to_simulation(opt_args["variable_to_simulation"][0], states, {972}));

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
	x.resize(972);

	Eigen::MatrixXd V;
	state.get_vertices(V);
	Eigen::VectorXd V_flat = utils::flatten(V);
	auto b_idx = variable_to_simulations[0]->get_output_indexing(x);
	std::cout << "b_idx " << b_idx.size() << std::endl;
	for (int i = 0; i < b_idx.size(); ++i)
		x(i) = V_flat(b_idx(i));
	velocity_discrete = velocity(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-7, 1e-3);
}

TEST_CASE("control-pressure-nodes-3d", "[.][test_adjoint]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/differentiable/input/");
	json in_args;
	load_json(path + "control-pressure-nodes-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states({state_ptr});

	json opt_args;
	load_json(path + "control-pressure-nodes-3d-opt.json", opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	VariableToSimulationGroup variable_to_simulations;
	{
		auto v2s = std::make_shared<PressureVariableToSimulation>(state_ptr, CompositeParametrization());

		json composite_map_args = R"({
			"composite_map_type": "indices",
			"composite_map_indices": [0,1,2,3,4],
			"surface_selection": [2]
		})"_json;

		v2s->set_output_indexing(composite_map_args);
		variable_to_simulations.push_back(v2s);
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

	Eigen::VectorXd x = variable_to_simulations[0]->inverse_eval();
	velocity_discrete = velocity(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(obj, variable_to_simulations, states, opt_args);

	verify_adjoint(*nl_problem, x, velocity_discrete, 1e-8, 1e-3);
}
