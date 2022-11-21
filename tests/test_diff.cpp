////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/solver/InitialConditionParameter.hpp>
#include <polyfem/solver/TopologyOptimizationParameter.hpp>
#include <polyfem/solver/Objective.hpp>
#include <polyfem/solver/AdjointForm.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <jse/jse.h>

#include <polyfem/solver/DampingParameter.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>

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

	void perturb_mesh(State &state, const Eigen::MatrixXd &perturbation)
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;

		state.get_vf(V, F);
		V += utils::unflatten(perturbation, V.cols());

		state.set_mesh_vertices(V);
		state.build_basis();
	}

	void perturb_material(assembler::AssemblerUtils &assembler, const Eigen::MatrixXd &perturbation)
	{
		const auto &cur_lambdas = assembler.lame_params().lambda_mat_;
		const auto &cur_mus = assembler.lame_params().mu_mat_;

		Eigen::MatrixXd lambda_update(cur_lambdas.size(), 1), mu_update(cur_mus.size(), 1);
		for (int i = 0; i < lambda_update.size(); i++)
		{
			lambda_update(i) = perturbation(i);
			mu_update(i) = perturbation(i + lambda_update.size());
		}

		assembler.update_lame_params(cur_lambdas + lambda_update, cur_mus + mu_update);
	}

	// TODO: call parameter class instead
	void perturb(State &state, const Eigen::MatrixXd &dx, const std::string &type)
	{
		if (type == "shape")
			perturb_mesh(state, dx);
		else if (type == "initial")
		{
			assert(dx.cols() == 1);
			Eigen::VectorXd dx_ = dx;
			state.initial_sol_update += dx_.head(state.ndof());
			state.initial_vel_update += dx_.tail(state.ndof());
		}
		else if (type == "material")
			perturb_material(state.assembler, dx);
		else if (type == "damping")
		{
			state.args["materials"]["psi"] = state.args["materials"]["psi"].get<double>() + dx(0);
			state.args["materials"]["phi"] = state.args["materials"]["phi"].get<double>() + dx(1);
			state.set_materials();
		}
		else
			log_and_throw_error("Unknown type of perturbation!");
	}

	std::shared_ptr<State> create_state_and_solve(const json &args)
	{
		std::shared_ptr<State> state = create_state(args);
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
			const bool use_bases = order > 1;
			const int n_current_bases = use_bases ? state.n_bases : state.n_geom_bases;
			const auto &current_bases = use_bases ? state.bases : state.geom_bases();
			discrete_field.setZero(n_current_bases * actual_dim, 1);

			for (int e = 0; e < state.bases.size(); e++)
			{
				Eigen::MatrixXd local_pts, pts;
				if (!state.mesh->is_volume())
					autogen::p_nodes_2d(current_bases[e].bases.front().order(), local_pts);
				else
					autogen::p_nodes_3d(current_bases[e].bases.front().order(), local_pts);

				current_bases[e].eval_geom_mapping(local_pts, pts);
				Eigen::MatrixXd result = field(pts);
				for (int i = 0; i < local_pts.rows(); i++)
				{
					assert(current_bases[e].bases[i].global().size() == 1);
					for (int d = 0; d < actual_dim; d++)
						discrete_field(current_bases[e].bases[i].global()[0].index * actual_dim + d) = result(i, d);
				}
			}
		}
		else if (order == 0)
		{
			discrete_field.setZero(state.bases.size() * actual_dim, 1);
			Eigen::MatrixXd centers;
			if (state.mesh->is_volume())
				state.mesh->cell_barycenters(centers);
			else
				state.mesh->face_barycenters(centers);
			Eigen::MatrixXd result = field(centers);
			for (int e = 0; e < state.bases.size(); e++)
				for (int d = 0; d < actual_dim; d++)
					discrete_field(e * actual_dim + d) = result(e, d);
		}
	}

	void verify_adjoint(Objective &obj, State &state, const std::shared_ptr<Parameter> &param, const std::string &type, const Eigen::MatrixXd &theta, const double dt, const double tol)
	{
		double functional_val = obj.value();

		state.solve_adjoint(obj.compute_adjoint_rhs(state));
		Eigen::VectorXd one_form = obj.gradient(state, *param);
		double derivative = (one_form.array() * theta.array()).sum();

		perturb(state, theta * dt, type);
		solve_pde(state);
		double next_functional_val = obj.value();

		perturb(state, theta * (-2 * dt), type);
		solve_pde(state);
		double former_functional_val = obj.value();

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
		std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

		REQUIRE(derivative == Approx(finite_difference).epsilon(tol));
	}

} // namespace

TEST_CASE("deformed_boundary_smoothing", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "linear_elasticity-surface.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "linear_elasticity-surface-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);

	json obj_args = R"(
	{
		"type": "deformed_boundary_smoothing",
		"surface_selection": [2, 3, 4],
		"power": 4
	})"_json;

	DeformedBoundarySmoothingObjective obj(state, shape_param, obj_args);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
	for (int i = 0; i < state.n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}

	velocity_discrete.normalize();

	verify_adjoint(obj, state, shape_param, "shape", velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("laplacian", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "laplacian.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "laplacian-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);
	StressObjective func(state, shape_param, NULL, opt_args["functionals"][0], false);

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

	verify_adjoint(func, state, shape_param, "shape", velocity_discrete, 1e-7, 3e-5);
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

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);
	PositionObjective obj(state, shape_param, opt_args["functionals"][0]);
	obj.set_integral_type(AdjointForm::SpatialIntegralType::SURFACE);

	auto velocity = [](const Eigen::MatrixXd &position) {
		Eigen::MatrixXd vel;
		vel.setZero(position.rows(), position.cols());
		for (int i = 0; i < vel.rows(); i++)
		{
			vel(i, 0) = position(i, 0);
			vel(i, 1) = position(i, 0) * position(i, 0);
			vel(i, 2) = position(i, 0);
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(state, velocity, velocity_discrete);

	verify_adjoint(obj, state, shape_param, "shape", velocity_discrete, 1e-7, 1e-4);
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

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);
	PositionObjective obj(state, shape_param, opt_args["functionals"][0]);
	obj.set_integral_type(AdjointForm::SpatialIntegralType::SURFACE);

	auto velocity = [](const Eigen::MatrixXd &position) {
		Eigen::MatrixXd vel;
		vel.setZero(position.rows(), position.cols());
		for (int i = 0; i < vel.rows(); i++)
		{
			vel(i, 0) = position(i, 0);
			vel(i, 1) = position(i, 0) * position(i, 0);
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(state, velocity, velocity_discrete);

	verify_adjoint(obj, state, shape_param, "shape", velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("topology-compliance", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "topology-compliance.json", in_args);

	json opt_args;
	load_json(path + "topology-compliance-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::shared_ptr<State> state_ptr = std::make_shared<State>(1);
	state_ptr->init_logger("", spdlog::level::level_enum::warn, false);
	state_ptr->init(in_args, false);
	state_ptr->load_mesh();
	state_ptr->build_basis();
	State &state = *state_ptr;

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<TopologyOptimizationParameter> topo_param = std::make_shared<TopologyOptimizationParameter>(states_ptr, opt_args["parameters"][0]);
	ComplianceObjective func(state, NULL, NULL, topo_param, opt_args["functionals"][0]);

	solve_pde(state);

	Eigen::MatrixXd theta(state.bases.size(), 1);
	for (int e = 0; e < state.bases.size(); e++)
		theta(e) = (rand() % 1000) / 1000.0;

	// verify_adjoint(func, state, topo_param, "topology", theta, 1e-6, 1e-6);
	const double dt = 1e-6;
	const double tol = 1e-6;
	double functional_val = func.value();

	state.solve_adjoint(func.compute_adjoint_rhs(state));
	Eigen::VectorXd one_form = topo_param->map_grad(topo_param->initial_guess(), func.gradient(state, *topo_param));
	double derivative = (one_form.array() * theta.array()).sum();

	topo_param->pre_solve(topo_param->initial_guess() + dt * theta);
	solve_pde(state);
	double next_functional_val = func.value();

	topo_param->pre_solve(topo_param->initial_guess() - dt * theta);
	solve_pde(state);
	double former_functional_val = func.value();

	double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << former_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

	REQUIRE(derivative == Approx(finite_difference).epsilon(tol));
}

TEST_CASE("neohookean-stress-3d", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "neohookean-stress-3d.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "neohookean-stress-3d-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);
	StressObjective func(state, shape_param, NULL, opt_args["functionals"][0]);

	double functional_val = func.value();

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 1000) / 1000.0;
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(state, velocity, velocity_discrete);

	verify_adjoint(func, state, shape_param, "shape", velocity_discrete, 1e-6, 1e-3);
}

TEST_CASE("shape-contact", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-contact.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "shape-contact-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);
	StressObjective func(state, shape_param, NULL, opt_args["functionals"][0]);

	state.pre_sol = state.diff_cached[0].u;
	double functional_val = func.value();

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 10000) / 1.0e4;
		}
		return vel;
	};

	Eigen::MatrixXd velocity_discrete;
	sample_field(state, velocity, velocity_discrete);

	verify_adjoint(func, state, shape_param, "shape", velocity_discrete, 1e-6, 5e-4);
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

	Eigen::MatrixXd targets(state.n_bases, state.mesh->dimension());
	std::vector<int> actives;
	for (int i = 0; i < targets.size(); i++)
		targets(i) = (rand() % 10) / 10.;
	for (int i = 0; i < targets.rows(); i++)
		actives.push_back(i);

	NodeTargetObjective func(state, actives, targets);

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ElasticParameter> elastic_param = std::make_shared<ElasticParameter>(states_ptr, opt_args["parameters"][0]);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = (rand() % 10000) / 1.0e4;
		}
		return vel;
	};

	Eigen::MatrixXd velocity_discrete;
	sample_field(state, velocity, velocity_discrete, 0);

	verify_adjoint(func, state, elastic_param, "material", velocity_discrete, 1e-5, 1e-5);
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

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr, state_reference};
	std::shared_ptr<DampingParameter> damping_param = std::make_shared<DampingParameter>(states_ptr, opt_args["parameters"][0]);
	std::vector<std::shared_ptr<Parameter>> parameters = {damping_param};

	std::shared_ptr<Objective> func = Objective::create(opt_args["functionals"][0], root_path, parameters, states_ptr);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(2);

	verify_adjoint(*func, state, damping_param, "damping", velocity_discrete, 1e-5, 1e-4);
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

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ElasticParameter> elastic_param = std::make_shared<ElasticParameter>(states_ptr, opt_args["parameters"][0]);
	std::shared_ptr<TargetObjective> func_aux = std::make_shared<TargetObjective>(state, std::shared_ptr<ShapeParameter>(), opt_args["functionals"][0]);
	func_aux->set_reference(state_reference, {1, 3});
	TransientObjective func(state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], func_aux);

	Eigen::VectorXd velocity_discrete;
	velocity_discrete.setOnes(state.bases.size() * 2);
	velocity_discrete *= 1e3;

	verify_adjoint(func, state, elastic_param, "material", velocity_discrete, 1e-5, 1e-4);
}

TEST_CASE("shape-transient-friction", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-transient-friction.json", in_args);
	auto state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json opt_args;
	load_json(path + "shape-transient-friction-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, opt_args["parameters"][0]);
	json functional_args = opt_args["functionals"][0];
	std::shared_ptr<StaticObjective> func_aux = std::make_shared<StressObjective>(state, shape_param, std::shared_ptr<ElasticParameter>(), functional_args, false);
	TransientObjective func(state.args["time"]["time_steps"], state.args["time"]["dt"], functional_args["transient_integral_type"], func_aux);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
	for (int i = 0; i < state.n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}

	velocity_discrete.normalize();

	verify_adjoint(func, state, shape_param, "shape", velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("shape-transient-friction-sdf", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "shape-transient-friction-sdf.json", in_args);

	Eigen::MatrixXd control_points, tangents, delta;
	control_points.setZero(2, 2);
	control_points << 1, 0.4,
		0.1, 1;
	tangents.setZero(2, 2);
	tangents << -1, 1,
		-1, 0;
	delta.setZero(1, 2);
	delta << 0.05, 0.05;
	SDFTrajectoryFunctional func;
	func.set_spline_target(control_points, tangents, delta);
	func.set_interested_ids({}, {2});
	func.set_surface_integral();
	func.set_transient_integral_type("final");

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;
	double functional_val = func.energy(state);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.n_geom_bases * 2, 1);
	for (int i = 0; i < state.n_geom_bases; ++i)
	{
		velocity_discrete(i * 2 + 0) = rand() % 1000;
		velocity_discrete(i * 2 + 1) = rand() % 1000;
	}

	velocity_discrete.normalize();

	Eigen::VectorXd one_form = func.gradient(state, "shape");
	double derivative = (one_form.array() * velocity_discrete.array()).sum();

	// Check that the answer given is correct via finite difference.
	// First alter the mesh according to the velocity.
	const double t = 1e-6;
	perturb_mesh(state, velocity_discrete * t);

	solve_pde(state);
	double next_functional_val = func.energy(state);

	perturb_mesh(state, velocity_discrete * (-2 * t));

	solve_pde(state);
	double prev_functional_val = func.energy(state);

	double finite_difference = (next_functional_val - prev_functional_val) / (2 * t);

	std::cout << std::setprecision(16) << "f(x) " << functional_val << " f(x-dt) " << prev_functional_val << " f(x+dt) " << next_functional_val << "\n";
	std::cout << std::setprecision(12) << "derivative: " << derivative << ", fd: " << finite_difference << "\n";

	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-5));
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

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr, state_reference};
	std::shared_ptr<InitialConditionParameter> initial_param = std::make_shared<InitialConditionParameter>(states_ptr, opt_args["parameters"][0]);
	std::vector<std::shared_ptr<Parameter>> parameters = {initial_param};

	std::shared_ptr<Objective> func = Objective::create(opt_args["functionals"][0], root_path, parameters, states_ptr);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.ndof() * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(state.ndof() + i * 2 + 0) = -2.;
		velocity_discrete(state.ndof() + i * 2 + 1) = -1.;
	}

	verify_adjoint(*func, state, initial_param, "initial", velocity_discrete, 1e-5, 1e-5);
}

TEST_CASE("barycenter", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "barycenter.json", in_args);

	json opt_args;
	load_json(path + "barycenter-opt.json", opt_args);
	opt_args = apply_opt_json_spec(opt_args, false);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	json shape_arg = opt_args["parameters"][1];

	Eigen::MatrixXd centers;
	{
		auto in_args_ref = in_args;
		in_args_ref["initial_conditions"]["velocity"][0]["value"][0] = 4;
		in_args_ref["initial_conditions"]["velocity"][0]["value"][1] = -1;
		std::shared_ptr<State> state_reference = create_state_and_solve(in_args_ref);
		std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
		std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, shape_arg);
		BarycenterTargetObjective func_aux(*state_reference, shape_param, opt_args["functionals"][0], Eigen::MatrixXd::Zero(state_reference->diff_cached.size(), state_reference->mesh->dimension()));
		centers.setZero(state_reference->diff_cached.size(), state_reference->mesh->dimension());
		for (int t = 0; t < state_reference->diff_cached.size(); t++)
		{
			func_aux.set_time_step(t);
			centers.row(t) = func_aux.get_barycenter();
		}
	}

	std::vector<std::shared_ptr<State>> states_ptr = {state_ptr};
	std::shared_ptr<ShapeParameter> shape_param = std::make_shared<ShapeParameter>(states_ptr, shape_arg);
	std::shared_ptr<InitialConditionParameter> initial_param = std::make_shared<InitialConditionParameter>(states_ptr, opt_args["parameters"][0]);

	std::shared_ptr<StaticObjective> func_aux = std::make_shared<BarycenterTargetObjective>(state, shape_param, opt_args["functionals"][0], centers);
	TransientObjective func(state.args["time"]["time_steps"], state.args["time"]["dt"], opt_args["functionals"][0]["transient_integral_type"], func_aux);

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(state.ndof() * 2, 1);
	for (int i = 0; i < state.n_bases; i++)
	{
		velocity_discrete(state.ndof() + i * 2 + 0) = -2.;
		velocity_discrete(state.ndof() + i * 2 + 1) = -1.;
	}

	verify_adjoint(func, state, initial_param, "initial", velocity_discrete, 1e-6, 1e-5);
}

TEST_CASE("dirichlet-sdf", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "dirichlet-sdf.json", in_args);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	Eigen::MatrixXd control_points, tangents, delta;
	control_points.setZero(2, 2);
	control_points << -2.5, -0.1,
		2.5, -0.1;
	tangents.setZero(2, 2);
	tangents << 1.5, -2,
		1.5, 2;
	delta.setZero(1, 2);
	delta << 0.5, 0.5;
	SDFTrajectoryFunctional func;
	func.set_spline_target(control_points, tangents, delta);
	func.set_interested_ids({}, {4});
	func.set_surface_integral();
	func.set_transient_integral_type("step_10");

	const double functional_val = func.energy(state);

	int time_steps = state.args["time"]["time_steps"].get<int>();

	Eigen::VectorXd one_form = func.gradient(state, "dirichlet");
	// std::cout << "one form " << one_form << std::endl;

	// srand(time(0));

	double derivative;
	double finite_difference;

	derivative = 0;
	finite_difference = 0;

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(time_steps, state.mesh->dimension());
	for (int j = 0; j < time_steps; ++j)
	{
		for (int i = 0; i < state.mesh->dimension(); ++i)
		{
			double random_val = (rand() % 200) / 100. - 1.;
			velocity_discrete(j, i) = random_val;
		}
	}

	const double step_size = 1e-7;

	json temp_args = in_args;
	for (int i = 0; i < 2; ++i)
	{
		for (int t = 0; t < time_steps; ++t)
		{
			temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
		}
	}
	std::shared_ptr<State> state_fd = create_state_and_solve(temp_args);
	double next_functional_val = func.energy(*state_fd);

	finite_difference = (next_functional_val - functional_val) / step_size;
	for (int j = 0; j < time_steps; ++j)
		for (int i = 0; i < state.boundary_nodes.size(); ++i)
			derivative += one_form(j * state.boundary_nodes.size() + i) * velocity_discrete(j, i % 2);
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}

TEST_CASE("dirichlet-ref", "[adjoint_method]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../differentiable/");
	json in_args;
	load_json(path + "dirichlet-ref.json", in_args);

	std::shared_ptr<State> state_ptr = create_state_and_solve(in_args);
	State &state = *state_ptr;

	int time_steps = state.args["time"]["time_steps"].get<int>();

	json ref_args = in_args;
	for (int t = 0; t < time_steps; ++t)
	{
		ref_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][0][t] = ref_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][0][t].get<double>() - 0.5 * t;
		ref_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][0][t] = ref_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][0][t].get<double>() + 0.5 * t;
	}
	std::shared_ptr<State> state_ref = create_state_and_solve(ref_args);

	TrajectoryFunctional func;
	func.set_reference(state_ref.get(), state, {2});
	func.set_interested_ids({}, {4});
	func.set_surface_integral();
	func.set_transient_integral_type("uniform");

	const double functional_val = func.energy(state);

	Eigen::VectorXd one_form = func.gradient(state, "dirichlet");
	// std::cout << "one form " << one_form << std::endl;

	// srand(time(0));

	double derivative;
	double finite_difference;

	derivative = 0;
	finite_difference = 0;

	Eigen::MatrixXd velocity_discrete;
	velocity_discrete.setZero(time_steps, state.mesh->dimension());
	for (int j = 0; j < time_steps; ++j)
	{
		for (int i = 0; i < state.mesh->dimension(); ++i)
		{
			double random_val = (rand() % 200) / 100. - 1.;
			velocity_discrete(j, i) = random_val;
		}
	}

	const double step_size = 1e-7;

	json temp_args = in_args;
	for (int i = 0; i < 2; ++i)
	{
		for (int t = 0; t < time_steps; ++t)
		{
			temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][0]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][1]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
			temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t] = temp_args["boundary_conditions"]["dirichlet_boundary"][2]["value"][i][t].get<double>() + velocity_discrete(t, i) * step_size;
		}
	}
	std::shared_ptr<State> state_fd = create_state_and_solve(temp_args);
	double next_functional_val = func.energy(*state_fd);

	finite_difference = (next_functional_val - functional_val) / step_size;
	for (int j = 0; j < time_steps; ++j)
		for (int i = 0; i < state.boundary_nodes.size(); ++i)
			derivative += one_form(j * state.boundary_nodes.size() + i) * velocity_discrete(j, i % 2);
	std::cout << "derivative: " << derivative << ", fd: " << finite_difference << "\n";
	REQUIRE(derivative == Approx(finite_difference).epsilon(1e-4));
}