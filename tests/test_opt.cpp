////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/AdjointNLProblem.hpp>
#include <polyfem/solver/BFGSSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/LBFGSBSolver.hpp>
#include <polyfem/solver/MMASolver.hpp>
#include <polyfem/solver/GradientDescentSolver.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <jse/jse.h>

#include <iostream>
#include <fstream>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polysolve;

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

	std::string resolve_output_path(const std::string &output_dir, const std::string &path)
	{
		if (std::filesystem::path(path).is_absolute())
			return path;
		else
			return std::filesystem::weakly_canonical(std::filesystem::path(output_dir) / path).string();
	}

	json apply_opt_json_spec(const json &input_args, bool strict_validation)
	{
		json args_in = input_args;

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			const std::string polyfem_input_spec = POLYFEM_OPT_INPUT_SPEC;
			std::ifstream file(polyfem_input_spec);

			if (file.is_open())
				file >> rules;
			else
			{
				logger().error("unable to open {} rules", polyfem_input_spec);
				throw std::runtime_error("Invald spec file");
			}
		}

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}

		json args = jse.inject_defaults(args_in, rules);
		return args;
	}

	std::shared_ptr<State> create_state(const json &args)
	{
		std::shared_ptr<State> state = std::make_shared<State>();
		state->init_logger("", spdlog::level::level_enum::err, false);
		state->init(args, false);
		state->args["optimization"]["enabled"] = true;
		state->load_mesh();
		Eigen::MatrixXd sol, pressure;
		state->build_basis();
		state->assemble_rhs();
		state->assemble_stiffness_mat();

		return state;
	}

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver(const json &solver_params)
	{
		const std::string name = solver_params["solver"].template get<std::string>();
		if (name == "GradientDescent" || name == "gradientdescent" || name == "gradient")
		{
			return std::make_shared<cppoptlib::GradientDescentSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "bfgs" || name == "BFGS" || name == "BFGS")
		{
			return std::make_shared<cppoptlib::BFGSSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "lbfgsb" || name == "LBFGSB" || name == "L-BFGS-B")
		{
			return std::make_shared<cppoptlib::LBFGSBSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "mma" || name == "MMA")
		{
			return std::make_shared<cppoptlib::MMASolver<ProblemType>>(
				solver_params);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	void solve_pde(State &state)
	{
		state.assemble_rhs();
		state.assemble_stiffness_mat();
		Eigen::MatrixXd sol, pressure;
		state.solve_problem(sol, pressure);
	}

	std::vector<double> read_energy(const std::string &file)
	{
		std::ifstream energy_out(file);
		std::vector<double> energies;
		std::string line;
		if (energy_out.is_open())
		{
			while (getline(energy_out, line))
			{
				energies.push_back(std::stod(line.substr(0, line.find(","))));
			}
		}
		double starting_energy = energies[0];
		double optimized_energy = energies[energies.size() - 1];

		std::cout << "initial " << energies[0] << std::endl;
		std::cout << "final " << energies[energies.size() - 1] << std::endl;

		return energies;
	}

	void run_trajectory_opt(const std::string &name)
	{
		const std::string path = POLYFEM_DATA_DIR + std::string("/../optimizations/" + name);

		json target_args, in_args;
		load_json(path + "/run.json", in_args);
		load_json(path + "/target.json", target_args);
		auto state = create_state(in_args);
		auto target_state = create_state(target_args);
		solve_pde(*target_state);

		auto opt_params = state->args["optimization"];
		auto objective_params = opt_params["functionals"][0];

		std::string matching_type = objective_params["matching"];
		std::shared_ptr<CompositeFunctional> func;
		if (objective_params["type"] == "trajectory")
		{
			if (matching_type == "exact")
				func = CompositeFunctional::create("Trajectory");
			else if (matching_type == "sdf")
				func = CompositeFunctional::create("SDFTrajectory");
			else
				logger().error("Invalid matching type!");
		}
		else if (objective_params["type"] == "height")
		{
			func = CompositeFunctional::create("Height");
		}

		std::string transient_integral_type = objective_params["transient_integral_type"];
		if (transient_integral_type != "")
			func->set_transient_integral_type(transient_integral_type);

		std::set<int> interested_body_ids;
		std::vector<int> interested_bodies = objective_params["volume_selection"];
		interested_body_ids = std::set(interested_bodies.begin(), interested_bodies.end());

		std::set<int> interested_boundary_ids;
		std::vector<int> interested_boundaries = objective_params["surface_selection"];
		interested_boundary_ids = std::set(interested_boundaries.begin(), interested_boundaries.end());

		func->set_interested_ids(interested_body_ids, interested_boundary_ids);

		if (matching_type == "exact")
		{
			auto &f = *dynamic_cast<TrajectoryFunctional *>(func.get());

			std::set<int> reference_cached_body_ids;
			if (objective_params["reference_cached_body_ids"].size() > 0)
			{
				std::vector<int> ref_cached = objective_params["reference_cached_body_ids"];
				reference_cached_body_ids = std::set(ref_cached.begin(), ref_cached.end());
			}
			else
				reference_cached_body_ids = interested_body_ids;

			f.set_reference(target_state.get(), *state, reference_cached_body_ids);
		}

		CHECK_THROWS_WITH(general_optimization(*state, func), Catch::Matchers::Contains("Reached iteration limit"));
	}

	void run_opt_new(const std::string &name)
	{
		const std::string root_folder = POLYFEM_DATA_DIR + std::string("/../optimizations/") + name + "/";
		json opt_args;
		if (!load_json(resolve_output_path(root_folder, "run.json"), opt_args))
			log_and_throw_error("Failed to load optimization json file!");

		opt_args = apply_opt_json_spec(opt_args, false);

		// create states
		json state_args = opt_args["states"];
		assert(state_args.is_array() && state_args.size() > 0);
		std::vector<std::shared_ptr<State>> states(state_args.size());
		std::map<int, int> id_to_state;
		int i = 0;
		for (const json &args : state_args)
		{
			json cur_args;
			if (!load_json(resolve_output_path(root_folder, args["path"]), cur_args))
				log_and_throw_error("Can't find json for State {}", args["id"]);

			states[i] = create_state(cur_args);
			id_to_state[args["id"].get<int>()] = i++;
		}

		// create parameters
		json param_args = opt_args["parameters"];
		assert(param_args.is_array() && param_args.size() > 0);
		std::vector<std::shared_ptr<Parameter>> parameters(param_args.size());
		i = 0;
		for (const json &args : param_args)
		{
			std::vector<std::shared_ptr<State>> some_states;
			for (int id : args["states"])
			{
				some_states.push_back(states[id_to_state[id]]);
			}
			parameters[i++] = Parameter::create(args, some_states);
		}

		// create objectives
		json obj_args = opt_args["functionals"];
		assert(obj_args.is_array() && obj_args.size() > 0);
		std::vector<std::shared_ptr<solver::Objective>> objs(obj_args.size());
		Eigen::VectorXd weights;
		weights.setOnes(objs.size());
		i = 0;
		for (const json &args : obj_args)
		{
			weights[i] = args["weight"];
			objs[i++] = solver::Objective::create(args, parameters, states);
		}
		std::shared_ptr<solver::SumObjective> sum_obj = std::make_shared<solver::SumObjective>(objs, weights);

		solver::AdjointNLProblem nl_problem(sum_obj, parameters, states, opt_args);
		std::shared_ptr<cppoptlib::NonlinearSolver<solver::AdjointNLProblem>> nlsolver = make_nl_solver<solver::AdjointNLProblem>(opt_args["solver"]["nonlinear"]);

		Eigen::VectorXd x;
		x.setZero(nl_problem.full_size());
		int cumulative = 0;
		for (const auto &p : parameters)
		{
			x.segment(cumulative, p->optimization_dim()) = p->initial_guess();
			cumulative += p->optimization_dim();
		}

		for (auto &state : states)
		{
			state->assemble_rhs();
			state->assemble_stiffness_mat();
			Eigen::MatrixXd sol, pressure;
			state->solve_problem(sol, pressure);
		}

		CHECK_THROWS_WITH(nlsolver->minimize(nl_problem, x), Catch::Matchers::Contains("Reached iteration limit"));
	}
} // namespace

#if defined(__linux__)

TEST_CASE("shape-trajectory-surface-opt", "[optimization]")
{
	run_trajectory_opt("shape-trajectory-surface-opt");
	auto energies = read_energy("shape-trajectory-surface-opt");

	REQUIRE(energies[0] == Approx(6.1658e-05).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(3.6194e-05).epsilon(1e-4));
}

TEST_CASE("shape-stress-opt", "[optimization]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../optimizations/shape-stress-opt");
	json in_args;
	load_json(path + "/run.json", in_args);

	auto state = create_state(in_args);

	std::shared_ptr<CompositeFunctional> func;
	for (const auto &param : state->args["optimization"]["functionals"])
	{
		if (param["type"] == "stress")
		{
			func = CompositeFunctional::create("Stress");
			func->set_power(param["power"]);
			break;
		}
	}

	CHECK_THROWS_WITH(single_optimization(*state, func), Catch::Matchers::Contains("Reached iteration limit"));

	auto energies = read_energy("shape-stress-opt");

	REQUIRE(energies[0] == Approx(12.0721).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(11.5431).epsilon(1e-4));
}

TEST_CASE("material-opt", "[optimization]")
{
	run_trajectory_opt("material-opt");
	auto energies = read_energy("material-opt");

	REQUIRE(energies[0] == Approx(0.00143472).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(1.10657e-05).epsilon(1e-4));
}

TEST_CASE("initial-opt", "[optimization]")
{
	run_trajectory_opt("initial-opt");
	auto energies = read_energy("initial-opt");

	REQUIRE(energies[0] == Approx(0.147092).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(0.109971).epsilon(1e-4));
}

TEST_CASE("topology-opt", "[optimization]")
{
	run_opt_new("topology-opt");
	auto energies = read_energy("topology-opt");

	REQUIRE(energies[0] == Approx(136.014).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(1.73135).epsilon(1e-4));
}

TEST_CASE("shape-stress-opt-new", "[optimization]")
{
	run_opt_new("shape-stress-opt-new");
	auto energies = read_energy("shape-stress-opt-new");

	REQUIRE(energies[0] == Approx(12.0735).epsilon(1e-4));
	REQUIRE(energies[energies.size() - 1] == Approx(11.5482).epsilon(1e-4));
}
#endif