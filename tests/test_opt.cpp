////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>

#include <iostream>
#include <fstream>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;
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

		for (auto &state_arg : opt_args["states"])
			state_arg["path"] = resolve_output_path(root_folder, state_arg["path"]);

		auto nl_problem = make_nl_problem(opt_args);

		Eigen::VectorXd x = nl_problem->initial_guess();

		std::shared_ptr<cppoptlib::NonlinearSolver<solver::AdjointNLProblem>> nlsolver = make_nl_solver<solver::AdjointNLProblem>(opt_args["solver"]["nonlinear"]);

		CHECK_THROWS_WITH(nlsolver->minimize(*nl_problem, x), Catch::Matchers::Contains("Reached iteration limit"));
	}
} // namespace

#if defined(__linux__)

TEST_CASE("shape-trajectory-surface-opt", "[optimization]")
{
	run_trajectory_opt("shape-trajectory-surface-opt");
	auto energies = read_energy("shape-trajectory-surface-opt");

	REQUIRE(energies[0] == Approx(6.1658e-05).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(3.6193e-05).epsilon(1e-3));
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

TEST_CASE("shape-trajectory-surface-opt-new", "[optimization]")
{
	run_opt_new("shape-trajectory-surface-opt-new");
	auto energies = read_energy("shape-trajectory-surface-opt-new");

	REQUIRE(energies[0] == Approx(6.1658e-05).epsilon(1e-3));
	REQUIRE(energies[energies.size() - 1] == Approx(3.6194e-05).epsilon(1e-3));
}
#endif