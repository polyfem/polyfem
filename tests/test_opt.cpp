////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/solver/Optimizations.hpp>

#include <iostream>
#include <fstream>
#include "polyfem/utils/JSONUtils.hpp"
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

namespace {

bool load_json(const std::string &json_file, json &out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

    out["root_path"] = json_file;

	return true;
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

void solve_pde(State &state)
{
	state.assemble_rhs();
	state.assemble_stiffness_mat();
	Eigen::MatrixXd sol, pressure;
	state.solve_problem(sol, pressure);
}

} // namespace

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
		else if (matching_type == "exact-center")
			func = CompositeFunctional::create("CenterTrajectory");
		else if (matching_type == "last-center")
			func = CompositeFunctional::create("CenterXZTrajectory");
		else if (matching_type == "sine")
			func = CompositeFunctional::create("TargetY");
		else if (matching_type == "center-data")
			func = CompositeFunctional::create("CenterXYTrajectory");
		else if (matching_type == "marker-data")
			func = CompositeFunctional::create("NodeTrajectory");
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

#if defined(__linux__)

TEST_CASE("shape-trajectory-surface-opt", "[optimization]")
{
	run_trajectory_opt("shape-trajectory-surface-opt");

	std::ifstream energy_out("shape-trajectory-surface-opt");
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

	std::cout << starting_energy << std::endl;
	std::cout << optimized_energy << std::endl;

	REQUIRE(optimized_energy == Approx(0.6 * starting_energy).epsilon(0.05));
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

	std::ifstream energy_out("shape-stress-opt");
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

	std::cout << starting_energy << std::endl;
	std::cout << optimized_energy << std::endl;

	REQUIRE(starting_energy  == Approx(12.0721).epsilon(1e-4));
	REQUIRE(optimized_energy == Approx(11.3404).epsilon(1e-4));
}

TEST_CASE("material-opt", "[optimization]")
{
	run_trajectory_opt("material-opt");

	std::ifstream energy_out("material-opt");
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

	std::cout << starting_energy << std::endl;
	std::cout << optimized_energy << std::endl;

	REQUIRE(starting_energy  == Approx(0.00143472).epsilon(1e-4));
	REQUIRE(optimized_energy == Approx(1.10657e-05).epsilon(1e-4));
}

TEST_CASE("initial-opt", "[optimization]")
{
	run_trajectory_opt("initial-opt");

	std::ifstream energy_out("initial-opt");
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

	std::cout << starting_energy << std::endl;
	std::cout << optimized_energy << std::endl;

	REQUIRE(starting_energy  == Approx(0.147092).epsilon(1e-4));
	REQUIRE(optimized_energy == Approx(0.109971).epsilon(1e-4));
}

#endif