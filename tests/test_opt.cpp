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
	state->load_mesh();
	Eigen::MatrixXd sol, pressure;
	state->build_basis();

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

TEST_CASE("shape-trajectory-surface-opt", "[optimization]")
{
	const std::string path = POLYFEM_DATA_DIR + std::string("/../optimizations/shape-trajectory-surface-opt");
	json target_args, in_args;
    load_json(path + "/target.json", target_args);
	load_json(path + "/run.json", in_args);

	auto target_state = create_state(target_args);
	solve_pde(*target_state);

	auto state = create_state(in_args);

	std::shared_ptr<CompositeFunctional> func = CompositeFunctional::create("Trajectory");
	auto &f = *dynamic_cast<TrajectoryFunctional *>(func.get());
	f.set_interested_ids({2}, {});
	f.set_reference(target_state.get(), *state, {2});

	CHECK_THROWS_WITH(general_optimization(*state, func), Catch::Matchers::Contains("Reached iteration limit"));

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