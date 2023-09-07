////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>

#include "polyfem/utils/JSONUtils.hpp"
#include "polyfem/State.hpp"
#include "spdlog/spdlog.h"
#include <polyfem/Common.hpp>

#include <filesystem>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::utils;

bool load_json(const std::string &json_file, json &out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

	return true;
}

bool missing_tests_data(const json &j, const std::string &key)
{
	return !j.contains(key) || (j.at(key).size() == 1 && j.at(key).contains("time_steps"));
}

int authenticate_json(const std::string &json_file, const bool compute_validation)
{
	json in_args;
	if (!load_json(json_file, in_args))
	{
		spdlog::error("unable to open {} file", json_file);
		return 1;
	}

	const std::string tests_key = "tests";
	if (missing_tests_data(in_args, tests_key) && !compute_validation)
	{
		spdlog::error(
			"JSON file missing \"{}\" key. Add a * to the beginning of filename to allow appends.",
			tests_key);
		return 2;
	}

	// ------------------------------------------------------------------------
	// Patch the JSON file to run a single time step
	json args = in_args;
	args["root_path"] = json_file;
	utils::apply_common_params(args);

	json time_steps;
	if (!args.contains("time"))
		time_steps = "static";
	else if (!args.contains(tests_key) || !args[tests_key].contains("time_steps"))
		time_steps = 1;
	else
		time_steps = args[tests_key]["time_steps"];

	args["output"] = json({});
	args["output"]["advanced"]["save_time_sequence"] = false;

	if (time_steps.is_number())
	{
		json t_args = args["time"];
		if (t_args.contains("tend") && t_args.contains("dt"))
		{
			t_args.erase("tend");
			t_args["time_steps"] = time_steps.get<int>();
		}
		else if (t_args.contains("tend") && t_args.contains("time_steps"))
		{
			t_args["dt"] = t_args["tend"].get<double>() / t_args["time_steps"].get<int>();
			t_args["time_steps"] = time_steps.get<int>();
			t_args.erase("tend");
		}
		else if (t_args.contains("dt") && t_args.contains("time_steps"))
		{
			t_args["time_steps"] = time_steps.get<int>();
		}
		else
		{
			// Required to have two of tend, dt, time_steps
			spdlog::error("Missing time parameters");
			REQUIRE(false);
		}
		args["time"] = t_args;
	}
	// ------------------------------------------------------------------------

	args["/solver/linear/solver"_json_pointer] =
		json_file.find("navier") == std::string::npos
			? "Eigen::SimplicialLDLT"
			: "Eigen::SparseLU";

	State state;
	args["/output/log/level"_json_pointer] = "error";
	state.init(args, true);
	state.set_max_threads(1);
	spdlog::set_level(spdlog::level::info);
	state.load_mesh();

	if (state.mesh == nullptr)
	{
		spdlog::warn("No Mesh is Read!!");
		return 1;
	}

	// state.compute_mesh_stats();

	state.build_basis();

	state.assemble_rhs();
	state.assemble_mass_mat();

	Eigen::MatrixXd sol;
	Eigen::MatrixXd pressure;

	state.solve_problem(sol, pressure);

	state.compute_errors(sol);

	state.save_json(sol);
	state.export_data(sol, pressure);

	json out = json({});
	out["err_l2"] = state.stats.l2_err;
	out["err_h1"] = state.stats.h1_err;
	out["err_h1_semi"] = state.stats.h1_semi_err;
	out["err_linf"] = state.stats.linf_err;
	out["err_linf_grad"] = state.stats.grad_max_err;
	out["err_lp"] = state.stats.lp_err;
	out["margin"] = 1e-5;
	out["time_steps"] = time_steps;

	std::vector<std::string> test_keys =
		{"err_l2", "err_h1", "err_h1_semi", "err_linf", "err_linf_grad", "err_lp"};

	if (!compute_validation)
	{
		spdlog::info("Authenticating...");
		json authen = in_args.at(tests_key);
		double margin = authen.value("margin", 1e-5);
		for (const std::string &key : test_keys)
		{
			const double prev_val = authen[key];
			const double curr_val = out[key];
			const double relerr = std::abs((curr_val - prev_val) / std::max(std::abs(prev_val), 1e-5));
			if (relerr > margin)
			{
				spdlog::error("Violating Authenticate prev_{0}={1} curr_{0}={2}", key, prev_val, curr_val);
				return 2;
			}
		}
		spdlog::info("Authenticated ✅");
	}
	else
	{
		spdlog::warn("Appending JSON...");

		in_args[tests_key] = out;
		std::ofstream file(json_file);
		file << in_args;
	}

	return 0;
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsrun = "[run]";
#else
std::string tagsrun = "[.][run]";
#endif
TEST_CASE("runners", tagsrun)
{
	// Disabled on Windows CI, due to the requirement for Pardiso.
	std::ifstream file(POLYFEM_TEST_DIR "/system_test_list.txt");
	std::vector<std::string> failing_tests;
	std::string line;
	while (std::getline(file, line))
	{
		bool compute_validation = false;
		if (line[0] == '#')
			continue;
		else if (line[0] == '*')
		{
			compute_validation = true;
			line = line.substr(1);
		}
		spdlog::info("Processing {}", line);
		auto flag = authenticate_json(POLYFEM_DATA_DIR "/" + line, compute_validation);
		CAPTURE(line);
		CHECK(flag == 0);
		if (flag != 0)
			failing_tests.push_back(line);
	}
	if (failing_tests.size() > 0)
	{
		std::cout << "Failing tests:" << std::endl;
		for (auto &t : failing_tests)
			std::cout << t << std::endl;
	}
}
