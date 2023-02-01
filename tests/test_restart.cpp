////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch.hpp>

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

bool load_json(const std::string &json_file, json &out);

json load_sim_json(const std::string filename, const int time_steps)
{
	json in_args;
	if (!load_json(filename, in_args))
	{
		spdlog::error("unable to open {} file", filename);
		FAIL();
	}

	json args = in_args;
	// args["output"] = json({});
	// args["output"]["advanced"]["save_time_sequence"] = false;
	{
		json t_args = args["time"];
		if (t_args.contains("tend") && t_args.contains("dt"))
		{
			t_args.erase("tend");
			t_args["time_steps"] = time_steps;
		}
		else if (t_args.contains("tend") && t_args.contains("time_steps"))
		{
			t_args["dt"] = t_args["tend"].get<double>() / t_args["time_steps"].get<int>();
			t_args["time_steps"] = time_steps;
			t_args.erase("tend");
		}
		else if (t_args.contains("dt") && t_args.contains("time_steps"))
		{
			t_args["time_steps"] = time_steps;
		}
		else
		{
			// Required to have two of tend, dt, time_steps
			FAIL();
		}
		args["time"] = t_args;
	}
	args["root_path"] = filename;
	args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
	// args["/output/log/level"_json_pointer] = "error";

	return args;
}

json run_sim(State &state, const json &args)
{
	state.init(args, true);
	state.set_max_threads(1);
	spdlog::set_level(spdlog::level::info);
	state.load_mesh();

	if (state.mesh == nullptr)
	{
		spdlog::warn("No Mesh is Read!!");
		FAIL();
	}

	state.build_basis();

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	Eigen::MatrixXd sol;
	Eigen::MatrixXd pressure;

	state.solve_problem(sol, pressure);

	state.compute_errors(sol);

	json out = json({});
	out["err_l2"] = state.stats.l2_err;
	out["err_h1"] = state.stats.h1_err;
	out["err_h1_semi"] = state.stats.h1_semi_err;
	out["err_linf"] = state.stats.linf_err;
	out["err_linf_grad"] = state.stats.grad_max_err;
	out["err_lp"] = state.stats.lp_err;

	return out;
}

TEST_CASE("restart", "[restart]")
{
	// const std::string scene_file = POLYFEM_DATA_DIR "/contact/examples/3D/stress-tests/mat-twist.json";
	const std::string scene_file = POLYFEM_DATA_DIR "/contact/examples/3D/unit-tests/5-cubes-fast.json";
	constexpr int total_time_steps = 10;
	REQUIRE(total_time_steps % 2 == 0);
	constexpr int restart_time_steps = total_time_steps / 2;
	constexpr double margin = 1e-5;
	const std::vector<std::string> test_keys =
		{"err_l2", "err_h1", "err_h1_semi", "err_linf", "err_linf_grad", "err_lp"};

	const std::filesystem::path outdir = std::filesystem::current_path() / "DELETE_ME_restart_test_output";
	const std::filesystem::path full_outdir = outdir / "full";
	const std::filesystem::path part1_outdir = outdir / "part1";
	const std::filesystem::path part2_outdir = outdir / "part2";

	json args = load_sim_json(scene_file, total_time_steps);

	State state;

	args["/output/directory"_json_pointer] = full_outdir.string();
	const json full_out = run_sim(state, args);

	args["/output/directory"_json_pointer] = part1_outdir.string();
	args["/output/data/u_path"_json_pointer] = "restart_sol.bin";
	args["/output/data/v_path"_json_pointer] = "restart_vel.bin";
	args["/output/data/a_path"_json_pointer] = "restart_acc.bin";
	args["time"]["time_steps"] = restart_time_steps;
	run_sim(state, args); // partial sim

	args["/output/directory"_json_pointer] = part2_outdir.string();
	args["/input/data/u_path"_json_pointer] = (part1_outdir / "restart_sol.bin").string();
	args["/input/data/v_path"_json_pointer] = (part1_outdir / "restart_vel.bin").string();
	args["/input/data/a_path"_json_pointer] = (part1_outdir / "restart_acc.bin").string();
	args["/time/t0"_json_pointer] = args["/time/dt"_json_pointer].get<double>() * restart_time_steps;
	const json restart_out = run_sim(state, args);

	for (const std::string &key : test_keys)
	{
		const double full_val = full_out[key];
		const double restart_val = restart_out[key];
		const double relerr = std::abs((restart_val - full_val) / std::max(std::abs(full_val), 1e-5));
		CHECK(relerr < margin);
	}

	// std::filesystem::remove_all(outdir);
}
