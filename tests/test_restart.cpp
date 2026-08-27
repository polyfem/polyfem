////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/io/Checkpoint.hpp>

#include <filesystem>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
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
	args["output"] = json({});
	args["output"]["advanced"]["save_time_sequence"] = false;
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

Eigen::MatrixXd run_sim(State &state, const json &args)
{
	state.init(args, true);
	state.set_max_threads(1);
	logger().set_level(spdlog::level::info);
	state.load_mesh();

	Eigen::MatrixXd sol;

	state.solve(sol);

	return sol;
}

Eigen::MatrixXd run_checkpoint(State &state, const std::filesystem::path &path)
{
	io::CheckpointReader checkpoint(path);
	state.init(checkpoint, true);
	state.set_max_threads(1);
	state.load_mesh();
	Eigen::MatrixXd sol;
	state.solve(sol);
	return sol;
}

#ifdef NDEBUG
TEST_CASE("restart", "[restart]")
#else
TEST_CASE("restart", "[.][restart]")
#endif
{
	const std::string scene_file = POLYFEM_DATA_DIR "/contact/examples/3D/unit-tests/2-cubes.json";
	constexpr int total_time_steps = 10;
	REQUIRE(total_time_steps % 2 == 0);
	constexpr int restart_time_steps = total_time_steps / 2;
	constexpr double margin = 1e-3;

	const std::filesystem::path outdir = std::filesystem::current_path() / "DELETE_ME_restart_test_output";
	const std::filesystem::path full_outdir = outdir / "full";
	json args = load_sim_json(scene_file, total_time_steps);

	State state;

	args["/output/directory"_json_pointer] = full_outdir.string();
	args["/output/checkpoint/path"_json_pointer] = "checkpoint_{:d}.h5";
	const auto full_sol = run_sim(state, args);

	State resumed_state;
	const auto restart_sol = run_checkpoint(
		resumed_state, full_outdir / fmt::format("checkpoint_{:d}.h5", restart_time_steps));

	CHECK(full_sol.rows() == restart_sol.rows());
	CHECK(full_sol.cols() == restart_sol.cols());
	CAPTURE((full_sol - restart_sol).lpNorm<Eigen::Infinity>());
	CHECK(full_sol.isApprox(restart_sol, margin));

	// Resuming continues output indices stored in checkpoint metadata.
	for (int t = restart_time_steps + 1; t <= total_time_steps; ++t)
	{
		const auto state_file = full_outdir / fmt::format("checkpoint_{:d}.h5", t);
		CHECK(std::filesystem::exists(state_file));
	}

	std::filesystem::remove_all(outdir);
}
