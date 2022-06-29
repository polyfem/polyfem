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

int authenticate_json(std::string json_file, const bool allow_append)
{
	json in_args = json({});
	{
		std::ifstream file(json_file);

		if (file.is_open())
		{
			file >> in_args;
			file.close();
		}
		else
		{
			logger().error("unable to open {} file", json_file);
			return 1;
		}
	}

	std::string authent1 = "authen_t1";
	if (!in_args.contains(authent1) && !allow_append)
	{
		logger().error("Not Allowed appending JSON!! Add a * to the beginning of filename.");
		return 2;
	}

	auto args = in_args;
	if (true)
	{
		args["output"] = json({});
		args["output"]["advanced"]["save_time_sequence"] = false;
		if (args.contains("time"))
		{
			json t_args = args["time"];
			if (t_args.contains("tend") && t_args.contains("dt"))
			{
				t_args.erase("tend");
				t_args["time_steps"] = 1;
			}
			else if (t_args.contains("tend") && t_args.contains("time_steps"))
			{
				t_args["dt"] = t_args["tend"].get<double>() / t_args["time_steps"].get<int>();
				t_args["time_steps"] = 1;
				t_args.erase("tend");
			}
			else if (t_args.contains("dt") && t_args.contains("time_steps"))
			{
				t_args["time_steps"] = 1;
			}
			else
			{
				// Required to have at two of tend, dt, time_steps
				REQUIRE(false);
			}
			args["time"] = t_args;
		}
		args["root_path"] = json_file;
	}
	else
	{
		// old
		args["export"] = json({});
		args["save_time_sequence"] = false;
		args["output"] = "";
		if (args.contains("tend") || args.contains("dt"))
		{
			if (!args.contains("dt"))
			{
				args["dt"] = args["tend"].get<double>() / args["time_steps"].get<int>();
			}
			if (!args.contains("time_steps"))
			{
				args["time_steps"] = args["tend"].get<double>() / args["dt"].get<double>();
			}
			args["tend"] = args["dt"].get<double>();
		}
		args["time_steps"] = 1;
		args["root_path"] = json_file;
	}

	State state(1);
	state.init_logger("", spdlog::level::info, false);
	state.init(args, "");
	state.load_mesh();

	if (state.mesh == nullptr)
	{
		spdlog::warn("No Mesh is Read!!");
		return 1;
	}

	state.compute_mesh_stats();

	state.build_basis();

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();

	state.compute_errors();

	json out = json({});
	out["err_l2"] = state.l2_err;
	out["err_h1"] = state.h1_err;
	out["err_h1_semi"] = state.h1_semi_err;
	out["err_linf"] = state.linf_err;
	out["err_linf_grad"] = state.grad_max_err;
	out["err_lp"] = state.lp_err;
	out["margin"] = 1e-8;

	if (in_args.contains(authent1))
	{
		logger().info("Authenticating..");
		auto authen = in_args.at(authent1);
		auto margin = authen.at("margin").get<double>();
		margin = 1e-5;
		for (auto &el : out.items())
		{
			auto gt_val = authen[el.key()].get<double>();
			auto relerr = std::abs((gt_val - el.value().get<double>()) / std::max(std::abs(gt_val), 1e-5));
			if (relerr > margin)
			{
				logger().error("Violating Authenticate {}", el.key());
				return 2;
			}
		}
		logger().info("Authenticated ✅");
	}
	else
	{
		logger().warn("Appending JSON..");

		in_args[authent1] = out;
		std::ofstream file(json_file);
		file << in_args;
	}

	return 0;
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tags = "[run]";
#else
std::string tags = "[.]";
#endif
TEST_CASE("runners", tags)
{
	// Disabled on Windows CI, due to the requirement for Pardiso.
	std::ifstream file(POLYFEM_DATA_DIR "/system_test_list.txt");
	std::string line;
	spdlog::set_level(spdlog::level::info);
	while (std::getline(file, line))
	{
		DYNAMIC_SECTION(line)
		{
			auto allow_append = false;
			if (line[0] == '#')
				continue;
			if (line[0] == '*')
			{
				allow_append = true;
				line = line.substr(1);
			}
			spdlog::info("Processing {}", line);
			auto flag = authenticate_json(POLYFEM_DATA_DIR "/" + line, allow_append);
			CAPTURE(line);
			CHECK(flag == 0);
		}
	}
}
