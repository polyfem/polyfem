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
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::utils;
namespace fs = std::filesystem;

int authenticate_json(std::string json_file)
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

	in_args["root_path"] = json_file;
	size_t max_threads = std::numeric_limits<size_t>::max();

	State state(max_threads);
	state.init_logger("", 1, false);
	state.init(in_args, "");
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

	if (in_args.contains("authentication"))
	{
		logger().info("Authenticating..");
		auto authen = in_args.at("authentication");
		auto margin = authen.at("margin").get<double>();
		for (auto &el : out.items())
		{
			auto gt_val = authen[el.key()].get<double>();
			auto relerr = std::abs((gt_val - el.value().get<double>()) / std::max(std::abs(gt_val), margin));
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

		in_args["authentication"] = out;
		std::ofstream file(json_file);
		file << in_args;
	}

	return 0;
}

TEST_CASE("runners", "[.]")
{
	std::string path = "json_dir/";
	for (const auto &entry : fs::directory_iterator(path))
	{
		if (entry.path().extension() == ".json")
			CHECK(authenticate_json(entry.path()) == 0);
	}
}