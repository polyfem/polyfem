////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <filesystem>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::utils;

bool load_json(const std::string &json_file, json &out);

TEST_CASE("full sim", "[full_sim]")
{
	const std::string scene_file = POLYFEM_DATA_DIR "/contact/examples/3D/unit-tests/5-cubes-fast.json";

	json args;
	if (!load_json(filename, args))
	{
		spdlog::error("unable to open {} file", filename);
		FAIL();
	}

	args["root_path"] = filename;
	const std::filesystem::path outdir = std::filesystem::current_path() / "DELETE_ME_full_sim_test_output";
	args["/output/directory"_json_pointer] = outdir.string();
	args["/output/log/level"_json_pointer] = "warning";

	State state;

	state.init(args, true);
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

	std::filesystem::remove_all(outdir);
}
