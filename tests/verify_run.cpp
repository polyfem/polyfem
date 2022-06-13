////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch.hpp>

#include "polyfem/utils/JSONUtils.hpp"
#include "polyfem/State.hpp"
#include <polyfem/Common.hpp>

#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::utils;

int authenticate_json(std::string json_file)
{
	json in_args = json({});
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

	in_args["root_path"] = json_file;
	size_t max_threads = std::numeric_limits<size_t>::max();
	if (in_args.contains("authentication"))
	{
		// authenticate mode enabled
		logger().info("Authenticating..");
		auto authen_fields = in_args["authentication"];
	}

	State state(max_threads);
	state.init_logger("", 6, false);
	state.init(in_args, "");

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
	{
		// Cannot proceed without a mesh.
		return 1;
	}

	state.compute_mesh_stats();

	state.build_basis();

	state.assemble_rhs();
	state.assemble_stiffness_mat();

	state.solve_problem();

	state.compute_errors();

	// state.save_json();
	// state.export_data();
	return 0;
}

TEST_CASE("runners", "[.]")
{
	authenticate_json("../data/data/contact/examples/3D/unit-tests/5-cubes-fast.json");
}