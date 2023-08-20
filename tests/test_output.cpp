////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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

#ifdef NDEBUG
TEST_CASE("full sim", "[full_sim]")
#else
TEST_CASE("full sim", "[.][full_sim]")
#endif
{
	const std::string scene = GENERATE("2D/unit-tests/5-squares", "3D/unit-tests/5-cubes-fast", "3D/unit-tests/edge-edge-parallel");
	const std::string scene_file = fmt::format("{}/contact/examples/{}.json", POLYFEM_DATA_DIR, scene);

	json args;
	if (!load_json(scene_file, args))
	{
		spdlog::error("unable to open {} file", scene_file);
		FAIL();
	}

	args["root_path"] = scene_file;
	const std::filesystem::path outdir = std::filesystem::current_path() / "DELETE_ME_full_sim_test_output";
	args["/output/directory"_json_pointer] = outdir.string();
	args["/output/paraview"_json_pointer] = R"({
		"file_name": "sim.pvd",
		"volume": true,
		"surface": true,
		"wireframe": true,
		"points": true,
		"vismesh_rel_area": 1e-05,
		"options": {
			"material": true,
			"body_ids": true,
			"contact_forces": true,
			"friction_forces": true,
			"velocity": true,
			"acceleration": true
		}
	})"_json;
	args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
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
	state.assemble_mass_mat();

	Eigen::MatrixXd sol;
	Eigen::MatrixXd pressure;

	state.solve_problem(sol, pressure);

	state.compute_errors(sol);

	CHECK(std::filesystem::exists(outdir));
	CHECK(std::filesystem::exists(outdir / "sim.pvd"));

	std::filesystem::remove_all(outdir);
}
