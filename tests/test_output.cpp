////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/io/OutStatsData.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include <filesystem>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::utils;

bool load_json(const std::string &json_file, json &out);

TEST_CASE("output statistics are initialized", "[output]")
{
	io::OutRuntimeData timings;
	CHECK(timings.total_time() == 0);
	CHECK(timings.loading_mesh_time == 0);
	CHECK(timings.assigning_rhs_time == 0);
	timings.building_basis_time = 1;
	timings.loading_mesh_time = 2;
	timings.assigning_rhs_time = 3;
	timings.solving_time = 4;
	CHECK(timings.total_time() == 10);

	io::OutStatsData stats;
	CHECK(stats.spectrum.isZero());
	CHECK(stats.mesh_size == 0);
	CHECK(stats.min_edge_length == 0);
	CHECK(stats.nn_zero == 0);
	CHECK(stats.simplex_count == 0);
}

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

	Eigen::MatrixXd sol;

	state.solve(sol);
	state.variational_formulation->compute_errors(sol);

	CHECK(std::filesystem::exists(outdir));
	CHECK(std::filesystem::exists(outdir / "sim.pvd"));

	std::filesystem::remove_all(outdir);
}
