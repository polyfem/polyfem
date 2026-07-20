#include <polyfem/State.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <filesystem>
#include <string>
#include <utility>

using namespace polyfem;

namespace
{
	json transient_args()
	{
		json args = json::object();
		args["time"] = json::object();
		return args;
	}

	json load_scene(const std::string &path)
	{
		std::ifstream file(path);
		REQUIRE(file.is_open());

		json args;
		file >> args;
		args["root_path"] = path;
		args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
		args["/output/directory"_json_pointer] = "";
		args["/output/log/quiet"_json_pointer] = true;
		args["/output/log/level"_json_pointer] = "error";
		args["/output/advanced/save_time_sequence"_json_pointer] = false;
		args["/output/paraview/file_name"_json_pointer] = "";
		args["/output/data/state"_json_pointer] = "";

		return args;
	}
} // namespace

TEST_CASE("varform factory supports migrated formulations", "[varform]")
{
	const json args = transient_args();

	for (const std::string formulation : {
			 "NeoHookean",
			 "LinearElasticity",
			 "Laplacian",
			 "Stokes",
			 "NavierStokes",
			 "OperatorSplitting",
			 "IncompressibleLinearElasticity",
			 "Bilaplacian",
		 })
	{
		CHECK(varform::VarFormFactory::supports(formulation, args));
		CHECK(varform::VarFormFactory::create(formulation, args) != nullptr);
	}
	CHECK(varform::VarFormFactory::supports("NavierStokesFSI", args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", args) != nullptr);
	json static_args = args;
	static_args["time"] = nullptr;
	CHECK_FALSE(varform::VarFormFactory::supports("NavierStokesFSI", static_args));
	CHECK(varform::VarFormFactory::create("NavierStokesFSI", static_args) == nullptr);

	json periodic_args = args;
	periodic_args["/boundary_conditions/periodic_boundary/enabled"_json_pointer] = true;
	CHECK_FALSE(varform::VarFormFactory::supports("Stokes", periodic_args));
	CHECK(varform::VarFormFactory::create("Stokes", periodic_args) == nullptr);
	CHECK_FALSE(varform::VarFormFactory::supports("NeoHookean", periodic_args));
	CHECK(varform::VarFormFactory::create("NeoHookean", periodic_args) == nullptr);
}

TEST_CASE("state can opt into migrated varforms", "[varform][state]")
{
	for (const auto &[scene, expected_name] : {
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json", std::string("Stokes")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/units/navier_stokes_static.json", std::string("NavierStokes")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/navier_stokes_split.json", std::string("OperatorSplitting")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/incompressible.json", std::string("IncompressibleElastic")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/bilaplace.json", std::string("Bilaplacian")},
		 })
	{
		State state;
		state.init(load_scene(scene), true);

		REQUIRE(state.variational_formulation != nullptr);
		CHECK(state.variational_formulation->name() == expected_name);
	}
}

TEST_CASE("ALE Navier-Stokes FSI runs through State", "[varform][state][navier_stokes][fsi]")
{
	const std::filesystem::path state_pattern =
		std::filesystem::temp_directory_path() / "polyfem-navier-stokes-fsi-state-{:d}.h5";
	const std::string state_path =
		(std::filesystem::temp_directory_path() / "polyfem-navier-stokes-fsi-state-1.h5").string();
	std::filesystem::remove(state_path);
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/navier_stokes_transient.json");
	args.erase("preset_problem");
	args["geometry"] = {
		{"mesh", std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D/simple/square.obj"},
		{"surface_selection", json::array({{{"id", 7}, {"box", {{-10, -10}, {10, 10}}}}})}};
	args["materials"] = {
		{"type", "NavierStokesFSI"},
		{"viscosity", 0.1},
		{"rho", 1.0},
		{"velocity_space_id", 0},
		{"pressure_space_id", 1},
		{"mesh_displacement_space_id", 2},
		{"mesh_material", {{"type", "NeoHookean"}, {"E", 10.0}, {"nu", 0.3}, {"rho", 1.0}}}};
	args["boundary_conditions"] = {
		{"dirichlet_boundary", json::array({{{"id", 7}, {"fe_space", 0}, {"value", {0, 0}}},
											{{"id", 7}, {"fe_space", 2}, {"value", {"0.1*t", 0}}}})}};
	args["space"]["discr_order"] = json::array({{{"fe_space", 0}, {"order", 2}},
												{{"fe_space", 1}, {"order", 1}},
												{{"fe_space", 2}, {"order", 1}}});
	args["time"] = {{"t0", 0}, {"tend", 0.01}, {"time_steps", 1}};
	args["/output/data/state"_json_pointer] = state_pattern.string();

	State state;
	state.init(args, true);
	state.load_mesh();
	Eigen::MatrixXd solution;
	state.solve(solution);

	REQUIRE(state.variational_formulation != nullptr);
	CHECK(state.variational_formulation->name() == "NavierStokesFSI");
	CHECK(solution.rows() > 0);
	CHECK(solution.allFinite());
	REQUIRE(std::filesystem::is_regular_file(state_path));

	json restart_args = args;
	restart_args["/input/data/state"_json_pointer] = state_path;
	restart_args["/output/data/state"_json_pointer] = "";
	restart_args["time"] = {{"t0", 0.01}, {"tend", 0.02}, {"time_steps", 1}};
	State restarted_state;
	restarted_state.init(restart_args, true);
	restarted_state.load_mesh();
	Eigen::MatrixXd restarted_solution;
	restarted_state.solve(restarted_solution);
	CHECK(restarted_solution.rows() == solution.rows());
	CHECK(restarted_solution.allFinite());
	std::filesystem::remove(state_path);
}

TEST_CASE("periodic boundary conditions remain on legacy state path", "[varform][state]")
{
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json");
	args["/boundary_conditions/periodic_boundary/enabled"_json_pointer] = true;

	CHECK_FALSE(varform::uses_varform_state(args));

	legacy::State state;
	state.init(args, false);

	REQUIRE(state.assembler != nullptr);
	CHECK(state.assembler->name() == "Stokes");
	REQUIRE(state.pressure_assembler != nullptr);
	CHECK(state.pressure_assembler->name() == "StokesPressure");
	CHECK(state.mixed_assembler != nullptr);
}

TEST_CASE("optimization keeps varforms on the legacy state path", "[varform][state]")
{
	json args = load_scene(std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json");
	legacy::State state;
	state.optimization_enabled = true;
	state.init(args, true);

	REQUIRE(state.assembler != nullptr);
	CHECK(state.assembler->name() == "Stokes");
	REQUIRE(state.pressure_assembler != nullptr);
	CHECK(state.pressure_assembler->name() == "StokesPressure");
	CHECK(state.mixed_assembler != nullptr);
}
