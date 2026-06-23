#include <polyfem/State.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>

#include <catch2/catch_test_macros.hpp>

#include <fstream>
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
			 "IncompressibleLinearElasticity",
			 "Bilaplacian",
		 })
	{
		CHECK(varform::VarFormFactory::supports(formulation, args));
		CHECK(varform::VarFormFactory::create(formulation, args) != nullptr);
	}

	CHECK_FALSE(varform::VarFormFactory::supports("OperatorSplitting", args));
	CHECK(varform::VarFormFactory::create("OperatorSplitting", args) == nullptr);

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
