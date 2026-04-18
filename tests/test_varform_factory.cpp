#include <polyfem/State.hpp>
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

		return args;
	}
} // namespace

TEST_CASE("varform factory guards migrated formulations", "[varform]")
{
	const json args = transient_args();

	CHECK(varform::VarFormFactory::supports("NeoHookean", args));
	CHECK(varform::VarFormFactory::create("NeoHookean", args) != nullptr);

	CHECK_FALSE(varform::VarFormFactory::supports("NeoHookean", json::object()));
	CHECK(varform::VarFormFactory::create("NeoHookean", json::object()) == nullptr);

	json output_args = args;
	output_args["/output/advanced/save_time_sequence"_json_pointer] = true;
	CHECK_FALSE(varform::VarFormFactory::supports("NeoHookean", output_args));
	CHECK(varform::VarFormFactory::create("NeoHookean", output_args) == nullptr);

	json restart_args = args;
	restart_args["/output/data/state"_json_pointer] = "restart_{:d}.hdf5";
	CHECK_FALSE(varform::VarFormFactory::supports("NeoHookean", restart_args));
	CHECK(varform::VarFormFactory::create("NeoHookean", restart_args) == nullptr);

	for (const std::string formulation : {"Stokes", "NavierStokes", "OperatorSplitting", "Laplacian", "LinearElasticity"})
	{
		CHECK_FALSE(varform::VarFormFactory::supports(formulation, args));
		CHECK(varform::VarFormFactory::create(formulation, args) == nullptr);
	}
}

TEST_CASE("unmigrated fluid state uses legacy assemblers", "[varform][state]")
{
	for (const auto &[scene, formulation] : {
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/standard/stokes_static.json", std::string("Stokes")},
			 std::pair{std::string(POLYFEM_DATA_DIR) + "/units/navier_stokes_static.json", std::string("NavierStokes")},
		 })
	{
		State state;
		state.init(load_scene(scene), true);

		CHECK(state.variational_formulation == nullptr);
		REQUIRE(state.assembler != nullptr);
		CHECK(state.assembler->name() == formulation);
		REQUIRE(state.pressure_assembler != nullptr);
		CHECK(state.pressure_assembler->name() == "StokesPressure");
		CHECK(state.mixed_assembler != nullptr);
	}
}

TEST_CASE("nonlinear transient state can opt into varform", "[varform][state]")
{
	const std::string scene = std::string(POLYFEM_DATA_DIR) + "/interpolation/main.json";
	json args = load_scene(scene);
	args["output"] = json::object();
	args["/output/log/quiet"_json_pointer] = true;
	args["/output/log/level"_json_pointer] = "error";
	args["/output/advanced/save_time_sequence"_json_pointer] = false;
	args["/output/paraview/file_name"_json_pointer] = "";
	args["/output/data/state"_json_pointer] = "";

	State state;
	state.init(args, true);

	CHECK(state.variational_formulation != nullptr);
}
