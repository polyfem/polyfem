////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>

#include <h5pp/h5pp.h>

#include "polyfem/io/Checkpoint.hpp"
#include "polyfem/io/InputLoader.hpp"
#include "polyfem/utils/JSONUtils.hpp"
#include "polyfem/utils/MatrixUtils.hpp"
#include "polyfem/State.hpp"
#include "polyfem/legacy/State.hpp"
#include "polyfem/varforms/VarForm.hpp"
#include "polyfem/varforms/VarFormFactory.hpp"
#include "spdlog/spdlog.h"
#include <polyfem/Common.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::utils;

bool load_json(const std::string &json_file, json &out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

	return true;
}

bool missing_tests_data(const json &j, const std::string &key)
{
	return !j.contains(key) || (j.at(key).size() == 1 && j.at(key).contains("time_steps"));
}

enum AuthenticateResult
{
	SUCCESS,
	MISSING_FILE,
	MISSING_TEST_DATA,
	SOLVE_FAILED,
	AUTHETICATION_FAILED
};

AuthenticateResult run_legacy_state(json &args, json &out)
{
	legacy::State state;
	args["/output/log/level"_json_pointer] = "error";
	state.init(args, true);
	state.set_max_threads(1);
	spdlog::set_level(spdlog::level::info);
	state.load_mesh();

	if (state.mesh == nullptr)
	{
		spdlog::warn("No Mesh is Read!!");
		return MISSING_FILE;
	}

	// state.compute_mesh_stats();

	state.build_basis();

	state.assemble_rhs();
	state.assemble_mass_mat();

	Eigen::MatrixXd sol;
	Eigen::MatrixXd pressure;

	try
	{
		state.solve_problem(sol, pressure);
	}
	catch (...)
	{
		return SOLVE_FAILED;
	}

	state.compute_errors(sol);

	state.save_json(sol);
	state.export_data(sol, pressure);

	out["err_l2"] = state.stats.l2_err;
	out["err_h1"] = state.stats.h1_err;
	out["err_h1_semi"] = state.stats.h1_semi_err;
	out["err_linf"] = state.stats.linf_err;
	out["err_linf_grad"] = state.stats.grad_max_err;
	out["err_lp"] = state.stats.lp_err;

	return SUCCESS;
}

void solve_initialized_state(State &state, Eigen::MatrixXd &solution)
{
	state.set_max_threads(1);
	spdlog::set_level(spdlog::level::info);
	state.load_mesh();
	state.solve(solution);
}

AuthenticateResult run_varform_state(json &args, json &out, const io::ResourceIO &resources)
{
	State state;
	args["/output/log/level"_json_pointer] = "error";
	state.init(args, resources, true);

	Eigen::MatrixXd sol;
	try
	{
		solve_initialized_state(state, sol);
	}
	catch (...)
	{
		return SOLVE_FAILED;
	}

	const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);

	state.variational_formulation->save_json(sol);
	state.variational_formulation->export_data(sol);

	out["err_l2"] = stats.l2_err;
	out["err_h1"] = stats.h1_err;
	out["err_h1_semi"] = stats.h1_semi_err;
	out["err_linf"] = stats.linf_err;
	out["err_linf_grad"] = stats.grad_max_err;
	out["err_lp"] = stats.lp_err;

	return SUCCESS;
}

void configure_time_steps(json &args, const int time_steps)
{
	REQUIRE(args.contains("time"));
	json &time = args["time"];
	const double t0 = time.value("t0", 0.0);
	if (time.contains("tend") && time.contains("dt"))
	{
		time.erase("tend");
		time["time_steps"] = time_steps;
	}
	else if (time.contains("tend") && time.contains("time_steps"))
	{
		time["dt"] = (time["tend"].get<double>() - t0) / time["time_steps"].get<int>();
		time["time_steps"] = time_steps;
		time.erase("tend");
	}
	else if (time.contains("dt") && time.contains("time_steps"))
	{
		time["time_steps"] = time_steps;
	}
	else
	{
		FAIL("A transient regression requires two of time.tend, time.dt, and time.time_steps");
	}
}

AuthenticateResult authenticate_input(
	io::LoadedInput loaded,
	const std::string &input_name,
	const bool compute_validation,
	json *computed_validation = nullptr)
{
	const json in_args = loaded.config;

	const std::string tests_key = "tests";
	if (missing_tests_data(in_args, tests_key) && !compute_validation)
	{
		spdlog::error(
			"JSON file missing \"{}\" key. Add a * to the beginning of filename to allow appends.",
			tests_key);
		return MISSING_TEST_DATA;
	}

	// ------------------------------------------------------------------------
	// Patch the JSON file to run a single time step
	json args = in_args;
	auto common_resources = utils::apply_common_params(args, *loaded.resources);
	const io::ResourceIO &resources = common_resources ? *common_resources : *loaded.resources;

	json time_steps;
	if (!args.contains("time"))
		time_steps = "static";
	else if (!args.contains(tests_key) || !args[tests_key].contains("time_steps"))
		time_steps = 1;
	else
		time_steps = args[tests_key]["time_steps"];

	// args["output"] = json({});
	// args["output"]["advanced"]["save_time_sequence"] = false;

	if (time_steps.is_number())
		configure_time_steps(args, time_steps.get<int>());
	// ------------------------------------------------------------------------

	if (input_name.find("/standard/mooney_rivlin_p2.json") == std::string::npos)
	{
		args["/solver/linear/solver"_json_pointer] =
			(input_name.find("navier") == std::string::npos && input_name.find("bilaplace") == std::string::npos && input_name.find("thermoelastic") == std::string::npos)
				? "Eigen::SimplicialLDLT"
				: "Eigen::SparseLU";
	}

	json out = json({});
	AuthenticateResult run_result;
	if (varform::uses_varform_state(args, resources))
	{
		run_result = run_varform_state(args, out, resources);
	}
	else
	{
		if (dynamic_cast<const io::FileSystemIO *>(&resources) == nullptr)
		{
			spdlog::error("Legacy State cannot read bundled input {}", input_name);
			return SOLVE_FAILED;
		}
		args["root_path"] = resources.describe("");
		run_result = run_legacy_state(args, out);
	}
	if (run_result != SUCCESS)
		return run_result;

	out["margin"] = 1e-5;
	out["time_steps"] = time_steps;

	std::vector<std::string> test_keys =
		{"err_l2", "err_h1", "err_h1_semi", "err_linf", "err_linf_grad", "err_lp"};

	if (!compute_validation)
	{
		spdlog::info("Authenticating...");
		json authen = in_args.at(tests_key);
		double margin = authen.value("margin", 1e-5);
		bool authenticated = true;
		for (const std::string &key : test_keys)
		{
			const double prev_val = authen[key];
			const double curr_val = out[key];
			const double relerr = std::abs((curr_val - prev_val) / std::max(std::abs(prev_val), 1e-5));
			if (relerr > margin)
			{
				spdlog::error("Violating Authenticate prev_{0}={1} curr_{0}={2} relerr_{0}={3}", key, prev_val, curr_val, relerr);
				authenticated = false;
			}
		}
		if (!authenticated)
		{
			spdlog::error("Computed tests: {}", out.dump());
			return AUTHETICATION_FAILED;
		}
		spdlog::info("Authenticated ✅");
	}
	else
	{
		if (computed_validation == nullptr)
			return AUTHETICATION_FAILED;
		*computed_validation = std::move(out);
	}

	return SUCCESS;
}

AuthenticateResult authenticate_json(const std::string &json_file, const bool compute_validation)
{
	io::LoadedInput loaded;
	try
	{
		loaded = io::load_json_input(json_file);
	}
	catch (const std::exception &e)
	{
		spdlog::error("unable to load {}: {}", json_file, e.what());
		return MISSING_FILE;
	}

	json computed_validation;
	const AuthenticateResult result = authenticate_input(
		std::move(loaded), json_file, compute_validation,
		compute_validation ? &computed_validation : nullptr);
	if (result != SUCCESS || !compute_validation)
		return result;

	json original;
	if (!load_json(json_file, original))
		return MISSING_FILE;
	spdlog::warn("Appending JSON...");
	original["tests"] = computed_validation;
	std::ofstream file(json_file);
	file << original;
	return file ? SUCCESS : MISSING_FILE;
}

std::string shell_quote(const std::string &value)
{
#ifdef WIN32
	std::string quoted = "\"";
	for (const char character : value)
		quoted += character == '"' ? "\\\"" : std::string(1, character);
	return quoted + "\"";
#else
	std::string quoted = "'";
	for (const char character : value)
		quoted += character == '\'' ? "'\\''" : std::string(1, character);
	return quoted + "'";
#endif
}

std::string hdf5_bundle_name(const std::string &scene)
{
	std::string name;
	name.reserve(scene.size());
	for (const char character : scene)
		name += character == '/' || character == '\\' ? "__" : std::string(1, character);
	if (name.size() >= 5 && name.substr(name.size() - 5) == ".json")
		name.resize(name.size() - 5);
	return name + ".h5";
}

void disable_regression_outputs(json &args)
{
	if (!args.contains("output") || !args["output"].is_object())
		args["output"] = json::object();
	args["/output/directory"_json_pointer] = "";
	args["/output/json"_json_pointer] = "";
	args["/output/paraview/file_name"_json_pointer] = "";
	args["/output/advanced/save_time_sequence"_json_pointer] = false;
}

json checkpoint_config(json args, const std::filesystem::path &output, const int time_steps)
{
	configure_time_steps(args, time_steps);
	args["output"] = json::object();
	args["/output/directory"_json_pointer] = output.string();
	args["/output/checkpoint/path"_json_pointer] = "checkpoint_{:d}.h5";
	args["/output/advanced/save_time_sequence"_json_pointer] = false;
	args["/output/log/level"_json_pointer] = "error";
	args["/solver/max_threads"_json_pointer] = 1;
	args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
	args["/solver/adjoint_linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
	return args;
}

Eigen::MatrixXd run_simulation(const json &args, const io::ResourceIO &resources)
{
	State state;
	state.init(args, resources, true);
	Eigen::MatrixXd solution;
	solve_initialized_state(state, solution);
	return solution;
}

Eigen::MatrixXd resume_checkpoint(const std::filesystem::path &path, const bool reorder)
{
	io::CheckpointReader checkpoint(path);
	json continuation = checkpoint.config();
	continuation["/input/checkpoint/reorder"_json_pointer] = reorder;
	State state;
	state.init(continuation, checkpoint, true);
	Eigen::MatrixXd solution;
	solve_initialized_state(state, solution);
	return solution;
}

Eigen::MatrixXd canonical_checkpoint_solution(const std::filesystem::path &path)
{
	io::CheckpointReader checkpoint(path);
	Eigen::MatrixXd solution = checkpoint.read_matrix("/checkpoint/state/solution");
	const int dimension = checkpoint.read_mesh("/checkpoint/meshes/active")->dimension();

	const auto ordering = [&checkpoint](const std::string &name) {
		const std::string path = "/checkpoint/state/orderings/" + name;
		if (!checkpoint.exists(path))
			return Eigen::VectorXi();
		const std::vector<int> values = checkpoint.read_int_vector(path);
		return Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(values.data(), values.size()));
	};

	const Eigen::VectorXi primary = ordering("primary");
	const Eigen::VectorXi pressure = ordering("pressure");
	const Eigen::VectorXi mesh_motion = ordering("mesh_motion");
	const Eigen::VectorXi solid = ordering("solid");
	const Eigen::VectorXi temperature = ordering("temperature");
	if (primary.size() == 0)
		return solution;

	const int secondary_rows = pressure.size() + dimension * mesh_motion.size()
							   + dimension * solid.size() + temperature.size();
	const int available_primary_rows = solution.rows() - secondary_rows;
	const int primary_block_size = available_primary_rows >= dimension * primary.size()
									   ? dimension
									   : 1;
	REQUIRE(available_primary_rows >= primary_block_size * primary.size());

	Eigen::MatrixXd canonical = solution;
	int offset = 0;
	const auto convert = [&](const Eigen::VectorXi &map, const int block_size) {
		if (map.size() == 0)
			return;
		const int rows = map.size() * block_size;
		canonical.middleRows(offset, rows) = utils::unreorder_matrix(
			solution.middleRows(offset, rows), map, -1, block_size);
		offset += rows;
	};
	convert(primary, primary_block_size);
	convert(pressure, 1);
	convert(mesh_motion, dimension);
	convert(solid, dimension);
	convert(temperature, 1);
	REQUIRE(offset <= solution.rows());
	return canonical;
}

void check_checkpoint_equivalent(
	const json &config,
	const io::ResourceIO &resources,
	const std::filesystem::path &output,
	const int total_steps,
	const int checkpoint_step,
	const bool reorder,
	const double margin)
{
	namespace fs = std::filesystem;
	REQUIRE(checkpoint_step > 0);
	REQUIRE(checkpoint_step < total_steps);
	const json args = checkpoint_config(config, output, total_steps);
	run_simulation(args, resources);
	const fs::path checkpoint_path = output / fmt::format("checkpoint_{:d}.h5", checkpoint_step);
	const fs::path final_checkpoint = output / fmt::format("checkpoint_{:d}.h5", total_steps);
	const fs::path uninterrupted_checkpoint = output / "uninterrupted.h5";
	REQUIRE(fs::is_regular_file(checkpoint_path));
	REQUIRE(fs::is_regular_file(final_checkpoint));
	fs::copy_file(final_checkpoint, uninterrupted_checkpoint, fs::copy_options::overwrite_existing);
	resume_checkpoint(checkpoint_path, reorder);
	const Eigen::MatrixXd uninterrupted = canonical_checkpoint_solution(uninterrupted_checkpoint);
	const Eigen::MatrixXd resumed = canonical_checkpoint_solution(final_checkpoint);
	REQUIRE(uninterrupted.rows() == resumed.rows());
	REQUIRE(uninterrupted.cols() == resumed.cols());
	CAPTURE((uninterrupted - resumed).lpNorm<Eigen::Infinity>(), margin, reorder);
	CHECK(uninterrupted.isApprox(resumed, margin));
	for (int step = checkpoint_step + 1; step <= total_steps; ++step)
		CHECK(fs::is_regular_file(output / fmt::format("checkpoint_{:d}.h5", step)));
}

bool has_multiple_time_steps(const json &args)
{
	if (!args.contains("time") || !args["time"].is_object())
		return false;
	const json &time = args["time"];
	if (time.contains("time_steps"))
		return time["time_steps"].is_number_integer() && time["time_steps"].get<int>() > 1;
	if (time.contains("tend") && time.contains("dt"))
		return time["tend"].get<double>() - time.value("t0", 0.0) > time["dt"].get<double>();
	return false;
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsrun = "[run]";
#else
std::string tagsrun = "[.][run]";
#endif

void run_data(const std::string &test_file, const std::string &dir)
{
	// Disabled on Windows CI, due to the requirement for Pardiso.
	std::ifstream file(POLYFEM_TEST_DIR "/" + test_file + ".txt");
	std::vector<std::string> failing_tests;
	const char *scene_filter = std::getenv("POLYFEM_VERIFY_SCENE");
	int processed_scenes = 0;
	std::string line;
	while (std::getline(file, line))
	{
		if (line.empty())
			continue;

		bool compute_validation = false;
		if (line[0] == '#')
			continue;
		else if (line[0] == '*')
		{
			compute_validation = true;
			line = line.substr(1);
		}
		if (scene_filter != nullptr && line != scene_filter)
			continue;
		++processed_scenes;
		spdlog::info("Processing {}", line);
		AuthenticateResult result = authenticate_json(dir + "/" + line, compute_validation);
		CAPTURE(line);
		CHECK(result == SUCCESS);
		if (result != SUCCESS)
			failing_tests.push_back(line);
	}
	if (failing_tests.size() > 0)
	{
		std::stringstream ss;
		ss << "Failing tests:" << std::endl;
		for (auto &t : failing_tests)
			ss << t << std::endl;

		logger().error(ss.str());
	}
	if (scene_filter != nullptr)
		CHECK(processed_scenes == 1);
}

TEST_CASE("all PolyFEM data JSON files are classified", "[data]")
{
	const std::vector<std::string> manifests = {
		"contact_2d", "contact_3d", "adhesion", "selection", "thermo",
		"standard", "hybrid", "time_int", "miso", "triangle", "slow", "known_issues"};
	std::set<std::string> classified;

	for (const std::string &manifest : manifests)
	{
		std::ifstream file(POLYFEM_TEST_DIR "/" + manifest + ".txt");
		REQUIRE(file.is_open());

		std::string line;
		while (std::getline(file, line))
		{
			if (line.empty() || line[0] == '#')
				continue;
			if (line[0] == '*')
				line = line.substr(1);
			CAPTURE(manifest, line);
			CHECK(classified.insert(line).second);
			CHECK(std::filesystem::is_regular_file(std::filesystem::path(POLYFEM_DATA_DIR) / line));
		}
	}

	std::set<std::string> dependencies;
	for (const auto &entry : std::filesystem::recursive_directory_iterator(POLYFEM_DATA_DIR))
	{
		if (!entry.is_regular_file() || entry.path().extension() != ".json")
			continue;

		const std::string relative = std::filesystem::relative(entry.path(), POLYFEM_DATA_DIR).generic_string();
		if (relative.rfind("old-tolerances/", 0) == 0)
			continue;

		json input;
		REQUIRE(load_json(entry.path().string(), input));
		if (input.contains("common"))
		{
			REQUIRE(input["common"].is_string());
			const std::filesystem::path dependency =
				(entry.path().parent_path() / input["common"].get<std::string>()).lexically_normal();
			CAPTURE(relative, dependency);
			CHECK(std::filesystem::is_regular_file(dependency));
			dependencies.insert(std::filesystem::relative(dependency, POLYFEM_DATA_DIR).generic_string());
		}
	}

	for (const auto &entry : std::filesystem::recursive_directory_iterator(POLYFEM_DATA_DIR))
	{
		if (!entry.is_regular_file() || entry.path().extension() != ".json")
			continue;

		const std::string relative = std::filesystem::relative(entry.path(), POLYFEM_DATA_DIR).generic_string();
		if (relative.rfind("old-tolerances/", 0) == 0)
			continue;

		json input;
		REQUIRE(load_json(entry.path().string(), input));
		const bool is_polyfem_input =
			input.contains("geometry") || input.contains("common") || input.contains("problem")
			|| input.contains("materials") || input.contains("space") || input.contains("time")
			|| input.contains("contact") || input.contains("boundary_conditions")
			|| input.contains("solver") || input.contains("output") || input.contains("tests");

		CAPTURE(relative);
		if (is_polyfem_input)
			CHECK((classified.count(relative) == 1 || dependencies.count(relative) == 1));
		else
			CHECK(classified.count(relative) == 0);
	}
}

const std::string CONTACT_TEST_FOLDER = POLYFEM_TEST_DIR + std::string("/../contact-tests/");

TEST_CASE("contact_2d", tagsrun)
{
	run_data("contact_2d", POLYFEM_DATA_DIR);
}

TEST_CASE("contact_3d", tagsrun)
{
	run_data("contact_3d", POLYFEM_DATA_DIR);
}

TEST_CASE("adhesion", tagsrun)
{
	run_data("adhesion", POLYFEM_DATA_DIR);
}

TEST_CASE("selection", tagsrun)
{
	run_data("selection", POLYFEM_DATA_DIR);
}

TEST_CASE("thermo", tagsrun)
{
	run_data("thermo", POLYFEM_DATA_DIR);
}

TEST_CASE("standard", tagsrun)
{
	run_data("standard", POLYFEM_DATA_DIR);
}

TEST_CASE("hybrid", tagsrun)
{
	run_data("hybrid", POLYFEM_DATA_DIR);
}

TEST_CASE("time_int", tagsrun)
{
	run_data("time_int", POLYFEM_DATA_DIR);
}

#if defined(NDEBUG) && !defined(WIN32)
TEST_CASE("quick filesystem scenes run from generated HDF5 bundles", "[hdf5_regression]")
#else
TEST_CASE("quick filesystem scenes run from generated HDF5 bundles", "[.][hdf5_regression]")
#endif
{
	namespace fs = std::filesystem;
	const fs::path manifest_path = fs::path(POLYFEM_DATA_DIR) / "io-tests/hdf5-quick.json";
	json manifest;
	REQUIRE(load_json(manifest_path.string(), manifest));
	REQUIRE(manifest["scenes"].size() >= 20);

	const fs::path output = fs::temp_directory_path() / "polyfem-hdf5-regression";
	fs::remove_all(output);
	fs::create_directories(output);
	const char *environment_python = std::getenv("POLYFEM_TEST_PYTHON");
	const std::string python = environment_python == nullptr
								   ? std::string(POLYFEM_TEST_PYTHON_EXECUTABLE)
								   : std::string(environment_python);
	const std::string command =
		shell_quote(python) + " "
		+ shell_quote(std::string(POLYFEM_SOURCE_DIR) + "/tools/package_json_hdf5.py")
		+ " --manifest " + shell_quote(manifest_path.string())
		+ " --data-root " + shell_quote(POLYFEM_DATA_DIR)
		+ " --output-dir " + shell_quote(output.string());
	REQUIRE(std::system(command.c_str()) == 0);

	const char *scene_filter = std::getenv("POLYFEM_HDF5_SCENE");
	int tested_scenes = 0;
	for (const json &entry : manifest["scenes"])
	{
		const std::string scene = entry["path"];
		if (scene_filter != nullptr && scene != scene_filter)
			continue;
		CAPTURE(scene, entry["category"]);
		++tested_scenes;

		io::LoadedInput loaded = io::load_hdf5_input(output / hdf5_bundle_name(scene));
		disable_regression_outputs(loaded.config);
		CHECK(authenticate_input(std::move(loaded), scene, false) == SUCCESS);
	}
	CHECK(tested_scenes == (scene_filter == nullptr ? int(manifest["scenes"].size()) : 1));
	fs::remove_all(output);
}

#if defined(NDEBUG) && !defined(WIN32)
TEST_CASE("checkpoint resume matches uninterrupted transient simulations", "[checkpoint]")
#else
TEST_CASE("checkpoint resume matches uninterrupted transient simulations", "[.][checkpoint]")
#endif
{
	namespace fs = std::filesystem;
	const fs::path manifest_path = fs::path(POLYFEM_DATA_DIR) / "io-tests/checkpoint-quick.json";
	json manifest;
	REQUIRE(load_json(manifest_path.string(), manifest));
	REQUIRE(manifest["scenes"].size() >= 6);
	const fs::path root = fs::temp_directory_path() / "polyfem-checkpoint-regression";
	fs::remove_all(root);

	int index = 0;
	const char *family_filter = std::getenv("POLYFEM_CHECKPOINT_FAMILY");
	for (const json &entry : manifest["scenes"])
	{
		if (family_filter != nullptr && entry["family"] != family_filter)
			continue;
		const std::string scene = entry["path"];
		CAPTURE(scene, entry["family"]);
		io::LoadedInput loaded = io::load_json_input(fs::path(POLYFEM_DATA_DIR) / scene);
		json scene_config = loaded.config;
		auto common_resources = utils::apply_common_params(scene_config, *loaded.resources);
		const io::ResourceIO &resources = common_resources ? *common_resources : *loaded.resources;
		const json overrides = entry.value("overrides", json::object());
		for (const auto &[pointer, value] : overrides.items())
			scene_config[json::json_pointer(pointer)] = value;
		check_checkpoint_equivalent(
			scene_config, resources, root / std::to_string(index),
			entry["steps"], entry["checkpoint_step"],
			/*reorder=*/true, entry.value("margin", 1e-8));
		++index;
	}
	fs::remove_all(root);
}

TEST_CASE("all testable transient scenes support checkpoint continuation", "[.][checkpoint_all]")
{
	namespace fs = std::filesystem;
	const std::vector<std::string> manifests = {
		"contact_2d", "contact_3d", "adhesion", "selection", "thermo",
		"standard", "hybrid", "time_int", "slow"};
	const fs::path root = fs::temp_directory_path() / "polyfem-checkpoint-all";
	fs::remove_all(root);
	int scene_index = 0;
	for (const std::string &manifest : manifests)
	{
		std::ifstream stream(std::filesystem::path(POLYFEM_TEST_DIR) / (manifest + ".txt"));
		REQUIRE(stream.good());
		std::string scene;
		while (std::getline(stream, scene))
		{
			if (scene.empty() || scene.front() == '#')
				continue;
			if (scene.front() == '*')
				scene.erase(scene.begin());
			io::LoadedInput loaded = io::load_json_input(fs::path(POLYFEM_DATA_DIR) / scene);
			json effective = loaded.config;
			auto common_resources = utils::apply_common_params(effective, *loaded.resources);
			const io::ResourceIO &resources = common_resources ? *common_resources : *loaded.resources;
			if (!has_multiple_time_steps(effective) || !varform::uses_varform_state(effective, resources))
				continue;

			CAPTURE(scene, manifest);
			check_checkpoint_equivalent(
				effective, resources, root / std::to_string(scene_index),
				/*total_steps=*/2, /*checkpoint_step=*/1,
				/*reorder=*/true, /*margin=*/1e-7);
			++scene_index;
		}
	}
	CHECK(scene_index >= 20);
	fs::remove_all(root);
}

#if defined(NDEBUG) && !defined(WIN32)
TEST_CASE("checkpoint embeds all HDF5 input dependencies", "[checkpoint][hdf5]")
#else
TEST_CASE("checkpoint embeds all HDF5 input dependencies", "[.][checkpoint][hdf5]")
#endif
{
	namespace fs = std::filesystem;
	const fs::path root = fs::temp_directory_path() / "polyfem-checkpoint-self-contained";
	const fs::path bundle = root / "input.h5";
	const fs::path output = root / "output";
	fs::remove_all(root);
	fs::create_directories(root);
	const char *environment_python = std::getenv("POLYFEM_TEST_PYTHON");
	const std::string python = environment_python == nullptr
								   ? std::string(POLYFEM_TEST_PYTHON_EXECUTABLE)
								   : std::string(environment_python);
	const std::string command =
		shell_quote(python) + " "
		+ shell_quote(std::string(POLYFEM_SOURCE_DIR) + "/tools/package_json_hdf5.py")
		+ " --input " + shell_quote(std::string(POLYFEM_DATA_DIR) + "/time-int/bdf1.json")
		+ " --output " + shell_quote(bundle.string());
	REQUIRE(std::system(command.c_str()) == 0);

	io::LoadedInput loaded = io::load_hdf5_input(bundle);
	const json args = checkpoint_config(loaded.config, output, 4);
	run_simulation(args, *loaded.resources);
	fs::copy_file(output / "checkpoint_4.h5", root / "uninterrupted.h5", fs::copy_options::overwrite_existing);
	loaded.resources.reset();
	fs::remove(bundle);
	REQUIRE_FALSE(fs::exists(bundle));

	const fs::path checkpoint_path = output / "checkpoint_2.h5";
	REQUIRE(fs::is_regular_file(checkpoint_path));
	CHECK_THROWS(resume_checkpoint(checkpoint_path, false));
	resume_checkpoint(checkpoint_path, true);
	const Eigen::MatrixXd uninterrupted = canonical_checkpoint_solution(root / "uninterrupted.h5");
	const Eigen::MatrixXd resumed = canonical_checkpoint_solution(output / "checkpoint_4.h5");
	REQUIRE(uninterrupted.rows() == resumed.rows());
	REQUIRE(uninterrupted.cols() == resumed.cols());
	CAPTURE((uninterrupted - resumed).lpNorm<Eigen::Infinity>());
	CHECK(uninterrupted.isApprox(resumed, 1e-8));
	fs::remove_all(root);
}

TEST_CASE("checkpoint continuation rejects incompatible runtime state", "[.][checkpoint][checkpoint_negative]")
{
	namespace fs = std::filesystem;
	const fs::path root = fs::temp_directory_path() / "polyfem-invalid-checkpoint-state";
	const fs::path output = root / "output";
	fs::remove_all(root);
	io::LoadedInput loaded = io::load_json_input(fs::path(POLYFEM_DATA_DIR) / "time-int/bdf1.json");
	json quick_config = loaded.config;
	auto common_resources = utils::apply_common_params(quick_config, *loaded.resources);
	const io::ResourceIO &resources = common_resources ? *common_resources : *loaded.resources;
	quick_config["/geometry/n_refs"_json_pointer] = 1;
	const json args = checkpoint_config(quick_config, output, 2);
	run_simulation(args, resources);
	const fs::path valid = output / "checkpoint_1.h5";
	REQUIRE(fs::is_regular_file(valid));

	const fs::path wrong_formulation = root / "wrong-formulation.h5";
	fs::copy_file(valid, wrong_formulation);
	{
		h5pp::File file(wrong_formulation.string(), h5pp::FileAccess::READWRITE);
		file.deleteLink("/checkpoint/metadata/formulation");
		file.writeDataset(std::string("WrongFormulation"), "/checkpoint/metadata/formulation");
	}
	CHECK_THROWS([&] {
		io::CheckpointReader checkpoint(wrong_formulation);
		State state;
		state.init(checkpoint, true);
	}());

	CHECK_THROWS([&] {
		io::CheckpointReader checkpoint(valid);
		json continuation = checkpoint.config();
		continuation["time"]["dt"] = 2 * continuation["time"]["dt"].get<double>();
		State state;
		state.init(continuation, checkpoint, true);
	}());

	const fs::path wrong_solution = root / "wrong-solution.h5";
	fs::copy_file(valid, wrong_solution);
	{
		h5pp::File file(wrong_solution.string(), h5pp::FileAccess::READWRITE);
		file.deleteLink("/checkpoint/state/solution");
		file.writeDataset(Eigen::MatrixXd::Zero(1, 1), "/checkpoint/state/solution");
	}
	CHECK_THROWS(resume_checkpoint(wrong_solution, true));

	const fs::path missing_history = root / "missing-history.h5";
	fs::copy_file(valid, missing_history);
	{
		h5pp::File file(missing_history.string(), h5pp::FileAccess::READWRITE);
		file.deleteLink("/checkpoint/state/primary_integrator/x");
	}
	CHECK_THROWS(resume_checkpoint(missing_history, true));

	const fs::path wrong_dynamic_order = root / "wrong-dynamic-order.h5";
	fs::copy_file(valid, wrong_dynamic_order);
	{
		h5pp::File file(wrong_dynamic_order.string(), h5pp::FileAccess::READWRITE);
		const long stored_order = file.readDataset<long>("/checkpoint/state/primary_integrator/dynamic_order");
		file.deleteLink("/checkpoint/state/primary_integrator/dynamic_order");
		file.writeDataset(
			stored_order == 1 ? long(2) : long(1),
			"/checkpoint/state/primary_integrator/dynamic_order");
	}
	CHECK_THROWS(resume_checkpoint(wrong_dynamic_order, true));

	const fs::path wrong_history_dimensions = root / "wrong-history-dimensions.h5";
	fs::copy_file(valid, wrong_history_dimensions);
	{
		h5pp::File file(wrong_history_dimensions.string(), h5pp::FileAccess::READWRITE);
		file.deleteLink("/checkpoint/state/primary_integrator/x");
		file.writeDataset(Eigen::MatrixXd::Zero(1, 1), "/checkpoint/state/primary_integrator/x");
	}
	CHECK_THROWS(resume_checkpoint(wrong_history_dimensions, true));
	fs::remove_all(root);
}

#ifdef POLYFEM_WITH_TRIANGLE
TEST_CASE("triangle_data", tagsrun)
{
	run_data("triangle", POLYFEM_DATA_DIR);
}
#endif

TEST_CASE("slow", "[.][slow]")
{
	run_data("slow", POLYFEM_DATA_DIR);
}

TEST_CASE("runners-pref", tagsrun)
{
	run_data("pref_test_list", POLYFEM_PREF_DIR);
}

TEST_CASE("runners-polyspline", tagsrun)
{
	run_data("polyspline_test_list", POLYFEM_POLYSPLINE_DIR);
}

#ifdef POLYFEM_WITH_MISO
TEST_CASE("miso", tagsrun)
{
	run_data("miso", POLYFEM_DATA_DIR);
}
#endif
