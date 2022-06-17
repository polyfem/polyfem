#include <CLI/CLI.hpp>
#include <polyfem/State.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/problem/ProblemFactory.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <filesystem>

using namespace polyfem;
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::utils;
using namespace polysolve;
using namespace Eigen;
using namespace HighFive;

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

int main(int argc, char **argv)
{
	using namespace std::filesystem;

	CLI::App command_line{"polyfem"};
	// Eigen::setNbThreads(1);

	// Input
	std::string mesh_file = "";
	std::string json_file = "";
	std::string hdf5_file = "";
	std::string febio_file = "";

	// Output
	std::string output_dir = "";
	std::string output_json = "";
	std::string output_vtu = "";
	std::string screenshot = "";

	// Problem
	std::string problem_name = "";
	std::string formulation = "";
	std::string time_integrator_name = "";
	std::string solver = "";

	std::string bc_method = "";

	int discr_order = 1;

	int n_refs = 0;
	bool use_splines = false;
	bool count_flipped_els = true;
	bool skip_normalization = true;
	bool no_ui = false;
	bool p_ref = false;
	bool force_linear = false;
	bool isoparametric = false;
	bool serendipity = false;
	bool export_material_params = false;
	bool save_solve_sequence_debug = false;
	bool compute_errors = false;

	std::string log_file = "";
	bool is_quiet = false;
	bool stop_after_build_basis = false;
	bool lump_mass_mat = false;
	spdlog::level::level_enum log_level = spdlog::level::debug;
	int cache_size = -1;
	size_t max_threads = std::numeric_limits<size_t>::max();
	double f_delta = 0;

	bool use_al = false;
	int min_component = -1;

	double vis_mesh_res = -1;

	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);
	command_line.add_option("--hdf5", hdf5_file, "Simulation hdf5 file")->check(CLI::ExistingFile);


	// IO
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");
	command_line.add_option("--log_file", log_file, "Log to a file");
	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

	std::vector<std::string> names;
	std::vector<Eigen::MatrixXi> cells;
	std::vector<Eigen::MatrixXd> vertices;

	json in_args = json({});

	if (!json_file.empty())
	{
		std::ifstream file(json_file);

		if (file.is_open())
			file >> in_args;
		else
			logger().error("unable to open {} file", json_file);
		file.close();

		if (!in_args.contains("root_path"))
		{
			in_args["root_path"] = json_file;
		}

		if (in_args.contains("common"))
		{
			apply_default_params(in_args);
			in_args.erase("common"); // Remove this so state does not redo the apply
		}
	}
	else if (!hdf5_file.empty())
	{
		File file(hdf5_file, File::ReadOnly);
		std::string json_string = H5Easy::load<std::string>(file, "json");

		in_args = json::parse(json_string);
		in_args["root_path"] = hdf5_file;

		Group meshes = file.getGroup("meshes");
		names = meshes.listObjectNames();
		cells.resize(names.size());
		vertices.resize(names.size());

		for (size_t i = 0; i < names.size(); ++i)
		{
			const auto &s = names[i];
			const auto &tmp = meshes.getGroup(s);

			tmp.getDataSet("c").read(cells[i]);
			tmp.getDataSet("v").read(vertices[i]);
		}
	}

	if (!output_dir.empty())
	{
		create_directories(output_dir);
	}

		State state(max_threads);
		state.init_logger(log_file, log_level, is_quiet);
		state.init(in_args, output_dir);
			state.load_mesh(false, names, cells, vertices);

		// Mesh was not loaded successfully; load_mesh() logged the error.
		if (state.mesh == nullptr)
		{
			// Cannot proceed without a mesh.
			return EXIT_FAILURE;
		}

		state.compute_mesh_stats();

		state.build_basis();

		state.assemble_rhs();
		state.assemble_stiffness_mat();

		state.solve_problem();

		state.compute_errors();

		state.save_json();
		state.export_data();

	return EXIT_SUCCESS;
}
