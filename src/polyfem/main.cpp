#include <filesystem>

#include <CLI/CLI.hpp>

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <polyfem/State.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polysolve/LinearSolver.hpp>

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

int main(int argc, char **argv)
{
	using namespace polyfem;

	CLI::App command_line{"polyfem"};

	// Eigen::setNbThreads(1);
	size_t max_threads = std::numeric_limits<size_t>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	std::string json_file = "";
	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);

	std::string hdf5_file = "";
	command_line.add_option("--hdf5", hdf5_file, "Simulation hdf5 file")->check(CLI::ExistingFile);

	std::string output_dir = "";
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	bool is_quiet = false;
	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");

	std::string log_file = "";
	command_line.add_option("--log_file", log_file, "Log to a file");

	const std::vector<std::string> solvers = polysolve::LinearSolver::availableSolvers();
	std::string solver;
	command_line.add_option("--solver", solver, "Used to print the list of linear solvers available")->check(CLI::IsMember(solvers));

	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	spdlog::level::level_enum log_level = spdlog::level::debug;
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
	}
	else if (!hdf5_file.empty())
	{
		HighFive::File file(hdf5_file, HighFive::File::ReadOnly);
		std::string json_string = H5Easy::load<std::string>(file, "json");

		in_args = json::parse(json_string);
		in_args["root_path"] = hdf5_file;

		HighFive::Group meshes = file.getGroup("meshes");
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
		std::filesystem::create_directories(output_dir);
	}

	State state(max_threads);
	state.init_logger(log_file, log_level, is_quiet);
	state.init(in_args, output_dir);
	state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

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
