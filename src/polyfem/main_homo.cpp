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

Eigen::MatrixXd generate_linear_field(const polyfem::State &state, const Eigen::MatrixXd &grad)
{
	const int problem_dim = grad.rows();
	const int dim = state.mesh->dimension();
	assert(dim == grad.cols());

	Eigen::MatrixXd func(state.n_bases * problem_dim, 1);
	func.setZero();

	for (int i = 0; i < state.n_bases; i++)
	{
		func.block(i * problem_dim, 0, problem_dim, 1) = grad * state.mesh_nodes->node_position(i).transpose();
	}

	return func;
}

int main(int argc, char **argv)
{
	using namespace polyfem;

	CLI::App command_line{"polyfem"};

	// Eigen::setNbThreads(1);
	size_t max_threads = std::numeric_limits<size_t>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	std::string json_file = "";
	command_line.add_option("-j,--json", json_file, "Simulation JSON file")->check(CLI::ExistingFile);

	std::string output_dir = "";
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	bool is_quiet = false;
	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");

	bool is_strict = false;

	bool fallback_solver = false;
	command_line.add_flag("--enable_overwrite_solver", fallback_solver, "If solver in json is not present, falls back to default");

	std::string log_file = "";
	command_line.add_option("--log_file", log_file, "Log to a file");

	// const std::vector<std::string> solvers = polysolve::LinearSolver::availableSolvers();
	// std::string solver;
	// command_line.add_option("--solver", solver, "Used to print the list of linear solvers available")->check(CLI::IsMember(solvers));

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
			log_and_throw_error(fmt::format("unable to open {} file", json_file));
		file.close();

		if (!in_args.contains("root_path"))
		{
			in_args["root_path"] = json_file;
		}
	}
	else
	{
		logger().error("No input file specified!");
		return command_line.exit(CLI::RequiredError("--json or --hdf5"));
	}

	if (!output_dir.empty())
	{
		std::filesystem::create_directories(output_dir);
	}

	State state(max_threads);
	state.init_logger(log_file, log_level, is_quiet);
	state.init(in_args, is_strict, output_dir, fallback_solver);
	state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
		// Cannot proceed without a mesh.
		return EXIT_FAILURE;

	state.stats.compute_mesh_stats(*state.mesh);
	state.build_basis();
	state.assemble_rhs();
	state.assemble_stiffness_mat();

	auto &micro_assembler = state.assembler.get_microstructure_local_assembler(state.formulation());
	std::shared_ptr<State> micro_state = micro_assembler.get_microstructure_state();
	assert(micro_state);

	const int dim = micro_state->mesh->dimension();
	Eigen::MatrixXd F(dim, dim);
	int i = 0;
	for (auto &vec : state.args["def_grad"])
	{
		int j = 0;
		for (auto &val : vec)
		{
			F(i, j) = val;
			j++;
		}
		i++;
	}

	Eigen::MatrixXd fluctuated;
	for (int l = 200; l > 100; l--)
	{
		F(1, 1) = l / 200.0;
		
		{
			Eigen::MatrixXd disp_grad = F - Eigen::MatrixXd::Identity(dim, dim);
			Eigen::MatrixXd x;
			micro_state->solve_homogenized_field(disp_grad, fluctuated, x);
			fluctuated = x;
		}

		// effective energy = average energy over unit cell
		double energy;
		energy = micro_assembler.homogenize_energy(fluctuated);

		// effective stress = average stress over unit cell
		Eigen::MatrixXd stress;
		micro_assembler.homogenize_stress(fluctuated, stress);

		std::cout << "homogenized energy " << energy << "\n";
		std::cout << "homogenized stress\n" << stress << "\n";

		Eigen::MatrixXd pressure(micro_state->n_pressure_bases, 1);
		micro_state->args["output"]["paraview"]["file_name"] = "load_" + std::to_string(l) + ".vtu";
		micro_state->export_data(fluctuated, pressure);
	}

	return EXIT_SUCCESS;
}