#include <filesystem>

#include <CLI/CLI.hpp>

#include <polyfem/State.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/assembler/Multiscale.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/io/Evaluator.hpp>

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
	size_t max_threads = 32;
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

	int min_strain = 0;
	command_line.add_option("--min", min_strain, "min strain in the sweep");

	int max_strain = 40;
	command_line.add_option("--max", max_strain, "max strain in the sweep");

	int stride = 1;
	command_line.add_option("--stride", stride, "stride in the sweep");

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
		return command_line.exit(CLI::RequiredError("--json"));
	}

	if (has_arg(command_line, "max_threads"))
	{
		auto tmp = R"({
				"solver": {
					"max_threads": -1
				}
			})"_json;

		tmp["solver"]["max_threads"] = max_threads;

		in_args.merge_patch(tmp);
	}

	if (has_arg(command_line, "log_level"))
	{
		auto tmp = R"({
				"output": {
					"log": {
						"level": -1
					}
				}
			})"_json;

		tmp["output"]["log"]["level"] = int(log_level);

		in_args.merge_patch(tmp);
	}

	if (!output_dir.empty())
	{
		std::filesystem::create_directories(output_dir);
	}

	State state;
	state.init(in_args, is_strict);
	state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
		// Cannot proceed without a mesh.
		return EXIT_FAILURE;

	state.stats.compute_mesh_stats(*state.mesh);
	state.build_basis();
	state.assemble_rhs();
	state.assemble_mass_mat();

	assembler::Multiscale &micro_assembler = *dynamic_cast<assembler::Multiscale*>(state.assembler.get());
	std::shared_ptr<State> micro_state = micro_assembler.get_microstructure_state();
	assert(micro_state);

	const int dim = micro_state->mesh->dimension();
	Eigen::MatrixXd F;
	F.setZero(dim, dim);

	for (int l = -min_strain; l >= -max_strain; l -= stride)
	{
		F << 0, 0, 0, l / 100.0;

		micro_state->args["output"]["paraview"]["file_name"] = "load_" + std::to_string(-l) + ".vtu";

		Eigen::MatrixXd fluctuated;
		micro_state->solve_homogenized_field(F, fluctuated, micro_state->args["boundary_conditions"]["fixed_macro_strain"], false);
		
		// recover extended solution as the initial guess for the next solve
		Eigen::VectorXd extended(fluctuated.size() + F.size());
		{
			extended.head(fluctuated.size()) = fluctuated - io::Evaluator::generate_linear_field(micro_state->n_bases, micro_state->mesh_nodes, F);
			extended.tail(F.size()) = utils::flatten(F);
			micro_state->homo_initial_guess = extended;
		}

		// effective energy = average energy over unit cell
		double energy = micro_assembler.homogenize_energy(fluctuated);

		// effective stress = average stress over unit cell
		Eigen::MatrixXd stress;
		micro_assembler.homogenize_stress(fluctuated, stress);

		// Eigen::MatrixXd def_grad = micro_assembler.homogenize_def_grad(fluctuated) + Eigen::MatrixXd::Identity(dim, dim);

		std::cout << "disp grad " << utils::flatten(F).transpose() << "\n";
		std::cout << "homogenized energy " << energy << "\n";
		// std::cout << "homogenized def grad\n" << def_grad << "\n";
		std::cout << "homogenized stress\n" << stress << "\n";

		micro_state->export_data(fluctuated, Eigen::MatrixXd());
	}

	return EXIT_SUCCESS;
}