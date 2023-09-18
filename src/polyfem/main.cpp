#include <filesystem>

#include <CLI/CLI.hpp>

#include <h5pp/h5pp.h>

#include <polyfem/State.hpp>

#include <polyfem/solver/AdjointNLProblem.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polysolve/LinearSolver.hpp>

using namespace polyfem;
using namespace solver;

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

bool load_json(const std::string &json_file, json &out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

	if (!out.contains("root_path"))
		out["root_path"] = json_file;

	return true;
}

int forward_simulation(const CLI::App &command_line,
					   const std::string &hdf5_file,
					   const std::string output_dir,
					   const size_t max_threads,
					   const bool is_strict,
					   const bool fallback_solver,
					   const spdlog::level::level_enum &log_level,
					   json &in_args);

int optimization_simulation(const CLI::App &command_line,
							const size_t max_threads,
							const bool is_strict,
							const spdlog::level::level_enum &log_level,
							json &opt_args);

int main(int argc, char **argv)
{
	using namespace polyfem;

	CLI::App command_line{"polyfem"};

	command_line.ignore_case();
	command_line.ignore_underscore();

	// Eigen::setNbThreads(1);
	size_t max_threads = std::numeric_limits<size_t>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	std::string json_file = "";
	command_line.add_option("-j,--json", json_file, "Simulation JSON file")->check(CLI::ExistingFile);

	std::string hdf5_file = "";
	command_line.add_option("--hdf5", hdf5_file, "Simulation hdf5 file")->check(CLI::ExistingFile);

	std::string output_dir = "";
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	bool is_strict = true;
	command_line.add_flag("-s,--strict_validation,!--ns,!--no_strict_validation", is_strict, "Disables strict validation of input JSON");

	bool fallback_solver = false;
	command_line.add_flag("--enable_overwrite_solver", fallback_solver, "If solver in json is not present, falls back to default");

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

	json in_args = json({});

	if (!json_file.empty())
	{
		const bool ok = load_json(json_file, in_args);
		std::ifstream file(json_file);

		if (!ok)
			log_and_throw_error(fmt::format("unable to open {} file", json_file));

		if (in_args.contains("states"))
			return optimization_simulation(command_line, max_threads, is_strict, log_level, in_args);
		else
			return forward_simulation(command_line, "", output_dir, max_threads,
									  is_strict, fallback_solver, log_level, in_args);
	}
	else
		return forward_simulation(command_line, hdf5_file, output_dir, max_threads,
								  is_strict, fallback_solver, log_level, in_args);
}

int forward_simulation(const CLI::App &command_line,
					   const std::string &hdf5_file,
					   const std::string output_dir,
					   const size_t max_threads,
					   const bool is_strict,
					   const bool fallback_solver,
					   const spdlog::level::level_enum &log_level,
					   json &in_args)
{
	std::vector<std::string> names;
	std::vector<Eigen::MatrixXi> cells;
	std::vector<Eigen::MatrixXd> vertices;

	if (in_args.empty() && hdf5_file.empty())
	{
		logger().error("No input file specified!");
		return command_line.exit(CLI::RequiredError("--json or --hdf5"));
	}

	if (in_args.empty() && !hdf5_file.empty())
	{
		using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

		h5pp::File file(hdf5_file, h5pp::FileAccess::READONLY);
		std::string json_string = file.readDataset<std::string>("json");

		in_args = json::parse(json_string);
		in_args["root_path"] = hdf5_file;

		names = file.findGroups("", "/meshes");
		cells.resize(names.size());
		vertices.resize(names.size());

		for (size_t i = 0; i < names.size(); ++i)
		{
			const std::string &name = names[i];
			cells[i] = file.readDataset<MatrixXl>("/meshes/" + name + "/c").cast<int>();
			vertices[i] = file.readDataset<Eigen::MatrixXd>("/meshes/" + name + "/v");
		}
	}

	json tmp = json::object();
	if (has_arg(command_line, "log_level"))
		tmp["/output/log/level"_json_pointer] = int(log_level);
	if (has_arg(command_line, "max_threads"))
		tmp["/solver/max_threads"_json_pointer] = max_threads;
	if (has_arg(command_line, "output_dir"))
		tmp["/output/directory"_json_pointer] = std::filesystem::absolute(output_dir);
	if (has_arg(command_line, "enable_overwrite_solver"))
		tmp["/solver/linear/enable_overwrite_solver"_json_pointer] = fallback_solver;
	assert(tmp.is_object());
	in_args.merge_patch(tmp);

	State state;
	state.init(in_args, is_strict);
	state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

	// Mesh was not loaded successfully; load_mesh() logged the error.
	if (state.mesh == nullptr)
	{
		// Cannot proceed without a mesh.
		return EXIT_FAILURE;
	}

	state.stats.compute_mesh_stats(*state.mesh);

	state.build_basis();

	state.assemble_rhs();
	state.assemble_mass_mat();

	Eigen::MatrixXd sol;
	Eigen::MatrixXd pressure;

	state.solve_problem(sol, pressure);

	state.compute_errors(sol);

	logger().info("total time: {}s", state.timings.total_time());

	state.save_json(sol);
	state.export_data(sol, pressure);

	return EXIT_SUCCESS;
}

int optimization_simulation(const CLI::App &command_line,
							const size_t max_threads,
							const bool is_strict,
							const spdlog::level::level_enum &log_level,
							json &opt_args)
{
	// TODO fix gobal stuff threads log level etc

	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, is_strict);

	/* states */
	json state_args = opt_args["states"];
	std::vector<std::shared_ptr<State>> states(state_args.size());
	{
		int i = 0;
		for (const json &args : state_args)
		{
			json cur_args;
			if (!load_json(args["path"], cur_args))
				log_and_throw_error("Can't find json for State {}", i);

			{
				auto tmp = R"({
						"output": {
							"log": {
								"level": -1
							}
						}
					})"_json;

				tmp["output"]["log"]["level"] = int(log_level);

				cur_args.merge_patch(tmp);
			}

			states[i++] = AdjointOptUtils::create_state(cur_args, max_threads);
		}
	}

	/* DOF */
	int ndof = 0;
	std::vector<int> variable_sizes;
	for (const auto &arg : opt_args["parameters"])
	{
		int size = AdjointOptUtils::compute_variable_size(arg, states);
		ndof += size;
		variable_sizes.push_back(size);
	}

	/* variable to simulations */
	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	for (const auto &arg : opt_args["variable_to_simulation"])
		variable_to_simulations.push_back(
			AdjointOptUtils::create_variable_to_simulation(arg, states,
														   variable_sizes));

	/* forms */
	std::shared_ptr<SumCompositeForm> obj =
		std::dynamic_pointer_cast<SumCompositeForm>(AdjointOptUtils::create_form(
			opt_args["functionals"], variable_to_simulations, states));

	/* stopping conditions */
	std::vector<std::shared_ptr<AdjointForm>> stopping_conditions;
	for (const auto &arg : opt_args["stopping_conditions"])
		stopping_conditions.push_back(
			AdjointOptUtils::create_form(arg, variable_to_simulations, states));

	Eigen::VectorXd x;
	x.setZero(ndof);
	int accumulative = 0;
	int var = 0;
	for (const auto &arg : opt_args["parameters"])
	{
		Eigen::VectorXd tmp(variable_sizes[var]);
		if (arg["initial"].is_array() && arg["initial"].size() > 0)
		{
			nlohmann::adl_serializer<Eigen::VectorXd>::from_json(arg["initial"], tmp);
			x.segment(accumulative, tmp.size()) = tmp;
		}
		else if (arg["initial"].is_number())
		{
			tmp.setConstant(arg["initial"].get<double>());
			x.segment(accumulative, tmp.size()) = tmp;
		}
		else
			x += variable_to_simulations[var]->inverse_eval();

		accumulative += tmp.size();
		var++;
	}

	for (auto &v2s : variable_to_simulations)
		v2s->update(x);

	auto nl_problem = std::make_shared<AdjointNLProblem>(
		obj, stopping_conditions, variable_to_simulations, states, opt_args);

	// TODO this should be a json arg
	//  if (only_compute_energy)
	//  {
	//  	nl_problem->solution_changed(x);
	//  	logger().info("Energy is {}", nl_problem->value(x));
	//  	return EXIT_SUCCESS;
	//  }

	std::shared_ptr<cppoptlib::NonlinearSolver<AdjointNLProblem>> nl_solver =
		AdjointOptUtils::make_nl_solver(opt_args["solver"]["nonlinear"], states.front()->units.characteristic_length());
	nl_solver->minimize(*nl_problem, x);

	return EXIT_SUCCESS;
}
