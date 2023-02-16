#include <filesystem>

#include <CLI/CLI.hpp>

#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <time.h>

using namespace polyfem;
using namespace solver;
using namespace polysolve;

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
	{
		out["root_path"] = json_file;
	}

	return true;
}

int main(int argc, char **argv)
{
	using namespace std::filesystem;

	CLI::App command_line{"polyfem"};

	// Eigen::setNbThreads(1);
	size_t max_threads = std::numeric_limits<size_t>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	std::string json_file = "";
	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);

	std::string log_file = "";
	command_line.add_option("--log_file", log_file, "Log to a file");

	bool only_compute_energy = false;
	command_line.add_flag("--only_compute_energy", only_compute_energy, "Compute energy and exit");

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

	json opt_args;
	if (!load_json(json_file, opt_args))
		log_and_throw_error("Failed to load optimization json file!");

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

		opt_args.merge_patch(tmp);
	}

	if (has_arg(command_line, "max_threads"))
	{
		auto tmp = R"({
				"solver": {
					"max_threads": -1
				}
			})"_json;

		tmp["solver"]["max_threads"] = max_threads;

		opt_args.merge_patch(tmp);
	}

	auto nl_problem = make_nl_problem(opt_args, log_level);

	std::shared_ptr<cppoptlib::NonlinearSolver<AdjointNLProblem>> nlsolver = make_nl_solver<AdjointNLProblem>(opt_args["solver"]["nonlinear"]);

	Eigen::VectorXd x = nl_problem->initial_guess();

	if (only_compute_energy)
	{
		nl_problem->solution_changed(x);
		logger().info("Energy is {}", nl_problem->value(x));
		auto state = nl_problem->get_state(0);
		if (!state->problem->is_time_dependent())
		{
			state->save_json(state->diff_cached[0].u);
			state->export_data(state->diff_cached[0].u, Eigen::MatrixXd());
		}
		return EXIT_SUCCESS;
	}

	nlsolver->minimize(*nl_problem, x);

	return EXIT_SUCCESS;
}
