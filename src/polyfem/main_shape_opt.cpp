#include <filesystem>

#include <CLI/CLI.hpp>
#include <polyfem/solver/Optimizations.hpp>

#include <polyfem/State.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <algorithm>

using namespace polyfem;
using namespace Eigen;
using namespace std::filesystem;

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

	// Input
	std::string json_file = "";
	std::string output_dir = "";

	std::string log_file = "";
	spdlog::level::level_enum log_level = spdlog::level::debug;
	size_t max_threads = std::min((size_t)32, std::numeric_limits<size_t>::max());

	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);

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
	else
	{
		logger().error("Empty json!");
		return EXIT_FAILURE;
	}

	// create solver
	State state(max_threads);
	state.init_logger(log_file, log_level, false);
	state.init(in_args, false, output_dir);

	if (state.is_contact_enabled() && !state.args["solver"]["contact"]["barrier_stiffness"].is_number())
	{
		logger().error("Not fixing the barrier stiffness!");
		return EXIT_FAILURE;
	}

	// load mesh
	state.load_mesh();

	if (state.mesh == nullptr)
		return EXIT_FAILURE;

	std::shared_ptr<CompositeFunctional> func;
	for (const auto &param : state.args["optimization"]["functionals"])
	{
		if (param["type"] == "stress")
		{
			func = CompositeFunctional::create("Stress");
			func->set_power(param["power"]);
			break;
		}
	}

	state.stats.compute_mesh_stats(*state.mesh);
	state.build_basis();

	single_optimization(state, func);

	return EXIT_SUCCESS;
}
