#include <filesystem>

#include <CLI/CLI.hpp>

#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/SpatialIntegralForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/TransientForm.hpp>

#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

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
	command_line.add_option("-j,--json", json_file, "Optimization json file")->check(CLI::ExistingFile);

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
	
	opt_args = apply_opt_json_spec(opt_args, false);

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

			states[i++] = create_state(cur_args, log_level, max_threads);
		}
	}

	/* variable to simulations */
	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;
	{
		Eigen::Matrix2d Affine;
		Affine << 1.2, 0, 0, 1.5;

		json iso_options;
		iso_options["maxArea"] = 1e-3;
		iso_options["dump_shape_velocity"] = "tmp-vel.msh";
		// iso_options["curveSimplifier"] = "NONE";
		// iso_options["forceMaxBdryEdgeLen"] = 0.001;
		iso_options["marchingSquaresGridSize"] = 1024;
		iso_options["forceMSGridSize"] = true;

		std::vector<std::shared_ptr<Parametrization>> map_list = {
			std::make_shared<AppendConstantMap>(8, 0.01),
			std::make_shared<SDF2Mesh>(std::string("bistable.obj"), std::string("tmp-unit.msh"), iso_options),
			std::make_shared<MeshTiling>(Eigen::Vector2i(2, 2), "tmp-unit.msh", "tmp-tiled.msh"),
			std::make_shared<MeshAffine>(Affine, Eigen::Vector2d(1.0, 1.0), "tmp-tiled.msh", "tmp-scaled.msh")};
		CompositeParametrization composite_map(map_list);

		json options;
		options["mesh"] = "tmp-scaled.msh";
		options["mesh_id"] = 0;

		variable_to_simulations.push_back(std::make_shared<SDFShapeVariableToSimulation>(states[0], composite_map, options));
	}

	/* forms */
	auto obj = std::make_shared<StressNormForm>(variable_to_simulations, *(states[0]), opt_args["functionals"][0]);

	Eigen::VectorXd x(20);
	x << 0.3, 0.10, 0.333333, 0.40, 0.666667, 0.50, 0.50, 0.75, 0.60, 0.666667, 0.90, 0.333333, 0.30, 0.20, 0.05, 0.05, 0.30, 0.20, 0.05, 0.05;// 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;

	for (auto &v2s : variable_to_simulations)
		v2s->update(x);
	solve_pde(*(states[0]));

	std::vector<std::shared_ptr<AdjointForm>> forms({obj});
	auto sum = std::make_shared<SumCompositeForm>(variable_to_simulations, forms);

	auto nl_problem = std::make_shared<AdjointNLProblem>(sum, variable_to_simulations, states, opt_args);
	std::shared_ptr<cppoptlib::NonlinearSolver<AdjointNLProblem>> nl_solver = make_nl_solver<AdjointNLProblem>(opt_args["solver"]["nonlinear"]);

	if (only_compute_energy)
	{
		nl_problem->solution_changed(x);
		logger().info("Energy is {}", nl_problem->value(x));
		// auto state = nl_problem->get_state(0);
		// state->save_json(state->diff_cached[0].u);
		// state->export_data(state->diff_cached[0].u, Eigen::MatrixXd());
		return EXIT_SUCCESS;
	}

	nl_solver->minimize(*nl_problem, x);

	return EXIT_SUCCESS;
}
