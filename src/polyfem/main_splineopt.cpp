#include <filesystem>

#include <CLI/CLI.hpp>

#include <polyfem/solver/Optimizations.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/State.hpp>

#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/solver/forms/parametrization/NodeCompositeParametrizations.hpp>
#include <polyfem/solver/forms/parametrization/SplineParametrizations.hpp>

#include <polyfem/solver/forms/adjoint_forms/SpatialIntegralForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/AMIPSForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/TransientForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/BarrierForms.hpp>

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

std::string resolve_output_path(const std::string &output_dir, const std::string &path)
{
	if (std::filesystem::path(path).is_absolute())
		return path;
	else
		return std::filesystem::weakly_canonical(std::filesystem::path(output_dir) / path).string();
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

	std::string root_path = "";
	if (utils::is_param_valid(opt_args, "root_path"))
		root_path = opt_args["root_path"].get<std::string>();

	for (auto &state_arg : opt_args["states"])
		state_arg["path"] = resolve_output_path(root_path, state_arg["path"]);

	json state_args = opt_args["states"];
	std::vector<std::shared_ptr<State>> states(state_args.size());
	int i = 0;
	for (const json &args : state_args)
	{
		json cur_args;
		if (!load_json(utils::resolve_path(args["path"], root_path, false), cur_args))
			log_and_throw_error("Can't find json for State {}", i);

		states[i++] = create_state(cur_args);
	}

	states[0]->set_log_level(log_level);

	std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations;

	Eigen::VectorXd x;
	{
		std::vector<int> opt_body_ids = {1, 3, 4, 5, 7, 8, 9, 10};

		auto state = states[0];
		const auto &mesh = state->mesh;
		const auto &bases = state->bases;
		const auto &gbases = state->geom_bases();
		int dim = mesh->dimension();

		for (const auto &volume_selection : opt_body_ids)
		{

			std::set<int> node_ids;
			for (int e = 0; e < mesh->n_elements(); e++)
			{
				const int body_id = mesh->get_body_id(e);
				if (volume_selection == body_id)
				{
					for (int i = 0; i < mesh->dimension() + 1; i++)
					{
						const int vid = mesh->element_vertex(e, i);
						if (!mesh->is_boundary_vertex(vid))
							node_ids.insert(vid);
					}
				}
			}
			int opt_inodes = node_ids.size();
			int free_vars = opt_inodes * dim;

			std::vector<std::shared_ptr<Parametrization>> param_map_list = {};
			param_map_list.push_back(std::make_shared<SliceMap>(x.size(), x.size() + free_vars, -1));
			variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(states[0], VariableToInteriorNodes(param_map_list, *states[0], volume_selection)));

			int start_idx = x.size();
			x.conservativeResize(x.size() + free_vars);

			Eigen::MatrixXd V;
			state->get_vertices(V);
			Eigen::VectorXd V_flat = utils::flatten(V);
			auto b_idx = variable_to_simulations.back()->get_parametrization().get_output_indexing(x);
			assert(b_idx.size() == (opt_inodes * dim));
			Eigen::VectorXd y(opt_inodes * dim);
			for (int i = 0; i < opt_inodes; ++i)
				for (int k = 0; k < dim; ++k)
					y(i * dim + k) = V_flat(b_idx(i * dim + k));

			x.segment(start_idx, free_vars) = variable_to_simulations.back()->get_parametrization().inverse_eval(y);
		}
	}

	for (const auto &param : opt_args["parameters"])
	{
		if (param["type"] == "shape")
		{
			if (param["restriction"] == "bspline")
			{
				int dim = states[param["states"][0]]->mesh->dimension();
				Eigen::MatrixXd control_points(param["spline_specification"][0]["control_point"].size(), dim);
				for (int i = 0; i < control_points.rows(); ++i)
					for (int j = 0; j < control_points.cols(); ++j)
						control_points(i, j) = param["spline_specification"][0]["control_point"][i][j].get<double>();
				int free_vars = (control_points.rows() - 2) * control_points.cols();
				Eigen::VectorXd knots(param["spline_specification"][0]["knot"].size());
				for (int i = 0; i < knots.size(); ++i)
					knots(i) = param["spline_specification"][0]["knot"][i].get<double>();

				int opt_bnodes = 0;
				int surface_selection = param["surface_selection"][0];
				{
					auto state = states[param["states"][0]];
					const auto &mesh = state->mesh;
					const auto &bases = state->bases;
					const auto &gbases = state->geom_bases();

					std::set<int> node_ids;
					for (const auto &lb : state->total_local_boundary)
					{
						const int e = lb.element_id();
						for (int i = 0; i < lb.size(); ++i)
						{
							const int primitive_global_id = lb.global_primitive_id(i);
							const int boundary_id = mesh->get_boundary_id(primitive_global_id);
							const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

							if (boundary_id == surface_selection)
								for (long n = 0; n < nodes.size(); ++n)
									node_ids.insert(gbases[e].bases[nodes(n)].global()[0].index);
						}
					}
					opt_bnodes = node_ids.size();
				}

				std::vector<std::shared_ptr<Parametrization>> param_map_list = {};
				param_map_list.push_back(std::make_shared<SliceMap>(x.size(), x.size() + free_vars, -1));
				param_map_list.push_back(std::make_shared<BSplineParametrization1DTo2D>(control_points, knots, opt_bnodes, true));

				variable_to_simulations.push_back(std::make_shared<ShapeVariableToSimulation>(states[param["states"][0]], VariableToBoundaryNodes(param_map_list, *states[param["states"][0]], surface_selection)));

				int start_idx = x.size();
				x.conservativeResize(x.size() + free_vars);

				Eigen::MatrixXd V;
				states[param["states"][0]]->get_vertices(V);
				Eigen::VectorXd V_flat = utils::flatten(V);
				auto b_idx = variable_to_simulations.back()->get_parametrization().get_output_indexing(x);
				assert(b_idx.size() == (opt_bnodes * dim));
				Eigen::VectorXd y(opt_bnodes * dim);
				for (int i = 0; i < opt_bnodes; ++i)
					for (int k = 0; k < dim; ++k)
						y(i * dim + k) = V_flat(b_idx(i * dim + k));

				x.segment(start_idx, free_vars) = variable_to_simulations.back()->get_parametrization().inverse_eval(y);
			}
		}
	}

	int dim = states[0]->mesh->dimension();
	Eigen::MatrixXd control_points(opt_args["functionals"][0]["control_points"].size(), dim);
	for (int i = 0; i < control_points.rows(); ++i)
		for (int j = 0; j < control_points.cols(); ++j)
			control_points(i, j) = opt_args["functionals"][0]["control_points"][i][j].get<double>();
	Eigen::VectorXd knots(opt_args["functionals"][0]["knots"].size());
	for (int i = 0; i < knots.size(); ++i)
		knots(i) = opt_args["functionals"][0]["knots"][i].get<double>();

	auto target = std::make_shared<SDFTargetForm>(variable_to_simulations, *states[0], opt_args["functionals"][0]);
	target->set_bspline_target(control_points, knots, 0.001);
	auto obj1 = std::make_shared<TransientForm>(variable_to_simulations, states[0]->args["time"]["time_steps"], states[0]->args["time"]["dt"], "final", std::vector<int>(), target);
	obj1->set_weight(1.0);

	auto obj2 = std::make_shared<AMIPSForm>(variable_to_simulations, *states[0]);
	obj2->set_weight(0.01);

	auto obj3 = std::make_shared<CollisionBarrierForm>(variable_to_simulations, *states[0], 1e-3);
	obj3->set_weight(1.0);

	std::vector<std::shared_ptr<AdjointForm>> forms({obj1, obj2, obj3});

	auto sum = std::make_shared<SumCompositeForm>(variable_to_simulations, forms);
	sum->set_weight(1.0);

	std::shared_ptr<solver::AdjointNLProblem> nl_problem = std::make_shared<solver::AdjointNLProblem>(sum, variable_to_simulations, states, opt_args);

	nl_problem->solution_changed(x);

	auto nl_solver = make_nl_solver<AdjointNLProblem>(opt_args["solver"]["nonlinear"]);
	nl_solver->minimize(*nl_problem, x);

	return EXIT_SUCCESS;
}