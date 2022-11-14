#include <filesystem>

#include <CLI/CLI.hpp>

#include <polyfem/solver/AdjointNLProblem.hpp>

#include <polyfem/solver/BFGSSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/LBFGSBSolver.hpp>
#include <polyfem/solver/MMASolver.hpp>
#include <polyfem/solver/GradientDescentSolver.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <jse/jse.h>

#include <time.h>

using namespace polyfem;
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

    out["root_path"] = json_file;

	return true;
}

json apply_opt_json_spec(const json &input_args, bool strict_validation)
{
    json args_in = input_args;

    // CHECK validity json
    json rules;
    jse::JSE jse;
    {
        jse.strict = strict_validation;
        const std::string polyfem_input_spec = POLYFEM_OPT_INPUT_SPEC;
        std::ifstream file(polyfem_input_spec);

        if (file.is_open())
            file >> rules;
        else
        {
            logger().error("unable to open {} rules", polyfem_input_spec);
            throw std::runtime_error("Invald spec file");
        }
    }

    const bool valid_input = jse.verify_json(args_in, rules);
   
    if (!valid_input)
    {
        logger().error("invalid input json:\n{}", jse.log2str());
        throw std::runtime_error("Invald input json file");
    }

    json args = jse.inject_defaults(args_in, rules);
    return args;
}

template <typename ProblemType>
std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver(const json &solver_params)
{
    const std::string name = solver_params["solver"].template get<std::string>();
    if (name == "GradientDescent" || name == "gradientdescent" || name == "gradient")
    {
        return std::make_shared<cppoptlib::GradientDescentSolver<ProblemType>>(
            solver_params);
    }
    else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
    {
        return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(
            solver_params);
    }
    else if (name == "bfgs" || name == "BFGS" || name == "BFGS")
    {
        return std::make_shared<cppoptlib::BFGSSolver<ProblemType>>(
            solver_params);
    }
    else if (name == "lbfgsb" || name == "LBFGSB" || name == "L-BFGS-B")
    {
        return std::make_shared<cppoptlib::LBFGSBSolver<ProblemType>>(
            solver_params);
    }
    else if (name == "mma" || name == "MMA")
    {
        return std::make_shared<cppoptlib::MMASolver<ProblemType>>(
            solver_params);
    }
    else
    {
        throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
    }
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

    if (max_threads > 32)
    {
        logger().warn("Using {} threads may slow down the optimization!", max_threads);
    }

    json opt_args;
    if (!load_json(json_file, opt_args))
        log_and_throw_error("Failed to load optimization json file!");
    
    opt_args = apply_opt_json_spec(opt_args, false);

    // create states
    json state_args = opt_args["states"];
    assert(state_args.is_array() && state_args.size() > 0);
    std::vector<std::shared_ptr<State>> states(state_args.size());
    std::map<int, int> id_to_state;
    int i = 0;
    for (const json &args : state_args)
    {
        json cur_args;
        if (!load_json(args["path"], cur_args))
            log_and_throw_error("Can't find json for State {}", args["id"]);

        auto& state = states[i];
        state = std::make_shared<State>(max_threads);
        state->init_logger(log_file, log_level, false);
        state->init(cur_args, false);
        state->args["optimization"]["enabled"] = true;
        state->load_mesh(/*non_conforming=*/false);
        state->build_basis();
        state->assemble_rhs();
        state->assemble_stiffness_mat();
        // Eigen::MatrixXd sol, pressure;
        // state->solve_problem(sol, pressure);
        id_to_state[args["id"].get<int>()] = i++;
    }

    // create parameters
    json param_args = opt_args["parameters"];
    assert(param_args.is_array() && param_args.size() > 0);
    std::vector<std::shared_ptr<Parameter>> parameters(param_args.size());
    i = 0;
    for (const json &args : param_args)
    {
        std::vector<std::shared_ptr<State>> some_states;
        for (int id : args["states"])
        {
            some_states.push_back(states[id_to_state[id]]);
        }
        parameters[i++] = Parameter::create(args, some_states);
    }

    // create objectives
    json obj_args = opt_args["functionals"];
    assert(obj_args.is_array() && obj_args.size() > 0);
    std::vector<std::shared_ptr<solver::Objective>> objs(obj_args.size());
    Eigen::VectorXd weights;
    weights.setOnes(objs.size());
    i = 0;
    for (const json &args : obj_args)
    {
        weights[i] = args["weight"];
        objs[i++] = solver::Objective::create(args, parameters, states);
    }
    std::shared_ptr<solver::SumObjective> sum_obj = std::make_shared<solver::SumObjective>(objs, weights);

    solver::AdjointNLProblem nl_problem(sum_obj, parameters, states, opt_args);
    std::shared_ptr<cppoptlib::NonlinearSolver<solver::AdjointNLProblem>> nlsolver = make_nl_solver<solver::AdjointNLProblem>(opt_args["solver"]["nonlinear"]);

    Eigen::VectorXd x(nl_problem.full_size());
    int cumulative = 0;
    for (const auto &p : parameters)
    {
        x.segment(cumulative, p->optimization_dim()) = p->initial_guess();
        cumulative += p->optimization_dim();
    }

    nl_problem.solve_pde();
    nlsolver->minimize(nl_problem, x);

	return EXIT_SUCCESS;
}
