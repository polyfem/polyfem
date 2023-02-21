#include "Optimizations.hpp"

#include <polyfem/utils/CubicHermiteSplineParametrization.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>

#include <polyfem/utils/MaybeParallelFor.hpp>

#include <map>

namespace polyfem::solver
{
	namespace
	{
		bool load_json(const std::string &json_file, json &out)
		{
			std::ifstream file(json_file);

			if (!file.is_open())
				return false;

			file >> out;

			out["root_path"] = json_file;

			return true;
		}

		double cross2(double x, double y)
		{
			x = abs(x);
			y = abs(y);
			if (x > y)
				std::swap(x, y);

			if (x < 0.1)
				return 0.05;
			return 0.95;
		}

		double cross3(double x, double y, double z)
		{
			x = abs(x);
			y = abs(y);
			z = abs(z);
			if (x > y)
				std::swap(x, y);
			if (y > z)
				std::swap(y, z);
			if (x > y)
				std::swap(x, y);

			if (y < 0.2)
				return 0.001;
			return 1;
		}

		double matrix_dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }
	} // namespace

	std::shared_ptr<State> create_state(const json &args, spdlog::level::level_enum log_level, const int max_threads)
	{
		std::shared_ptr<State> state = std::make_shared<State>();
		state->set_max_threads(max_threads);

		json in_args = args;
		in_args["solver"]["max_threads"] = max_threads;
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

		state->init(in_args, false);
		state->args["optimization"]["enabled"] = true;
		state->load_mesh();
		Eigen::MatrixXd sol, pressure;
		state->build_basis();
		state->assemble_rhs();
		state->assemble_stiffness_mat();

		return state;
	}

	void solve_pde(State &state)
	{
		state.assemble_rhs();
		state.assemble_stiffness_mat();
		Eigen::MatrixXd sol, pressure;
		state.solve_problem(sol, pressure);
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

	std::shared_ptr<AdjointNLProblem> make_nl_problem(json &opt_args, spdlog::level::level_enum log_level)
	{
		std::string root_path = "";
		if (utils::is_param_valid(opt_args, "root_path"))
			root_path = opt_args["root_path"].get<std::string>();

		opt_args = apply_opt_json_spec(opt_args, false);

		// create states
		json state_args = opt_args["states"];
		assert(state_args.is_array() && state_args.size() > 0);
		std::vector<std::shared_ptr<State>> states(state_args.size());
		int i = 0;
		for (const json &args : state_args)
		{
			json cur_args;
			if (!load_json(utils::resolve_path(args["path"], root_path, false), cur_args))
				log_and_throw_error("Can't find json for State {}", i);

			states[i++] = create_state(cur_args, log_level);
		}
		json variables = opt_args["variable"];
		json variable_to_objective = opt_args["variable_to_objective"];
		json variable_to_simulation = opt_args["variable_to_objective"];

		// // create parameters
		// json param_args = opt_args["parameters"];
		// assert(param_args.is_array() && param_args.size() > 0);
		// std::vector<std::shared_ptr<Parameter>> parameters(param_args.size());
		// i = 0;
		// int cumulative_dim = 0;
		// for (const json &args : param_args)
		// {
		// 	std::vector<std::shared_ptr<State>> some_states;
		// 	for (int id : args["states"])
		// 	{
		// 		some_states.push_back(states[id]);
		// 	}
		// 	parameters[i] = Parameter::create(args, some_states);
		// 	parameters[i]->set_optimization_variable_position(cumulative_dim);
		// 	cumulative_dim += parameters[i]->optimization_dim();
		// 	i++;
		// }

		// const int cur_log = states[0]->current_log_level;
		// states[0]->set_log_level(static_cast<spdlog::level::level_enum>(opt_args["output"]["solve_log_level"])); // log level is global, only need to change in one state
		// utils::maybe_parallel_for(states.size(), [&](int start, int end, int thread_id) {
		// 	for (int i = start; i < end; i++)
		// 	{
		// 		auto state = states[i];
		// 		solve_pde(*state);
		// 	}
		// });
		// states[0]->set_log_level(static_cast<spdlog::level::level_enum>(cur_log));

		// create objectives
		// json obj_args = opt_args["functionals"];
		// assert(obj_args.is_array() && obj_args.size() > 0);
		// std::vector<std::shared_ptr<Objective>> objs(obj_args.size());
		// Eigen::VectorXd weights;
		// weights.setOnes(objs.size());
		// i = 0;
		// for (const json &args : obj_args)
		// {
		// 	weights[i] = args["weight"];
		// 	objs[i++] = Objective::create(args, root_path, parameters, states);
		// }
		auto vec = std::vector<std::shared_ptr<AdjointForm>>();
		std::vector<std::shared_ptr<VariableToSimulation>> v2sim;
		std::shared_ptr<SumCompositeForm> sum = std::make_shared<SumCompositeForm>(v2sim, vec);

		std::shared_ptr<AdjointNLProblem> nl_problem; // = std::make_shared<AdjointNLProblem>(sum, states, opt_args);

		return nl_problem;
	}
} // namespace polyfem::solver