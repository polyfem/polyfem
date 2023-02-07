#include "Optimizations.hpp"

#include "InitialConditionProblem.hpp"
#include "GeneralOptimizationProblem.hpp"

#include <polyfem/utils/CubicHermiteSplineParametrization.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

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

	std::shared_ptr<OptimizationProblem> setup_initial_condition_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];

		std::shared_ptr<InitialConditionProblem> initial_problem = std::make_shared<InitialConditionProblem>(state, j);

		json initial_params;
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "initial")
			{
				initial_params = param;
				break;
			}
		}

		// fix certain object
		std::set<int> optimize_body_ids;
		if (initial_params["volume_selection"].size() > 0)
		{
			for (int i : initial_params["volume_selection"])
				optimize_body_ids.insert(i);
		}
		else
			logger().info("No optimization body specified, optimize initial condition of every mesh...");

		// to get the initial velocity
		{
			state.solve_data.rhs_assembler = state.build_rhs_assembler();
			state.solve_data.rhs_assembler->initial_solution(state.initial_sol_update);
			state.solve_data.rhs_assembler->initial_velocity(state.initial_vel_update);
		}
		auto initial_guess_vel = state.initial_vel_update, initial_guess_sol = state.initial_sol_update;

		const int dim = state.mesh->dimension();
		const int dof = state.n_bases;

		std::map<int, std::array<int, 2>> body_id_map; // from body_id to {node_id, index}
		int n = 0;
		for (int e = 0; e < state.bases.size(); e++)
		{
			const int body_id = state.mesh->get_body_id(e);
			if (!body_id_map.count(body_id) && (optimize_body_ids.count(body_id) || optimize_body_ids.size() == 0))
			{
				body_id_map[body_id] = {{state.bases[e].bases[0].global()[0].index, n}};
				n++;
			}
		}
		logger().info("{} objects found, each object has a constant initial velocity and position...", body_id_map.size());

		if (initial_params["restriction"].get<std::string>() == "velocity")
		{
			initial_problem->x_to_param = [initial_guess_sol, initial_guess_vel, body_id_map, dim](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol = initial_guess_sol;
				init_vel = initial_guess_vel;
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
								init_vel(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
				}
				std::cout << "initial velocity: " << std::setprecision(16) << x.transpose() << "\n";
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				for (auto i : body_id_map)
					for (int d = 0; d < dim; d++)
						x(i.second[1] * dim + d) = init_vel(i.second[0] * dim + d);
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
								x(dim * body_id_map.at(body_id)[1] + d) += init_vel(g.index * dim + d);
						}
				}
			};
		}
		else if (initial_params["restriction"].get<std::string>() == "velocityXZ")
		{
			initial_problem->x_to_param = [initial_guess_sol, initial_guess_vel, body_id_map, dim](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol = initial_guess_sol;
				init_vel = initial_guess_vel;
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
								init_vel(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
				}
				logger().debug("initial velocity: {}", x.transpose());
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				for (auto i : body_id_map)
					for (int d = 0; d < dim; d++)
						x(i.second[1] * dim + d) = init_vel(i.second[0] * dim + d);
				logger().debug("initial velocity: {}", x.transpose());
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
							{
								if (d == 1)
									continue;
								x(dim * body_id_map.at(body_id)[1] + d) += init_vel(g.index * dim + d);
							}
						}
				}
			};
		}
		else if (initial_params["restriction"].get<std::string>() == "position")
		{
			initial_problem->x_to_param = [body_id_map, dim, dof](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol.setZero(dof * dim, 1);
				init_vel.setZero(dof * dim, 1);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
								init_sol(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
				}
				logger().debug("initial solution: {}", x.transpose());
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				for (auto i : body_id_map)
				{
					for (int d = 0; d < dim; d++)
						x(i.second[1] * dim + d) = init_sol(i.second[0] * dim + d);
				}
				logger().debug("initial solution: {}", x.transpose());
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size());
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					if (!body_id_map.count(body_id))
						continue;
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
								x(dim * body_id_map.at(body_id)[1] + d) += init_sol(g.index * dim + d);
						}
				}
			};
		}
		else
		{
			initial_problem->x_to_param = [initial_guess_sol, initial_guess_vel, body_id_map, dim](const InitialConditionProblem::TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state) {
				init_sol = initial_guess_sol;
				init_vel = initial_guess_vel;
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
							for (int d = 0; d < dim; d++)
							{
								init_sol(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d);
								init_vel(g.index * dim + d) = x(body_id_map.at(body_id)[1] * dim + d + dim * body_id_map.size());
							}
				}
				logger().debug("initial velocity: {}", x.tail(x.size() / 2).transpose());
				logger().debug("initial position: {}", x.head(x.size() / 2).transpose());
			};

			initial_problem->param_to_x = [body_id_map, dim](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size() * 2);
				for (auto i : body_id_map)
					for (int d = 0; d < dim; d++)
					{
						x(i.second[1] * dim + d) = init_sol(i.second[0] * dim + d);
						x(i.second[1] * dim + d + dim * body_id_map.size()) = init_vel(i.second[0] * dim + d);
					}
				logger().debug("initial velocity: {}", x.tail(x.size() / 2).transpose());
				logger().debug("initial position: {}", x.head(x.size() / 2).transpose());
			};

			initial_problem->dparam_to_dx = [body_id_map, dim, dof](InitialConditionProblem::TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state) {
				x.setZero(dim * body_id_map.size() * 2);
				std::vector<bool> visited(dof, false);
				for (int e = 0; e < state.bases.size(); e++)
				{
					const int body_id = state.mesh->get_body_id(e);
					for (auto &bs : state.bases[e].bases)
						for (auto &g : bs.global())
						{
							if (!visited[g.index])
								visited[g.index] = true;
							else
								continue;
							for (int d = 0; d < dim; d++)
							{
								x(dim * body_id_map.at(body_id)[1] + d) += init_sol(g.index * dim + d);
								x(dim * body_id_map.at(body_id)[1] + d + dim * body_id_map.size()) += init_vel(g.index * dim + d);
							}
						}
				}
			};
		}
		initial_problem->param_to_x(x_initial, state.initial_sol_update, state.initial_vel_update, state);
		initial_problem->set_optimization_dim(x_initial.size());

		return initial_problem;
	}

	std::shared_ptr<OptimizationProblem> setup_optimization(const std::string &type, State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		std::map<std::string, std::function<std::shared_ptr<OptimizationProblem>(State &, const std::shared_ptr<CompositeFunctional>, Eigen::VectorXd &)>> setup_functions{{"initial", setup_initial_condition_optimization}};

		return setup_functions[type](state, j, x_initial);
	}

	std::shared_ptr<GeneralOptimizationProblem> setup_general_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, Eigen::VectorXd &x_initial)
	{
		const auto &opt_params = state.args["optimization"];

		std::vector<std::shared_ptr<OptimizationProblem>> problems;
		std::vector<Eigen::VectorXd> x_initial_list;
		int x_initial_size = 0;
		for (const auto &param : opt_params["parameters"])
		{
			Eigen::VectorXd tmp;
			problems.push_back(setup_optimization(param["type"], state, j, tmp));
			x_initial_size += tmp.size();
			x_initial_list.push_back(tmp);
		}

		x_initial.resize(x_initial_size);
		int count = 0;
		for (const auto &x : x_initial_list)
		{
			x_initial.segment(count, x.size()) = x;
			count += x.size();
			logger().trace("Size of initial guess is {}", x.size());
		}

		std::shared_ptr<GeneralOptimizationProblem>
			general_optimization_problem = std::make_shared<GeneralOptimizationProblem>(problems, j);

		return general_optimization_problem;
	}

	void general_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		std::shared_ptr<cppoptlib::NonlinearSolver<GeneralOptimizationProblem>> nlsolver = make_nl_solver<GeneralOptimizationProblem>(state.args["optimization"]["solver"]["nonlinear"]);

		Eigen::VectorXd x;
		auto general_optimization_problem = setup_general_optimization(state, j, x);
		nlsolver->minimize(*general_optimization_problem, x);

		json solver_info;
		nlsolver->get_info(solver_info);
		std::cout << solver_info << std::endl;
	}

	std::shared_ptr<State> create_state(const json &args, spdlog::level::level_enum log_level, const int max_threads)
	{
		std::shared_ptr<State> state = std::make_shared<State>();
		state->set_max_threads(max_threads);

		json in_args = args;
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

		// create parameters
		json param_args = opt_args["parameters"];
		assert(param_args.is_array() && param_args.size() > 0);
		std::vector<std::shared_ptr<Parameter>> parameters(param_args.size());
		i = 0;
		int cumulative_dim = 0;
		for (const json &args : param_args)
		{
			std::vector<std::shared_ptr<State>> some_states;
			for (int id : args["states"])
			{
				some_states.push_back(states[id]);
			}
			parameters[i] = Parameter::create(args, some_states);
			parameters[i]->set_optimization_variable_position(cumulative_dim);
			cumulative_dim += parameters[i]->optimization_dim();
			i++;
		}

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
		json obj_args = opt_args["functionals"];
		assert(obj_args.is_array() && obj_args.size() > 0);
		std::vector<std::shared_ptr<Objective>> objs(obj_args.size());
		Eigen::VectorXd weights;
		weights.setOnes(objs.size());
		i = 0;
		for (const json &args : obj_args)
		{
			weights[i] = args["weight"];
			objs[i++] = Objective::create(args, root_path, parameters, states);
		}
		std::shared_ptr<SumObjective> sum_obj = std::make_shared<SumObjective>(objs, weights);

		std::shared_ptr<AdjointNLProblem> nl_problem = std::make_shared<AdjointNLProblem>(sum_obj, parameters, states, opt_args);

		return nl_problem;
	}
} // namespace polyfem::solver