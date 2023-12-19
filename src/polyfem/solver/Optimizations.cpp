#include "Optimizations.hpp"

#include <polyfem/mesh/GeometryReader.hpp>
#include <jse/jse.h>
#include <polyfem/State.hpp>

#include "AdjointNLProblem.hpp"

#include <polyfem/solver/forms/adjoint_forms/SpatialIntegralForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/SumCompositeForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/CompositeForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/TransientForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/SmoothingForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/AMIPSForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/BarrierForms.hpp>
#include <polyfem/solver/forms/adjoint_forms/TargetForms.hpp>

#include <polyfem/solver/forms/parametrization/Parametrizations.hpp>
#include <polyfem/solver/forms/parametrization/NodeCompositeParametrizations.hpp>

#include <polyfem/solver/forms/adjoint_forms/ParametrizedProductForm.hpp>

#include <polyfem/io/OBJReader.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polysolve/nonlinear/BoxConstraintSolver.hpp>

namespace spdlog::level
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		spdlog::level::level_enum,
		{{spdlog::level::level_enum::trace, "trace"},
		 {spdlog::level::level_enum::debug, "debug"},
		 {spdlog::level::level_enum::info, "info"},
		 {spdlog::level::level_enum::warn, "warning"},
		 {spdlog::level::level_enum::err, "error"},
		 {spdlog::level::level_enum::critical, "critical"},
		 {spdlog::level::level_enum::off, "off"},
		 {spdlog::level::level_enum::trace, 0},
		 {spdlog::level::level_enum::debug, 1},
		 {spdlog::level::level_enum::info, 2},
		 {spdlog::level::level_enum::warn, 3},
		 {spdlog::level::level_enum::err, 3},
		 {spdlog::level::level_enum::critical, 4},
		 {spdlog::level::level_enum::off, 5}})
}

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
	} // namespace

	std::shared_ptr<polysolve::nonlinear::Solver> AdjointOptUtils::make_nl_solver(const json &solver_params, const json &linear_solver_params, const double characteristic_length)
	{
		auto names = polysolve::nonlinear::Solver::available_solvers();
		if (std::find(names.begin(), names.end(), solver_params["solver"]) != names.end())
			return polysolve::nonlinear::Solver::create(solver_params, linear_solver_params, characteristic_length, adjoint_logger());

		names = polysolve::nonlinear::BoxConstraintSolver::available_solvers();
		if (std::find(names.begin(), names.end(), solver_params["solver"]) != names.end())
			return polysolve::nonlinear::BoxConstraintSolver::create(solver_params, linear_solver_params, characteristic_length, adjoint_logger());

		log_and_throw_adjoint_error("Invalid nonlinear solver name!");
	}

	std::shared_ptr<AdjointForm> AdjointOptUtils::create_form(const json &args, const std::vector<std::shared_ptr<VariableToSimulation>> &var2sim, const std::vector<std::shared_ptr<State>> &states)
	{
		std::shared_ptr<AdjointForm> obj;
		if (args.is_array())
		{
			std::vector<std::shared_ptr<AdjointForm>> forms;
			for (const auto &arg : args)
				forms.push_back(create_form(arg, var2sim, states));

			obj = std::make_shared<SumCompositeForm>(var2sim, forms);
		}
		else
		{
			const std::string type = args["type"];
			if (type == "transient_integral")
			{
				std::shared_ptr<StaticForm> static_obj = std::dynamic_pointer_cast<StaticForm>(create_form(args["static_objective"], var2sim, states));
				const auto &state = states[args["state"]];
				obj = std::make_shared<TransientForm>(var2sim, state->args["time"]["time_steps"], state->args["time"]["dt"], args["integral_type"], args["steps"].get<std::vector<int>>(), static_obj);
			}
			else if (type == "power")
			{
				std::shared_ptr<AdjointForm> obj_aux = create_form(args["objective"], var2sim, states);
				obj = std::make_shared<PowerForm>(obj_aux, args["power"]);
			}
			else if (type == "divide")
			{
				std::shared_ptr<AdjointForm> obj1 = create_form(args["objective"][0], var2sim, states);
				std::shared_ptr<AdjointForm> obj2 = create_form(args["objective"][1], var2sim, states);
				std::vector<std::shared_ptr<AdjointForm>> objs({obj1, obj2});
				obj = std::make_shared<DivideForm>(objs);
			}
			else if (type == "plus-const")
			{
				obj = std::make_shared<PlusConstCompositeForm>(create_form(args["objective"], var2sim, states), args["value"]);
			}
			else if (type == "compliance")
			{
				obj = std::make_shared<ComplianceForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "acceleration")
			{
				obj = std::make_shared<AccelerationForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "kinetic")
			{
				obj = std::make_shared<AccelerationForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "target")
			{
				std::shared_ptr<TargetForm> tmp = std::make_shared<TargetForm>(var2sim, *(states[args["state"]]), args);
				auto reference_cached = args["reference_cached_body_ids"].get<std::vector<int>>();
				tmp->set_reference(states[args["target_state"]], std::set(reference_cached.begin(), reference_cached.end()));
				obj = tmp;
			}
			else if (type == "center-target")
			{
				obj = std::make_shared<BarycenterTargetForm>(var2sim, args, states[args["state"]], states[args["target_state"]]);
			}
			else if (type == "function-target")
			{
				std::shared_ptr<TargetForm> tmp = std::make_shared<TargetForm>(var2sim, *(states[args["state"]]), args);
				tmp->set_reference(args["target_function"], args["target_function_gradient"]);
				obj = tmp;
			}
			else if (type == "position")
			{
				obj = std::make_shared<PositionForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "stress")
			{
				obj = std::make_shared<StressForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "stress_norm")
			{
				obj = std::make_shared<StressNormForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "elastic_energy")
			{
				obj = std::make_shared<ElasticEnergyForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "max_stress")
			{
				obj = std::make_shared<MaxStressForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "volume")
			{
				obj = std::make_shared<VolumeForm>(var2sim, *(states[args["state"]]), args);
			}
			else if (type == "soft_constraint")
			{
				std::vector<std::shared_ptr<AdjointForm>> forms({create_form(args["objective"], var2sim, states)});
				Eigen::VectorXd bounds = args["soft_bound"];
				obj = std::make_shared<InequalityConstraintForm>(forms, bounds, args["power"]);
			}
			else if (type == "AMIPS")
			{
				obj = std::make_shared<AMIPSForm>(var2sim, *(states[args["state"]]));
			}
			else if (type == "boundary_smoothing")
			{
				obj = std::make_shared<BoundarySmoothingForm>(var2sim, *(states[args["state"]]), args["scale_invariant"], args["power"]);
			}
			else if (type == "collision_barrier")
			{
				obj = std::make_shared<CollisionBarrierForm>(var2sim, *(states[args["state"]]), args["dhat"]);
			}
			else if (type == "deformed_collision_barrier")
			{
				obj = std::make_shared<DeformedCollisionBarrierForm>(var2sim, *(states[args["state"]]), args["dhat"]);
			}
			else if (type == "parametrized_product")
			{
				if (args["parametrization"].contains("parameter_index"))
					log_and_throw_adjoint_error("Parametrizations in parametrized forms don't support parameter_index!");

				std::vector<std::shared_ptr<Parametrization>> map_list;
				for (const auto &arg : args["parametrization"])
					map_list.push_back(create_parametrization(arg, states, {}));
				CompositeParametrization composite_map = CompositeParametrization(map_list);

				obj = std::make_shared<ParametrizedProductForm>(composite_map);
			}
			else
				log_and_throw_adjoint_error("Objective not implemented!");

			obj->set_weight(args["weight"]);
			if (args["print_energy"].get<std::string>() != "")
				obj->enable_energy_print(args["print_energy"]);
		}

		return obj;
	}

	std::shared_ptr<Parametrization> AdjointOptUtils::create_parametrization(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes)
	{
		std::shared_ptr<Parametrization> map;
		const std::string type = args["type"];
		if (type == "per-body-to-per-elem")
		{
			map = std::make_shared<PerBody2PerElem>(*(states[args["state"]]->mesh));
		}
		else if (type == "per-body-to-per-node")
		{
			map = std::make_shared<PerBody2PerNode>(*(states[args["state"]]->mesh), states[args["state"]]->bases, states[args["state"]]->n_bases);
		}
		else if (type == "E-nu-to-lambda-mu")
		{
			map = std::make_shared<ENu2LambdaMu>(args["is_volume"]);
		}
		else if (type == "slice")
		{
			if (args.contains("from") && args.contains("to"))
			{
				assert(args["from"] != -1);
				assert(args["to"] != -1);
				map = std::make_shared<SliceMap>(args["from"], args["to"], args["last"]);
			}
			else if (args.contains("parameter_index"))
			{
				assert(args["parameter_index"] != -1);
				int idx = args["parameter_index"].get<int>();
				int from, to, last;
				int cumulative = 0;
				for (int i = 0; i < variable_sizes.size(); ++i)
				{
					if (i == idx)
					{
						from = cumulative;
						to = from + variable_sizes[i];
					}
					cumulative += variable_sizes[i];
				}
				last = cumulative;
				map = std::make_shared<SliceMap>(from, to, last);
			}
			else
				log_and_throw_adjoint_error("Incorrect spec for SliceMap!");
		}
		else if (type == "exp")
		{
			map = std::make_shared<ExponentialMap>();
		}
		else if (type == "scale")
		{
			map = std::make_shared<Scaling>(args["value"]);
		}
		else if (type == "power")
		{
			map = std::make_shared<PowerMap>(args["power"]);
		}
		else if (type == "append-values")
		{
			Eigen::VectorXd vals = args["values"];
			map = std::make_shared<InsertConstantMap>(vals);
		}
		else if (type == "append-const")
		{
			map = std::make_shared<InsertConstantMap>(args["size"], args["value"], args["start"]);
		}
		else if (type == "linear-filter")
		{
			map = std::make_shared<LinearFilter>(*(states[args["state"]]->mesh), args["radius"]);
		}
		else
			log_and_throw_adjoint_error("Unkown parametrization!");

		return map;
	}

	std::shared_ptr<VariableToSimulation> AdjointOptUtils::create_variable_to_simulation(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes)
	{
		std::shared_ptr<VariableToSimulation> var2sim;
		const std::string type = args["type"];

		std::vector<std::shared_ptr<Parametrization>> map_list;
		for (const auto &arg : args["composition"])
			map_list.push_back(create_parametrization(arg, states, variable_sizes));
		CompositeParametrization composite_map;

		std::vector<std::shared_ptr<State>> cur_states;
		if (args["state"].is_array())
			for (int i : args["state"])
				cur_states.push_back(states[i]);
		else
			cur_states.push_back(states[args["state"]]);

		composite_map = CompositeParametrization(map_list);

		const std::string composite_map_type = args["composite_map_type"];
		Eigen::VectorXi output_indexing;
		if (composite_map_type == "none")
		{
		}
		else if (composite_map_type == "interior")
		{
			assert(type == "shape");
			VariableToInteriorNodes variable_to_node(*cur_states[0], args["volume_selection"][0]);
			output_indexing = variable_to_node.get_output_indexing();
		}
		else if (composite_map_type == "boundary")
		{
			assert(type == "shape");
			VariableToBoundaryNodes variable_to_node(*cur_states[0], args["surface_selection"][0]);
			output_indexing = variable_to_node.get_output_indexing();
		}
		else if (composite_map_type == "boundary_excluding_surface")
		{
			assert(type == "shape");
			std::vector<int> excluded_surfaces;
			for (const auto &s : args["surface_selection"])
				excluded_surfaces.push_back(s);
			VariableToBoundaryNodesExclusive variable_to_node(*cur_states[0], excluded_surfaces);
			output_indexing = variable_to_node.get_output_indexing();
		}
		else if (composite_map_type == "indices")
		{
			if (args["composite_map_indices"].is_string())
			{
				Eigen::MatrixXi tmp_mat;
				polyfem::io::read_matrix(states[0]->resolve_input_path(args["composite_map_indices"].get<std::string>()), tmp_mat);
				output_indexing = tmp_mat;
			}
			else if (args["composite_map_indices"].is_array())
				output_indexing = args["composite_map_indices"];
			else
				log_and_throw_adjoint_error("Invalid composite map indices type!");
		}

		if (type == "shape")
		{
			var2sim = std::make_shared<ShapeVariableToSimulation>(cur_states, composite_map);
			var2sim->set_output_indexing(output_indexing);
		}
		else if (type == "elastic")
		{
			var2sim = std::make_shared<ElasticVariableToSimulation>(cur_states, composite_map);
		}
		else if (type == "friction")
		{
			var2sim = std::make_shared<FrictionCoeffientVariableToSimulation>(cur_states, composite_map);
		}
		else if (type == "damping")
		{
			var2sim = std::make_shared<DampingCoeffientVariableToSimulation>(cur_states, composite_map);
		}
		else if (type == "initial")
		{
			var2sim = std::make_shared<InitialConditionVariableToSimulation>(cur_states, composite_map);
		}
		else if (type == "dirichlet")
		{
			var2sim = std::make_shared<DirichletVariableToSimulation>(cur_states, composite_map);
		}

		return var2sim;
	}

	Eigen::VectorXd AdjointOptUtils::inverse_evaluation(const json &args, const int ndof, const std::vector<int> &variable_sizes, std::vector<std::shared_ptr<VariableToSimulation>> &var2sim)
	{
		Eigen::VectorXd x;
		x.setZero(ndof);
		int accumulative = 0;
		int var = 0;
		for (const auto &arg : args)
		{
			Eigen::VectorXd tmp(variable_sizes[var]);
			if (arg["initial"].is_array() && arg["initial"].size() > 0)
			{
				tmp = arg["initial"];
				x.segment(accumulative, tmp.size()) = tmp;
			}
			else if (arg["initial"].is_number())
			{
				tmp.setConstant(arg["initial"].get<double>());
				x.segment(accumulative, tmp.size()) = tmp;
			}
			else
				x += var2sim[var]->inverse_eval();

			accumulative += tmp.size();
			var++;
		}

		return x;
	}

	std::shared_ptr<State> AdjointOptUtils::create_state(const json &args, CacheLevel level, const size_t max_threads)
	{
		std::shared_ptr<State> state = std::make_shared<State>();
		state->set_max_threads(max_threads);

		json in_args = args;
		in_args["solver"]["max_threads"] = max_threads;
		if (!args.contains("output") || !args["output"].contains("log") || !args["output"]["log"].contains("level"))
		{
			auto tmp = R"({
					"output": {
						"log": {
							"level": -1
						}
					}
				})"_json;

			tmp["output"]["log"]["level"] = "error";

			in_args.merge_patch(tmp);
		}

		state->optimization_enabled = level;
		state->init(in_args, true);
		state->load_mesh();
		Eigen::MatrixXd sol, pressure;
		state->build_basis();
		state->assemble_rhs();
		state->assemble_mass_mat();

		return state;
	}

	std::vector<std::shared_ptr<State>> AdjointOptUtils::create_states(const json &state_args, const CacheLevel &level, const size_t max_threads)
	{
		std::vector<std::shared_ptr<State>> states(state_args.size());
		int i = 0;
		for (const json &args : state_args)
		{
			json cur_args;
			if (!load_json(args["path"], cur_args))
				log_and_throw_adjoint_error("Can't find json for State {}", i);

			states[i++] = AdjointOptUtils::create_state(cur_args, level, max_threads);
		}
		return states;
	}

	void AdjointOptUtils::solve_pde(State &state)
	{
		state.assemble_rhs();
		state.assemble_mass_mat();
		Eigen::MatrixXd sol, pressure;
		state.solve_problem(sol, pressure);
	}

	void apply_objective_json_spec(json &args, const json &rules)
	{
		if (args.is_array())
		{
			for (auto &arg : args)
				apply_objective_json_spec(arg, rules);
		}
		else if (args.is_object())
		{
			jse::JSE jse;
			const bool valid_input = jse.verify_json(args, rules);

			if (!valid_input)
			{
				logger().error("invalid objective json:\n{}", jse.log2str());
				throw std::runtime_error("Invald objective json file");
			}

			args = jse.inject_defaults(args, rules);

			for (auto &it : args.items())
			{
				if (it.key().find("objective") != std::string::npos)
					apply_objective_json_spec(it.value(), rules);
			}
		}
	}

	json AdjointOptUtils::apply_opt_json_spec(const json &input_args, bool strict_validation)
	{
		json args_in = input_args;

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			std::ifstream file(POLYFEM_OPT_INPUT_SPEC);

			if (file.is_open())
				file >> rules;
			else
			{
				logger().error("unable to open {} rules", POLYFEM_OPT_INPUT_SPEC);
				throw std::runtime_error("Invald spec file");
			}

			jse.include_directories.push_back(POLYFEM_JSON_SPEC_DIR);
			jse.include_directories.push_back(POLYSOLVE_JSON_SPEC_DIR);
			rules = jse.inject_include(rules);

			// polysolve::linear::Solver::apply_default_solver(rules, "/solver/linear");
		}

		// polysolve::linear::Solver::select_valid_solver(args_in["solver"]["linear"], logger());

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}

		json args = jse.inject_defaults(args_in, rules);

		json obj_rules;
		{
			const std::string polyfem_objective_spec = POLYFEM_OBJECTIVE_INPUT_SPEC;
			std::ifstream file(polyfem_objective_spec);

			if (file.is_open())
				file >> obj_rules;
			else
			{
				logger().error("unable to open {} rules", polyfem_objective_spec);
				throw std::runtime_error("Invald spec file");
			}
		}
		apply_objective_json_spec(args["functionals"], obj_rules);

		if (args.contains("stopping_conditions"))
			apply_objective_json_spec(args["stopping_conditions"], obj_rules);

		return args;
	}

	int AdjointOptUtils::compute_variable_size(const json &args, const std::vector<std::shared_ptr<State>> &states)
	{
		if (args["number"].is_number())
		{
			return args["number"].get<int>();
		}
		else if (args["number"].is_null() && args["initial"].size() > 0)
		{
			return args["initial"].size();
		}
		else if (args["number"].is_object())
		{
			auto selection = args["number"];
			if (selection.contains("surface_selection"))
			{
				auto surface_selection = selection["surface_selection"].get<std::vector<int>>();
				auto state_id = selection["state"];
				std::set<int> node_ids = {};
				for (const auto &surface : surface_selection)
				{
					std::vector<int> ids;
					states[state_id]->compute_surface_node_ids(surface, ids);
					for (const auto &i : ids)
						node_ids.insert(i);
				}
				return node_ids.size() * states[state_id]->mesh->dimension();
			}
			else if (selection.contains("volume_selection"))
			{
				auto volume_selection = selection["volume_selection"].get<std::vector<int>>();
				auto state_id = selection["state"];
				std::set<int> node_ids = {};
				for (const auto &volume : volume_selection)
				{
					std::vector<int> ids;
					states[state_id]->compute_volume_node_ids(volume, ids);
					for (const auto &i : ids)
						node_ids.insert(i);
				}

				if (selection["exclude_boundary_nodes"])
				{
					std::vector<int> ids;
					states[state_id]->compute_total_surface_node_ids(ids);
					for (const auto &i : ids)
						node_ids.erase(i);
				}

				return node_ids.size() * states[state_id]->mesh->dimension();
			}
			else
			{
				log_and_throw_adjoint_error("Incorrect specification for parameters.");
			}
		}
		else
			log_and_throw_adjoint_error("Incorrect specification for parameters.");

		return -1;
	}
} // namespace polyfem::solver