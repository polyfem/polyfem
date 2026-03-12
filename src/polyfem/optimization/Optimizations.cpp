#include <polyfem/optimization/Optimizations.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/optimization/CacheLevel.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/forms/SpatialIntegralForms.hpp>
#include <polyfem/optimization/forms/SumCompositeForm.hpp>
#include <polyfem/optimization/forms/CompositeForms.hpp>
#include <polyfem/optimization/forms/TransientForm.hpp>
#include <polyfem/optimization/forms/SmoothingForms.hpp>
#include <polyfem/optimization/forms/AMIPSForm.hpp>
#include <polyfem/optimization/forms/BarrierForms.hpp>
#include <polyfem/optimization/forms/SurfaceTractionForms.hpp>
#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/optimization/forms/ParametrizedProductForm.hpp>
#include <polyfem/optimization/parametrization/Parametrizations.hpp>
#include <polyfem/optimization/parametrization/SplineParametrizations.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/mesh/GeometryReader.hpp>

#include <polyfem/io/OBJReader.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polysolve/nonlinear/BoxConstraintSolver.hpp>

#include <jse/jse.h>

#include <Eigen/Core>

#include <memory>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <stdexcept>

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

	Eigen::VectorXd AdjointOptUtils::inverse_evaluation(const json &args, const int ndof, const std::vector<int> &variable_sizes, VariableToSimulationGroup &var2sim)
	{
		Eigen::VectorXd x;
		x.setZero(ndof);
		int accumulative = 0;
		int var = 0;
		for (const auto &arg : args)
		{
			const auto &arg_initial = arg["initial"];
			Eigen::VectorXd tmp(variable_sizes[var]);
			if (arg_initial.is_array() && arg_initial.size() > 0)
			{
				tmp = arg_initial;
				x.segment(accumulative, tmp.size()) = tmp;
			}
			else if (arg_initial.is_number())
			{
				tmp.setConstant(arg_initial.get<double>());
				x.segment(accumulative, tmp.size()) = tmp;
			}
			else // arg["initial"] is empty array
				x += var2sim.data[var]->inverse_eval();

			accumulative += tmp.size();
			var++;
		}

		return x;
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
					compute_surface_node_ids(*states[state_id], surface, ids);
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
					compute_volume_node_ids(*states[state_id], volume, ids);
					for (const auto &i : ids)
						node_ids.insert(i);
				}

				if (selection["exclude_boundary_nodes"])
				{
					std::vector<int> ids;
					compute_total_surface_node_ids(*states[state_id], ids);
					for (const auto &i : ids)
						node_ids.erase(i);
				}

				return node_ids.size() * states[state_id]->mesh->dimension();
			}
		}

		log_and_throw_adjoint_error("Incorrect specification for parameters.");
		return -1;
	}
} // namespace polyfem::solver
