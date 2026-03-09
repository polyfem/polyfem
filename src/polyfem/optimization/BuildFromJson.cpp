#include <polyfem/optimization/BuildFromJson.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/io/OBJReader.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <polyfem/optimization/CacheLevel.hpp>
#include <polyfem/optimization/forms/AMIPSForm.hpp>
#include <polyfem/optimization/forms/AdjointForm.hpp>
#include <polyfem/optimization/forms/BarrierForms.hpp>
#include <polyfem/optimization/forms/CompositeForms.hpp>
#include <polyfem/optimization/forms/ParametrizedProductForm.hpp>
#include <polyfem/optimization/forms/SmoothingForms.hpp>
#include <polyfem/optimization/forms/SpatialIntegralForms.hpp>
#include <polyfem/optimization/forms/SumCompositeForm.hpp>
#include <polyfem/optimization/forms/SurfaceTractionForms.hpp>
#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/optimization/forms/TransientForm.hpp>
#include <polyfem/optimization/forms/VariableToSimulation.hpp>
#include <polyfem/optimization/parametrization/Parametrization.hpp>
#include <polyfem/optimization/parametrization/Parametrizations.hpp>
#include <polyfem/optimization/parametrization/SplineParametrizations.hpp>

#include <Eigen/Core>
#include <spdlog/fmt/fmt.h>

#include <string>
#include <memory>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <set>
#include <utility>
#include <vector>

namespace polyfem::from_json
{
	namespace
	{
		bool load_json(const std::string &json_file, json &out)
		{
			std::ifstream file(json_file);
			if (!file.is_open())
			{
				return false;
			}

			file >> out;

			out["root_path"] = json_file;

			return true;
		}
	} // namespace

	std::shared_ptr<State> build_state(
		const json &args,
		const solver::CacheLevel &level,
		const size_t max_threads)
	{
		std::shared_ptr<State> state = std::make_shared<State>();
		state->set_max_threads(max_threads);

		json in_args = args;
		in_args["solver"]["max_threads"] = max_threads;
		if (!args.contains("output") || !args["output"].contains("log") || !args["output"]["log"].contains("level"))
		{
			const json tmp = R"({
					"output": {
						"log": {
							"level": "error"
						}
					}
				})"_json;

			in_args.merge_patch(tmp);
		}

		state->optimization_enabled = level;
		state->init(in_args, true);
		state->load_mesh();
		state->build_basis();
		state->assemble_rhs();
		state->assemble_mass_mat();

		return state;
	}

	std::vector<std::shared_ptr<State>> build_states(
		const std::string &root_path,
		const json &args,
		const solver::CacheLevel &level,
		const size_t max_threads)
	{
		std::vector<std::shared_ptr<State>> states(args.size());
		for (int i = 0; i < args.size(); ++i)
		{
			json cur_args;
			std::string abs_path = utils::resolve_path(args[i]["path"], root_path, false);
			if (!load_json(abs_path, cur_args))
			{
				log_and_throw_adjoint_error("Can't find json for State {}", i);
			}

			states[i] = build_state(cur_args, level, max_threads);
		}
		return states;
	}

	std::shared_ptr<solver::Parametrization> build_parametrization(
		const json &args,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<int> &variable_sizes)
	{
		using namespace polyfem::solver;

		std::shared_ptr<Parametrization> map;
		const std::string type = args["type"];
		if (type == "per-body-to-per-elem")
		{
			map = std::make_shared<PerBody2PerElem>(*(states[args["state"]]->mesh));
		}
		else if (type == "per-body-to-per-node")
		{
			map = std::make_shared<PerBody2PerNode>(*(states[args["state"]]->mesh),
													states[args["state"]]->bases,
													states[args["state"]]->n_bases);
		}
		else if (type == "E-nu-to-lambda-mu")
		{
			map = std::make_shared<ENu2LambdaMu>(args["is_volume"]);
		}
		else if (type == "slice")
		{
			if (args["from"] != -1 || args["to"] != -1)
			{
				map = std::make_shared<SliceMap>(args["from"], args["to"], args["last"]);
			}
			else if (args["parameter_index"] != -1)
			{
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
			{
				log_and_throw_adjoint_error("Incorrect spec for SliceMap!");
			}
		}
		else if (type == "exp")
		{
			map = std::make_shared<ExponentialMap>(args["from"], args["to"]);
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
			map = std::make_shared<InsertConstantMap>(vals, args["start"]);
		}
		else if (type == "append-const")
		{
			map = std::make_shared<InsertConstantMap>(args["size"], args["value"], args["start"]);
		}
		else if (type == "linear-filter")
		{
			map = std::make_shared<LinearFilter>(*(states[args["state"]]->mesh), args["radius"]);
		}
		else if (type == "bounded-biharmonic-weights")
		{
			map = std::make_shared<BoundedBiharmonicWeights2Dto3D>(
				args["num_control_vertices"], args["num_vertices"],
				*states[args["state"]], args["allow_rotations"]);
		}
		else if (type == "scalar-velocity-parametrization")
		{
			map = std::make_shared<ScalarVelocityParametrization>(args["start_val"], args["dt"]);
		}
		else
		{
			log_and_throw_adjoint_error("Unkown parametrization!");
		}

		return map;
	}

	std::unique_ptr<solver::VariableToSimulation> build_variable_to_simulation(
		const json &args,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<std::shared_ptr<DiffCache>> &diff_caches,
		const std::vector<int> &variable_sizes)
	{
		using namespace polyfem::solver;

		// Collect relevant states from state index json.
		std::vector<std::shared_ptr<State>> rel_states;
		std::vector<std::shared_ptr<DiffCache>> rel_diff_caches;
		if (args["state"].is_array())
		{
			for (int i : args["state"])
			{
				rel_states.push_back(states[i]);
				rel_diff_caches.push_back(diff_caches[i]);
			}
		}
		else
		{
			const int state_id = args["state"];
			rel_states.push_back(states[state_id]);
			rel_diff_caches.push_back(diff_caches[state_id]);
		}

		// Build all parametrizations.
		std::vector<std::shared_ptr<Parametrization>> map_list;
		for (const auto &arg : args["composition"])
		{
			map_list.push_back(build_parametrization(arg, states, variable_sizes));
		}
		CompositeParametrization compo{std::move(map_list)};

		// Build VariableToSimulation based on type string.
		std::string type = args["type"];
		std::unique_ptr<VariableToSimulation> var2sim;
		if (type == "shape")
		{
			var2sim = std::make_unique<ShapeVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "elastic")
		{
			var2sim = std::make_unique<ElasticVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "friction")
		{
			var2sim = std::make_unique<FrictionCoeffientVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "damping")
		{
			var2sim = std::make_unique<DampingCoeffientVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "initial")
		{
			var2sim = std::make_unique<InitialConditionVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "dirichlet")
		{
			var2sim = std::make_unique<DirichletVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "dirichlet-nodes")
		{
			var2sim = std::make_unique<DirichletNodesVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "pressure")
		{
			var2sim = std::make_unique<PressureVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else if (type == "periodic-shape")
		{
			var2sim = std::make_unique<PeriodicShapeVariableToSimulation>(std::move(rel_states), std::move(rel_diff_caches), std::move(compo));
		}
		else
		{
			log_and_throw_adjoint_error("Invalid type of VariableToSimulation!");
		}

		var2sim->set_output_indexing(args);

		return var2sim;
	}

	solver::VariableToSimulationGroup build_variable_to_simulation_group(
		const json &args,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<std::shared_ptr<DiffCache>> &diff_caches,
		const std::vector<int> &variable_sizes)
	{
		solver::VariableToSimulationGroup v2s_group;
		for (const auto &arg : args)
		{
			v2s_group.data.push_back(
				build_variable_to_simulation(arg, states, diff_caches, variable_sizes));
		}
		return v2s_group;
	}

	std::shared_ptr<solver::AdjointForm> build_form(
		const json &args,
		const solver::VariableToSimulationGroup &var2sim,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<std::shared_ptr<DiffCache>> &diff_caches)
	{
		using namespace polyfem::solver;

		std::shared_ptr<AdjointForm> obj;
		if (args.is_array())
		{
			std::vector<std::shared_ptr<AdjointForm>> forms;
			for (const auto &arg : args)
			{
				forms.push_back(build_form(arg, var2sim, states, diff_caches));
			}

			obj = std::make_shared<SumCompositeForm>(var2sim, forms);
		}
		else
		{
			const std::string type = args["type"];
			if (type == "transient_integral")
			{
				std::shared_ptr<StaticForm> static_obj =
					std::dynamic_pointer_cast<StaticForm>(build_form(args["static_objective"], var2sim, states, diff_caches));
				if (!static_obj)
				{
					log_and_throw_adjoint_error("Transient integral objective must have a static objective!");
				}
				const auto &state = states[args["state"]];
				obj = std::make_shared<TransientForm>(
					var2sim, state->args["time"]["time_steps"], state->args["time"]["dt"],
					args["integral_type"], args["steps"].get<std::vector<int>>(),
					static_obj);
			}
			else if (type == "proxy_transient_integral")
			{
				std::shared_ptr<StaticForm> static_obj =
					std::dynamic_pointer_cast<StaticForm>(build_form(args["static_objective"], var2sim, states, diff_caches));
				if (!static_obj)
				{
					log_and_throw_adjoint_error("Transient integral objective must have a static objective!");
				}
				if (args["steps"].size() == 0)
				{
					log_and_throw_adjoint_error("ProxyTransientForm requires non-empty \"steps\"!");
				}
				const auto &state = states[args["state"]];
				obj = std::make_shared<ProxyTransientForm>(
					var2sim, state->args["time"]["time_steps"], state->args["time"]["dt"],
					args["integral_type"], args["steps"].get<std::vector<int>>(),
					static_obj);
			}
			else if (type == "power")
			{
				std::shared_ptr<AdjointForm> obj_aux =
					build_form(args["objective"], var2sim, states, diff_caches);
				obj = std::make_shared<PowerForm>(obj_aux, args["power"]);
			}
			else if (type == "divide")
			{
				std::shared_ptr<AdjointForm> obj1 =
					build_form(args["objective"][0], var2sim, states, diff_caches);
				std::shared_ptr<AdjointForm> obj2 =
					build_form(args["objective"][1], var2sim, states, diff_caches);
				std::vector<std::shared_ptr<AdjointForm>> objs({obj1, obj2});
				obj = std::make_shared<DivideForm>(objs);
			}
			else if (type == "plus-const")
			{
				obj = std::make_shared<PlusConstCompositeForm>(
					build_form(args["objective"], var2sim, states, diff_caches), args["value"]);
			}
			else if (type == "log")
			{
				obj = std::make_shared<LogCompositeForm>(
					build_form(args["objective"], var2sim, states, diff_caches));
			}
			else if (type == "compliance")
			{
				obj = std::make_shared<ComplianceForm>(var2sim, states[args["state"]], diff_caches[args["state"]],
													   args);
			}
			else if (type == "acceleration")
			{
				obj = std::make_shared<AccelerationForm>(var2sim,
														 states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "kinetic")
			{
				obj = std::make_shared<AccelerationForm>(var2sim,
														 states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "target")
			{
				std::shared_ptr<TargetForm> tmp =
					std::make_shared<TargetForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
				auto reference_cached =
					args["reference_cached_body_ids"].get<std::vector<int>>();
				tmp->set_reference(
					states[args["target_state"]],
					diff_caches[args["target_state"]],
					std::set(reference_cached.begin(), reference_cached.end()));
				obj = tmp;
			}
			else if (type == "displacement-target")
			{
				std::shared_ptr<TargetForm> tmp =
					std::make_shared<TargetForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);

				Eigen::VectorXd target_displacement;
				target_displacement.setZero(states[args["state"]]->mesh->dimension());
				if (target_displacement.size() != args["target_displacement"].size())
				{
					log_and_throw_error("Target displacement shape must match the dimension of the simulation");
				}
				for (int i = 0; i < target_displacement.size(); ++i)
				{
					target_displacement(i) = args["target_displacement"][i].get<double>();
				}
				if (args["active_dimension"].size() > 0)
				{
					if (target_displacement.size() != args["active_dimension"].size())
					{
						log_and_throw_error("Active dimension shape must match the dimension of the simulation");
					}
					std::vector<bool> active_dimension_mask(args["active_dimension"].size());
					for (int i = 0; i < args["active_dimension"].size(); ++i)
					{
						active_dimension_mask[i] = args["active_dimension"][i].get<bool>();
					}
					tmp->set_active_dimension(active_dimension_mask);
				}
				tmp->set_reference(target_displacement);
				obj = tmp;
			}
			else if (type == "center-target")
			{
				obj = std::make_shared<BarycenterTargetForm>(
					var2sim, args, states[args["state"]], diff_caches[args["state"]], states[args["target_state"]], diff_caches[args["target_state"]]);
			}
			else if (type == "sdf-target")
			{
				std::shared_ptr<SDFTargetForm> tmp = std::make_shared<SDFTargetForm>(
					var2sim, states[args["state"]], diff_caches[args["state"]], args);
				double delta = args["delta"].get<double>();
				if (!states[args["state"]]->mesh->is_volume())
				{
					int dim = 2;
					Eigen::MatrixXd control_points(args["control_points"].size(), dim);
					for (int i = 0; i < control_points.rows(); ++i)
					{
						for (int j = 0; j < control_points.cols(); ++j)
						{
							control_points(i, j) = args["control_points"][i][j].get<double>();
						}
					}
					Eigen::VectorXd knots(args["knots"].size());
					for (int i = 0; i < knots.size(); ++i)
					{
						knots(i) = args["knots"][i].get<double>();
					}
					tmp->set_bspline_target(control_points, knots, delta);
				}
				else
				{
					int dim = 3;
					Eigen::MatrixXd control_points_grid(args["control_points_grid"].size(), dim);
					for (int i = 0; i < control_points_grid.rows(); ++i)
					{
						for (int j = 0; j < control_points_grid.cols(); ++j)
						{
							control_points_grid(i, j) = args["control_points_grid"][i][j].get<double>();
						}
					}
					Eigen::VectorXd knots_u(args["knots_u"].size());
					for (int i = 0; i < knots_u.size(); ++i)
					{
						knots_u(i) = args["knots_u"][i].get<double>();
					}
					Eigen::VectorXd knots_v(args["knots_v"].size());
					for (int i = 0; i < knots_v.size(); ++i)
					{
						knots_v(i) = args["knots_v"][i].get<double>();
					}
					tmp->set_bspline_target(control_points_grid, knots_u, knots_v, delta);
				}

				obj = tmp;
			}
			else if (type == "mesh-target")
			{
				std::shared_ptr<MeshTargetForm> tmp = std::make_shared<MeshTargetForm>(
					var2sim, states[args["state"]], diff_caches[args["state"]], args);
				double delta = args["delta"].get<double>();

				std::string mesh_path =
					states[args["state"]]->resolve_input_path(args["mesh_path"].get<std::string>());
				Eigen::MatrixXd V;
				Eigen::MatrixXi E, F;
				bool read = polyfem::io::OBJReader::read(mesh_path, V, E, F);
				if (!read)
				{
					log_and_throw_error(fmt::format("Could not read mesh! {}", mesh_path));
				}
				tmp->set_surface_mesh_target(V, F, delta);
				obj = tmp;
			}
			else if (type == "function-target")
			{
				std::shared_ptr<TargetForm> tmp =
					std::make_shared<TargetForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
				tmp->set_reference(args["target_function"], args["target_function_gradient"]);
				obj = tmp;
			}
			else if (type == "node-target")
			{
				obj = std::make_shared<NodeTargetForm>(states[args["state"]], diff_caches[args["state"]], var2sim, args);
			}
			else if (type == "min-dist-target")
			{
				obj = std::make_shared<MinTargetDistForm>(
					var2sim, args["steps"], args["target"], args, states[args["state"]], diff_caches[args["state"]]);
			}
			else if (type == "position")
			{
				obj = std::make_shared<PositionForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "stress")
			{
				obj = std::make_shared<StressForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "stress_norm")
			{
				obj = std::make_shared<StressNormForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "dirichlet_energy")
			{
				obj = std::make_shared<DirichletEnergyForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "elastic_energy")
			{
				obj = std::make_shared<ElasticEnergyForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "quadratic_contact_force_norm")
			{
				obj = std::make_shared<ProxyContactForceForm>(
					var2sim, states[args["state"]], diff_caches[args["state"]], args["dhat"], true, args);
			}
			else if (type == "log_contact_force_norm")
			{
				obj = std::make_shared<ProxyContactForceForm>(
					var2sim, states[args["state"]], diff_caches[args["state"]], args["dhat"], false, args);
			}
			else if (type == "max_stress")
			{
				obj = std::make_shared<MaxStressForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "smooth_contact_force_norm")
			{
				// assert(states[args["state"]]->args["contact"]["use_gcp_formulation"]);
				obj = std::make_shared<SmoothContactForceForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "volume")
			{
				obj = std::make_shared<VolumeForm>(var2sim, states[args["state"]], diff_caches[args["state"]], args);
			}
			else if (type == "soft_constraint")
			{
				std::vector<std::shared_ptr<AdjointForm>> forms({build_form(args["objective"], var2sim, states, diff_caches)});
				Eigen::VectorXd bounds = args["soft_bound"];
				obj = std::make_shared<InequalityConstraintForm>(forms, bounds, args["power"]);
			}
			else if (type == "min_jacobian")
			{
				obj = std::make_shared<MinJacobianForm>(var2sim, states[args["state"]]);
			}
			else if (type == "AMIPS")
			{
				obj = std::make_shared<AMIPSForm>(var2sim, states[args["state"]]);
			}
			else if (type == "boundary_smoothing")
			{
				if (args["surface_selection"].is_array())
				{
					obj = std::make_shared<BoundarySmoothingForm>(
						var2sim, states[args["state"]], args["scale_invariant"],
						args["power"], args["surface_selection"].get<std::vector<int>>(),
						args["dimensions"].get<std::vector<int>>());
				}
				else
				{
					obj = std::make_shared<BoundarySmoothingForm>(
						var2sim, states[args["state"]], args["scale_invariant"],
						args["power"],
						std::vector<int>{args["surface_selection"].get<int>()},
						args["dimensions"].get<std::vector<int>>());
				}
			}
			else if (type == "collision_barrier")
			{
				obj = std::make_shared<CollisionBarrierForm>(
					var2sim, states[args["state"]], args["dhat"]);
			}
			else if (type == "layer_thickness")
			{
				obj = std::make_shared<LayerThicknessForm>(
					var2sim, states[args["state"]],
					args["boundary_ids"].get<std::vector<int>>(), args["dhat"]);
			}
			else if (type == "layer_thickness_log")
			{
				obj = std::make_shared<LayerThicknessForm>(
					var2sim, states[args["state"]],
					args["boundary_ids"].get<std::vector<int>>(), args["dhat"], true,
					args["dmin"]);
			}
			else if (type == "deformed_collision_barrier")
			{
				obj = std::make_shared<DeformedCollisionBarrierForm>(
					var2sim, states[args["state"]], diff_caches[args["state"]], args["dhat"]);
			}
			else if (type == "parametrized_product")
			{
				std::vector<std::shared_ptr<Parametrization>> map_list;
				for (const auto &arg : args["parametrization"])
				{
					map_list.push_back(build_parametrization(arg, states, {}));
				}
				obj = std::make_shared<ParametrizedProductForm>(
					CompositeParametrization(std::move(map_list)));
			}
			else
			{
				log_and_throw_adjoint_error("Objective not implemented!");
			}

			obj->set_weight(args["weight"]);
			if (args["print_energy"].get<std::string>() != "")
			{
				obj->enable_energy_print(args["print_energy"]);
			}
		}

		return obj;
	}

} // namespace polyfem::from_json
