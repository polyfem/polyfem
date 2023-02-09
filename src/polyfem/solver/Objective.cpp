#include "Objective.hpp"
#include "HomoObjective.hpp"
#include "IntegralObjective.hpp"

#include <polyfem/utils/CubicHermiteSplineParametrization.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <igl/adjacency_list.h>
#include "ControlParameter.hpp"
#include "ShapeParameter.hpp"

using namespace polyfem::utils;

namespace polyfem::solver
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}

		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

		template <typename T>
		T inverse_Lp_norm(const Eigen::Matrix<T, Eigen::Dynamic, 1> &F, const double p)
		{
			T val = T(0);
			for (int i = 0; i < F.size(); i++)
			{
				val += pow(F(i), p);
			}
			return T(1) / pow(val, 1. / p);
		}

		Eigen::VectorXd inverse_Lp_norm_grad(const Eigen::VectorXd &F, const double p)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, 1> full_diff(F.size());
			for (int i = 0; i < F.size(); i++)
				full_diff(i) = Diff(i, F(i));
			auto reduced_diff = inverse_Lp_norm(full_diff, p);

			Eigen::VectorXd grad(F.size());
			for (int i = 0; i < F.size(); ++i)
				grad(i) = reduced_diff.getGradient()(i);

			return grad;
		}

		double barrier_func(double x, double dhat)
		{
			double y = x / dhat;
			if (0 < y && y < 1)
				return -(y - 1) * (y - 1) * log(y) * dhat;
			else if (x > dhat)
				return 0;
			else
				return std::nan("");
		}

		double barrier_func_derivative(double x, double dhat)
		{
			double y = x / dhat;
			if (0 < y && y < 1)
				return -(1 - y) * (1 - y - 2 * y * log(y)) / y;
			else if (y > 1)
				return 0;
			else
				return std::nan("");
		}
	} // namespace

	std::shared_ptr<Objective> Objective::create(const json &args, const std::string &root_path, const std::vector<std::shared_ptr<Parameter>> &parameters, const std::vector<std::shared_ptr<State>> &states)
	{
		std::shared_ptr<Objective> obj;
		std::shared_ptr<StaticObjective> static_obj;
		const std::string type = args["type"];

		std::string transient_integral_type;
		if (args.contains("transient_integral_type"))
		{
			if (args["transient_integral_type"] == "steps")
			{
				auto steps = args["steps"].get<std::vector<int>>();
				transient_integral_type = "[";
				for (auto s : steps)
					transient_integral_type += std::to_string(s) + ",";
				transient_integral_type.pop_back();
				transient_integral_type += "]";
			}
			else
				transient_integral_type = args["transient_integral_type"];
		}

		if (type == "trajectory")
		{
			State &state = *(states[args["state"]]);
			const std::string matching = args["matching"];
			if (matching == "exact")
			{
				std::shared_ptr<Parameter> shape_param;
				if (args["shape_parameter"] >= 0)
				{
					shape_param = parameters[args["shape_parameter"]];
					if (!shape_param->contains_state(state))
						logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
				}

				auto target = states[args["target_state"]];
				auto target_obj = std::make_shared<TargetObjective>(state, shape_param, args);
				auto reference_cached = args["reference_cached_body_ids"].get<std::vector<int>>();
				target_obj->set_reference(target, std::set(reference_cached.begin(), reference_cached.end()));
				static_obj = target_obj;
			}
			else if (matching == "sdf")
			{
				Eigen::MatrixXd control_points, tangents, delta;
				Eigen::VectorXd knots, knots_u, knots_v;
				int dim = states[args["state"]]->mesh->dimension();
				if (dim == 2)
				{
					control_points.setZero(args["control_points"].size(), dim);
					for (int i = 0; i < args["control_points"].size(); ++i)
					{
						for (int j = 0; j < args["control_points"][i].size(); ++j)
							control_points(i, j) = args["control_points"][i][j].get<double>();
					}

					knots.setZero(args["knots"].size());
					for (int i = 0; i < args["knots"].size(); ++i)
						knots(i) = args["knots"][i].get<double>();
				}
				else if (dim == 3)
				{
					control_points.setZero(args["control_points_grid"].size(), dim);
					for (int i = 0; i < args["control_points_grid"].size(); ++i)
					{
						for (int j = 0; j < args["control_points_grid"][i].size(); ++j)
							control_points(i, j) = args["control_points_grid"][i][j].get<double>();
					}

					knots_u.setZero(args["knots_u"].size());
					for (int i = 0; i < args["knots_u"].size(); ++i)
						knots_u(i) = args["knots_u"][i].get<double>();
					knots_v.setZero(args["knots_v"].size());
					for (int i = 0; i < args["knots_v"].size(); ++i)
						knots_v(i) = args["knots_v"][i].get<double>();
				}

				delta.setZero(1, args["delta"].size());
				for (int i = 0; i < delta.size(); ++i)
					delta(i) = args["delta"][i].get<double>();

				std::shared_ptr<Parameter> shape_param;
				if (args["shape_parameter"] >= 0)
				{
					shape_param = parameters[args["shape_parameter"]];
					if (!shape_param->contains_state(state))
						logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
				}

				auto target_obj = std::make_shared<SDFTargetObjective>(state, shape_param, args);
				if (dim == 2)
					target_obj->set_bspline_target(control_points, knots, delta(0));
				else if (dim == 3)
					target_obj->set_bspline_target(control_points, knots_u, knots_v, delta(0));

				static_obj = target_obj;
			}
			else if (matching == "marker-data" || matching == "exact-marker")
			{
				State &state = *(states[args["state"]]);
				const int dim = state.mesh->dimension();
				const std::string target_data_path = utils::resolve_path(args["marker_data_path"], root_path, false);

				if (!std::filesystem::is_regular_file(target_data_path))
				{
					throw std::runtime_error("Marker path invalid!");
				}
				Eigen::MatrixXd tmp;
				io::read_matrix(target_data_path, tmp);
				Eigen::VectorXi nodes = tmp.col(0).cast<int>();

				Eigen::MatrixXd targets;
				targets.setZero(nodes.size(), dim);
				std::vector<int> active_nodes;

				if (matching == "exact-marker")
				{
					const State &state_reference = *(states[args["target_state"]]);
					const Eigen::MatrixXd &sol = state_reference.diff_cached[0].u;

					for (int s = 0; s < nodes.size(); s++)
					{
						const int node_id = state.in_node_to_node(nodes(s));
						targets.row(s) = sol.block(node_id * dim, 0, dim, 1).transpose() + state_reference.mesh_nodes->node_position(node_id);
						active_nodes.push_back(node_id);
					}
				}
				else
				{
					for (int s = 0; s < nodes.size(); s++)
					{
						const int node_id = state.in_node_to_node(nodes(s));
						targets.row(s) = tmp.block(s, 1, 1, tmp.cols() - 1);
						active_nodes.push_back(node_id);
					}
				}

				static_obj = std::make_shared<NodeTargetObjective>(state, active_nodes, targets);
			}
			else if (matching == "target-position")
			{
				std::shared_ptr<Parameter> shape_param;
				if (args["shape_parameter"] >= 0)
				{
					shape_param = parameters[args["shape_parameter"]];
					if (!shape_param->contains_state(state))
						logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
				}

				auto target_obj = std::make_shared<TargetObjective>(state, shape_param, args);
				Eigen::VectorXd target_disp = args["target_displacement"];
				target_obj->set_reference(target_disp);
				if (args["active_dimension"].size() > 0)
					target_obj->set_active_dimension(args["active_dimension"].get<std::vector<bool>>());
				static_obj = target_obj;
			}
			else if (matching == "target-function")
			{
				std::shared_ptr<Parameter> shape_param;
				if (args["shape_parameter"] >= 0)
				{
					shape_param = parameters[args["shape_parameter"]];
					if (!shape_param->contains_state(state))
						logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
				}

				auto target_obj = std::make_shared<TargetObjective>(state, shape_param, args);

				target_obj->set_reference(args["target_function"], args["target_function_gradient"]);
				static_obj = target_obj;
			}
			else
			{
				assert(false);
			}

			if (state.problem->is_time_dependent())
				obj = std::make_shared<TransientObjective>(state.args["time"]["time_steps"], state.args["time"]["dt"], transient_integral_type, static_obj);
			else
				obj = static_obj;
		}
		else if (type == "stress")
		{
			State &state = *(states[args["state"]]);
			std::shared_ptr<Parameter> shape_param;
			std::shared_ptr<Parameter> elastic_param;
			if (args["shape_parameter"] >= 0)
			{
				shape_param = parameters[args["shape_parameter"]];
				if (!shape_param->contains_state(state))
					logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
			}
			else
				logger().warn("No shape parameter is assigned to functional");

			if (args["material_parameter"] >= 0)
				elastic_param = parameters[args["material_parameter"]];

			std::shared_ptr<solver::StaticObjective> tmp = std::make_shared<solver::StressObjective>(state, shape_param, elastic_param, args);
			if (state.problem->is_time_dependent())
				obj = std::make_shared<solver::TransientObjective>(state.args["time"]["time_steps"], state.args["time"]["dt"], transient_integral_type, tmp);
			else
				obj = tmp;
		}
		else if (type == "homogenized_energy")
		{
			State &state = *(states[args["state"]]);
			std::shared_ptr<Parameter> shape_param, elastic_param, macro_strain_param;
			if (args["shape_parameter"] >= 0)
			{
				shape_param = parameters[args["shape_parameter"]];
				if (!shape_param->contains_state(state))
					logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
			}
			else
				logger().warn("No shape parameter is assigned to functional");

			if (args["material_parameter"] >= 0)
				elastic_param = parameters[args["material_parameter"]];

			if (args["macro_strain_parameter"] >= 0)
				macro_strain_param = parameters[args["macro_strain_parameter"]];

			obj = std::make_shared<solver::HomogenizedEnergyObjective>(state, shape_param, macro_strain_param, elastic_param, args);
		}
		else if (type == "homogenized_stress")
		{
			State &state = *(states[args["state"]]);
			std::shared_ptr<Parameter> shape_param, elastic_param, macro_strain_param;
			if (args["shape_parameter"] >= 0)
			{
				shape_param = parameters[args["shape_parameter"]];
				if (!shape_param->contains_state(state))
					logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
			}
			else
				logger().warn("No shape parameter is assigned to functional");

			if (args["material_parameter"] >= 0)
				elastic_param = parameters[args["material_parameter"]];

			if (args["macro_strain_parameter"] >= 0)
				macro_strain_param = parameters[args["macro_strain_parameter"]];

			obj = std::make_shared<solver::CompositeHomogenizedStressObjective>(state, shape_param, macro_strain_param, elastic_param, args);
		}
		else if (type == "position")
		{
			State &state = *(states[args["state"]]);
			std::shared_ptr<Parameter> shape_param;
			if (args["shape_parameter"] >= 0)
			{
				shape_param = parameters[args["shape_parameter"]];
				if (!shape_param->contains_state(state))
					logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
			}
			else
				logger().warn("No shape parameter is assigned to functional");

			std::shared_ptr<solver::PositionObjective> tmp = std::make_shared<solver::PositionObjective>(state, shape_param, args);
			tmp->set_dim(args["dim"]);
			if (state.problem->is_time_dependent())
				obj = std::make_shared<solver::TransientObjective>(state.args["time"]["time_steps"], state.args["time"]["dt"], transient_integral_type, tmp);
			else
				obj = tmp;
		}
		else if (type == "boundary_smoothing")
		{
			std::shared_ptr<Parameter> shape_param = parameters[args["shape_parameter"]];
			obj = std::make_shared<solver::BoundarySmoothingObjective>(shape_param, args);
		}
		else if (type == "deformed_boundary_smoothing")
		{
			State &state = *(states[args["state"]]);
			std::shared_ptr<Parameter> shape_param;
			if (args["shape_parameter"] >= 0)
			{
				shape_param = parameters[args["shape_parameter"]];
				if (!shape_param->contains_state(state))
					logger().error("Shape parameter {} is inconsistent with state {} in functional", args["shape_parameter"], args["state"]);
			}
			obj = std::make_shared<solver::DeformedBoundarySmoothingObjective>(state, shape_param, args);
		}
		else if (type == "material_bound")
		{
			if (args["material_parameter"] < 0)
				log_and_throw_error("No material parameter assigned to material bound objective!");
			std::shared_ptr<Parameter> elastic_param = parameters[args["material_parameter"]];
			obj = std::make_shared<MaterialBoundObjective>(elastic_param, args);
		}
		else if (type == "control_smoothing")
		{
			State &state = *(states[args["state"]]);
			std::shared_ptr<Parameter> control_param = parameters[args["control_parameter"]];
			auto static_obj = std::make_shared<ControlSmoothingObjective>(control_param, args);
			if (state.problem->is_time_dependent())
				obj = std::make_shared<TransientObjective>(state.args["time"]["time_steps"], state.args["time"]["dt"], "uniform", static_obj);
			else
				assert(false);
		}
		else if (type == "material_smoothing")
		{
			assert(false);
		}
		else if (type == "volume_constraint")
		{
			std::shared_ptr<Parameter> shape_param = parameters[args["shape_parameter"]];
			obj = std::make_shared<solver::VolumePenaltyObjective>(shape_param, args);
		}
		else if (type == "collision_barrier")
		{
			std::shared_ptr<Parameter> shape_param = parameters[args["shape_parameter"]];
			obj = std::make_shared<solver::CollisionBarrierObjective>(shape_param, args);
		}
		else if (type == "compliance")
		{
			State &state = *(states[args["state"]]);

			std::shared_ptr<Parameter> shape_param;
			if (args["shape_parameter"] >= 0)
				shape_param = parameters[args["shape_parameter"]];

			std::shared_ptr<Parameter> elastic_param;
			if (args["material_parameter"] >= 0)
				elastic_param = parameters[args["material_parameter"]];

			if (!shape_param && !elastic_param)
				logger().warn("No parameter is assigned to functional");

			obj = std::make_shared<solver::ComplianceObjective>(state, shape_param, elastic_param, args);
		}
		else if (type == "strain_norm")
		{
			State &state = *(states[args["state"]]);

			std::shared_ptr<Parameter> shape_param;
			if (args["shape_parameter"] >= 0)
				shape_param = parameters[args["shape_parameter"]];

			obj = std::make_shared<solver::StrainObjective>(state, shape_param, args);
		}
		else if (type == "layer_thickness")
		{
			std::shared_ptr<Parameter> first_shape_param = parameters[args["shape_parameter"]];
			if (args["adjacent_shape_parameter"] >= 0)
			{
				std::shared_ptr<Parameter> second_shape_param = parameters[args["adjacent_shape_parameter"]];
				obj = std::make_shared<solver::LayerThicknessObjective>(first_shape_param, second_shape_param, args);
			}
			else if (args["adjacent_boundary_id"] >= 0)
			{
				int adjacent_boundary_id = args["adjacent_boundary_id"];
				obj = std::make_shared<solver::LayerThicknessObjective>(first_shape_param, adjacent_boundary_id, args);
			}
			else
				log_and_throw_error("Invalid specification of boundaries for layer_thickness objective!");
		}
		else
			log_and_throw_error("Unkown functional type {}!", type);

		return obj;
	}

	Eigen::VectorXd Objective::compute_adjoint_term(const State &state, const Eigen::MatrixXd &adjoints, const Parameter &param)
	{
		Eigen::VectorXd term;
		term.setZero(param.full_dim());

		if (param.contains_state(state))
		{
			AdjointTools::compute_adjoint_term(state, adjoints, param.name(), term);
		}

		return term;
	}

	Eigen::MatrixXd SumObjective::compute_adjoint_rhs(const State &state)
	{
		Eigen::MatrixXd rhs;
		rhs.setZero(state.ndof(), state.problem->is_time_dependent() ? state.args["time"]["time_steps"].get<int>() + 1 : 1);
		int i = 0;
		for (const auto &obj : objs_)
		{
			rhs += weights_(i++) * obj->compute_adjoint_rhs(state);
		}
		return rhs;
	}

	Eigen::VectorXd SumObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd grad;
		grad.setZero(param.optimization_dim());
		int i = 0;
		for (const auto &obj : objs_)
		{
			grad += weights_(i++) * obj->compute_partial_gradient(param, param_value);
		}
		return grad;
	}

	double SumObjective::value()
	{
		double val = 0;
		int i = 0;
		for (const auto &obj : objs_)
		{
			val += weights_(i++) * obj->value();
		}
		return val;
	}

	void BoundarySmoothingObjective::init()
	{
		const auto &state_ = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);

		const int dim = V.cols();
		const int n_verts = V.rows();

		// collect active nodes
		std::vector<bool> active_mask;
		active_mask.assign(n_verts, false);
		std::vector<int> tmp = args_["surface_selection"];
		std::set<int> surface_ids = std::set(tmp.begin(), tmp.end());

		const auto &gbases = state_.geom_bases();
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); i++)
			{
				const int global_primitive_id = lb.global_primitive_id(i);
				const int boundary_id = state_.mesh->get_boundary_id(global_primitive_id);
				if (!surface_ids.empty() && surface_ids.find(boundary_id) == surface_ids.end())
					continue;

				const auto nodes = gbases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *state_.mesh);

				for (int n = 0; n < nodes.size(); n++)
				{
					const auto &global = gbases[e].bases[nodes(n)].global();
					for (int g = 0; g < global.size(); g++)
						active_mask[global[g].index] = true;
				}
			}
		}

		adj.setZero();
		adj.resize(n_verts, n_verts);
		std::vector<Eigen::Triplet<bool>> T_adj;

		ipc::CollisionMesh collision_mesh;
		state_.build_collision_mesh(collision_mesh, state_.n_geom_bases, state_.geom_bases());
		for (int e = 0; e < collision_mesh.num_edges(); e++)
		{
			int v1 = collision_mesh.to_full_vertex_id(collision_mesh.edges()(e, 0));
			int v2 = collision_mesh.to_full_vertex_id(collision_mesh.edges()(e, 1));
			if (active_mask[v1] && active_mask[v2])
			{
				T_adj.emplace_back(v1, v2, true);
				T_adj.emplace_back(v2, v1, true);
			}
		}
		adj.setFromTriplets(T_adj.begin(), T_adj.end());

		std::vector<int> degrees(n_verts, 0);
		for (int k = 0; k < adj.outerSize(); ++k)
			for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
				degrees[k]++;

		L.setZero();
		L.resize(n_verts, n_verts);
		if (!args_["scale_invariant"])
		{
			std::vector<Eigen::Triplet<double>> T_L;
			for (int k = 0; k < adj.outerSize(); ++k)
			{
				if (degrees[k] == 0 || !active_mask[k])
					continue;
				T_L.emplace_back(k, k, degrees[k]);
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
				{
					assert(it.row() == k);
					T_L.emplace_back(it.row(), it.col(), -1);
				}
			}
			L.setFromTriplets(T_L.begin(), T_L.end());
			L.prune([](int i, int j, double val) { return abs(val) > 1e-12; });
		}
	}

	BoundarySmoothingObjective::BoundarySmoothingObjective(const std::shared_ptr<const Parameter> shape_param, const json &args) : shape_param_(shape_param), args_(args)
	{
		init();
	}

	double BoundarySmoothingObjective::value()
	{
		const auto &state_ = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();
		const int power = args_["power"];

		double val = 0;
		if (args_["scale_invariant"])
		{
			for (int b = 0; b < adj.rows(); b++)
			{
				polyfem::RowVectorNd s;
				s.setZero(V.cols());
				double sum_norm = 0;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					s += V.row(b) - V.row(it.col());
					sum_norm += (V.row(b) - V.row(it.col())).norm();
					valence += 1;
				}
				if (valence)
				{
					s = s / sum_norm;
					val += pow(s.norm(), power);
				}
			}
		}
		else
			val = (L * V).eval().squaredNorm();

		return val;
	}

	Eigen::MatrixXd BoundarySmoothingObjective::compute_adjoint_rhs(const State &state)
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.problem->is_time_dependent() ? state.args["time"]["time_steps"].get<int>() + 1 : 1);
	}

	Eigen::VectorXd BoundarySmoothingObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param != shape_param_.get())
			return Eigen::VectorXd::Zero(param.optimization_dim());

		const auto &state_ = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();
		const int power = args_["power"];

		Eigen::VectorXd grad;
		if (args_["scale_invariant"])
		{
			grad.setZero(V.size());
			for (int b = 0; b < adj.rows(); b++)
			{
				polyfem::RowVectorNd s;
				s.setZero(dim);
				double sum_norm = 0;
				auto sum_normalized = s;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					auto x = V.row(b) - V.row(it.col());
					s += x;
					sum_norm += x.norm();
					sum_normalized += x.normalized();
					valence += 1;
				}
				if (valence)
				{
					s = s / sum_norm;

					for (int d = 0; d < dim; d++)
					{
						grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * power * pow(s.norm(), power - 2.) / sum_norm;
					}

					for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
					{
						for (int d = 0; d < dim; d++)
						{
							grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (V(it.col(), d) - V(b, d)) / (V.row(b) - V.row(it.col())).norm()) * power * pow(s.norm(), power - 2.) / sum_norm;
						}
					}
				}
			}
		}
		else
			grad = utils::flatten(2 * (L.transpose() * (L * V)));

		return param.map_grad(param_value, grad);
	}

	void DeformedBoundarySmoothingObjective::init()
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);

		const int dim = V.cols();
		const int n_verts = V.rows();

		// collect active nodes
		std::vector<bool> active_mask;
		active_mask.assign(n_verts, false);
		std::vector<int> tmp = args_["surface_selection"];
		std::set<int> surface_ids = std::set(tmp.begin(), tmp.end());

		const auto &gbases = state_.geom_bases();
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); i++)
			{
				const int global_primitive_id = lb.global_primitive_id(i);
				const int boundary_id = state_.mesh->get_boundary_id(global_primitive_id);
				if (!surface_ids.empty() && surface_ids.find(boundary_id) == surface_ids.end())
					continue;

				const auto nodes = gbases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *state_.mesh);

				for (int n = 0; n < nodes.size(); n++)
				{
					const auto &global = gbases[e].bases[nodes(n)].global();
					for (int g = 0; g < global.size(); g++)
						active_mask[global[g].index] = true;
				}
			}
		}

		adj.setZero();
		adj.resize(n_verts, n_verts);
		std::vector<Eigen::Triplet<bool>> T_adj;

		ipc::CollisionMesh collision_mesh;
		state_.build_collision_mesh(collision_mesh, state_.n_geom_bases, state_.geom_bases());
		for (int e = 0; e < collision_mesh.num_edges(); e++)
		{
			int v1 = collision_mesh.to_full_vertex_id(collision_mesh.edges()(e, 0));
			int v2 = collision_mesh.to_full_vertex_id(collision_mesh.edges()(e, 1));
			if (active_mask[v1] && active_mask[v2])
			{
				T_adj.emplace_back(v1, v2, true);
				T_adj.emplace_back(v2, v1, true);
			}
		}
		adj.setFromTriplets(T_adj.begin(), T_adj.end());
	}

	DeformedBoundarySmoothingObjective::DeformedBoundarySmoothingObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : state_(state), shape_param_(shape_param), args_(args)
	{
		init();
	}

	double DeformedBoundarySmoothingObjective::value()
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();
		const int power = args_["power"];
		Eigen::MatrixXd displaced = V + utils::unflatten(state_.down_sampling_mat * state_.diff_cached[0].u, dim);

		double val = 0;
		for (int b = 0; b < adj.rows(); b++)
		{
			polyfem::RowVectorNd s;
			s.setZero(dim);
			double sum_norm = 0;
			int valence = 0;
			for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
			{
				assert(it.col() != b);
				s += displaced.row(b) - displaced.row(it.col());
				sum_norm += (displaced.row(b) - displaced.row(it.col())).norm();
				valence += 1;
			}
			if (valence)
			{
				s = s / sum_norm;
				val += pow(s.norm(), power);
			}
		}

		return val;
	}

	Eigen::MatrixXd DeformedBoundarySmoothingObjective::compute_adjoint_rhs(const State &state)
	{
		if (&state != &state_)
			return Eigen::MatrixXd::Zero(state.ndof(), state.problem->is_time_dependent() ? state.args["time"]["time_steps"].get<int>() + 1 : 1);

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();
		const int power = args_["power"];
		Eigen::MatrixXd displaced = V + utils::unflatten(state.down_sampling_mat * state.diff_cached[0].u, dim);

		Eigen::VectorXd grad;
		grad.setZero(displaced.size());
		for (int b = 0; b < adj.rows(); b++)
		{
			polyfem::RowVectorNd s;
			s.setZero(dim);
			double sum_norm = 0;
			auto sum_normalized = s;
			int valence = 0;
			for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
			{
				assert(it.col() != b);
				auto x = displaced.row(b) - displaced.row(it.col());
				s += x;
				sum_norm += x.norm();
				sum_normalized += x.normalized();
				valence += 1;
			}
			if (valence)
			{
				s = s / sum_norm;

				for (int d = 0; d < dim; d++)
				{
					grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * power * pow(s.norm(), power - 2.) / sum_norm;
				}

				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					for (int d = 0; d < dim; d++)
					{
						grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (displaced(it.col(), d) - displaced(b, d)) / (displaced.row(b) - displaced.row(it.col())).norm()) * power * pow(s.norm(), power - 2.) / sum_norm;
					}
				}
			}
		}

		return state.down_sampling_mat.transpose() * grad;
	}

	Eigen::VectorXd DeformedBoundarySmoothingObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param != shape_param_.get())
			return Eigen::VectorXd::Zero(param.optimization_dim());

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state_.get_vf(V, F);
		const int dim = V.cols();
		const int n_verts = V.rows();
		const int power = args_["power"];
		Eigen::MatrixXd displaced = V + utils::unflatten(state_.down_sampling_mat * state_.diff_cached[0].u, dim);

		Eigen::VectorXd grad;
		grad.setZero(displaced.size());
		for (int b = 0; b < adj.rows(); b++)
		{
			polyfem::RowVectorNd s;
			s.setZero(dim);
			double sum_norm = 0;
			auto sum_normalized = s;
			int valence = 0;
			for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
			{
				assert(it.col() != b);
				auto x = displaced.row(b) - displaced.row(it.col());
				s += x;
				sum_norm += x.norm();
				sum_normalized += x.normalized();
				valence += 1;
			}
			if (valence)
			{
				s = s / sum_norm;

				for (int d = 0; d < dim; d++)
				{
					grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * power * pow(s.norm(), power - 2.) / sum_norm;
				}

				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					for (int d = 0; d < dim; d++)
					{
						grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (displaced(it.col(), d) - displaced(b, d)) / (displaced.row(b) - displaced.row(it.col())).norm()) * power * pow(s.norm(), power - 2.) / sum_norm;
					}
				}
			}
		}

		return param.map_grad(param_value, grad);
	}

	VolumeObjective::VolumeObjective(const std::shared_ptr<const Parameter> shape_param, const json &args) : shape_param_(shape_param)
	{
		if (!shape_param)
			log_and_throw_error("Volume Objective needs non-empty shape parameter!");
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		weights_.setOnes(0);
	}

	double VolumeObjective::value()
	{
		assert(weights_.size() == 0 || weights_.size() == shape_param_->get_state().bases.size());

		IntegrableFunctional j;
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setOnes(u.rows(), 1);
			const int e = params["elem"];
			if (weights_.size() > e)
				val *= this->weights_(e);
		});

		const State &state = shape_param_->get_state();

		return AdjointTools::integrate_objective(state, j, Eigen::MatrixXd::Zero(state.ndof(), 1), interested_ids_, SpatialIntegralType::VOLUME, 0);
	}

	Eigen::MatrixXd VolumeObjective::compute_adjoint_rhs(const State &state)
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.problem->is_time_dependent() ? state.args["time"]["time_steps"].get<int>() + 1 : 1); // Important: it's state, not state_
	}

	Eigen::VectorXd VolumeObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param == shape_param_.get())
		{
			assert(weights_.size() == 0 || weights_.size() == shape_param_->get_state().bases.size());

			IntegrableFunctional j;
			j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setOnes(u.rows(), 1);
				const int e = params["elem"];
				if (weights_.size() > e)
					val *= this->weights_(e);
			});

			const State &state = shape_param_->get_state();
			Eigen::VectorXd term;
			AdjointTools::compute_shape_derivative_functional_term(state, Eigen::MatrixXd::Zero(state.ndof(), 1), j, interested_ids_, SpatialIntegralType::VOLUME, term, 0);
			return param.map_grad(param_value, term);
		}
		else
			return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	VolumePenaltyObjective::VolumePenaltyObjective(const std::shared_ptr<const Parameter> shape_param, const json &args)
	{
		if (args["soft_bound"].get<std::vector<double>>().size() == 2)
			bound = args["soft_bound"];
		else
			bound << 0, std::numeric_limits<double>::max();

		obj = std::make_shared<VolumeObjective>(shape_param, args);
	}

	double VolumePenaltyObjective::value()
	{
		double vol = obj->value();

		logger().debug("Current volume: {}", vol);

		if (vol < bound[0])
			return pow(vol - bound[0], 2);
		else if (vol > bound[1])
			return pow(vol - bound[1], 2);
		else
			return 0;
	}
	Eigen::MatrixXd VolumePenaltyObjective::compute_adjoint_rhs(const State &state)
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.problem->is_time_dependent() ? state.args["time"]["time_steps"].get<int>() + 1 : 1);
	}
	Eigen::VectorXd VolumePenaltyObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		double vol = obj->value();
		Eigen::VectorXd grad = obj->compute_partial_gradient(param, param_value);

		if (vol < bound[0])
			return (2 * (vol - bound[0])) * grad;
		else if (vol > bound[1])
			return (2 * (vol - bound[1])) * grad;
		else
			return Eigen::VectorXd::Zero(grad.size(), 1);
	}

	Eigen::MatrixXd StaticObjective::compute_adjoint_rhs(const State &state)
	{
		Eigen::MatrixXd term(state.ndof(), state.diff_cached.size());
		term.col(time_step_) = compute_adjoint_rhs_step(state);

		return term;
	}

	TransientObjective::TransientObjective(const int time_steps, const double dt, const std::string &transient_integral_type, const std::shared_ptr<StaticObjective> &obj)
	{
		time_steps_ = time_steps;
		dt_ = dt;
		transient_integral_type_ = transient_integral_type;
		obj_ = obj;
	}

	std::vector<double> TransientObjective::get_transient_quadrature_weights() const
	{
		std::vector<double> weights;
		weights.assign(time_steps_ + 1, dt_);
		if (transient_integral_type_ == "uniform")
		{
			weights[0] = 0;
		}
		else if (transient_integral_type_ == "trapezoidal")
		{
			weights[0] = dt_ / 2.;
			weights[weights.size() - 1] = dt_ / 2.;
		}
		else if (transient_integral_type_ == "simpson")
		{
			weights[0] = dt_ / 3.;
			weights[weights.size() - 1] = dt_ / 3.;
			for (int i = 1; i < weights.size() - 1; i++)
			{
				if (i % 2)
					weights[i] = dt_ * 4. / 3.;
				else
					weights[i] = dt_ * 2. / 4.;
			}
		}
		else if (transient_integral_type_ == "final")
		{
			weights.assign(time_steps_ + 1, 0);
			weights[time_steps_] = 1;
		}
		else if (transient_integral_type_.find("step_") != std::string::npos)
		{
			weights.assign(time_steps_ + 1, 0);
			int step = std::stoi(transient_integral_type_.substr(5));
			assert(step > 0 && step < weights.size());
			weights[step] = 1;
		}
		else if (json::parse(transient_integral_type_).is_array())
		{
			weights.assign(time_steps_ + 1, 0);
			auto steps = json::parse(transient_integral_type_);
			for (const int step : steps)
			{
				assert(step > 0 && step < weights.size());
				weights[step] = 1. / steps.size();
			}
		}
		else
			assert(false);

		return weights;
	}

	double TransientObjective::value()
	{
		double value = 0;
		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			value += weights[i] * obj_->value();
		}
		return value;
	}

	Eigen::MatrixXd TransientObjective::compute_adjoint_rhs(const State &state)
	{
		Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), time_steps_ + 1);

		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			terms.col(i) = weights[i] * obj_->compute_adjoint_rhs_step(state);
		}

		return terms;
	}

	Eigen::VectorXd TransientObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		term.setZero(param.optimization_dim());

		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			term += weights[i] * obj_->compute_partial_gradient(param, param_value);
		}

		return term;
	}

	NodeTargetObjective::NodeTargetObjective(const State &state, const json &args) : state_(state)
	{
		std::string target_data_path = args["target_data_path"];
		if (!std::filesystem::is_regular_file(target_data_path))
		{
			throw std::runtime_error("Marker path invalid!");
		}
		Eigen::MatrixXd tmp;
		io::read_matrix(target_data_path, tmp);

		// markers to nodes
		Eigen::VectorXi nodes = tmp.col(0).cast<int>();
		target_vertex_positions.setZero(nodes.size(), state_.mesh->dimension());
		active_nodes.reserve(nodes.size());
		for (int s = 0; s < nodes.size(); s++)
		{
			const int node_id = state_.in_node_to_node(nodes(s));
			target_vertex_positions.row(s) = tmp.block(s, 1, 1, tmp.cols() - 1);
			active_nodes.push_back(node_id);
		}
	}

	NodeTargetObjective::NodeTargetObjective(const State &state, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_) : state_(state), target_vertex_positions(target_vertex_positions_), active_nodes(active_nodes_)
	{
	}

	double NodeTargetObjective::value()
	{
		const int dim = state_.mesh->dimension();
		double val = 0;
		int i = 0;
		for (int v : active_nodes)
		{
			RowVectorNd cur_pos = state_.mesh_nodes->node_position(v) + state_.diff_cached[time_step_].u.block(v * dim, 0, dim, 1).transpose();
			val += (cur_pos - target_vertex_positions.row(i++)).squaredNorm();
		}
		return val;
	}

	Eigen::VectorXd NodeTargetObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd rhs;
		rhs.setZero(state.diff_cached[0].u.size());

		const int dim = state_.mesh->dimension();

		if (&state == &state_)
		{
			int i = 0;
			for (int v : active_nodes)
			{
				RowVectorNd cur_pos = state_.mesh_nodes->node_position(v) + state_.diff_cached[time_step_].u.block(v * dim, 0, dim, 1).transpose();

				rhs.segment(v * dim, dim) = 2 * (cur_pos - target_vertex_positions.row(i++));
			}
		}

		return rhs;
	}

	Eigen::VectorXd NodeTargetObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (param.name() == "shape")
			log_and_throw_error("Not implemented!");

		return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	MaterialBoundObjective::MaterialBoundObjective(const std::shared_ptr<const Parameter> elastic_param, const json &args) : elastic_param_(elastic_param), is_volume(elastic_param->get_state().mesh->is_volume())
	{
		volume_selections.clear();
		if (args["volume_selection"].is_array())
		{
			for (int body : args["volume_selection"])
				volume_selections.insert(body);
		}
		else if (args["volume_selection"].is_string())
		{
			Eigen::MatrixXi tmp;
			io::read_matrix(args["volume_selection"].get<std::string>(), tmp);
			for (int i = 0; i < tmp.size(); i++)
				volume_selections.insert(tmp(i));
		}

		for (const auto &arg : args["bounds"])
		{
			if (arg["type"] == "E")
			{
				min_E = arg["min"];
				max_E = arg["max"];
				kappa_E = arg["kappa"];
				dhat_E = arg["dhat"];
			}
			else if (arg["type"] == "nu")
			{
				min_nu = arg["min"];
				max_nu = arg["max"];
				kappa_nu = arg["kappa"];
				dhat_nu = arg["dhat"];
			}
			else if (arg["type"] == "lambda")
			{
				min_lambda = arg["min"];
				max_lambda = arg["max"];
				kappa_lambda = arg["kappa"];
				dhat_lambda = arg["dhat"];
			}
			else if (arg["type"] == "mu")
			{
				min_mu = arg["min"];
				max_mu = arg["max"];
				kappa_mu = arg["kappa"];
				dhat_mu = arg["dhat"];
			}
		}
	}

	double MaterialBoundObjective::value()
	{
		const auto &lambdas = elastic_param_->get_state().assembler.lame_params().lambda_mat_;
		const auto &mus = elastic_param_->get_state().assembler.lame_params().mu_mat_;

		double val = 0;
		for (int e = 0; e < lambdas.size(); e++)
		{
			const int body_id = elastic_param_->get_state().mesh->get_body_id(e);
			if (!volume_selections.empty() && volume_selections.count(body_id) == 0)
				continue;
			const double lambda = lambdas(e);
			const double mu = mus(e);
			const double E = convert_to_E(is_volume, lambda, mu);
			const double nu = convert_to_nu(is_volume, lambda, mu);

			if (kappa_E > 0 && dhat_E > 0)
			{
				val += barrier_func(E - min_E, dhat_E) * kappa_E;
				val += barrier_func(max_E - E, dhat_E) * kappa_E;
			}

			if (kappa_nu > 0 && dhat_nu > 0)
			{
				val += barrier_func(nu - min_nu, dhat_nu) * kappa_nu;
				val += barrier_func(max_nu - nu, dhat_nu) * kappa_nu;
			}

			if (kappa_mu > 0 && dhat_mu > 0)
			{
				val += barrier_func(mu - min_mu, dhat_mu) * kappa_mu;
				val += barrier_func(max_mu - mu, dhat_mu) * kappa_mu;
			}

			if (kappa_lambda > 0 && dhat_lambda > 0)
			{
				val += barrier_func(lambda - min_lambda, dhat_lambda) * kappa_lambda;
				val += barrier_func(max_lambda - lambda, dhat_lambda) * kappa_lambda;
			}
		}

		return val / lambdas.size();
	}

	Eigen::MatrixXd MaterialBoundObjective::compute_adjoint_rhs(const State &state)
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
	}

	Eigen::VectorXd MaterialBoundObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd grad;
		grad.setZero(param.full_dim());

		const auto &lambdas = elastic_param_->get_state().assembler.lame_params().lambda_mat_;
		const auto &mus = elastic_param_->get_state().assembler.lame_params().mu_mat_;

		assert(grad.size() == lambdas.size() + mus.size());
		for (int e = 0; e < lambdas.size(); e++)
		{
			const int body_id = elastic_param_->get_state().mesh->get_body_id(e);
			if (!volume_selections.empty() && volume_selections.count(body_id) == 0)
				continue;
			const double lambda = lambdas(e);
			const double mu = mus(e);
			const double E = convert_to_E(is_volume, lambda, mu);
			const double nu = convert_to_nu(is_volume, lambda, mu);
			Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(is_volume, E, nu);
			jacobian = jacobian.inverse().eval();

			if (kappa_E > 0 && dhat_E > 0)
			{
				double val = 0;
				val += barrier_func_derivative(E - min_E, dhat_E) * kappa_E;
				val += -barrier_func_derivative(max_E - E, dhat_E) * kappa_E;
				grad(e) += val * jacobian(0, 0);
				grad(e + lambdas.size()) += val * jacobian(0, 1);
			}

			if (kappa_nu > 0 && dhat_nu > 0)
			{
				double val = 0;
				val += barrier_func_derivative(nu - min_nu, dhat_nu) * kappa_nu;
				val += -barrier_func_derivative(max_nu - nu, dhat_nu) * kappa_nu;
				grad(e) += val * jacobian(1, 0);
				grad(e + lambdas.size()) += val * jacobian(1, 1);
			}

			if (kappa_mu > 0 && dhat_mu > 0)
			{
				grad(e + lambdas.size()) += barrier_func_derivative(mu - min_mu, dhat_mu) * kappa_mu;
				grad(e + lambdas.size()) += -barrier_func_derivative(max_mu - mu, dhat_mu) * kappa_mu;
			}

			if (kappa_lambda > 0 && dhat_lambda > 0)
			{
				grad(e) += barrier_func_derivative(lambda - min_lambda, dhat_lambda) * kappa_lambda;
				grad(e) += -barrier_func_derivative(max_lambda - lambda, dhat_lambda) * kappa_lambda;
			}
		}

		return param.map_grad(param_value, grad) / lambdas.size();
	}

	CollisionBarrierObjective::CollisionBarrierObjective(const std::shared_ptr<const Parameter> shape_param, const json &args) : shape_param_(shape_param)
	{
		if (!shape_param_)
			log_and_throw_error("CollisionBarrierObjective needs non-empty shape parameter!");
		else if (shape_param_->name() != "shape")
			log_and_throw_error("CollisionBarrierObjective wrong parameter type input!");

		const auto &state = shape_param_->get_state();

		state.build_collision_mesh(collision_mesh_, state.n_geom_bases, state.geom_bases());

		dhat = args["dhat"];
		broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	}

	void CollisionBarrierObjective::build_constraint_set(const Eigen::MatrixXd &displaced_surface)
	{
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		constraint_set.build(collision_mesh_, displaced_surface, dhat, 0, broad_phase_method);

		cached_displaced_surface = displaced_surface;
	}

	double CollisionBarrierObjective::value()
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(V);
		build_constraint_set(displaced_surface);

		return ipc::compute_barrier_potential(collision_mesh_, displaced_surface, constraint_set, dhat);
	}

	Eigen::MatrixXd CollisionBarrierObjective::compute_adjoint_rhs(const State &state)
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
	}

	Eigen::VectorXd CollisionBarrierObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param == shape_param_.get())
		{
			const auto &state = shape_param_->get_state();
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			state.get_vf(V, F);
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(V);
			build_constraint_set(displaced_surface);

			Eigen::VectorXd grad = ipc::compute_barrier_potential_gradient(collision_mesh_, displaced_surface, constraint_set, dhat);
			return param.map_grad(param_value, collision_mesh_.to_full_dof(grad));
		}
		else
			return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	bool CollisionBarrierObjective::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V_rest;
		Eigen::MatrixXi F;
		state.get_vf(V_rest, F);
		auto cast_shape_param = std::dynamic_pointer_cast<const ShapeParameter>(shape_param_);
		Eigen::MatrixXd V0, V1;
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x0), V_rest, V0);
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x1), V_rest, V1);

		// Skip CCD if the displacement is zero.
		if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
			return true;

		bool is_valid;
		is_valid = ipc::is_step_collision_free(
			collision_mesh_,
			collision_mesh_.vertices(V0),
			collision_mesh_.vertices(V1),
			broad_phase_method, 1e-6, 1000000);

		return is_valid;
	}

	double CollisionBarrierObjective::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V_rest;
		Eigen::MatrixXi F;
		state.get_vf(V_rest, F);
		auto cast_shape_param = std::dynamic_pointer_cast<const ShapeParameter>(shape_param_);
		Eigen::MatrixXd V0, V1;
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x0), V_rest, V0);
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x1), V_rest, V1);

		double max_step = 1;
		assert(!ShapeParameter::is_flipped(V0, F));
		while (ShapeParameter::is_flipped(V0 + max_step * (V1 - V0), F))
			max_step /= 2.;

		// Extract surface only
		V0 = collision_mesh_.vertices(V0);
		V1 = collision_mesh_.vertices(V1);

		auto Vmid = V0 + max_step * (V1 - V0);
		max_step *= ipc::compute_collision_free_stepsize(
			collision_mesh_, V0, Vmid,
			broad_phase_method, 1e-6, 1000000);
		// polyfem::logger().trace("best step {}", max_step);

		return max_step;
	}

	ControlSmoothingObjective::ControlSmoothingObjective(const std::shared_ptr<const Parameter> control_param, const json &args) : control_param_(control_param)
	{
		if (!control_param_)
			log_and_throw_error("ControlSmoothingObjective needs non-empty control parameter!");
		else if (control_param_->name() != "dirichlet")
			log_and_throw_error("ControlSmoothingObjective wrong parameter type!");
	}

	Eigen::VectorXd ControlSmoothingObjective::compute_adjoint_rhs_step(const State &state)
	{
		return Eigen::VectorXd::Zero(state.ndof());
	}

	double ControlSmoothingObjective::value()
	{
		if (time_step_ == 0)
			return 0.;
		const auto &state = control_param_->get_state();
		double dt = state.args["time"]["dt"];
		auto control_param = std::dynamic_pointer_cast<const ControlParameter>(control_param_);
		auto dirichlet_val_i = control_param->get_current_dirichlet(time_step_);
		auto dirichlet_val_i_prev = control_param->get_current_dirichlet(time_step_ - 1);
		// auto val = pow(((dirichlet_val_i - dirichlet_val_i_prev) / dt).array().pow(8).sum(), 1. / 8.);
		auto val = pow(((dirichlet_val_i - dirichlet_val_i_prev) / dt).array().pow(2).sum(), 1. / 2.);
		return val;
	}

	Eigen::VectorXd ControlSmoothingObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param == control_param_.get() && time_step_ != 0)
		{
			Eigen::VectorXd term;
			term.setZero(param.optimization_dim());
			const auto &state = control_param_->get_state();
			double dt = state.args["time"]["dt"];
			auto control_param = std::dynamic_pointer_cast<const ControlParameter>(control_param_);
			int timestep_dim = control_param->get_timestep_dim();
			auto dirichlet_val_i = control_param->get_current_dirichlet(time_step_);
			auto dirichlet_val_i_prev = control_param->get_current_dirichlet(time_step_ - 1);
			auto x = ((dirichlet_val_i - dirichlet_val_i_prev) / dt);
			auto y = x / dt / value();
			term.segment((time_step_ - 1) * timestep_dim, timestep_dim) = y;
			if (time_step_ > 1)
				term.segment((time_step_ - 2) * timestep_dim, timestep_dim) = -y;
			return term;
		}
		else
			return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	LayerThicknessObjective::LayerThicknessObjective(const std::shared_ptr<const Parameter> first_shape_param, const std::shared_ptr<const Parameter> second_shape_param, const json &args) : shape_param_(first_shape_param), adjacent_shape_param_(second_shape_param)
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);

		if (!shape_param_)
			log_and_throw_error("LayerThicknessObjective needs non-empty first shape parameter!");
		else if (shape_param_->name() != "shape")
			log_and_throw_error("LayerThicknessObjective wrong first parameter type input!");

		auto first_param = std::dynamic_pointer_cast<const ShapeParameter>(shape_param_);
		auto first_nodes = first_param->get_constrained_nodes();
		auto first_edges = get_boundary_edges(F, first_nodes);

		if (!adjacent_shape_param_)
			log_and_throw_error("LayerThicknessObjective needs non-empty second shape parameter!");
		else if (adjacent_shape_param_->name() != "shape")
			log_and_throw_error("LayerThicknessObjective wrong second parameter type input!");

		auto second_param = std::dynamic_pointer_cast<const ShapeParameter>(adjacent_shape_param_);
		auto second_nodes = second_param->get_constrained_nodes();
		auto second_edges = get_boundary_edges(F, second_nodes);

		boundary_node_ids_ = {};
		boundary_node_ids_.insert(boundary_node_ids_.end(), first_nodes.begin(), first_nodes.end());
		boundary_node_ids_.insert(boundary_node_ids_.end(), second_nodes.begin(), second_nodes.end());
		Eigen::MatrixXd layer_vertices = extract_boundaries(V, boundary_node_ids_);

		Eigen::MatrixXi layer_edges(first_edges.rows() + second_edges.rows(), 2);
		layer_edges.block(0, 0, first_edges.rows(), 2) = first_edges;
		second_edges.array() += first_nodes.size();
		layer_edges.block(first_edges.rows(), 0, second_edges.rows(), 2) = second_edges;

		io::OBJWriter::write("layer_thickness.obj", layer_vertices, layer_edges);
		collision_mesh_ = ipc::CollisionMesh(layer_vertices, layer_edges, Eigen::MatrixXi::Zero(0, 3));

		dmin = args["dmin"];
		dhat = args["dhat"];
		broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	}

	LayerThicknessObjective::LayerThicknessObjective(const std::shared_ptr<const Parameter> shape_param, const int adjacent_boundary_id, const json &args) : shape_param_(shape_param), adjacent_boundary_id_(adjacent_boundary_id)
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);

		if (!shape_param_)
			log_and_throw_error("LayerThicknessObjective needs non-empty shape parameter!");
		else if (shape_param_->name() != "shape")
			log_and_throw_error("LayerThicknessObjective wrong parameter type input!");

		auto first_param = std::dynamic_pointer_cast<const ShapeParameter>(shape_param_);
		auto first_nodes = first_param->get_constrained_nodes();
		auto first_edges = get_boundary_edges(F, first_nodes);

		std::vector<int> second_nodes;
		{
			const auto &mesh = state.mesh;
			const auto &gbases = state.geom_bases();

			for (const auto &lb : state.total_local_boundary)
			{
				const int e = lb.element_id();
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const int boundary_id = mesh->get_boundary_id(primitive_global_id);
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

					for (long n = 0; n < nodes.size(); ++n)
					{
						const int g_id = gbases[e].bases[nodes(n)].global()[0].index;
						if ((boundary_id == adjacent_boundary_id) && (std::find(second_nodes.begin(), second_nodes.end(), g_id) == second_nodes.end()))
							second_nodes.push_back(g_id);
					}
				}
			}
		}
		auto second_edges = get_boundary_edges(F, second_nodes);

		boundary_node_ids_ = {};
		boundary_node_ids_.insert(boundary_node_ids_.end(), first_nodes.begin(), first_nodes.end());
		boundary_node_ids_.insert(boundary_node_ids_.end(), second_nodes.begin(), second_nodes.end());
		Eigen::MatrixXd layer_vertices = extract_boundaries(V, boundary_node_ids_);

		Eigen::MatrixXi layer_edges(first_edges.rows() + second_edges.rows(), 2);
		layer_edges.block(0, 0, first_edges.rows(), 2) = first_edges;
		second_edges.array() += first_nodes.size();
		layer_edges.block(first_edges.rows(), 0, second_edges.rows(), 2) = second_edges;

		io::OBJWriter::write("layer_thickness.obj", layer_vertices, layer_edges);
		collision_mesh_ = ipc::CollisionMesh(layer_vertices, layer_edges, Eigen::MatrixXi::Zero(0, 3));

		dmin = args["dmin"];
		dhat = args["dhat"];
		broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	}

	void LayerThicknessObjective::build_constraint_set(const Eigen::MatrixXd &displaced_surface)
	{
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		constraint_set.build(collision_mesh_, displaced_surface, dhat, dmin, broad_phase_method);

		cached_displaced_surface = displaced_surface;
	}

	Eigen::MatrixXd LayerThicknessObjective::extract_boundaries(const Eigen::MatrixXd &V, const std::vector<int> &boundary_node_ids)
	{
		Eigen::MatrixXd V_boundary;
		V_boundary.setZero(boundary_node_ids.size(), V.cols());
		for (int i = 0; i < boundary_node_ids.size(); ++i)
			V_boundary.row(i) = V.row(boundary_node_ids[i]);
		return V_boundary;
	}

	Eigen::MatrixXi LayerThicknessObjective::get_boundary_edges(const Eigen::MatrixXi &F, const std::vector<int> &boundary_node_ids)
	{
		Eigen::MatrixXi boundary_edges;
		std::vector<std::vector<int>> adjacency_list;
		igl::adjacency_list(F, adjacency_list);
		std::vector<int> queue = {0};
		std::set<int> visited_nodes;
		while (!queue.empty())
		{
			auto node = *queue.begin();
			queue.erase(queue.begin());
			visited_nodes.insert(node);

			for (auto adj : adjacency_list[boundary_node_ids[node]])
			{
				auto result = std::find(boundary_node_ids.begin(), boundary_node_ids.end(), adj);
				if (result != boundary_node_ids.end())
				{
					int adj_loc = result - boundary_node_ids.begin();
					boundary_edges.conservativeResize(boundary_edges.rows() + 1, 2);
					boundary_edges(boundary_edges.rows() - 1, 0) = node;
					boundary_edges(boundary_edges.rows() - 1, 1) = adj_loc;
					if (visited_nodes.count(adj_loc) == 0)
						queue.push_back(adj_loc);
				}
			}
		}
		return boundary_edges;
	}

	double LayerThicknessObjective::value()
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);
		const Eigen::MatrixXd displaced_surface = extract_boundaries(V, boundary_node_ids_);
		build_constraint_set(displaced_surface);

		return ipc::compute_barrier_potential(collision_mesh_, displaced_surface, constraint_set, dhat);
	}

	Eigen::MatrixXd LayerThicknessObjective::compute_adjoint_rhs(const State &state)
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
	}

	Eigen::VectorXd LayerThicknessObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if ((&param == shape_param_.get()) || (&param == adjacent_shape_param_.get()))
		{
			const auto &state = (&param == shape_param_.get()) ? shape_param_->get_state() : adjacent_shape_param_->get_state();
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			state.get_vf(V, F);
			const Eigen::MatrixXd displaced_surface = extract_boundaries(V, boundary_node_ids_);
			build_constraint_set(displaced_surface);

			Eigen::VectorXd grad = ipc::compute_barrier_potential_gradient(collision_mesh_, displaced_surface, constraint_set, dhat);

			Eigen::VectorXd dV;
			dV.setZero(V.rows() * dim_);
			for (int i = 0; i < boundary_node_ids_.size(); ++i)
				for (int j = 0; j < dim_; ++j)
					dV(boundary_node_ids_[i] * dim_ + j) = grad(i * dim_ + j);
			return param.map_grad(param_value, dV);
		}
		else
			return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	bool LayerThicknessObjective::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V_rest;
		Eigen::MatrixXi F;
		state.get_vf(V_rest, F);
		auto cast_shape_param = std::dynamic_pointer_cast<const ShapeParameter>(shape_param_);
		Eigen::MatrixXd V0, V1;
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x0), V_rest, V0);
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x1), V_rest, V1);

		if (adjacent_shape_param_)
		{
			auto cast_adjacent_shape_param = std::dynamic_pointer_cast<const ShapeParameter>(adjacent_shape_param_);
			Eigen::MatrixXd tmp_V0, tmp_V1;
			cast_adjacent_shape_param->get_updated_nodes(cast_adjacent_shape_param->get_optimization_variable_part(x0), V0, tmp_V0);
			cast_adjacent_shape_param->get_updated_nodes(cast_adjacent_shape_param->get_optimization_variable_part(x1), V1, tmp_V1);
			V0 = tmp_V0;
			V1 = tmp_V1;
		}

		// Skip CCD if the displacement is zero.
		if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
			return true;

		bool is_valid;
		is_valid = ipc::is_step_collision_free(
			collision_mesh_,
			extract_boundaries(V0, boundary_node_ids_),
			extract_boundaries(V1, boundary_node_ids_),
			broad_phase_method, dmin);

		return is_valid;
	}

	double LayerThicknessObjective::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		const auto &state = shape_param_->get_state();
		Eigen::MatrixXd V_rest;
		Eigen::MatrixXi F;
		state.get_vf(V_rest, F);
		auto cast_shape_param = std::dynamic_pointer_cast<const ShapeParameter>(shape_param_);
		Eigen::MatrixXd V0, V1;
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x0), V_rest, V0);
		cast_shape_param->get_updated_nodes(cast_shape_param->get_optimization_variable_part(x1), V_rest, V1);

		if (adjacent_shape_param_)
		{
			auto cast_adjacent_shape_param = std::dynamic_pointer_cast<const ShapeParameter>(adjacent_shape_param_);
			Eigen::MatrixXd tmp_V0, tmp_V1;
			cast_adjacent_shape_param->get_updated_nodes(cast_adjacent_shape_param->get_optimization_variable_part(x0), V0, tmp_V0);
			cast_adjacent_shape_param->get_updated_nodes(cast_adjacent_shape_param->get_optimization_variable_part(x1), V1, tmp_V1);
			V0 = tmp_V0;
			V1 = tmp_V1;
		}

		double max_step = 1;
		assert(!ShapeParameter::is_flipped(V0, F));
		while (ShapeParameter::is_flipped(V0 + max_step * (V1 - V0), F))
			max_step /= 2.;

		// Extract surface only
		V0 = extract_boundaries(V0, boundary_node_ids_);
		V1 = extract_boundaries(V1, boundary_node_ids_);

		auto Vmid = V0 + max_step * (V1 - V0);
		max_step *= ipc::compute_collision_free_stepsize(
			collision_mesh_, V0, Vmid,
			broad_phase_method, dmin);
		// polyfem::logger().trace("best step {}", max_step);

		return max_step;
	}

} // namespace polyfem::solver