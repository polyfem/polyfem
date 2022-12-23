#include "Objective.hpp"

#include <polyfem/utils/CubicHermiteSplineParametrization.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include "HomoObjective.hpp"
#include "ControlParameter.hpp"

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
				Eigen::VectorXd knots;
				control_points.setZero(args["control_points"].size(), args["control_points"][0].size());
				for (int i = 0; i < args["control_points"].size(); ++i)
				{
					for (int j = 0; j < args["control_points"][i].size(); ++j)
						control_points(i, j) = args["control_points"][i][j].get<double>();
				}
				// tangents.setZero(args["tangents"].size(), args["tangents"][0].size());
				// for (int i = 0; i < args["tangents"].size(); ++i)
				// 	for (int j = 0; j < args["tangents"][i].size(); ++j)
				// 		tangents(i, j) = args["tangents"][i][j].get<double>();

				knots.setZero(args["knots"].size());
				for (int i = 0; i < args["knots"].size(); ++i)
					knots(i) = args["knots"][i].get<double>();

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
				target_obj->set_bspline_target(control_points, knots, delta(0));
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
			AdjointForm::compute_adjoint_term(state, adjoints, param.name(), term);
		}

		return term;
	}

	SpatialIntegralObjective::SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const json &args) : state_(state), shape_param_(shape_param), macro_strain_param_(macro_strain_param)
	{
		if (shape_param_)
			assert(shape_param_->name() == "shape");

		if (macro_strain_param_)
			assert(macro_strain_param_->name() == "macro_strain");
	}

	SpatialIntegralObjective::SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : state_(state), shape_param_(shape_param)
	{
		if (shape_param_)
			assert(shape_param_->name() == "shape");

		macro_strain_param_ = NULL;
	}

	double SpatialIntegralObjective::value()
	{
		assert(time_step_ < state_.diff_cached.size());
		return AdjointForm::integrate_objective(state_, get_integral_functional(), state_.diff_cached[time_step_].u, interested_ids_, spatial_integral_type_, time_step_);
	}

	Eigen::VectorXd SpatialIntegralObjective::compute_adjoint_rhs_step(const State &state)
	{
		if (&state != &state_)
			return Eigen::VectorXd::Zero(state.ndof());

		assert(time_step_ < state_.diff_cached.size());

		Eigen::VectorXd rhs;
		AdjointForm::dJ_du_step(state, get_integral_functional(), state.diff_cached[time_step_].u, interested_ids_, spatial_integral_type_, time_step_, rhs);

		return rhs;
	}

	Eigen::VectorXd SpatialIntegralObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		if (&param == shape_param_.get())
		{
			assert(time_step_ < state_.diff_cached.size());
			AdjointForm::compute_shape_derivative_functional_term(state_, state_.diff_cached[time_step_].u, get_integral_functional(), interested_ids_, spatial_integral_type_, term, time_step_);
		}
		else if (&param == macro_strain_param_.get())
		{
			AdjointForm::compute_macro_strain_derivative_functional_term(state_, state_.diff_cached[time_step_].u, get_integral_functional(), interested_ids_, spatial_integral_type_, term, time_step_);
		}
		else
			term.setZero(param.full_dim());

		return param.map_grad(param_value, term);
	}

	StressObjective::StressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args, bool has_integral_sqrt) : SpatialIntegralObjective(state, shape_param, args), elastic_param_(elastic_param)
	{
		spatial_integral_type_ = AdjointForm::SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		formulation_ = state.formulation();
		in_power_ = args["power"];
		out_sqrt_ = has_integral_sqrt;
	}

	IntegrableFunctional StressObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (this->formulation_ == "Laplacian")
				{
					stress = grad_u.row(q);
				}
				else if (this->formulation_ == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (this->formulation_ == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val(q) = pow(stress.squaredNorm(), this->in_power_ / 2.);
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			const int actual_dim = (this->formulation_ == "Laplacian") ? 1 : dim;
			Eigen::MatrixXd grad_u_q, stress, stress_dstress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (this->formulation_ == "Laplacian")
				{
					stress = grad_u.row(q);
					stress_dstress = 2 * stress;
				}
				else if (this->formulation_ == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
				}
				else if (this->formulation_ == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
					stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
				}
				else
					logger().error("Unknown formulation!");

				const double coef = this->in_power_ * pow(stress.squaredNorm(), this->in_power_ / 2. - 1.);
				for (int i = 0; i < actual_dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = coef * stress_dstress(i, l);
			}
		});

		return j;
	}

	double StressObjective::value()
	{
		double val = SpatialIntegralObjective::value();
		if (out_sqrt_)
			return pow(val, 1. / in_power_);
		else
			return val;
	}

	Eigen::VectorXd StressObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd rhs = SpatialIntegralObjective::compute_adjoint_rhs_step(state);

		if (out_sqrt_)
		{
			double val = SpatialIntegralObjective::value();
			if (std::abs(val) < 1e-12)
				logger().warn("stress integral too small, may result in NAN grad!");
			return (pow(val, 1. / in_power_ - 1) / in_power_) * rhs;
		}
		else
			return rhs;
	}

	Eigen::VectorXd StressObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		if (&param == elastic_param_.get())
		{
			// TODO: differentiate stress wrt. lame param
			log_and_throw_error("Not implemented!");
		}
		else if (&param == shape_param_.get())
		{
			term = SpatialIntegralObjective::compute_partial_gradient(param, param_value);
		}
		else
			term.setZero(param.optimization_dim());

		if (out_sqrt_)
		{
			double val = SpatialIntegralObjective::value();
			if (std::abs(val) < 1e-12)
				logger().warn("stress integral too small, may result in NAN grad!");
			term *= (pow(val, 1. / in_power_ - 1) / in_power_);
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
		Eigen::MatrixXd boundary_nodes_pos;
		state_.build_collision_mesh(boundary_nodes_pos, collision_mesh, state_.n_geom_bases, state_.geom_bases());
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
		Eigen::MatrixXd boundary_nodes_pos;
		state_.build_collision_mesh(boundary_nodes_pos, collision_mesh, state_.n_geom_bases, state_.geom_bases());
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

		return AdjointForm::integrate_objective(state, j, Eigen::MatrixXd::Zero(state.ndof(), 1), interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, 0);
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
			AdjointForm::compute_shape_derivative_functional_term(state, Eigen::MatrixXd::Zero(state.ndof(), 1), j, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, term, 0);
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

	PositionObjective::PositionObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
	{
		spatial_integral_type_ = AdjointForm::SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
	}

	IntegrableFunctional PositionObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = u.col(this->dim_) + pts.col(this->dim_);
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			val.col(this->dim_).setOnes();
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			val.col(this->dim_).setOnes();
		});

		return j;
	}

	Eigen::MatrixXd StaticObjective::compute_adjoint_rhs(const State &state)
	{
		Eigen::MatrixXd term(state.ndof(), state.diff_cached.size());
		term.col(time_step_) = compute_adjoint_rhs_step(state);

		return term;
	}

	BarycenterTargetObjective::BarycenterTargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args, const Eigen::MatrixXd &target)
	{
		dim_ = state.mesh->dimension();
		target_ = target;

		objv = std::make_shared<VolumeObjective>(shape_param, args);
		objp.resize(dim_);
		for (int d = 0; d < dim_; d++)
		{
			objp[d] = std::make_shared<PositionObjective>(state, shape_param, args);
			objp[d]->set_dim(d);
		}
	}

	Eigen::VectorXd BarycenterTargetObjective::get_target() const
	{
		assert(target_.cols() == dim_);
		if (target_.rows() > 1)
			return target_.row(get_time_step());
		else
			return target_;
	}

	void BarycenterTargetObjective::set_time_step(int time_step)
	{
		StaticObjective::set_time_step(time_step);
		for (auto &obj : objp)
			obj->set_time_step(time_step);
	}

	double BarycenterTargetObjective::value()
	{
		return (get_barycenter() - get_target()).squaredNorm();
	}
	Eigen::VectorXd BarycenterTargetObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		term.setZero(param.optimization_dim());

		Eigen::VectorXd target = get_target();

		const double volume = objv->value();
		Eigen::VectorXd center(dim_);
		for (int d = 0; d < dim_; d++)
			center(d) = objp[d]->value() / volume;

		double coeffv = 0;
		for (int d = 0; d < dim_; d++)
			coeffv += 2 * (center(d) - target(d)) * (-center(d) / volume);

		term += coeffv * objv->compute_partial_gradient(param, param_value);

		for (int d = 0; d < dim_; d++)
			term += (2.0 / volume * (center(d) - target(d))) * objp[d]->compute_partial_gradient(param, param_value);

		return term;
	}
	Eigen::VectorXd BarycenterTargetObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd term;
		term.setZero(state.ndof());

		Eigen::VectorXd target = get_target();

		const double volume = objv->value();
		Eigen::VectorXd center(dim_);
		for (int d = 0; d < dim_; d++)
			center(d) = objp[d]->value() / volume;

		for (int d = 0; d < dim_; d++)
			term += (2.0 / volume * (center(d) - target(d))) * objp[d]->compute_adjoint_rhs_step(state);

		return term;
	}

	Eigen::VectorXd BarycenterTargetObjective::get_barycenter() const
	{
		const double volume = objv->value();
		Eigen::VectorXd center(dim_);
		for (int d = 0; d < dim_; d++)
			center(d) = objp[d]->value() / volume;

		return center;
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

	ComplianceObjective::ComplianceObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args) : SpatialIntegralObjective(state, shape_param, args), elastic_param_(elastic_param)
	{
		if (elastic_param_)
			assert(elastic_param_->name() == "material");
		spatial_integral_type_ = AdjointForm::SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		formulation_ = state.formulation();
	}

	IntegrableFunctional ComplianceObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				if (this->formulation_ == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else
					logger().error("Unknown formulation!");
				val(q) = (stress.array() * grad_u_q.array()).sum();
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());

				for (int i = 0; i < dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = 2 * stress(i, l);
			}
		});

		return j;
	}

	Eigen::VectorXd ComplianceObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param == shape_param_.get())
			return compute_partial_gradient(param, param_value);
		else if (&param == elastic_param_.get())
		{
			Eigen::VectorXd term;
			term.setZero(param.full_dim());

			const auto &bases = state_.bases;
			const auto &gbases = state_.geom_bases();
			auto df_dmu_dlambda_function = state_.assembler.get_dstress_dmu_dlambda_function(formulation_);
			const int dim = state_.mesh->dimension();

			for (int e = 0; e < bases.size(); e++)
			{
				assembler::ElementAssemblyValues vals;
				state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), bases[e], gbases[e], vals);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd u, grad_u;
				io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, state_.diff_cached[time_step_].u, u, grad_u);

				Eigen::MatrixXd grad_u_q;
				for (int q = 0; q < quadrature.weights.size(); q++)
				{
					double lambda, mu;
					state_.assembler.lame_params().lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

					vector2matrix(grad_u.row(q), grad_u_q);

					Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
					df_dmu_dlambda_function(e, quadrature.points.row(q), vals.val.row(q), grad_u_q, f_prime_dmu, f_prime_dlambda);

					term(e + bases.size()) += dot(f_prime_dmu, grad_u_q) * da(q);
					term(e) += dot(f_prime_dlambda, grad_u_q) * da(q);
				}
			}

			return param.map_grad(param_value, term);
		}
		else
			return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	IntegrableFunctional TargetObjective::get_integral_functional()
	{
		assert(target_state_);
		assert(target_state_->diff_cached.size() > 0);

		IntegrableFunctional j;

		auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			const int e = params["elem"];
			int e_ref;
			if (auto search = e_to_ref_e_.find(e); search != e_to_ref_e_.end())
				e_ref = search->second;
			else
				e_ref = e;
			const auto &gbase_ref = target_state_->geom_bases()[e_ref];

			Eigen::MatrixXd pts_ref;
			gbase_ref.eval_geom_mapping(local_pts, pts_ref);

			Eigen::MatrixXd u_ref, grad_u_ref;
			const Eigen::MatrixXd &sol_ref = target_state_->problem->is_time_dependent() ? target_state_->diff_cached[params["step"].get<int>()].u : target_state_->diff_cached[0].u;
			io::Evaluator::interpolate_at_local_vals(*(target_state_->mesh), target_state_->problem->is_scalar(), target_state_->bases, target_state_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

			for (int q = 0; q < u.rows(); q++)
			{
				val(q) = ((u_ref.row(q) + pts_ref.row(q)) - (u.row(q) + pts.row(q))).squaredNorm();
			}
		};

		auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			const int e = params["elem"];
			int e_ref;
			if (auto search = e_to_ref_e_.find(e); search != e_to_ref_e_.end())
				e_ref = search->second;
			else
				e_ref = e;
			const auto &gbase_ref = target_state_->geom_bases()[e_ref];

			Eigen::MatrixXd pts_ref;
			gbase_ref.eval_geom_mapping(local_pts, pts_ref);

			Eigen::MatrixXd u_ref, grad_u_ref;
			const Eigen::MatrixXd &sol_ref = target_state_->problem->is_time_dependent() ? target_state_->diff_cached[params["step"].get<int>()].u : target_state_->diff_cached[0].u;
			io::Evaluator::interpolate_at_local_vals(*(target_state_->mesh), target_state_->problem->is_scalar(), target_state_->bases, target_state_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

			for (int q = 0; q < u.rows(); q++)
			{
				auto x = (u.row(q) + pts.row(q)) - (u_ref.row(q) + pts_ref.row(q));
				val.row(q) = 2 * x;
			}
		};

		j.set_j(j_func);
		j.set_dj_du(djdu_func);
		j.set_dj_dx(djdu_func); // only used for shape derivative

		return j;
	}

	void TargetObjective::set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids)
	{
		target_state_ = target_state;

		std::map<int, std::vector<int>> ref_interested_body_id_to_e;
		int ref_count = 0;
		for (int e = 0; e < target_state_->bases.size(); ++e)
		{
			int body_id = target_state_->mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (ref_interested_body_id_to_e.find(body_id) != ref_interested_body_id_to_e.end())
				ref_interested_body_id_to_e[body_id].push_back(e);
			else
				ref_interested_body_id_to_e[body_id] = {e};
			ref_count++;
		}

		std::map<int, std::vector<int>> interested_body_id_to_e;
		int count = 0;
		for (int e = 0; e < state_.bases.size(); ++e)
		{
			int body_id = state_.mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (interested_body_id_to_e.find(body_id) != interested_body_id_to_e.end())
				interested_body_id_to_e[body_id].push_back(e);
			else
				interested_body_id_to_e[body_id] = {e};
			count++;
		}

		if (count != ref_count)
			logger().error("Number of interested elements in the reference and optimization examples do not match! {} {}", count, ref_count);
		else
			logger().trace("Found {} matching elements.", count);

		for (const auto &kv : interested_body_id_to_e)
		{
			for (int i = 0; i < kv.second.size(); ++i)
			{
				e_to_ref_e_[kv.second[i]] = ref_interested_body_id_to_e[kv.first][i];
			}
		}
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

	void SDFTargetObjective::compute_distance(const Eigen::MatrixXd &point, double &distance)
	{
		distance = DBL_MAX;
		Eigen::MatrixXd p = point.transpose();

		for (int i = 0; i < t_sampling.size() - 1; ++i)
		{
			const double l = (point_sampling.row(i + 1) - point_sampling.row(i)).norm();
			double distance_to_perpendicular = ((p - point_sampling.row(i)) * (point_sampling.row(i + 1) - point_sampling.row(i)).transpose())(0) / l;
			const float t = std::max(0., std::min(1., distance_to_perpendicular));
			const auto project = point_sampling.row(i) * (1 - t) + point_sampling.row(i + 1) * t;
			const double project_distance = (p - project).norm();
			if (project_distance < distance)
			{
				distance = project_distance;
			}
		}
	}

	void SDFTargetObjective::compute_distances(const Eigen::MatrixXd &point, Eigen::VectorXd &distances, double h = 1e-6)
	{
		std::vector<double> eps = {-delta_, 0, delta_};
		Eigen::MatrixXi bin(dim, 1);
		for (int k = 0; k < dim; ++k)
			bin(k) = (int)std::floor(point(k) / delta_);
		if (point.size() == 2)
		{
			distances.setZero(eps.size() * eps.size());
			int index = 0;
			for (int i = 0; i < eps.size(); ++i)
				for (int j = 0; j < eps.size(); ++j)
				{
					Eigen::MatrixXd new_point = point;
					new_point(0) += eps[i];
					new_point(1) += eps[j];
					double distance;
					compute_distance(new_point, distance);
					distances(index++) = distance;
				}
		}
		else if (point.size() == 3)
		{
			distances.setZero(eps.size() * eps.size() * eps.size());
			int index = 0;
			for (int i = 0; i < eps.size(); ++i)
				for (int j = 0; j < eps.size(); ++j)
					for (int l = 0; l < eps.size(); ++l)
					{
						Eigen::MatrixXd new_point = point;
						new_point(0) += eps[i];
						new_point(1) += eps[j];
						new_point(2) += eps[l];
						double distance;
						compute_distance(new_point, distance);
						distances(index++) = distance;
					}
		}
	}

	void SDFTargetObjective::bicubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad, const double h = 1e-6)
	{
		Eigen::MatrixXd corner_val(4, 1);
		Eigen::MatrixXd corner_grad(4, 2);
		Eigen::MatrixXd corner_grad_grad(4, 1);
		for (int i = 0; i < 4; ++i)
		{
			{
				std::shared_lock distance_lock(distance_mutex_);
				corner_val(i) = implicit_function_distance.at(keys[i]);
			}
			{
				std::shared_lock grad_lock(grad_mutex_);
				auto mixed_grads = implicit_function_grads.at(keys[i]);
				corner_grad(i, 0) = mixed_grads(0);
				corner_grad(i, 1) = mixed_grads(1);
				corner_grad_grad(i, 0) = mixed_grads(2);
			}
		}

		Eigen::MatrixXd x(16, 1);
		x << corner_val(0), corner_val(1), corner_val(2), corner_val(3),
			delta_ * corner_grad(0, 0), delta_ * corner_grad(1, 0), delta_ * corner_grad(2, 0), delta_ * corner_grad(3, 0),
			delta_ * corner_grad(0, 1), delta_ * corner_grad(1, 1), delta_ * corner_grad(2, 1), delta_ * corner_grad(3, 1),
			delta_ * delta_ * corner_grad_grad(0, 0), delta_ * delta_ * corner_grad_grad(1, 0), delta_ * delta_ * corner_grad_grad(2, 0), delta_ * delta_ * corner_grad_grad(3, 0);

		Eigen::MatrixXd coeffs = cubic_mat * x;

		auto bar_x = [&corner_point](double x_) { return (x_ - corner_point(0, 0)) / (corner_point(1, 0) - corner_point(0, 0)); };
		auto bar_y = [&corner_point](double y_) { return (y_ - corner_point(0, 1)) / (corner_point(2, 1) - corner_point(0, 1)); };

		val = 0;
		grad.setZero(2, 1);
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
			{
				val += coeffs(i + j * 4) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j);
				grad(0) += i == 0 ? 0 : (coeffs(i + j * 4) * i * pow(bar_x(point(0)), i - 1) * pow(bar_y(point(1)), j));
				grad(1) += j == 0 ? 0 : coeffs(i + j * 4) * pow(bar_x(point(0)), i) * j * pow(bar_y(point(1)), j - 1);
			}

		grad(0) /= (corner_point(1, 0) - corner_point(0, 0));
		grad(1) /= (corner_point(2, 1) - corner_point(0, 1));

		assert(!std::isnan(grad(0)) && !std::isnan(grad(1)));
	}

	void SDFTargetObjective::tricubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad)
	{
		Eigen::MatrixXd corner_val(8, 1);
		Eigen::MatrixXd corner_grad(8, 3);
		Eigen::MatrixXd corner_grad_grad(8, 3);
		Eigen::MatrixXd corner_grad_grad_grad(8, 1);
		for (int i = 0; i < 4; ++i)
		{
			{
				std::shared_lock distance_lock(distance_mutex_);
				corner_val(i) = implicit_function_distance.at(keys[i]);
			}
			{
				std::shared_lock grad_lock(grad_mutex_);
				auto mixed_grads = implicit_function_grads.at(keys[i]);
				corner_grad(i, 0) = mixed_grads(0);
				corner_grad(i, 1) = mixed_grads(1);
				corner_grad(i, 2) = mixed_grads(2);
				corner_grad_grad(i, 0) = mixed_grads(3);
				corner_grad_grad(i, 1) = mixed_grads(4);
				corner_grad_grad(i, 2) = mixed_grads(5);
				corner_grad_grad_grad(i, 0) = mixed_grads(6);
			}
		}
		Eigen::MatrixXd x(64, 1);
		x << corner_val(0), corner_val(1), corner_val(2), corner_val(3), corner_val(4), corner_val(5), corner_val(6), corner_val(7),
			delta_ * corner_grad(0, 0), delta_ * corner_grad(1, 0), delta_ * corner_grad(2, 0), delta_ * corner_grad(3, 0),
			delta_ * corner_grad(0, 1), delta_ * corner_grad(1, 1), delta_ * corner_grad(2, 1), delta_ * corner_grad(3, 1),
			delta_ * corner_grad(0, 2), delta_ * corner_grad(1, 2), delta_ * corner_grad(2, 2), delta_ * corner_grad(3, 2),
			delta_ * delta_ * corner_grad_grad(0, 0), delta_ * delta_ * corner_grad_grad(1, 0), delta_ * delta_ * corner_grad_grad(2, 0), delta_ * delta_ * corner_grad_grad(3, 0),
			delta_ * delta_ * corner_grad_grad(0, 1), delta_ * delta_ * corner_grad_grad(1, 1), delta_ * delta_ * corner_grad_grad(2, 1), delta_ * delta_ * corner_grad_grad(3, 1),
			delta_ * delta_ * corner_grad_grad(0, 2), delta_ * delta_ * corner_grad_grad(1, 2), delta_ * delta_ * corner_grad_grad(2, 2), delta_ * delta_ * corner_grad_grad(3, 2),
			delta_ * delta_ * delta_ * corner_grad_grad_grad(0, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(1, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(2, 0), delta_ * delta_ * delta_ * corner_grad_grad_grad(3, 0);

		Eigen::MatrixXd coeffs = cubic_mat * x;

		auto bar_x = [&corner_point](double x_) { return (x_ - corner_point(0, 0)) / (corner_point(1, 0) - corner_point(0, 0)); };
		auto bar_y = [&corner_point](double y_) { return (y_ - corner_point(0, 1)) / (corner_point(2, 1) - corner_point(0, 1)); };
		auto bar_z = [&corner_point](double z_) { return (z_ - corner_point(0, 2)) / (corner_point(4, 2) - corner_point(0, 2)); };

		val = 0;
		grad.setZero(3, 1);
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int l = 0; l < 4; ++l)
				{
					val += coeffs(i + j * 4 + l * 16) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j) * pow(bar_z(point(2)), l);
					grad(0) += i == 0 ? 0 : (coeffs(i + j * 4 + l * 16) * i * pow(bar_x(point(0)), i - 1) * pow(bar_y(point(1)), j)) * pow(bar_z(point(2)), l);
					grad(1) += j == 0 ? 0 : coeffs(i + j * 4 + l * 16) * pow(bar_x(point(0)), i) * j * pow(bar_y(point(1)), j - 1) * pow(bar_z(point(2)), l);
					grad(2) += l == 0 ? 0 : coeffs(i + j * 4 + l * 16) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j) * l * pow(bar_z(point(2)), l - 1);
				}

		grad(0) /= (corner_point(1, 0) - corner_point(0, 0));
		grad(1) /= (corner_point(2, 1) - corner_point(0, 1));
		grad(2) /= (corner_point(4, 2) - corner_point(0, 2));

		assert(!std::isnan(grad(0)) && !std::isnan(grad(1)) && !std::isnan(grad(2)));
	}

	void SDFTargetObjective::evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad)
	{
		grad.setZero(dim, 1);
		int num_corner_points = dim == 2 ? 4 : 8;
		Eigen::MatrixXi bin(dim, 1);
		Eigen::MatrixXi keys(num_corner_points, dim);
		std::vector<std::string> keys_string;
		for (int k = 0; k < dim; ++k)
			bin(k) = (int)std::floor(point(k) / delta_);
		if (dim == 2)
		{
			keys << bin(0), bin(1),
				bin(0) + 1, bin(1),
				bin(0), bin(1) + 1,
				bin(0) + 1, bin(1) + 1;
		}
		else if (dim == 3)
		{
			keys << bin(0), bin(1), bin(2),
				bin(0) + 1, bin(1), bin(2),
				bin(0), bin(1) + 1, bin(2),
				bin(0) + 1, bin(1) + 1, bin(2),
				bin(0), bin(1), bin(2) + 1,
				bin(0) + 1, bin(1), bin(2) + 1,
				bin(0), bin(1) + 1, bin(2) + 1,
				bin(0) + 1, bin(1) + 1, bin(2) + 1;
		}
		auto safe_compute_distance = [this](const Eigen::MatrixXd &clamped_point, const std::string &key_string) {
			std::unique_lock lock(distance_mutex_);
			if (implicit_function_distance.count(key_string) == 0)
				compute_distance(clamped_point, implicit_function_distance[key_string]);
		};
		auto setup_key = [this](const Eigen::VectorXi &key, std::string &key_string, Eigen::MatrixXd &clamped_point) {
			key_string = "";
			clamped_point.setZero(dim, 1);
			for (int k = 0; k < dim; ++k)
			{
				key_string += fmt::format("{:d},", key(k));
				clamped_point(k) = (double)key(k) * delta_;
			}
		};
		auto safe_distance = [this, setup_key, safe_compute_distance](const Eigen::VectorXi &key) {
			Eigen::MatrixXd point;
			std::string key_string;
			double distance;
			setup_key(key, key_string, point);
			safe_compute_distance(point, key_string);
			{
				std::shared_lock lock(distance_mutex_);
				distance = implicit_function_distance[key_string];
			}
			return distance;
		};
		auto centered_fd = [this, safe_distance](const Eigen::VectorXi &key, const int k) {
			Eigen::MatrixXi key_plus, key_minus;
			double distance_plus, distance_minus;
			key_plus = key;
			key_plus(k) += 1;
			distance_plus = safe_distance(key_plus);
			key_minus = key;
			key_minus(k) -= 1;
			distance_minus = safe_distance(key_minus);
			return (1. / 2. / delta_) * (distance_plus - distance_minus);
		};
		auto centered_mixed_fd = [this, centered_fd](const Eigen::VectorXi &key, const int k1, const int k2) {
			Eigen::VectorXi key_plus, key_minus;
			key_plus = key;
			key_plus(k1) += 1;
			key_minus = key;
			key_minus(k1) -= 1;
			return (1. / 2. / delta_) * (centered_fd(key_plus, k2) - centered_fd(key_minus, k2));
		};
		auto centered_mixed_fd_3d = [this, centered_mixed_fd](const Eigen::VectorXi &key) {
			Eigen::VectorXi key_plus, key_minus;
			key_plus = key;
			key_plus(0) += 1;
			key_minus = key;
			key_minus(0) -= 1;
			return (1. / 2. / delta_) * (centered_mixed_fd(key_plus, 1, 2) - centered_mixed_fd(key_minus, 1, 2));
		};
		auto compute_grad = [this, centered_fd, centered_mixed_fd, centered_mixed_fd_3d](const Eigen::VectorXi &key) {
			Eigen::VectorXd mixed_grads(dim == 2 ? 3 : 7);
			if (dim == 2)
			{
				mixed_grads(0) = centered_fd(key, 0);
				mixed_grads(1) = centered_fd(key, 1);
				mixed_grads(2) = centered_mixed_fd(key, 0, 1);
			}
			else if (dim == 3)
			{
				mixed_grads(0) = centered_fd(key, 0);
				mixed_grads(1) = centered_fd(key, 1);
				mixed_grads(2) = centered_fd(key, 2);
				mixed_grads(3) = centered_mixed_fd(key, 0, 1);
				mixed_grads(4) = centered_mixed_fd(key, 0, 2);
				mixed_grads(5) = centered_mixed_fd(key, 1, 2);
				mixed_grads(6) = centered_mixed_fd_3d(key);
			}
			return mixed_grads;
		};
		auto safe_compute_grad = [this, setup_key, compute_grad](const Eigen::VectorXi &key) {
			std::string key_string;
			Eigen::MatrixXd point;
			setup_key(key, key_string, point);
			{
				std::unique_lock lock(grad_mutex_);
				if (implicit_function_grads.count(key_string) == 0)
				{
					Eigen::MatrixXd grad = compute_grad(key);
					implicit_function_grads[key_string] = grad;
				}
			}
		};
		Eigen::MatrixXd corner_point(num_corner_points, dim);
		for (int i = 0; i < num_corner_points; ++i)
		{
			std::string key_string;
			Eigen::MatrixXd clamped_point;
			setup_key(keys.row(i), key_string, clamped_point);
			safe_compute_distance(clamped_point, key_string);
			keys_string.push_back(key_string);
			corner_point.row(i) = clamped_point.transpose();
			safe_compute_grad(keys.row(i));
		}

		bicubic_interpolation(corner_point, keys_string, point, val, grad);

		for (int i = 0; i < dim; ++i)
			if (std::isnan(grad(i)))
			{
				logger().error("Nan found in gradient computation.");
				break;
			}
	}

	IntegrableFunctional SDFTargetObjective::get_integral_functional()
	{
		IntegrableFunctional j;
		auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);

			for (int q = 0; q < u.rows(); q++)
			{
				double distance;
				Eigen::MatrixXd unused_grad;
				evaluate(u.row(q) + pts.row(q), distance, unused_grad);
				val(q) = pow(distance, 2);
			}
		};

		auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());

			for (int q = 0; q < u.rows(); q++)
			{
				double distance;
				Eigen::MatrixXd grad;
				evaluate(u.row(q) + pts.row(q), distance, grad);
				val.row(q) = 2 * distance * grad.transpose();
			}
		};

		j.set_j(j_func);
		j.set_dj_du(djdu_func);
		j.set_dj_dx(djdu_func);

		return j;
	}

	MaterialBoundObjective::MaterialBoundObjective(const std::shared_ptr<const Parameter> elastic_param, const json &args) : elastic_param_(elastic_param), is_volume(elastic_param->get_state().mesh->is_volume())
	{
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

		Eigen::MatrixXd boundary_nodes_pos_;
		state.build_collision_mesh(boundary_nodes_pos_, collision_mesh_, state.n_geom_bases, state.geom_bases());

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

} // namespace polyfem::solver