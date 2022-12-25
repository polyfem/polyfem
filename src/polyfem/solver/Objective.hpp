#pragma once

#include "AdjointForm.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>

// #include <polyfem/utils/CubicInterpolationMatrices.hpp>

#include <nanospline/BSpline.h>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <shared_mutex>
#include <array>

namespace polyfem::solver
{
	class Objective
	{
	public:
		Objective() = default;
		virtual ~Objective() = default;

		static std::shared_ptr<Objective> create(const json &args, const std::string &root_path, const std::vector<std::shared_ptr<Parameter>> &parameters, const std::vector<std::shared_ptr<State>> &states);

		virtual double value() = 0;
		Eigen::VectorXd gradient(const std::vector<std::shared_ptr<State>> &states, const std::vector<Eigen::MatrixXd> &adjoints, const Parameter &param, const Eigen::VectorXd &param_value)
		{
			Eigen::VectorXd adjoint_term;
			adjoint_term.setZero(param.full_dim());
			int i = 0;
			for (const auto &state : states)
				adjoint_term += compute_adjoint_term(*state, adjoints[i++], param);

			return compute_partial_gradient(param, param_value) + param.map_grad(param_value, adjoint_term);
		}

		// use only if there's only one state
		Eigen::VectorXd gradient(const State &state, const Eigen::MatrixXd &adjoints, const Parameter &param, const Eigen::VectorXd &param_value)
		{
			return compute_partial_gradient(param, param_value) + param.map_grad(param_value, compute_adjoint_term(state, adjoints, param));
		}

		virtual Eigen::MatrixXd compute_adjoint_rhs(const State &state) = 0; // compute $\partial_u J$

		virtual Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) = 0; // compute $\partial_q J$
		static Eigen::VectorXd compute_adjoint_term(const State &state, const Eigen::MatrixXd &adjoints, const Parameter &param);
	};

	// this objective either depends on solution in one time step, or one static solution
	class StaticObjective : public Objective
	{
	public:
		StaticObjective() = default;
		virtual ~StaticObjective() = default;

		virtual void set_time_step(int time_step) { time_step_ = time_step; }
		int get_time_step() const { return time_step_; }

		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		virtual Eigen::VectorXd compute_adjoint_rhs_step(const State &state) = 0;

	protected:
		int time_step_ = 0; // time step to integrate
	};

	class SpatialIntegralObjective : public StaticObjective
	{
	public:
		SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args);
		SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const json &args);
		virtual ~SpatialIntegralObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

		virtual IntegrableFunctional get_integral_functional() = 0;

		const State &get_state() { return state_; }

	protected:
		const State &state_;
		std::shared_ptr<const Parameter> shape_param_;
		std::shared_ptr<const Parameter> macro_strain_param_;
		AdjointForm::SpatialIntegralType spatial_integral_type_;
		std::set<int> interested_ids_;
	};

	class StressObjective : public SpatialIntegralObjective
	{
	public:
		StressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args, bool has_integral_sqrt = true);
		~StressObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		int in_power_;
		bool out_sqrt_;
		std::string formulation_;

		std::shared_ptr<const Parameter> elastic_param_; // stress depends on elastic param
	};

	class SumObjective : public Objective
	{
	public:
		SumObjective(const std::vector<std::shared_ptr<Objective>> &objs) : objs_(objs)
		{
			weights_.setOnes(objs_.size());
		}
		SumObjective(const std::vector<std::shared_ptr<Objective>> &objs, const Eigen::VectorXd &weights) : objs_(objs), weights_(weights) {}
		~SumObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

		int n_objs() const { return objs_.size(); }
		std::shared_ptr<Objective> get_obj(const int i) const { return objs_[i]; }
		double get_weight(const int i) const { return weights_[i]; }

	protected:
		std::vector<std::shared_ptr<Objective>> objs_;
		Eigen::VectorXd weights_;
	};

	// note: active nodes are selected by surface selection on the first state in shape_param
	class BoundarySmoothingObjective : public Objective
	{
	public:
		BoundarySmoothingObjective(const std::shared_ptr<const Parameter> shape_param, const json &args);
		~BoundarySmoothingObjective() = default;

		void init();

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::shared_ptr<const Parameter> shape_param_;

		const json args_;

		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

	class DeformedBoundarySmoothingObjective : public Objective
	{
	public:
		DeformedBoundarySmoothingObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args);
		~DeformedBoundarySmoothingObjective() = default;

		void init();

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		const State &state_;
		std::shared_ptr<const Parameter> shape_param_;
		const json args_;

		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
	};

	class VolumeObjective : public Objective
	{
	public:
		VolumeObjective(const std::shared_ptr<const Parameter> shape_param, const json &args);
		~VolumeObjective() = default;

		void set_weights(const Eigen::VectorXd &weights) { weights_ = weights; }

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::shared_ptr<const Parameter> shape_param_;
		std::set<int> interested_ids_;
		Eigen::VectorXd weights_;
	};

	class VolumePenaltyObjective : public Objective
	{
	public:
		VolumePenaltyObjective(const std::shared_ptr<const Parameter> shape_param, const json &args);
		~VolumePenaltyObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::shared_ptr<VolumeObjective> obj;
		Eigen::Vector2d bound;
	};

	class PositionObjective : public SpatialIntegralObjective
	{
	public:
		PositionObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args);
		~PositionObjective() = default;

		void set_dim(const int dim) { dim_ = dim; }
		void set_integral_type(const AdjointForm::SpatialIntegralType type) { spatial_integral_type_ = type; }

		IntegrableFunctional get_integral_functional() override;

	protected:
		int dim_ = 0; // integrate the "dim" dimension
	};

	class BarycenterTargetObjective : public StaticObjective
	{
	public:
		BarycenterTargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args, const Eigen::MatrixXd &target);
		~BarycenterTargetObjective() = default;

		double value() override;
		Eigen::VectorXd get_target() const;
		void set_time_step(int time_step) override;

		int dim() const { return dim_; }

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

		Eigen::VectorXd get_barycenter() const;

	protected:
		int dim_ = -1;
		std::vector<std::shared_ptr<PositionObjective>> objp;
		std::shared_ptr<VolumeObjective> objv;
		Eigen::MatrixXd target_; // N/1 by 3/2
	};

	class TransientObjective : public Objective
	{
	public:
		TransientObjective(const int time_steps, const double dt, const std::string &transient_integral_type, const std::shared_ptr<StaticObjective> &obj);
		virtual ~TransientObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::vector<double> get_transient_quadrature_weights() const;

		std::shared_ptr<StaticObjective> obj_;

		int time_steps_;
		double dt_;
		std::string transient_integral_type_;
	};

	class ComplianceObjective : public SpatialIntegralObjective
	{
	public:
		ComplianceObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args);
		~ComplianceObjective() = default;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		std::string formulation_;

		std::shared_ptr<const Parameter> elastic_param_; // stress depends on elastic param
	};

	class TargetObjective : public SpatialIntegralObjective
	{
	public:
		TargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
			active_dimension_mask.assign(true, state_.mesh->dimension());
		}
		~TargetObjective() = default;

		IntegrableFunctional get_integral_functional() override;
		void set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids); // target is another simulation solution
		void set_reference(const Eigen::VectorXd &disp) { target_disp = disp; } // target is a constant displacement
		void set_reference(const json &func, const json &grad_func); // target is a lambda function depending on deformed position
		void set_active_dimension(const std::vector<bool> &mask) { active_dimension_mask = mask; }

	protected:
		std::shared_ptr<const State> target_state_;
		std::map<int, int> e_to_ref_e_;

		std::vector<bool> active_dimension_mask;
		Eigen::VectorXd target_disp;

		bool have_target_func = false;
		utils::ExpressionValue target_func;
		std::array<utils::ExpressionValue, 3> target_func_grad;
	};

	class SDFTargetObjective : public SpatialIntegralObjective
	{
	public:
		SDFTargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			if (state.mesh->dimension() == 2)
			{
				cubic_mat.resize(16, 16);
				cubic_mat << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
					-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,
					9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
					-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
					2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
					-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
					4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1;
				// cubic_mat = utils::get_bicubic_mat();
			}
			else if (state.mesh->dimension() == 3)
			{
				// cubic_mat = utils::get_tricubic_mat();
			}
		}

		~SDFTargetObjective() = default;

		IntegrableFunctional get_integral_functional() override;

		void set_spline_target(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &tangents, const Eigen::MatrixXd &delta)
		{
			assert(false);
		}

		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots, const double delta)
		{
			control_points_ = control_points;
			knots_ = knots;
			dim = control_points.cols();
			if (dim != 2)
				logger().error("Only support 2d sdf at the moment.");
			delta_ = delta;
			curve.set_control_points(control_points);
			curve.set_knots(knots);

			t_sampling = Eigen::VectorXd::LinSpaced(100, 0, 1);
			point_sampling.setZero(100, 2);
			for (int i = 0; i < t_sampling.size(); ++i)
				point_sampling.row(i) = curve.evaluate(t_sampling(i));

			Eigen::MatrixXi edges(100, 2);
			edges.col(0) = Eigen::VectorXi::LinSpaced(100, 1, 100);
			edges.col(1) = Eigen::VectorXi::LinSpaced(100, 2, 101);
			io::OBJWriter::write(state_.resolve_output_path(fmt::format("spline_target_{:d}.obj", rand() % 100)), point_sampling, edges);
		}

		void evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);
		void compute_distance(const Eigen::MatrixXd &point, double &distance);
		void compute_distances(const Eigen::MatrixXd &point, Eigen::VectorXd &distances, double h);
		void bicubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad, const double h);
		void tricubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);

	protected:
		int dim;
		double t_cached;
		double delta_;
		std::unordered_map<std::string, double> implicit_function_distance;
		std::unordered_map<std::string, Eigen::VectorXd> implicit_function_grads;

		Eigen::MatrixXd cubic_mat;

		Eigen::MatrixXd control_points_;
		Eigen::VectorXd knots_;

		Eigen::VectorXd t_sampling;
		Eigen::MatrixXd point_sampling;

		nanospline::BSpline<double, 2, 3> curve;

		mutable std::shared_mutex distance_mutex_;
		mutable std::shared_mutex grad_mutex_;
	};

	class NodeTargetObjective : public StaticObjective
	{
	public:
		NodeTargetObjective(const State &state, const json &args);
		NodeTargetObjective(const State &state, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_);
		~NodeTargetObjective() = default;

		double value() override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		const State &state_;

		Eigen::MatrixXd target_vertex_positions;
		std::vector<int> active_nodes;
	};

	class MaterialBoundObjective : public Objective
	{
	public:
		MaterialBoundObjective(const std::shared_ptr<const Parameter> elastic_param, const json &args);
		~MaterialBoundObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::shared_ptr<const Parameter> elastic_param_;

		const bool is_volume;

		double min_E = 0, max_E = 0;
		double kappa_E = 0, dhat_E = 0;
		double min_lambda = 0, max_lambda = 0;
		double kappa_lambda = 0, dhat_lambda = 0;
		double min_mu = 0, max_mu = 0;
		double kappa_mu = 0, dhat_mu = 0;
		double min_nu = 0, max_nu = 0;
		double kappa_nu = 0, dhat_nu = 0;
	};

	class CollisionBarrierObjective : public Objective
	{
	public:
		CollisionBarrierObjective(const std::shared_ptr<const Parameter> shape_param, const json &args);
		~CollisionBarrierObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::shared_ptr<const Parameter> shape_param_;

		ipc::CollisionMesh collision_mesh_;
		ipc::Constraints constraint_set;
		void build_constraint_set(const Eigen::MatrixXd &displaced_surface);

		double dhat;
		ipc::BroadPhaseMethod broad_phase_method;
	};

	class ControlSmoothingObjective : public StaticObjective
	{
	public:
		ControlSmoothingObjective(const std::shared_ptr<const Parameter> control_param, const json &args);
		~ControlSmoothingObjective() = default;

		double value() override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

	protected:
		std::shared_ptr<const Parameter> control_param_;
		int p = 8;
	};
} // namespace polyfem::solver