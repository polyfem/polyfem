#pragma once

#include "ElasticParameter.hpp"
#include "ShapeParameter.hpp"
#include "TopologyOptimizationParameter.hpp"
#include "AdjointForm.hpp"

#include <array>

namespace polyfem::solver
{
	class Objective
	{
	public:
		Objective() = default;
		virtual ~Objective() = default;

		static std::shared_ptr<Objective> create(const json &args, const std::vector<std::shared_ptr<Parameter>> &parameters, const std::vector<std::shared_ptr<State>> &states);

		virtual double value() = 0;
		Eigen::VectorXd gradient(const std::vector<std::shared_ptr<State>> &states, const Parameter &param)
		{
			Eigen::VectorXd grad = compute_partial_gradient(param);
			for (const auto &state : states)
				grad += compute_adjoint_term(*state, param);

			return grad;
		}

		// use only if there's only one state
		Eigen::VectorXd gradient(const State &state, const Parameter &param)
		{
			return compute_partial_gradient(param) + compute_adjoint_term(state, param);
		}

		virtual Eigen::MatrixXd compute_adjoint_rhs(const State &state) = 0; // compute $\partial_u J$

		virtual Eigen::VectorXd compute_partial_gradient(const Parameter &param) = 0; // compute $\partial_q J$
		static Eigen::VectorXd compute_adjoint_term(const State &state, const Parameter &param);
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
		SpatialIntegralObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		virtual ~SpatialIntegralObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

		virtual IntegrableFunctional get_integral_functional() = 0;

	protected:
		const State &state_;
		std::shared_ptr<const ShapeParameter> shape_param_;
		AdjointForm::SpatialIntegralType spatial_integral_type_;
		std::set<int> interested_ids_;
	};

	class StressObjective : public SpatialIntegralObjective
	{
	public:
		StressObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const std::shared_ptr<const ElasticParameter> &elastic_param, const json &args, bool has_integral_sqrt = true);
		~StressObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		int in_power_;
		bool out_sqrt_;
		std::string formulation_;

		std::shared_ptr<const ElasticParameter> elastic_param_; // stress depends on elastic param
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
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		std::vector<std::shared_ptr<Objective>> objs_;
		Eigen::VectorXd weights_;
	};

	// note: active nodes are selected by surface selection on the first state in shape_param
	class BoundarySmoothingObjective : public Objective
	{
	public:
		BoundarySmoothingObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~BoundarySmoothingObjective() = default;

		void init();

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		std::shared_ptr<const ShapeParameter> shape_param_;

		const json args_;

		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

	class DeformedBoundarySmoothingObjective : public Objective
	{
	public:
		DeformedBoundarySmoothingObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~DeformedBoundarySmoothingObjective() = default;

		void init();

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		const State &state_;
		std::shared_ptr<const ShapeParameter> shape_param_;
		const json args_;

		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
	};

	class VolumeObjective : public Objective
	{
	public:
		VolumeObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~VolumeObjective() = default;

		void set_weights(const Eigen::VectorXd &weights) { weights_ = weights; }

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		std::shared_ptr<const ShapeParameter> shape_param_;
		std::set<int> interested_ids_;
		Eigen::VectorXd weights_;
	};

	class VolumePaneltyObjective : public Objective
	{
	public:
		VolumePaneltyObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~VolumePaneltyObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		std::shared_ptr<VolumeObjective> obj;
		Eigen::Vector2d bound;
	};

	class PositionObjective : public SpatialIntegralObjective
	{
	public:
		PositionObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
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
		BarycenterTargetObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args, const Eigen::MatrixXd &target);
		~BarycenterTargetObjective() = default;

		double value() override;
		Eigen::VectorXd get_target() const;
		void set_time_step(int time_step) override;

		int dim() const { return dim_; }

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;
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

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

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
		ComplianceObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const std::shared_ptr<const ElasticParameter> &elastic_param, const std::shared_ptr<const TopologyOptimizationParameter> topo_param, const json &args);
		~ComplianceObjective() = default;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		std::string formulation_;

		std::shared_ptr<const ElasticParameter> elastic_param_; // stress depends on elastic param
		std::shared_ptr<const TopologyOptimizationParameter> topo_param_;
	};

	class StrainObjective : public SpatialIntegralObjective
	{
	public:
		StrainObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args) {}
		~StrainObjective() = default;

		IntegrableFunctional get_integral_functional() override;
	};

	class NaiveNegativePoissonObjective : public Objective
	{
	public:
		NaiveNegativePoissonObjective(const State &state1, const json &args);
		~NaiveNegativePoissonObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		const State &state1_;

		int v1 = -1;
		int v2 = -1;

		double power_ = 2;
	};

	class TargetLengthObjective : public Objective
	{
	public:
		TargetLengthObjective(const State &state1, const json &args);
		~TargetLengthObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		const State &state1_;

		int v1 = -1;
		int v2 = -1;
		double target_length;

		double power_ = 2;
	};

	class TargetObjective : public SpatialIntegralObjective
	{
	public:
		TargetObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}
		~TargetObjective() = default;

		IntegrableFunctional get_integral_functional() override;
		void set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids);

	protected:
		std::shared_ptr<const State> target_state_;
		std::map<int, int> e_to_ref_e_;
	};

	class SDFTargetObjective : public SpatialIntegralObjective
	{
	public:
		SDFTargetObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			bicubic_mat.resize(16, 16);
			bicubic_mat << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
		}

		~SDFTargetObjective() = default;

		IntegrableFunctional get_integral_functional() override;

		void set_spline_target(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &tangents, const Eigen::MatrixXd &delta)
		{
			control_points_ = control_points;
			tangents_ = tangents;
			dim = control_points.cols();
			if (dim != 2)
				logger().error("Only support 2d sdf at the moment.");
			delta_ = delta;
		}

	protected:
		void evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);
		void compute_distance(const Eigen::MatrixXd &point, double &distance, Eigen::MatrixXd &grad);
		void bicubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);

		int dim;
		double t_cached;
		Eigen::MatrixXd delta_;
		std::unordered_map<std::string, double> implicit_function_distance;
		std::unordered_map<std::string, Eigen::MatrixXd> implicit_function_grad;

		Eigen::MatrixXd bicubic_mat;

		Eigen::MatrixXd control_points_;
		Eigen::MatrixXd tangents_;

		mutable std::shared_mutex mutex_;
	};

	class NodeTargetObjective : public StaticObjective
	{
	public:
		NodeTargetObjective(const State &state, const json &args);
		NodeTargetObjective(const State &state, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_);
		~NodeTargetObjective() = default;

		double value() override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) override;

	protected:
		const State &state_;

		Eigen::MatrixXd target_vertex_positions;
		std::vector<int> active_nodes;
	};
} // namespace polyfem::solver