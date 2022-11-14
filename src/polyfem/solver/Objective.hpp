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

		virtual double value() const = 0;
		Eigen::VectorXd gradient(const std::vector<std::shared_ptr<State>> &states, const Parameter &param) const
		{
			Eigen::VectorXd grad = compute_partial_gradient(param);
			for (const auto &state : states)
				grad += compute_adjoint_term(*state, param);
			
			return grad;
		}

		// use only if there's only one state
		Eigen::VectorXd gradient(const State& state, const Parameter &param) const
		{
			return compute_partial_gradient(param) + compute_adjoint_term(state, param);
		}

		virtual Eigen::MatrixXd compute_adjoint_rhs(const State& state) const = 0; // compute $\partial_u J$

		virtual Eigen::VectorXd compute_partial_gradient(const Parameter &param) const = 0; // compute $\partial_q J$
		static  Eigen::VectorXd compute_adjoint_term(const State& state, const Parameter &param);
	};

	// this objective either depends on solution in one time step, or one static solution
	class StaticObjective: public Objective
	{
	public:
		StaticObjective() = default;
		virtual ~StaticObjective() = default;

		virtual void set_time_step(int time_step) { time_step_ = time_step; }
		int  get_time_step() const { return time_step_; }

		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		virtual Eigen::VectorXd compute_adjoint_rhs_step(const State& state) const = 0;

	protected:
		int time_step_ = 0; // time step to integrate
	};

	class SpatialIntegralObjective: public StaticObjective
	{
	public:
		SpatialIntegralObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		virtual ~SpatialIntegralObjective() = default;

		double value() const override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State& state) const override;

		virtual IntegrableFunctional get_integral_functional() const = 0;

	protected:
		const State &state_;
		std::shared_ptr<const ShapeParameter> shape_param_;
		AdjointForm::SpatialIntegralType spatial_integral_type_;
		std::set<int> interested_ids_;
	};

	class StressObjective: public SpatialIntegralObjective
	{
	public:
		StressObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const std::shared_ptr<const ElasticParameter> &elastic_param, const json &args, bool has_integral_sqrt = true);
		~StressObjective() = default;

		double value() const override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State& state) const override;
		IntegrableFunctional get_integral_functional() const override;

	protected:
		int in_power_;
		bool out_sqrt_;
		std::string formulation_;

		std::shared_ptr<const ElasticParameter> elastic_param_; // stress depends on elastic param
	};

	class SumObjective: public Objective
	{
	public:
		SumObjective(const std::vector<std::shared_ptr<Objective>> &objs): objs_(objs) 
		{ weights_.setOnes(objs_.size()); }
		SumObjective(const std::vector<std::shared_ptr<Objective>> &objs, const Eigen::VectorXd &weights): objs_(objs), weights_(weights) {}
		~SumObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
	protected:
		std::vector<std::shared_ptr<Objective>> objs_;
		Eigen::VectorXd weights_;
	};

	// note: active nodes are selected by surface selection on the first state in shape_param
	class BoundarySmoothingObjective: public Objective
	{
	public:
		BoundarySmoothingObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~BoundarySmoothingObjective() = default;
		
		void init();

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		std::shared_ptr<const ShapeParameter> shape_param_;

		const json args_;

		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

	class DeformedBoundarySmoothingObjective: public Objective
	{
	public:
		DeformedBoundarySmoothingObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~DeformedBoundarySmoothingObjective() = default;
		
		void init();

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		const State &state_;
		std::shared_ptr<const ShapeParameter> shape_param_;
		const json args_;
		
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
	};

	class VolumeObjective: public Objective
	{
	public:
		VolumeObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~VolumeObjective() = default;
		
		void set_weights(const Eigen::VectorXd &weights) { weights_ = weights; }

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		std::shared_ptr<const ShapeParameter> shape_param_;
		std::set<int> interested_ids_;
		Eigen::VectorXd weights_;
	};

	class VolumePaneltyObjective: public Objective
	{
	public:
		VolumePaneltyObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~VolumePaneltyObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		std::shared_ptr<VolumeObjective> obj;
		Eigen::Vector2d bound;
	};

	class PositionObjective: public SpatialIntegralObjective
	{
	public:
		PositionObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~PositionObjective() = default;

		void set_dim(const int dim) { dim_ = dim; }
		void set_integral_type(const AdjointForm::SpatialIntegralType type) { spatial_integral_type_ = type; }

		IntegrableFunctional get_integral_functional() const override;

	protected:
		int dim_ = 0;		// integrate the "dim" dimension
	};

	class BarycenterTargetObjective: public StaticObjective
	{
	public:
		BarycenterTargetObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args, const Eigen::MatrixXd &target);
		~BarycenterTargetObjective() = default;

		double value() const override;
		Eigen::VectorXd get_target() const;
		void set_time_step(int time_step) override;

		int dim() const { return dim_; }

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State& state) const override;

		Eigen::VectorXd get_barycenter() const;

	protected:
		int dim_ = -1;
		std::vector<std::shared_ptr<PositionObjective>> objp;
		std::shared_ptr<VolumeObjective> objv;
		Eigen::MatrixXd target_; // N/1 by 3/2
	};
	
	class TransientObjective: public Objective
	{
	public:
		TransientObjective(const int time_steps, const double dt, const std::string &transient_integral_type, const std::shared_ptr<StaticObjective> &obj);
		virtual ~TransientObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
	protected:
		std::vector<double> get_transient_quadrature_weights() const;

		std::shared_ptr<StaticObjective> obj_;

		int time_steps_;
		double dt_;
		std::string transient_integral_type_;
	};

	class ComplianceObjective: public SpatialIntegralObjective
	{
	public:
		ComplianceObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const std::shared_ptr<const ElasticParameter> &elastic_param, const std::shared_ptr<const TopologyOptimizationParameter> topo_param, const json &args);
		~ComplianceObjective() = default;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		IntegrableFunctional get_integral_functional() const override;

	protected:
		std::string formulation_;

		std::shared_ptr<const ElasticParameter> elastic_param_; // stress depends on elastic param
		std::shared_ptr<const TopologyOptimizationParameter> topo_param_;
	};

	class StrainObjective: public SpatialIntegralObjective
	{
	public:
		StrainObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args): SpatialIntegralObjective(state, shape_param, args) {}
		~StrainObjective() = default;

		IntegrableFunctional get_integral_functional() const override;
	};

	class NaiveNegativePoissonObjective: public Objective
	{
	public:
		NaiveNegativePoissonObjective(const State &state1, const json &args);
		~NaiveNegativePoissonObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		const State& state1_;

		int v1 = -1;
		int v2 = -1;

		double power_ = 2;
	};

	class TargetLengthObjective: public Objective
	{
	public:
		TargetLengthObjective(const State &state1, const json &args);
		~TargetLengthObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		const State& state1_;

		int v1 = -1;
		int v2 = -1;
		double target_length;

		double power_ = 2;
	};

	class TargetObjective: public SpatialIntegralObjective
	{
	public:
		TargetObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args): SpatialIntegralObjective(state, shape_param, args) 
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
		}
		~TargetObjective() = default;

		IntegrableFunctional get_integral_functional() const override;
		void set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids);

	protected:
		std::shared_ptr<const State> target_state_;
		std::map<int, int> e_to_ref_e_;
	};
} // namespace polyfem::solver