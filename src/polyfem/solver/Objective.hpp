#pragma once

#include "ElasticParameter.hpp"
#include "ShapeParameter.hpp"

namespace polyfem::solver
{
	class Objective
	{
	public:
		Objective() = default;
		virtual ~Objective() = default;

		virtual double value() const = 0;
		Eigen::VectorXd gradient(const State& state, const Parameter &param) const
		{
			return compute_partial_gradient(param) + compute_adjoint_term(state, param);
		}

		Eigen::VectorXd gradient(const Parameter &param) const
		{
			log_and_throw_error("Shouldn't use this!");
			return Eigen::VectorXd();
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

	class StressObjective: public StaticObjective
	{
	public:
		StressObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const std::shared_ptr<const ElasticParameter> &elastic_param, const json &args);
		~StressObjective() = default;

		double value() const override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State& state) const override;

	protected:
		const State &state_;
		IntegrableFunctional j_;
		int power_;
		std::string formulation_;

		std::shared_ptr<const ShapeParameter> shape_param_; // integral depends on shape param
		std::shared_ptr<const ElasticParameter> elastic_param_; // stress depends on elastic param

		std::set<int> interested_ids_;
	};

	class SumObjective: public Objective
	{
	public:
		SumObjective(const json &args);
		~SumObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
	protected:
		std::vector<Objective> objs;
	};

	class BoundarySmoothingObjective: public Objective
	{
	public:
		BoundarySmoothingObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~BoundarySmoothingObjective() = default;
		
		void init(const std::shared_ptr<const ShapeParameter> shape_param);

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		std::shared_ptr<const ShapeParameter> shape_param_;

		const json args_;

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
		
		std::vector<bool> active_mask;
		std::vector<int> boundary_nodes;
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

	class VolumeObjective: public Objective
	{
	public:
		VolumeObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~VolumeObjective() = default;

		double value() const override;
		Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	protected:
		std::shared_ptr<const ShapeParameter> shape_param_;
		std::set<int> interested_ids_;
	};

	// class VolumePaneltyObjective: public Objective
	// {
	// public:
	// 	VolumePaneltyObjective(const State &state, const json &args);
	// 	~VolumePaneltyObjective() = default;

	// 	double value() const override;
	// 	Eigen::MatrixXd compute_adjoint_rhs(const State& state) const override;

	// protected:
	// 	Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;

	// 	const State &state_;
	// 	std::set<int> interested_ids_;
	// };

	class PositionObjective: public StaticObjective
	{
	public:
		PositionObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
		~PositionObjective() = default;

		double value() const override;

		void set_dim(const int dim) { dim_ = dim; }

		Eigen::VectorXd compute_partial_gradient(const Parameter &param) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State& state) const override;

	protected:
		const State &state_;
		std::shared_ptr<const ShapeParameter> shape_param_;
		std::set<int> interested_ids_;
		int dim_ = 0;		// integrate the "dim" dimension
	};

	class BarycenterTargetObjective: public StaticObjective
	{
	public:
		BarycenterTargetObjective(const State &state, const Eigen::MatrixXd &target, const std::shared_ptr<const ShapeParameter> shape_param, const json &args);
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
		TransientObjective(const int time_steps, const double dt, const std::string &transient_integral_type);
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

	class CenterTrajectoryObjective: public TransientObjective
	{
	public:
		CenterTrajectoryObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args, const Eigen::MatrixXd &targets);
		~CenterTrajectoryObjective() = default;

		Eigen::MatrixXd get_barycenters();
	};
} // namespace polyfem::solver