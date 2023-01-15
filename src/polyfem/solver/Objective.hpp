#pragma once

#include "AdjointForm.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>

#include <polyfem/utils/UnsignedDistanceFunction.hpp>
#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>
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
		std::set<int> volume_selections;

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

	class LayerThicknessObjective : public Objective
	{
	public:
		LayerThicknessObjective(const std::shared_ptr<const Parameter> first_shape_param, const std::shared_ptr<const Parameter> second_shape_parameter, const json &args);
		LayerThicknessObjective(const std::shared_ptr<const Parameter> shape_param, const int adjacent_boundary_id, const json &args);
		~LayerThicknessObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::shared_ptr<const Parameter> shape_param_;
		std::shared_ptr<const Parameter> adjacent_shape_param_;
		int adjacent_boundary_id_;

		const int dim_ = 2;

		std::vector<int> boundary_node_ids_;

		ipc::CollisionMesh collision_mesh_;
		ipc::Constraints constraint_set;
		void build_constraint_set(const Eigen::MatrixXd &displaced_surface);

		Eigen::MatrixXd extract_boundaries(const Eigen::MatrixXd &V, const std::vector<int> &boundary_node_ids);
		Eigen::MatrixXi get_boundary_edges(const Eigen::MatrixXi &F, const std::vector<int> &boundary_node_ids);

		double dmin;
		double dhat;
		ipc::BroadPhaseMethod broad_phase_method;
	};
} // namespace polyfem::solver