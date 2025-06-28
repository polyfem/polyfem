#pragma once

#include <polyfem/solver/forms/adjoint_forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <ipc/potentials/barrier_potential.hpp>
#include <ipc/smooth_contact/smooth_collisions.hpp>
#include <ipc/smooth_contact/smooth_contact_potential.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

namespace polyfem::solver
{
	class CollisionBarrierForm : public AdjointForm
	{
	public:
		CollisionBarrierForm(const VariableToSimulationGroup &variable_to_simulation, const State &state, const double dhat, const double dmin = 0);

		double value_unweighted(const Eigen::VectorXd &x) const override;

		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		void solution_changed(const Eigen::VectorXd &x) override;

		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		std::string name() const override { return "collision barrier"; }

		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	protected:
		void build_collision_set(const Eigen::MatrixXd &displaced_surface);

		Eigen::VectorXd get_updated_mesh_nodes(const Eigen::VectorXd &x) const;

		const State &state_;

		Eigen::VectorXd X_init;

		ipc::CollisionMesh collision_mesh_;
		ipc::NormalCollisions collision_set;
		const double dhat_;
		const double dmin_;
		ipc::BroadPhaseMethod broad_phase_method_;

		ipc::BarrierPotential barrier_potential_;
	};

	class LayerThicknessForm : public CollisionBarrierForm
	{
	public:
		LayerThicknessForm(const VariableToSimulationGroup &variable_to_simulations,
						   const State &state,
						   const std::vector<int> &boundary_ids,
						   const double dhat,
						   const bool use_log_barrier = false,
						   const double dmin = 0);

		std::string name() const override { return "layer thickness"; }

		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override { return 1.; }

	protected:
		void build_collision_mesh();

		std::vector<int> boundary_ids_;
		std::map<int, std::set<int>> boundary_ids_to_dof_;
		Eigen::MatrixXi can_collide_cache_;
	};

	class DeformedCollisionBarrierForm : public AdjointForm
	{
	public:
		DeformedCollisionBarrierForm(const VariableToSimulationGroup &variable_to_simulation, const State &state, const double dhat);

		std::string name() const override { return "deformed_collision_barrier"; }

		double value_unweighted(const Eigen::VectorXd &x) const override;

		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		void solution_changed(const Eigen::VectorXd &x) override;

		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	private:
		void build_collision_set(const Eigen::MatrixXd &displaced_surface);

		Eigen::VectorXd get_updated_mesh_nodes(const Eigen::VectorXd &x) const;

		const State &state_;

		Eigen::VectorXd X_init;

		ipc::CollisionMesh collision_mesh_;
		ipc::NormalCollisions collision_set;
		const double dhat_;
		ipc::BroadPhaseMethod broad_phase_method_;

		const ipc::BarrierPotential barrier_potential_;
	};

	class SmoothContactForceForm : public StaticForm
	{
	public:
		SmoothContactForceForm(
			const VariableToSimulationGroup &variable_to_simulations,
			const State &state,
			const json &args);
		~SmoothContactForceForm() = default;

		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		void solution_changed_step(const int time_step, const Eigen::VectorXd &x) override;

	protected:
		void build_collision_mesh();
		ipc::SmoothCollisions get_smooth_collision_set(const Eigen::MatrixXd &displaced_surface);

		const State &state_;
		std::set<int> boundary_ids_;
		std::map<int, std::set<int>> boundary_ids_to_dof_;

		ipc::CollisionMesh collision_mesh_;
		ipc::SmoothCollisions collisions_;
		const ipc::ParameterType params_;
		const double dmin_ = 0;

		ipc::SmoothContactPotential potential_;
	};
} // namespace polyfem::solver
