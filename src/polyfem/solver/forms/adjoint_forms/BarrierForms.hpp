#pragma once

#include <polyfem/solver/forms/adjoint_forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>
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
		ipc::Collisions collision_set;
		const double dhat_;
		const double dmin_;
		ipc::BroadPhaseMethod broad_phase_method_;

		ipc::BarrierPotential barrier_potential_;
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
		ipc::Collisions collision_set;
		const double dhat_;
		ipc::BroadPhaseMethod broad_phase_method_;

		const ipc::BarrierPotential barrier_potential_;
	};

	template <int dim>
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
		void solution_changed_step(const int time_step, const Eigen::VectorXd &x) override
		{
			if (curr_x.size() > 0 && (curr_x - x).norm() < 1e-8)
				return;

			collision_set_indicator_.setZero();
			build_collision_mesh();
			curr_x = x;
		}

	protected:
		void build_collision_mesh();
		const ipc::SmoothCollisions<dim> &get_or_compute_collision_set(const int time_step, const Eigen::MatrixXd &displaced_surface) const;

		const State &state_;
		std::set<int> boundary_ids_;
		std::map<int, std::set<int>> boundary_ids_to_dof_;
		Eigen::MatrixXi can_collide_cache_;

		mutable Eigen::VectorXi collision_set_indicator_;
		// std::vector<std::shared_ptr<ipc::Collisions>> collision_sets_;
		std::vector<std::shared_ptr<ipc::SmoothCollisions<dim>>> collision_sets_;

		ipc::CollisionMesh collision_mesh_;
		const ipc::ParameterType params_;
		const double dmin_ = 0;
		ipc::BroadPhaseMethod broad_phase_method_;

		Eigen::VectorXd curr_x;

		// ipc::BarrierPotential barrier_potential_;
		ipc::SmoothContactPotential<ipc::SmoothCollisions<dim>> potential_;
	};
} // namespace polyfem::solver