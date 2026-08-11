#pragma once

#include "ContactForm.hpp"

#include <polyfem/utils/Types.hpp>
#include <polysolve/nonlinear/PostStepData.hpp>

#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/potentials/barrier_potential.hpp>

namespace polyfem::solver
{
	class BarrierContactForm : public ContactForm
	{
		friend class BarrierContactForceDerivative;

	public:
		BarrierContactForm(const ipc::CollisionMesh &collision_mesh,
						   const double dhat,
						   const double avg_mass,
						   const bool use_area_weighting,
						   const bool use_improved_max_operator,
						   const bool use_physical_barrier,
						   const bool use_adaptive_barrier_stiffness,
						   const bool is_time_dependent,
						   const bool enable_shape_derivatives,
						   const ipc::BroadPhaseMethod broad_phase_method,
						   const double ccd_tolerance,
						   const int ccd_max_iterations);

		std::string name() const override { return "barrier-contact"; }

		void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

		/// @brief Update fields after a step in the optimization
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		bool use_convergent_formulation() const override { return use_area_weighting() && use_improved_max_operator() && use_physical_barrier(); }

		bool use_area_weighting() const { return collision_set().use_area_weighting(); }
		bool use_improved_max_operator() const { return collision_set().use_improved_max_approximator(); }
		bool use_physical_barrier() const { return barrier_potential_.use_physical_barrier(); }

		const ipc::NormalCollisions &collision_set() const { return collision_set_; }
		const ipc::BarrierPotential &barrier_potential() const { return barrier_potential_; }

	protected:
		/// Constructor for specialized barrier-contact forms with a custom potential.
		BarrierContactForm(const ipc::CollisionMesh &collision_mesh,
						   const double dhat,
						   const double avg_mass,
						   const bool use_area_weighting,
						   const bool use_improved_max_operator,
						   const bool use_adaptive_barrier_stiffness,
						   const bool is_time_dependent,
						   const bool enable_shape_derivatives,
						   const ipc::BroadPhaseMethod broad_phase_method,
						   const double ccd_tolerance,
						   const int ccd_max_iterations,
						   const ipc::BarrierPotential &barrier_potential);

		double value_unweighted(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

		void update_collision_set(const Eigen::MatrixXd &displaced_surface) override;

		ipc::NormalCollisions collision_set_;
		const ipc::BarrierPotential barrier_potential_;

		/// Per-form cache; this used to be shared by all instances through a static local.
		Eigen::MatrixXd cached_displaced_surface_;
	};
} // namespace polyfem::solver
