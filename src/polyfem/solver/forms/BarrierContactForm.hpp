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

		virtual std::string name() const override { return "barrier-contact"; }

        virtual void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		bool use_convergent_formulation() const override { return use_area_weighting() && use_improved_max_operator() && use_physical_barrier(); }

		/// @brief Get use_area_weighting
		bool use_area_weighting() const { return collision_set().use_area_weighting();}

		/// @brief Get use_improved_max_operator
		bool use_improved_max_operator() const { return collision_set().use_improved_max_approximator();}

		/// @brief Get use_physical_barrier
		bool use_physical_barrier() const { return barrier_potential_.use_physical_barrier(); }
		
		const ipc::NormalCollisions &collision_set() const { return collision_set_; }
		const ipc::BarrierPotential &barrier_potential() const { return barrier_potential_; }

	protected:
		/// @brief Compute the contact barrier potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the value of the form multiplied per element
		/// @param x Current solution
		/// @return Computed value
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		virtual void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

		void update_collision_set(const Eigen::MatrixXd &displaced_surface) override;

		/// @brief Cached constraint set for the current solution
		ipc::NormalCollisions collision_set_;

		/// @brief Contact potential
		const ipc::BarrierPotential barrier_potential_;
    };
}
