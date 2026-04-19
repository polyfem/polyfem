#pragma once

#include "ContactForm.hpp"
#include <ipc/high_order_contact/high_order_collisions.hpp>
#include <ipc/high_order_contact/high_order_contact_potential.hpp>
#include <cmath>

namespace polyfem::solver
{
	/// Create a barrier object from the "barrier" JSON field.
	/// Recognised values: "log", "normalized_log", "linear_inverse", "quadratic_inverse".
	std::shared_ptr<ipc::Barrier> barrier_from_params(const json &params);

    class HighOrderContactForm : public ContactForm
    {
    public:
		HighOrderContactForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat,
					const double avg_mass,
					const json high_order_contact_params,
					const bool skip_obstacles,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const bool enable_shape_derivatives,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		virtual std::string name() const override { return "high-order-contact"; }

        void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

		void force_shape_derivative(const ipc::HighOrderCollisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) const;

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		const ipc::HighOrderContactParameters &get_params() const { return params; }

		const ipc::HighOrderCollisions &collision_set() const { return collision_set_; }

		const ipc::HighOrderContactPotential &barrier_potential() const { return barrier_potential_; }

		const ipc::HighOrderContactPotential::CountMap &get_ee_qp_count() const {
			return barrier_potential_.get_edge_evaluation_count();
		}

	protected:
		/// @brief Compute the contact barrier potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the value of the form multiplied per element
		/// @param x Current solution
		/// @return Computed value
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

		double barrier_support_size() const override { return dhat_; }

		void update_collision_set(const Eigen::MatrixXd &displaced_surface) override;

	private:
		ipc::HighOrderContactParameters params;

		/// @brief Cached constraint set for the current solution
		ipc::HighOrderCollisions collision_set_;

		/// @brief Contact potential
		ipc::HighOrderContactPotential barrier_potential_;

    	Eigen::MatrixXd cached_displaced_surface;
	};
}
