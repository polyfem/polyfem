#pragma once

#include "Form.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/friction/friction_constraints.hpp>

namespace polyfem::solver
{
	class ContactForm;

	/// @brief Form of the lagged friction disapative potential and forces
	class FrictionForm : public Form
	{
	public:
		/// @brief Construct a new Friction Form object
		/// @param collision_mesh Reference to the collision mesh
		/// @param time_integrator Pointer to the time integrator
		/// @param epsv Smoothing factor between static and dynamic friction
		/// @param mu Global coefficient of friction
		/// @param dhat Barrier activation distance
		/// @param broad_phase_method Broad-phase method used for distance computation and collision detection
		/// @param contact_form Pointer to contact form; necessary to have the barrier stiffnes, maybe clean me
		/// @param n_lagging_iters Number of lagging iterations
		FrictionForm(
			const ipc::CollisionMesh &collision_mesh,
			const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator,
			const double epsv,
			const double mu,
			const double dhat,
			const ipc::BroadPhaseMethod broad_phase_method,
			const ContactForm &contact_form,
			const int n_lagging_iters);

		std::string name() const override { return "friction"; }

		void force_shape_derivative(const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint, const ipc::FrictionConstraints &friction_constraints_set, Eigen::VectorXd &term);

	protected:
		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Initialize lagged fields
		/// @param x Current solution
		void init_lagging(const Eigen::VectorXd &x) override { update_lagging(x, 0); }

		/// @brief Update lagged fields
		/// @param x Current solution
		void update_lagging(const Eigen::VectorXd &x, const int iter_num) override;

		/// @brief Update lagged fields
		/// @param x Current solution
		void update_lagging(const Eigen::VectorXd &x) { update_lagging(x, -1); };

		/// @brief Get the maximum number of lagging iteration allowable.
		int max_lagging_iterations() const override { return n_lagging_iters_; }

		/// @brief Does this form require lagging?
		/// @return True if the form requires lagging
		bool uses_lagging() const override { return true; }

		/// @brief Compute the displaced positions of the surface nodes
		Eigen::MatrixXd compute_displaced_surface(const Eigen::VectorXd &x) const;
		/// @brief Compute the surface velocities
		Eigen::MatrixXd compute_surface_velocities(const Eigen::VectorXd &x) const;
		/// @brief Compute the derivative of the velocities wrt x
		double dv_dx() const;

		double mu() const { return mu_; }
		double epsv() const { return epsv_; }
		ipc::FrictionConstraints get_friction_constraint_set() const { return friction_constraint_set_; }

	private:
		/// Reference to the collision mesh
		const ipc::CollisionMesh &collision_mesh_;

		/// Pointer to the time integrator
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator_;

		const double epsv_;                              ///< Smoothing factor between static and dynamic friction
		const double mu_;                                ///< Global coefficient of friction
		const double dhat_;                              ///< Barrier activation distance
		const ipc::BroadPhaseMethod broad_phase_method_; ///< Broad-phase method used for distance computation and collision detection
		const int n_lagging_iters_;                      ///< Number of lagging iterations

		ipc::FrictionConstraints friction_constraint_set_; ///< Lagged friction constraint set

		const ContactForm &contact_form_; ///< necessary to have the barrier stiffnes, maybe clean me
	};
} // namespace polyfem::solver
