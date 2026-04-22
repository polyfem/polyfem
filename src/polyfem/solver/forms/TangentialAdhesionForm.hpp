#pragma once

#include "Form.hpp"
#include "NormalAdhesionForm.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>
#include <ipc/potentials/tangential_adhesion_potential.hpp>
#include <ipc/broad_phase/create_broad_phase.hpp>

namespace polyfem::solver
{
	/// @brief Form of the lagged tangential adhesion disapative potential and forces
	class TangentialAdhesionForm : public Form
	{
		friend class TangentialAdhesionForceDerivative;

	public:
		/// @brief Construct a new Tangential Adhesion Form object
		/// @param collision_mesh Reference to the collision mesh
		/// @param time_integrator Pointer to the time integrator
		/// @param epsa Smoothing factor between static and dynamic tangential adhesion
		/// @param mu Global coefficient of tangential adhesion
		/// @param dhat Barrier activation distance
		/// @param broad_phase_method Broad-phase method used for distance computation and collision detection
		/// @param normal_adhesion_form Pointer to normal adhesion form; necessary to have the potential, maybe clean me
		/// @param n_lagging_iters Number of lagging iterations
		TangentialAdhesionForm(
			const ipc::CollisionMesh &collision_mesh,
			const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator,
			const double epsa,
			const double mu,
			const ipc::BroadPhaseMethod broad_phase_method,
			const NormalAdhesionForm &normal_adhesion_form,
			const int n_lagging_iters);

		std::string name() const override { return "tangential adhesion"; }

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
		double epsa() const { return epsa_; }
		const ipc::TangentialCollisions &tangential_collision_set() const { return tangential_collision_set_; }
		const ipc::TangentialAdhesionPotential &tangential_adhesion_potential() const { return tangential_adhesion_potential_; }

	private:
		/// Reference to the collision mesh
		const ipc::CollisionMesh &collision_mesh_;

		/// Pointer to the time integrator
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator_;

		const double epsa_;                              ///< Smoothing factor for turning on/off tangential adhesion
		const double mu_;                                ///< Global coefficient of tangential adhesion
		const ipc::BroadPhaseMethod broad_phase_method_; ///< Broad-phase method used for distance computation and collision detection
		const std::shared_ptr<ipc::BroadPhase> broad_phase_;
		const int n_lagging_iters_; ///< Number of lagging iterations

		ipc::TangentialCollisions tangential_collision_set_; ///< Lagged tangential constraint set

		const NormalAdhesionForm &normal_adhesion_form_; ///< necessary to have the barrier stiffnes, maybe clean me

		const ipc::TangentialAdhesionPotential tangential_adhesion_potential_;
	};
} // namespace polyfem::solver
