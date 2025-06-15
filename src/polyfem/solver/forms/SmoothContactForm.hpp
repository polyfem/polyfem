#pragma once

#include "ContactForm.hpp"
#include <ipc/smooth_contact/smooth_collisions.hpp>
#include <ipc/smooth_contact/smooth_contact_potential.hpp>
#include <cmath>

namespace polyfem::solver
{
	template <int _dim = 3>
    class SmoothContactForm : public ContactForm
    {
    public:
		SmoothContactForm(const ipc::CollisionMesh &collision_mesh,
					const json &args,
					const double avg_mass,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const bool enable_shape_derivatives,
					const BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		virtual std::string name() const override { return "smooth-contact"; }

        void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

		void force_shape_derivative(ipc::CollisionsBase *collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) override;

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		bool using_adaptive_dhat() const { return use_adaptive_dhat; }
		const ipc::ParameterType &get_params() const { return params; }
		ipc::SmoothCollisions<_dim> &get_smooth_collision_set() { return *std::dynamic_pointer_cast<ipc::SmoothCollisions<_dim>>(collision_set_); }
		const ipc::SmoothCollisions<_dim> &get_smooth_collision_set() const { return *std::dynamic_pointer_cast<ipc::SmoothCollisions<_dim>>(collision_set_); }
		const ipc::Potential<ipc::SmoothCollisions<_dim>> &get_potential() const { return *contact_potential_; }

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

		/// @brief Contact potential
		std::shared_ptr<ipc::SmoothContactPotential<ipc::SmoothCollisions<_dim>>> contact_potential_;

	private:
		ipc::ParameterType params;
		const bool use_adaptive_dhat;
	};
}