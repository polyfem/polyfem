#pragma once

#include "ContactForm.hpp"
#include <ipc/smooth_contact/smooth_contact_potential.hpp>
#include <cmath>

namespace polyfem::solver
{
    class SmoothContactForm : public ContactForm
    {
    public:
		SmoothContactForm(const ipc::CollisionMesh &collision_mesh,
					const json &args,
					const double avg_mass,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		virtual std::string name() const override { return "smooth-contact"; }

        void force_shape_derivative(const ipc::VirtualCollisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) override;

        void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;
	
	protected:
		double barrier_support_size() const override { return dhat_; }

	private:
		ipc::ParameterType params;
	};
}