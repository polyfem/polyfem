#pragma once

#include "ContactForm.hpp"
#include <ipc/smooth_contact/smooth_contact_potential.hpp>

namespace polyfem::solver
{
    class SmoothContactForm : public ContactForm
    {
    public:
		SmoothContactForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat,
					const double alpha,
					const double r,
					const double avg_mass,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations): ContactForm(collision_mesh, dhat, avg_mass, false, use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method, ccd_tolerance, ccd_max_iterations)
		{
			params.eps = dhat;
			params.alpha = alpha;
			params.r = r;

			contact_potential_ = std::make_shared<ipc::SmoothContactPotential>(params);
		}

		virtual std::string name() const override { return "smooth-contact"; }

        void force_shape_derivative(const ipc::Collisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) override;

        void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;
	
	protected:
		double barrier_support_size() const override { return sqrt(dhat_); }

	private:
		ipc::ParameterType params;
	};
}