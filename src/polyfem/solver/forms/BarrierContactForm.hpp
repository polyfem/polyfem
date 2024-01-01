#pragma once

#include "ContactForm.hpp"
#include <ipc/potentials/barrier_potential.hpp>

namespace polyfem::solver
{
    class BarrierContactForm : public ContactForm
    {
    public:
		BarrierContactForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat,
					const double avg_mass,
					const bool use_convergent_formulation,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const bool enable_shape_derivatives,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		virtual std::string name() const override { return "barrier-contact"; }

        void force_shape_derivative(const ipc::VirtualCollisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) override;
        
        void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

		const ipc::BarrierPotential &get_barrier_potential() const { return *std::dynamic_pointer_cast<ipc::BarrierPotential>(contact_potential_); }

    };
}