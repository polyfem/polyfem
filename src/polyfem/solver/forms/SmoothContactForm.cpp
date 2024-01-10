#include "SmoothContactForm.hpp"
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
    SmoothContactForm::SmoothContactForm(const ipc::CollisionMesh &collision_mesh,
                const json &args,
                const double avg_mass,
                const bool use_adaptive_barrier_stiffness,
                const bool is_time_dependent,
                const ipc::BroadPhaseMethod broad_phase_method,
                const double ccd_tolerance,
                const int ccd_max_iterations): ContactForm(collision_mesh, args["dhat"], avg_mass, false, use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method, ccd_tolerance, ccd_max_iterations), params(dhat_*dhat_, args["alpha"], args["a"], args["r"], args["high_order_quadrature"])
    {
        if (collision_mesh.dim() == 2)
		    collision_set_ = std::make_shared<ipc::SmoothCollisions<2>>(args["use_adaptive_epsilon"]);
        else
            collision_set_ = std::make_shared<ipc::SmoothCollisions<3>>(args["use_adaptive_epsilon"]);
        contact_potential_ = std::make_shared<ipc::SmoothContactPotential<ipc::VirtualCollisions>>(params);
        if (params.a > 0)
            logger().error("The contact candidate search size is likely wrong!");
        if (args["high_order_quadrature"] > 1)
            collision_set_->set_edge_quadrature_type(ipc::SurfaceQuadratureType::UniformSampling);
        else
            collision_set_->set_edge_quadrature_type(ipc::SurfaceQuadratureType::SinglePoint);
        
        candidates_.set_candidate_types(collision_set_->get_candidate_types(collision_mesh_.dim()));
    }
    
    void SmoothContactForm::force_shape_derivative(const ipc::VirtualCollisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term)
    {
        log_and_throw_error("[{}] Shape derivatives not implemented!", name());
    }

    void SmoothContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
    {
        log_and_throw_error("[{}] Barrier stiffness update not implemented!", name());
    }
}