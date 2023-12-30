#include "SmoothContactForm.hpp"
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
    void SmoothContactForm::force_shape_derivative(const ipc::Collisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term)
    {
        log_and_throw_error("[{}] Shape derivatives not implemented!", name());
    }

    void SmoothContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
    {
        log_and_throw_error("[{}] Barrier stiffness update not implemented!", name());
    }
}