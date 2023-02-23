#pragma once

#include "Parametrization.hpp"

namespace polyfem::solver
{
    class SDF2Mesh : public Parametrization
    {
    public:
        SDF2Mesh(const std::string inflator_path, const std::string sdf_velocity_path = "micro-tmp-velocity.msh", const std::string msh_path = "micro-tmp.msh");

        int size(const int x_size) const override;

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;
    
    private:
        bool isosurface_inflator(const Eigen::VectorXd &x) const;

        std::string inflator_path_, sdf_velocity_path_, msh_path_;

        mutable Eigen::VectorXd last_x;
    };
}