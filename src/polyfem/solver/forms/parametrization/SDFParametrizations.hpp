#pragma once

#include "Parametrization.hpp"

namespace polyfem::solver
{
    class SDF2Mesh : public Parametrization
    {
    public:
        SDF2Mesh(const std::string inflator_path, const std::string sdf_velocity_path = "micro-tmp-velocity.msh", const std::string msh_path = "micro-tmp.msh") : inflator_path_(inflator_path), sdf_velocity_path_(sdf_velocity_path), msh_path_(msh_path) {}

        int size(const int x_size) const override;

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;
    
    private:
        bool isosurface_inflator(const Eigen::VectorXd &x) const;

        std::string inflator_path_, sdf_velocity_path_, msh_path_;

        mutable Eigen::VectorXd last_x;
    };

    class MeshTiling : public Parametrization
    {
    public:
        MeshTiling(const Eigen::VectorXi &nums, const std::string in_path, const std::string out_path);

        int size(const int x_size) const override;

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;
    
    private:
        Eigen::VectorXi nums_;
        std::string in_path_, out_path_;

        bool tiling(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &Vnew, Eigen::MatrixXi &Fnew, Eigen::VectorXi &index_map) const;

        mutable Eigen::MatrixXd last_x;
    };
}