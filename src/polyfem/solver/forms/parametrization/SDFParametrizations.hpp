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

        const std::string inflator_path_, sdf_velocity_path_, msh_path_;

        mutable Eigen::VectorXd last_x;
        mutable Eigen::MatrixXd Vout, vertex_normals, shape_vel;
        mutable Eigen::MatrixXi Fout;
    };

    class MeshTiling : public Parametrization
    {
    public:
        MeshTiling(const Eigen::VectorXi &nums, const std::string in_path, const std::string out_path);

        int size(const int x_size) const override;

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;
    
    private:
        const Eigen::VectorXi nums_;
        const std::string in_path_, out_path_;

        bool tiling(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &Vnew, Eigen::MatrixXi &Fnew) const;

        mutable Eigen::MatrixXd last_V;
        mutable Eigen::VectorXi index_map;
    };

    class MeshAffine : public Parametrization
    {
    public:
        MeshAffine(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const std::string in_path, const std::string out_path);

        int size(const int x_size) const override { return x_size; }

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;
    
    private:
        const Eigen::MatrixXd A_;
        const Eigen::VectorXd b_;
        const std::string in_path_, out_path_;
    };
}