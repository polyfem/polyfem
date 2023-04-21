#pragma once

#include "Parametrization.hpp"
#include <polyfem/Common.hpp>

namespace polyfem::solver
{
    // generates a unit size mesh using isosurface inflator
    class SDF2Mesh : public Parametrization
    {
    public:
        SDF2Mesh(const std::string binary_path, const std::string wire_path, const std::string out_path, const bool volume_velocity, const json &opts) : volume_velocity_(volume_velocity), dim_(2), binary_path_(binary_path), wire_path_(wire_path), out_path_(out_path), opts_(opts) {}

        int size(const int x_size) const override;

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;
    
        Eigen::VectorXd displace_vertices(const Eigen::VectorXd &x) const;
    private:
        bool isosurface_inflator(const Eigen::VectorXd &x) const;
        void extend_to_internal() const;

        const bool volume_velocity_;

        const int dim_;
        const std::string binary_path_, wire_path_, out_path_;
        const json opts_;
        
        mutable Eigen::VectorXd last_x;
        mutable Eigen::MatrixXd Vout, shape_velocity;
        mutable Eigen::MatrixXi Fout;
        mutable Eigen::Matrix<bool, -1, 1> boundary_flags;
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

        mutable Eigen::VectorXd last_x;
    };

    class PeriodicMeshToMesh : public Parametrization
    {
    public:
        PeriodicMeshToMesh(const Eigen::MatrixXd &V);

        int size(const int x_size) const override { assert(x_size == input_size()); return dependent_map.size() * dim_; }
        int input_size() const { return n_periodic_dof_ * dim_ + dim_; }
        int n_periodic_dof() const { return n_periodic_dof_; }
        int n_full_dof() const { return dependent_map.size(); }
        int dim() const { return dim_; }

        Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
        Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
        Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const override;

        int full_to_periodic(int i) const { return dependent_map(i); }

    private:
        int dim_;
        int n_periodic_dof_;
        Eigen::VectorXi dependent_map;
        std::array<std::vector<std::array<int, 2>>, 3> periodic_dependence; // <id1, id2> for 2/3 axis
    };
}