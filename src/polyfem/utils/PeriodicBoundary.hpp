#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <vector>

namespace polyfem
{
    namespace basis
    {
        class ElementBases;
    }

    namespace mesh
    {
        class Mesh;
        class MeshNodes;
        class LocalBoundary;
    }
}

namespace polyfem::utils
{
    class PeriodicBoundary
    {
    public:
        PeriodicBoundary(
            const mesh::Mesh &mesh,
            const std::vector<mesh::LocalBoundary> &total_local_boundary,
            const bool is_scalar,
            const int n_bases,
            const std::vector<basis::ElementBases> &bases,
            const std::shared_ptr<mesh::MeshNodes> &mesh_nodes,
            const std::vector<bool> &periodic_dimensions);

        int full_to_periodic(StiffnessMatrix &A) const;
		Eigen::MatrixXd full_to_periodic(const Eigen::MatrixXd &b, bool accumulate) const;
		std::vector<int> full_to_periodic(const std::vector<int> &boundary_nodes) const;

        inline int n_periodic_dof() const { return full_to_periodic_map_.maxCoeff() + 1; }
        inline bool is_periodic_dof(const int idx) const { return periodic_mask_[idx]; }

		Eigen::MatrixXd periodic_to_full(const int ndofs, const Eigen::MatrixXd &x_periodic) const;

        bool all_direction_periodic() const;
        bool has_periodic_bc() const;

    private:
        int problem_dim_;
        Eigen::VectorXi full_to_periodic_map_;
        Eigen::VectorXi periodic_mask_;

        std::vector<bool> periodic_dimensions_;
    };
}