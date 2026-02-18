#pragma once

#include "Parametrization.hpp"
#include <polyfem/Common.hpp>
#include <set>

namespace polyfem::solver
{
    // aux parametrization that cannot be used for optimization
    class PeriodicMeshToMesh : public Parametrization
    {
    public:
        PeriodicMeshToMesh(const Eigen::MatrixXd &V);

        int size(const int x_size) const override { assert(x_size == input_size()); return dependent_map.size() * dim_; }
        int input_size() const { return n_periodic_dof_ * dim_ + dim_ * dim_; }
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
        std::array<std::set<std::array<int, 2>>, 3> periodic_dependence; // <id1, id2> for 2/3 axis
    };
}
