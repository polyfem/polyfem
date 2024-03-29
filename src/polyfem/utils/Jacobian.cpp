#include <jacobian/element_validity.hpp>
#include <polyfem/utils/Logger.hpp>
#include "Jacobian.hpp"

namespace polyfem::utils
{
    bool isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u)
    {
        int order = -1;
        for (const auto &b : bases)
        {
            for (const auto &bs : b.bases)
            {
                if (order < 0)
                    order = bs.order();
                else if (order != bs.order())
                    log_and_throw_error("All bases must have the same order");
            }
        }

        const int n_basis_per_cell = bases[0].bases.size();
        Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(bases.size() * n_basis_per_cell, dim);
        for (int e = 0; e < bases.size(); ++e)
        {
            for (int i = 0; i < bases[e].bases.size(); ++i)
            {
                const auto &g = bases[e].bases[i].global()[0];

                cp.row(e * n_basis_per_cell + i) = g.node + u.segment(g.index * dim, dim).transpose();
            }
        }

        if (dim == 2)
            return element_validity::isValid<2>(cp, element_validity::shapes::TRIANGLE, order);
        else
            return element_validity::isValid<3>(cp, element_validity::shapes::TETRAHEDRON, order);
    }

    template<unsigned int nVar>
    bool isValid(const Eigen::Matrix<double, -1, nVar> &cp, int order)
    {
        if constexpr (nVar == 2)
            return element_validity::isValid<2>(cp, element_validity::shapes::TRIANGLE, order);
        else if constexpr (nVar == 3)
            return element_validity::isValid<3>(cp, element_validity::shapes::TETRAHEDRON, order);
        else
            log_and_throw_error("Invalid dimension");
    }

    template bool isValid<2>(const Eigen::Matrix<double, -1, 2> &cp, int order);
    template bool isValid<3>(const Eigen::Matrix<double, -1, 3> &cp, int order);
}