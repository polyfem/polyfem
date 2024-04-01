#include <jacobian/element_validity.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Rational.hpp>
#include "Jacobian.hpp"
#include <iostream>
#include <fstream>

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

        bool flag = false;
        unsigned invalid_id = 0;
        if (dim == 2)
            flag = element_validity::isValid<2>(cp, element_validity::shapes::TRIANGLE, order, &invalid_id);
        else
            flag = element_validity::isValid<3>(cp, element_validity::shapes::TETRAHEDRON, order, &invalid_id);

        if (!flag)
        {
            std::ofstream file("invalid_element.txt", std::ios_base::app);
            file << std::setprecision(20) << cp.block(invalid_id * n_basis_per_cell, 0, n_basis_per_cell, dim) << "\n" << std::endl;
        }
        
        return flag;
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

    bool isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2)
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
        Eigen::MatrixXd cp1 = Eigen::MatrixXd::Zero(bases.size() * n_basis_per_cell, dim);
        Eigen::MatrixXd cp2 = Eigen::MatrixXd::Zero(bases.size() * n_basis_per_cell, dim);
        for (int e = 0; e < bases.size(); ++e)
        {
            for (int i = 0; i < bases[e].bases.size(); ++i)
            {
                const auto &g = bases[e].bases[i].global()[0];

                cp1.row(e * n_basis_per_cell + i) = g.node + u1.segment(g.index * dim, dim).transpose();
                cp2.row(e * n_basis_per_cell + i) = g.node + u2.segment(g.index * dim, dim).transpose();
            }
        }

        bool flag = false;
        unsigned invalid_id = 0;
        if (dim == 2)
            flag = element_validity::isValid<2>(cp1, cp2, element_validity::shapes::TRIANGLE, order, &invalid_id);
        else
            flag = element_validity::isValid<3>(cp1, cp2, element_validity::shapes::TETRAHEDRON, order, &invalid_id);
        
        return flag;
    }

    double maxTimeStep(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2,
        double precision)
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
        Eigen::MatrixXd cp1 = Eigen::MatrixXd::Zero(bases.size() * n_basis_per_cell, dim);
        Eigen::MatrixXd cp2 = Eigen::MatrixXd::Zero(bases.size() * n_basis_per_cell, dim);
        for (int e = 0; e < bases.size(); ++e)
        {
            for (int i = 0; i < bases[e].bases.size(); ++i)
            {
                const auto &g = bases[e].bases[i].global()[0];

                cp1.row(e * n_basis_per_cell + i) = g.node + u1.segment(g.index * dim, dim).transpose();
                cp2.row(e * n_basis_per_cell + i) = g.node + u2.segment(g.index * dim, dim).transpose();
            }
        }

        if (dim == 2)
            return element_validity::maxTimeStep<2>(cp1, cp2, element_validity::shapes::TRIANGLE, order, precision);
        else
            return element_validity::maxTimeStep<3>(cp1, cp2, element_validity::shapes::TETRAHEDRON, order, precision);
    }
}
