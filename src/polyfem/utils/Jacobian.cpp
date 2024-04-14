#include <jacobian/element_validity.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Rational.hpp>
#include "Jacobian.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <paraviewo/ParaviewWriter.hpp>
#include <paraviewo/VTUWriter.hpp>
#include <paraviewo/HDF5VTUWriter.hpp>
namespace polyfem::utils
{
    std::tuple<bool, int, Tree>
    isValid(
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
        Tree tree;
        if (dim == 2)
        {
            typename element_validity::BezierChecker<2>::SubdivisionHierarchy hierarchy(element_validity::shapes::TRIANGLE);
            flag = element_validity::isValid<2>(cp, element_validity::shapes::TRIANGLE, order, &invalid_id, &hierarchy);
            if (!flag)
            {
                std::function<void(const typename element_validity::BezierChecker<2>::SubdivisionHierarchy::Node &, Tree &)> copy_hierarchy = [&copy_hierarchy](const typename element_validity::BezierChecker<2>::SubdivisionHierarchy::Node &src, Tree &dst) {
                    if (src.hasChildren())
                    {
                        dst.add_children(4);
                        for (int i = 0; i < 4; i++)
                            copy_hierarchy(*(src.children[i]), dst.child(i));
                    }
                };

                copy_hierarchy(hierarchy.get_root(), tree);
            }
        }
        else
        {
            typename element_validity::BezierChecker<3>::SubdivisionHierarchy hierarchy(element_validity::shapes::TETRAHEDRON);
            flag = element_validity::isValid<3>(cp, element_validity::shapes::TETRAHEDRON, order, &invalid_id, &hierarchy);
            if (!flag)
            {
                std::function<void(const typename element_validity::BezierChecker<3>::SubdivisionHierarchy::Node &, Tree &)> copy_hierarchy = [&copy_hierarchy](const typename element_validity::BezierChecker<3>::SubdivisionHierarchy::Node &src, Tree &dst) {
                    if (src.hasChildren())
                    {
                        dst.add_children(8);
                        for (int i = 0; i < 8; i++)
                            copy_hierarchy(*(src.children[i]), dst.child(i));
                    }
                };

                copy_hierarchy(hierarchy.get_root(), tree);
            }
        }

        // export invalid element in rational form
        // if (!flag)
        // {
        //     static Eigen::MatrixXd invalid_cp;
        //     if (invalid_cp.rows() == 0)
        //         invalid_cp = cp.block(invalid_id * n_basis_per_cell, 0, n_basis_per_cell, dim);
        //     else
        //     {
        //         invalid_cp.conservativeResize(invalid_cp.rows() + n_basis_per_cell, Eigen::NoChange);
        //         invalid_cp.block(invalid_cp.rows() - n_basis_per_cell, 0, n_basis_per_cell, dim) = cp.block(invalid_id * n_basis_per_cell, 0, n_basis_per_cell, dim);
        //     }

        //     if (invalid_cp.rows() % (10*n_basis_per_cell) == 0)
        //     {
        //         std::vector<std::string> nodes_rational;
        //         nodes_rational.resize(invalid_cp.rows() * 2 * dim);

        //         for (int i = 0; i < invalid_cp.rows(); i++)
        //         {
        //             for (int d = 0; d < dim; d++)
        //             {
        //                 utils::Rational num(invalid_cp(i, d));
        //                 nodes_rational[i * (2 * dim) + d * 2 + 0] = num.get_numerator_str();
        //                 nodes_rational[i * (2 * dim) + d * 2 + 1] = num.get_denominator_str();
        //             }
        //         }
        //         std::string path = "/home/zizhou/positive-jacobian/result/positive-jacobian/validation/invalid_element.hdf5";
        //         std::filesystem::remove(path);
        //         paraviewo::HDF5MatrixWriter::write_matrix(path, dim, invalid_cp.rows() / n_basis_per_cell, n_basis_per_cell, nodes_rational);
        //         logger().debug("Saved invalid element to invalid_element.hdf5");
        //     }
        // }
        
        return {flag, invalid_id, tree};
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
