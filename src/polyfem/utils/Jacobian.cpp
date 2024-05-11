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
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/io/Evaluator.hpp>

using namespace element_validity;
using namespace polyfem::assembler;

namespace polyfem::utils
{
    Eigen::MatrixXd extract_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, int order)
    {
        Eigen::MatrixXd local_pts;
        if (dim == 3)
            autogen::p_nodes_3d(order, local_pts);
        else
            autogen::p_nodes_2d(order, local_pts);
        const int n_basis_per_cell = local_pts.rows();
        Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(bases.size() * n_basis_per_cell, dim);
        // auto tmp = cp;
        for (int e = 0; e < bases.size(); ++e)
        {
            // for (int i = 0; i < bases[e].bases.size(); ++i)
            // {
            //     const auto &g = bases[e].bases[i].global()[0];

            //     tmp.row(e * n_basis_per_cell + i) = g.node + u.segment(g.index * dim, dim).transpose();

            //     if (bases[e].bases[i].global().size() > 1 || g.val < 1)
            //         logger().error("More than 1 global! {}", g.val);
            // }

            ElementAssemblyValues vals;
            vals.compute(e, dim == 3, local_pts, bases[e], gbases[e]);
            for (int p = 0; p < local_pts.rows(); p++)
            {
                for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
                {
                    for (const auto &g : vals.basis_values[j].global)
                        cp.row(e * n_basis_per_cell + p) += g.val * u.segment(g.index * dim, dim).transpose() * vals.basis_values[j].val(p);
                }
            }

            Eigen::MatrixXd mapped;
            gbases[e].eval_geom_mapping(local_pts, mapped);
            cp.middleRows(e * n_basis_per_cell, n_basis_per_cell) += mapped;

            // if (order == bases[0].bases[0].order() && (tmp.middleRows(e * n_basis_per_cell, n_basis_per_cell) - cp.middleRows(e * n_basis_per_cell, n_basis_per_cell)).norm() > 1e-8)
            // {
            //     std::cout << std::setprecision(16) << "method 1:\n" << tmp.middleRows(e * n_basis_per_cell, n_basis_per_cell) << 
            //     "\nmethod 2:\n" << cp.middleRows(e * n_basis_per_cell, n_basis_per_cell) << 
            //     "\ndiff:\n" << (tmp.middleRows(e * n_basis_per_cell, n_basis_per_cell) - cp.middleRows(e * n_basis_per_cell, n_basis_per_cell)) << "\n";
            //     std::cout << "mapped:\n" << mapped << "\nnodes:\n";
            //     for (int i = 0; i < bases[e].bases.size(); ++i)
            //     {
            //         const auto &g = bases[e].bases[i].global()[0];
            //         std::cout << g.node << "\n";
            //     }
            //     std::cout << std::endl;
            //     std::terminate();
            // }
        }
        return cp;
    }

    Eigen::VectorXd robust_evaluate_jacobian(
        const int order,
        const Eigen::MatrixXd &cp,
        const Eigen::MatrixXd &uv)
    {
        if (cp.cols() == 2)
        {
            StaticChecker<2> check(cp, shapes::TRIANGLE, order);
            return check.jacobian(0, uv);
        }
        else
        {
            StaticChecker<3> check(cp, shapes::TETRAHEDRON, order);
            return check.jacobian(0, uv);
        }
    }

    std::vector<uint> count_invalid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases, 
        const Eigen::VectorXd &u)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());
        const Eigen::MatrixXd cp = extract_nodes(dim, bases, gbases, u, order);

        std::tuple<uint, uint, uint> counters{0,0,0};
        std::vector<uint> invalidList;
        if (dim == 2)
        {
            StaticChecker<2> check(cp, shapes::TRIANGLE, order);
            check.isValid(0, nullptr, nullptr, &counters, &invalidList);
        }
        else
        {
            StaticChecker<3> check(cp, shapes::TETRAHEDRON, order);
            check.isValid(0, nullptr, nullptr, &counters, &invalidList);
        }
        
        return invalidList;
    }

    std::tuple<bool, int, Tree>
    isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases, 
        const Eigen::VectorXd &u,
        const double threshold)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());
        Eigen::MatrixXd cp = extract_nodes(dim, bases, gbases, u, order);

        bool flag = false;
        unsigned invalid_id = 0;
        Tree tree;
        if (dim == 2)
        {
            SubdivisionHierarchy<2> hierarchy(shapes::TRIANGLE);
            StaticChecker<2> check(cp, shapes::TRIANGLE, order);
            const auto flag_ = check.isValid(threshold, &invalid_id, &hierarchy);
            flag = flag_ == Validity::Valid;

            if (!flag)
            {
                std::function<void(const SubdivisionHierarchy<2>::Node &, Tree &)> copy_hierarchy = [&copy_hierarchy](const SubdivisionHierarchy<2>::Node &src, Tree &dst) {
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
            SubdivisionHierarchy<3> hierarchy(shapes::TETRAHEDRON);
            StaticChecker<3> check(cp, shapes::TETRAHEDRON, order);
            const auto flag_ = check.isValid(threshold, &invalid_id, &hierarchy);
            flag = flag_ == Validity::Valid;

            if (!flag)
            {
                std::function<void(const SubdivisionHierarchy<3>::Node &, Tree &)> copy_hierarchy = [&copy_hierarchy](const SubdivisionHierarchy<3>::Node &src, Tree &dst) {
                    if (src.hasChildren())
                    {
                        dst.add_children(8);
                        for (int i = 0; i < 8; i++)
                            copy_hierarchy(*(src.children[i]), dst.child(i));
                    }
                };

                copy_hierarchy(hierarchy.get_root(), tree);
            }

            if (tree.depth() > 25)
            {
                std::cout << "hard element:" << threshold << " " << invalid_id << "\n";
                std::cout << std::setprecision(20) << cp.middleRows(n_basis_per_cell * invalid_id, n_basis_per_cell) << "\n";
                std::cout << hierarchy << "\n";
                std::ofstream file("hard-elem.txt");
                file << std::setprecision(20) << cp << "\n";
                std::terminate();
            }
        }

        // {
        //     std::vector<std::string> nodes_rational;
        //     nodes_rational.resize(cp.rows() * 2 * dim);

        //     for (int i = 0; i < cp.rows(); i++)
        //     {
        //         for (int d = 0; d < dim; d++)
        //         {
        //             utils::Rational num(cp(i, d));
        //             nodes_rational[i * (2 * dim) + d * 2 + 0] = num.get_numerator_str();
        //             nodes_rational[i * (2 * dim) + d * 2 + 1] = num.get_denominator_str();
        //         }
        //     }
        //     std::string path = "check.hdf5";
        //     std::filesystem::remove(path);
        //     paraviewo::HDF5MatrixWriter::write_matrix(path, dim, cp.rows() / n_basis_per_cell, n_basis_per_cell, nodes_rational);
        //     logger().debug("Saved invalid element to check.hdf5");
        //     std::terminate();
        // }
        
        return {flag, invalid_id, tree};
    }

    bool isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases, 
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2,
        const double threshold)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());

        const Eigen::MatrixXd cp1 = extract_nodes(dim, bases, gbases, u1, order);
        const Eigen::MatrixXd cp2 = extract_nodes(dim, bases, gbases, u2, order);

        bool flag = false;
        unsigned invalid_id = 0;
        if (dim == 2)
        {
            DynamicChecker<2> check(cp1, cp2, shapes::TRIANGLE, order);
            const auto flag_ = check.isValid(threshold, &invalid_id);
            flag = flag_ == Validity::Valid;
        }
        else
        {
            DynamicChecker<3> check(cp1, cp2, shapes::TETRAHEDRON, order);
            const auto flag_ = check.isValid(threshold, &invalid_id);
            flag = flag_ == Validity::Valid;
        }
        
        return flag;
    }

    std::tuple<double, int, Tree> maxTimeStep(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases, 
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2,
        double precision)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());
        Eigen::MatrixXd cp1 = extract_nodes(dim, bases, gbases, u1, order);
        Eigen::MatrixXd cp2 = extract_nodes(dim, bases, gbases, u2, order);

        unsigned invalidID = -1;
        bool gaveUp = false;
        double step = 1;
        Tree tree;
        if (dim == 2)
        {
            SubdivisionHierarchy<2> hierarchy(shapes::TRIANGLE);
            DynamicChecker<2> check(cp1, cp2, shapes::TRIANGLE, order);
            step = check.maxTimeStep(0, precision, &invalidID, &hierarchy, &gaveUp);
            if (step < 1)
            {
                std::function<void(const SubdivisionHierarchy<2>::Node &, Tree &)> copy_hierarchy = [&copy_hierarchy](const SubdivisionHierarchy<2>::Node &src, Tree &dst) {
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
            SubdivisionHierarchy<3> hierarchy(shapes::TETRAHEDRON);
            DynamicChecker<3> check(cp1, cp2, shapes::TETRAHEDRON, order);
            step = check.maxTimeStep(0, precision, &invalidID, &hierarchy, &gaveUp);
            if (step < 1)
            {
                std::function<void(const SubdivisionHierarchy<3>::Node &, Tree &)> copy_hierarchy = [&copy_hierarchy](const SubdivisionHierarchy<3>::Node &src, Tree &dst) {
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

        if (gaveUp || step == 0) {
            static int idx = 0;
            std::string path = "zero_step_" + std::to_string(idx++) + ".hdf5";
            const int n_elem = bases.size();
            std::vector<std::string> nodes_rational;
            nodes_rational.resize(n_elem * n_basis_per_cell * 4 * dim);
            // utils::maybe_parallel_for(n_elem, [&](int start, int end, int thread_id) {
                for (int e = 0; e < n_elem; e++)
                {
                    for (int i = 0; i < n_basis_per_cell; i++)
                    {
                        const int idx = i + n_basis_per_cell * e;
                        Eigen::Matrix<double, -1, 1, Eigen::ColMajor, 3, 1> pos = cp2.row(idx);

                        for (int d = 0; d < dim; d++)
                        {
                            utils::Rational num(pos(d));
                            nodes_rational[idx * (4 * dim) + d * 4 + 2] = num.get_numerator_str();
                            nodes_rational[idx * (4 * dim) + d * 4 + 3] = num.get_denominator_str();
                        }

                        pos = cp1.row(idx);

                        for (int d = 0; d < dim; d++)
                        {
                            utils::Rational num(pos(d));
                            nodes_rational[idx * (4 * dim) + d * 4 + 0] = num.get_numerator_str();
                            nodes_rational[idx * (4 * dim) + d * 4 + 1] = num.get_denominator_str();
                        }
                    }
                }
            // });
            // paraviewo::HDF5MatrixWriter::write_matrix(rawname + ".hdf5", dim, bases.size(), n_basis_per_cell, nodes);
            paraviewo::HDF5MatrixWriter::write_matrix(path, dim, n_elem, n_basis_per_cell, nodes_rational);
            // for (auto& s : nodes_rational)
            // 	std::cout << s << ",";
            logger().info("Save to {}", path);
            // std::terminate();
        }

        return {step, invalidID, tree};
    }
}
