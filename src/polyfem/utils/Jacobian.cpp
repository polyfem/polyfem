#include <element_validity.hpp>
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
    Eigen::MatrixXd extract_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, int order, int n_elem)
    {
        if (n_elem < 0)
            n_elem = bases.size();
        Eigen::MatrixXd local_pts;
        if (dim == 3)
            autogen::p_nodes_3d(order, local_pts);
        else
            autogen::p_nodes_2d(order, local_pts);
        const int n_basis_per_cell = local_pts.rows();
        Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(n_elem * n_basis_per_cell, dim);
        for (int e = 0; e < n_elem; ++e)
        {
            ElementAssemblyValues vals;
            vals.compute(e, dim == 3, local_pts, bases[e], gbases[e]);

            for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
                for (const auto &g : vals.basis_values[j].global)
                    cp.middleRows(e * n_basis_per_cell, n_basis_per_cell) += g.val * vals.basis_values[j].val * u.segment(g.index * dim, dim).transpose();

            Eigen::MatrixXd mapped;
            gbases[e].eval_geom_mapping(local_pts, mapped);
            cp.middleRows(e * n_basis_per_cell, n_basis_per_cell) += mapped;
        }
        return cp;
    }

    Eigen::MatrixXd extract_nodes(const int dim, const basis::ElementBases &basis, const basis::ElementBases &gbasis, const Eigen::VectorXd &u, int order)
    {
        Eigen::MatrixXd local_pts;
        if (dim == 3)
            autogen::p_nodes_3d(order, local_pts);
        else
            autogen::p_nodes_2d(order, local_pts);
            
        Eigen::MatrixXd cp;
        gbasis.eval_geom_mapping(local_pts, cp);

        ElementAssemblyValues vals;
        vals.compute(0, dim == 3, local_pts, basis, gbasis);
        for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
            for (const auto &g : vals.basis_values[j].global)
                cp += g.val * vals.basis_values[j].val * u.segment(g.index * dim, dim).transpose();

        return cp;
    }

    Eigen::VectorXd robust_evaluate_jacobian(
        const int order,
        const Eigen::MatrixXd &cp,
        const Eigen::MatrixXd &uv)
    {
        #define JAC_EVAL(n,s,p) \
            case p: { \
                JacobianEvaluator<n,s,p> evaluator(cp); \
                return evaluator.eval(uv, 0); \
            }

        if (cp.cols() == 2) {
            switch (order) {
                JAC_EVAL(2,2,1)
                JAC_EVAL(2,2,2)
                JAC_EVAL(2,2,3)
                JAC_EVAL(2,2,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }
        else {
            switch (order) {
                JAC_EVAL(3,3,1)
                JAC_EVAL(3,3,2)
                JAC_EVAL(3,3,3)
                // JAC_EVAL(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef JAC_EVAL
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

        std::vector<uint> invalidList;

        #define CHECK_STATIC(n,s,p) \
            case p: { \
                StaticValidator<n,s,p> check(16 /*no. of threads*/); \
                check.isValid(cp, nullptr, nullptr, &invalidList); \
                break; \
            }

        if (cp.cols() == 2) {
            switch (order) {
                CHECK_STATIC(2,2,1)
                CHECK_STATIC(2,2,2)
                CHECK_STATIC(2,2,3)
                CHECK_STATIC(2,2,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }
        else {
            switch (order) {
                CHECK_STATIC(3,3,1)
                CHECK_STATIC(3,3,2)
                CHECK_STATIC(3,3,3)
                // CHECK_STATIC(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef CHECK_STATIC
        
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

        #define CHECK_STATIC(n,s,p) \
            case p: { \
                std::vector<unsigned> hierarchy; \
                StaticValidator<n,s,p> check(16 /*no. of threads*/); \
                const auto flag_ = check.isValid(cp, &hierarchy, &invalid_id); \
                flag = flag_ == Validity::valid; \
                if (!flag) { \
                    Tree &dst = tree; \
                    for (const auto i : hierarchy) { \
                        dst.add_children(1<<n); \
                        dst = dst.child(i); \
                    } \
                } \
                break; \
            }

        if (dim == 2) {
            switch (order) {
                CHECK_STATIC(2,2,1)
                CHECK_STATIC(2,2,2)
                CHECK_STATIC(2,2,3)
                CHECK_STATIC(2,2,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }
        else {
            switch (order) {
                CHECK_STATIC(3,3,1)
                CHECK_STATIC(3,3,2)
                CHECK_STATIC(3,3,3)
                // CHECK_STATIC(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef CHECK_STATIC
        
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

        #define CHECK_CONTINUOUS(n,s,p) \
            case p: { \
                std::vector<unsigned> hierarchy; \
                ContinuousValidator<n,s,p> check(16 /*no. of threads*/); \
                check.setPrecisionTarget(1); \
                const auto flag_ = check.maxTimeStep(cp1, cp2, &hierarchy, &invalid_id); \
                flag = flag_ == 1.; \
                break; \
            }

        if (dim == 2) {
            switch (order) {
                CHECK_CONTINUOUS(2,2,1)
                CHECK_CONTINUOUS(2,2,2)
                CHECK_CONTINUOUS(2,2,3)
                CHECK_CONTINUOUS(2,2,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }
        else {
            switch (order) {
                CHECK_CONTINUOUS(3,3,1)
                CHECK_CONTINUOUS(3,3,2)
                CHECK_CONTINUOUS(3,3,3)
                // CHECK_CONTINUOUS(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef CHECK_CONTINUOUS
        
        return flag;
    }

    void print_eigen(const Eigen::MatrixXd &mat)
    {
        for (int i = 0; i < mat.rows(); i++)
        {
            for (int j = 0; j < mat.cols(); j++)
            {
                std::cout << mat(i, j);
                if (i < mat.rows() - 1 || j < mat.cols() - 1)
                    std::cout << ", ";
            }
        }
    }

    void export_static_hdf5(
        const std::string &path,
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases,
        const Eigen::VectorXd &u)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());
        Eigen::MatrixXd cp = extract_nodes(dim, bases, gbases, u, order);
        const int n_elem = bases.size();
        std::vector<std::string> nodes_rational;
        nodes_rational.resize(n_elem * n_basis_per_cell * 2 * dim);
        for (int e = 0; e < n_elem; e++)
        {
            for (int i = 0; i < n_basis_per_cell; i++)
            {
                const int idx = i + n_basis_per_cell * e;
                Eigen::Matrix<double, -1, 1, Eigen::ColMajor, 3, 1> pos = cp.row(idx);

                for (int d = 0; d < dim; d++)
                {
                    utils::Rational num(pos(d));
                    nodes_rational[idx * (2 * dim) + d * 2 + 0] = num.get_numerator_str();
                    nodes_rational[idx * (2 * dim) + d * 2 + 1] = num.get_denominator_str();
                }
            }
        }
        paraviewo::HDF5MatrixWriter::write_matrix(path, dim, n_elem, n_basis_per_cell, nodes_rational);
    }

    void export_transient_hdf5(
        const std::string &path,
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases,
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());
        Eigen::MatrixXd cp1 = extract_nodes(dim, bases, gbases, u1, order);
        Eigen::MatrixXd cp2 = extract_nodes(dim, bases, gbases, u2, order);
        const int n_elem = bases.size();
        std::vector<std::string> nodes_rational;
        nodes_rational.resize(n_elem * n_basis_per_cell * 4 * dim);
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
        paraviewo::HDF5MatrixWriter::write_matrix(path, dim, n_elem, n_basis_per_cell, nodes_rational);
    }

    std::tuple<double, int, double, Tree> maxTimeStep(
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

        // logger().debug("Jacobian check order {}, number of nodes per cell {}, number of total nodes {}", order, n_basis_per_cell, cp2.rows());

        unsigned invalid_id = -1;
        bool gaveUp = false;
        double step = 1;
        double invalid_step = 1.;
        Tree tree;

        #define MAX_TIME_STEP(n,s,p) \
            case p: { \
                std::vector<unsigned> hierarchy; \
                ContinuousValidator<n,s,p> check(16 /*no. of threads*/); \
                ContinuousValidator<n,s,p>::Info info; \
                step = check.maxTimeStep( \
                    cp1, cp2, &hierarchy, &invalid_id, &invalid_step, &info \
                ); \
                gaveUp = !info.success(); \
                if (step < 1) { \
                    Tree &dst = tree; \
                    for (const auto i : hierarchy) { \
                        dst.add_children(1<<n); \
                        dst = dst.child(i); \
                    } \
                } \
                break; \
            }

        if (dim == 2) {
            switch (order) {
                MAX_TIME_STEP(2,2,1)
                MAX_TIME_STEP(2,2,2)
                MAX_TIME_STEP(2,2,3)
                MAX_TIME_STEP(2,2,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }
        else {
            switch (order) {
                MAX_TIME_STEP(3,3,1)
                MAX_TIME_STEP(3,3,2)
                MAX_TIME_STEP(3,3,3)
                // MAX_TIME_STEP(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef MAX_TIME_STEP

        if (gaveUp)
            logger().warn("Jacobian check gave up!");

        // if (gaveUp) {
        //     static int idx = 0;
        //     std::string path = "transient_" + std::to_string(idx++) + ".hdf5";
        //     export_transient_hdf5(path, dim, bases, gbases, u1, u2);
        //     logger().info("Save to {}", path);
        //     // std::terminate();
        // }

        return {step, invalid_id, invalid_step, tree};
    }
}
