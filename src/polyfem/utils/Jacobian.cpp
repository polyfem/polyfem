#include <element_validity.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include "Jacobian.hpp"
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/io/Evaluator.hpp>

using namespace element_validity;
using namespace polyfem::assembler;

namespace polyfem::utils
{
    namespace {
        template <int n, int s, int p>
        void check_transient(
            const int n_threads, 
            const Eigen::MatrixXd &cp1,
            const Eigen::MatrixXd &cp2,
            double &step, 
            int &invalid_id, 
            double &invalid_step, 
            bool &gaveUp, 
            Tree &tree)
        { 
            std::vector<int> hierarchy;
            ContinuousValidator<n,s,p> check(n_threads);
            typename ContinuousValidator<n,s,p>::Info info;
            step = check.maxTimeStep( 
                cp1, cp2, &hierarchy, &invalid_id, &invalid_step, &info 
            );
            gaveUp = !info.success();
            if (step < 1) { 
                Tree *dst = &tree;
                for (const auto i : hierarchy) { 
                    dst->add_children(1<<n);
                    dst = &(dst->child(i));
                } 
            }
        }
    }
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
                JAC_EVAL(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef JAC_EVAL
    }

    std::vector<int> count_invalid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases, 
        const Eigen::VectorXd &u)
    {
        const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
        const int n_basis_per_cell = std::max(bases[0].bases.size(), gbases[0].bases.size());
        const Eigen::MatrixXd cp = extract_nodes(dim, bases, gbases, u, order);

        std::vector<int> invalidList;

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
                CHECK_STATIC(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef CHECK_STATIC
        
        return invalidList;
    }

    std::tuple<bool, int, Tree>
    is_valid(
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
        int invalid_id = 0;
        Tree tree;

        #define CHECK_STATIC(n,s,p) \
            case p: { \
                std::vector<int> hierarchy; \
                StaticValidator<n,s,p> check(16 /*no. of threads*/); \
                const auto flag_ = check.isValid(cp, &hierarchy, &invalid_id); \
                flag = flag_ == Validity::valid; \
                if (!flag) { \
                    Tree *dst = &tree; \
                    for (const auto i : hierarchy) { \
                        dst->add_children(1<<n); \
                        dst = &(dst->child(i)); \
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
                CHECK_STATIC(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef CHECK_STATIC
        
        return {flag, invalid_id, tree};
    }

    bool is_valid(
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
        int invalid_id = 0;

        #define CHECK_CONTINUOUS(n,s,p) \
            case p: { \
                std::vector<int> hierarchy; \
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
                CHECK_CONTINUOUS(3,3,4)
                default: throw std::invalid_argument("Order not supported");
            }
        }

        #undef CHECK_CONTINUOUS
        
        return flag;
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

        int invalid_id = -1;
        bool gaveUp = false;
        double step = 1;
        double invalid_step = 1.;
        Tree tree;

        const int n_threads = utils::NThread::get().num_threads();

        if (dim == 2) {
            switch (order) {
                case 1:
                    check_transient<2, 2, 1>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                case 2:
                    check_transient<2, 2, 2>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                case 3:
                    check_transient<2, 2, 3>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                case 4:
                    check_transient<2, 2, 4>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                default:
                    throw std::invalid_argument("Order not supported");
            }
        }
        else {
            switch (order) {
                case 1:
                    check_transient<3, 3, 1>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                case 2:
                    check_transient<3, 3, 2>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                case 3:
                    check_transient<3, 3, 3>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                case 4:
                    check_transient<3, 3, 4>(n_threads, cp1, cp2, step, invalid_id, invalid_step, gaveUp, tree);
                    break;
                default:
                    throw std::invalid_argument("Order not supported");
            }
        }

        #undef MAX_TIME_STEP

        // if (gaveUp)
        //     logger().warn("Jacobian check gave up!");

        return {step, invalid_id, invalid_step, tree};
    }
}
