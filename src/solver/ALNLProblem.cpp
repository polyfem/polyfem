#include <polyfem/ALNLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>

#include <igl/write_triangle_mesh.h>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
    using namespace polysolve;

    ALNLProblem::ALNLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const double weight)
        : super(state, rhs_assembler, t, dhat, project_to_psd, true), weight_(weight)
    {
        std::vector<Eigen::Triplet<double>> entries;

        // stop_dist_ = 1e-2 * state.min_edge_length;

        for (const auto bn : state.boundary_nodes)
            entries.emplace_back(bn, bn, 2 * weight_);

        hessian_.resize(state.n_bases * state.mesh->dimension(), state.n_bases * state.mesh->dimension());
        hessian_.setFromTriplets(entries.begin(), entries.end());
        hessian_.makeCompressed();

        displaced_.resize(hessian_.rows(), 1);
        displaced_.setZero();

        rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, displaced_, t);

        std::vector<bool> mask(hessian_.rows(), true);

        for (const auto bn : state.boundary_nodes)
            mask[bn] = false;

        for (int i = 0; i < mask.size(); ++i)
            if (mask[i])
                not_boundary_.push_back(i);
    }

    void ALNLProblem::compute_distance(const TVector &x, TVector &res)
    {
        res = x - displaced_;

        for (const auto bn : not_boundary_)
            res[bn] = 0;
    }

    double ALNLProblem::value(const TVector &x)
    {
        const double val = super::value(x);
        TVector distv;
        compute_distance(x, distv);
        const double dist = distv.squaredNorm();

        logger().trace("dist {}", sqrt(dist));

        return val + weight_ * dist / _barrier_stiffness;
        // return weight_ * dist / _barrier_stiffness;
    }

    void ALNLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv)
    {
        TVector tmp;
        super::gradient_no_rhs(x, gradv);
        compute_distance(x, tmp);
        //logger().trace("dist grad {}", tmp.norm());
        tmp *= 2 * weight_ / _barrier_stiffness;

        gradv += tmp;
        // gradv = tmp;
    }

    void ALNLProblem::hessian_full(const TVector &x, THessian &hessian)
    {
        super::hessian_full(x, hessian);
        hessian += hessian_ / _barrier_stiffness;
        // hessian = hessian_ / _barrier_stiffness;

        hessian.makeCompressed();
    }

    bool ALNLProblem::stop(const TVector &x)
    {
        // TVector distv;
        // compute_distance(x, distv);
        // const double dist = distv.norm();

        return false; //dist < stop_dist_;
    }
} // namespace polyfem
