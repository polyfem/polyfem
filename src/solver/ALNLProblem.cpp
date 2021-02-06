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

    ALNLProblem::ALNLProblem(State &state, const RhsAssembler &rhs_assembler, const double prev_t, const double t, const double dhat, const bool project_to_psd, const double weight)
        : nl_problem(state, rhs_assembler, prev_t, dhat, project_to_psd, true), state(state), weight_(weight)
    {
        std::vector<Eigen::Triplet<double>> entries;

        for (const auto bn : state.boundary_nodes)
            entries.emplace_back(bn, bn, 2);

        hessian_.resize(state.n_bases * state.mesh->dimension(), state.n_bases * state.mesh->dimension());
        hessian_.setFromTriplets(entries.begin(), entries.end());
        hessian_.makeCompressed();

        displaced_.resize(hessian_.rows(), 1);
        displaced_.setZero();

        rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, displaced_, t);
    }

    void ALNLProblem::init(const TVector &full)
    {
        nl_problem.init(full);
    }

    void ALNLProblem::init_timestep(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt)
    {
        nl_problem.init_timestep(x_prev, v_prev, a_prev, dt);
    }

    void ALNLProblem::compute_distance(const TVector &x, TVector &res)
    {
        res = x - displaced_;

        for (const auto bn : state.boundary_nodes)
            res[bn] = 0;
    }

    void ALNLProblem::update_quantities(const double t, const TVector &x)
    {
        nl_problem.update_quantities(t, x);
    }

    void ALNLProblem::substepping(const double t)
    {
        nl_problem.substepping(t);
    }

    double ALNLProblem::max_step_size(const TVector &x0, const TVector &x1)
    {
        return nl_problem.max_step_size(x0, x1);
    }

    bool ALNLProblem::is_step_valid(const TVector &x0, const TVector &x1)
    {
        return nl_problem.is_step_valid(x0, x1);
    }

    double ALNLProblem::value(const TVector &x)
    {
        const double val = nl_problem.value(x);
        TVector distv;
        compute_distance(x, distv);
        const double dist = distv.squaredNorm();

        std::cout << dist << std::endl;

        return val + weight_ * dist;
    }

    void ALNLProblem::gradient(const TVector &x, TVector &gradv)
    {
        TVector nl_grad;
        nl_problem.gradient(x, nl_grad);
        compute_distance(x, gradv);
        gradv *= 2 * weight_;

        gradv += nl_grad;
    }

    void ALNLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv)
    {
        TVector tmp;
        nl_problem.gradient_no_rhs(x, gradv);
        compute_distance(x, tmp);
        tmp *= 2 * weight_;

        gradv += tmp;
    }

    void ALNLProblem::hessian(const TVector &x, THessian &hessian)
    {
        nl_problem.hessian(x, hessian);
        hessian += hessian_;

        hessian.makeCompressed();
    }
} // namespace polyfem
