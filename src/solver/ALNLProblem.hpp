#pragma once

#include <polyfem/NLProblem.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/RhsAssembler.hpp>
#include <polyfem/State.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
    class ALNLProblem : public cppoptlib::Problem<double>
    {
    public:
        using typename cppoptlib::Problem<double>::Scalar;
        using typename cppoptlib::Problem<double>::TVector;
        typedef StiffnessMatrix THessian;

        ALNLProblem(State &state, const RhsAssembler &rhs_assembler, const double prev_t, const double t, const double dhat, const bool project_to_psd, const double weight);
        void init(const TVector &displacement);
        void init_timestep(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt);
        TVector initial_guess();

        double value(const TVector &x) override;
        void gradient(const TVector &x, TVector &gradv) override;
        void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv);

        bool is_step_valid(const TVector &x0, const TVector &x1);
        double max_step_size(const TVector &x0, const TVector &x1);

#include <polyfem/DisableWarnings.hpp>
        void hessian(const TVector &x, THessian &hessian);
        void hessian_full(const TVector &x, THessian &gradv) { hessian(x, gradv); }
#include <polyfem/EnableWarnings.hpp>

        void update_quantities(const double t, const TVector &x);
        void substepping(const double t);

        const Eigen::MatrixXd &current_rhs() { return nl_problem.current_rhs(); }

        void full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const { reduced = full; }
        void reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full) { full = reduced; }

    private:
        NLProblem nl_problem;
        State &state;
        const double weight_;
        THessian hessian_;
        Eigen::MatrixXd displaced_;

        void compute_distance(const TVector &x, TVector &res);
    };
} // namespace polyfem
