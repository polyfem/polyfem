#pragma once

#include <polyfem/NLProblem.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/RhsAssembler.hpp>
#include <polyfem/State.hpp>

#include <cppoptlib/problem.h>

namespace polyfem
{
    class ALNLProblem : public NLProblem
    {
    public:
        typedef NLProblem super;
        using typename cppoptlib::Problem<double>::Scalar;
        using typename cppoptlib::Problem<double>::TVector;
        using typename super::THessian;

        ALNLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const double weight);
        TVector initial_guess();

        double value(const TVector &x) override;
        void gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv) override;

        bool stop(const TVector &x) override;

#include <polyfem/DisableWarnings.hpp>
        void hessian_full(const TVector &x, THessian &gradv) override;
#include <polyfem/EnableWarnings.hpp>

    private:
        const double weight_;
        double stop_dist_;
        THessian hessian_;
        std::vector<int> not_boundary_;
        Eigen::MatrixXd displaced_;

        void compute_distance(const TVector &x, TVector &res);
    };
} // namespace polyfem
