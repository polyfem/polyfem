#pragma once

#include <polyfem/solver/forms/ParametrizationForm.hpp>
#include <polyfem/solver/AdjointTools.hpp>

namespace polyfem::solver
{
    class AdjointForm : public ParametrizationForm 
    {
    public:
        virtual ~AdjointForm() {}

    protected:
        virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;
        virtual Eigen::VectorXd compute_adjoint_term(const State &state, const ParameterType &param) const final;
        virtual Eigen::VectorXd compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;

        std::vector<std::tuple<CompositeParameterization, std::shared_ptr<State>, std::vector<ParameterType>>> variable_to_simulation; // parameterizations that map from opt variables to simulations
    };
}