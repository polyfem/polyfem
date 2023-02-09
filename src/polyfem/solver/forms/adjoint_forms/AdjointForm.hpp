#pragma once

#include <polyfem/solver/forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

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
        virtual void compute_adjoint_rhs(std::vector<Eigen::MatrixXd> &rhss) override;

        std::vector<VariablesToSimulation> *variable_to_simulations;
    };
}