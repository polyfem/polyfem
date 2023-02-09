#pragma once

#include <polyfem/solver/forms/ParameterizationForm.hpp>
#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
    class AdjointForm : public ParameterizationForm 
    {
    public:
        AdjointForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations) : variable_to_simulations_(variable_to_simulations) {}
        virtual ~AdjointForm() {}

        virtual Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) override;

    protected:
        virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;
        
        virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

        std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations_;

    private:
        virtual Eigen::VectorXd compute_adjoint_term(const State &state, const ParameterType &param) const final;
    };
}