#include "AdjointForm.hpp"

namespace polyfem::solver
{
    void AdjointForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        gradv.setZero(x.size());
        for (const auto &param_map : variable_to_simulations_)
        {
            const auto &parametrization = param_map->get_parameterization();
            const auto &state = param_map->get_state();
            const auto &param_type = param_map->get_parameter_type();

            gradv += parametrization.apply_jacobian(compute_adjoint_term(state, param_type), x);
        }

        Eigen::VectorXd partial_grad;
        compute_partial_gradient(x, partial_grad);
        gradv += partial_grad;
    }

    void AdjointForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        log_and_throw_error("Should override this function in any AdjointForm!");
    }

    Eigen::MatrixXd AdjointForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state)
    {
        log_and_throw_error("Should override this function in any AdjointForm!");
        return Eigen::MatrixXd();
    }

    Eigen::VectorXd AdjointForm::compute_adjoint_term(const State &state, const ParameterType &param) const
    {
        Eigen::VectorXd term;
        AdjointTools::compute_adjoint_term(state, state.get_adjoint_mat(), param, term);
        return term;
    }
}