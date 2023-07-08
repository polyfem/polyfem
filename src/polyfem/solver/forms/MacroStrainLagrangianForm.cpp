#include "MacroStrainLagrangianForm.hpp"

namespace polyfem::solver
{
    MacroStrainLagrangianForm::MacroStrainLagrangianForm(const Eigen::VectorXi &indices, const Eigen::VectorXd &values) : indices_(indices), values_(values)
    {
        assert(indices_.size() == values_.size());
        lagr_mults_.setZero(indices_.size());
    }

    double MacroStrainLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        return lagr_mults_.transpose() * (x(indices_) - values_);
    }

    void MacroStrainLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        gradv.setZero(x.size());
        for (int i = 0; i < indices_.size(); i++)
            gradv(indices_(i)) = lagr_mults_(i);
    }

    void MacroStrainLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        hessian.resize(x.size(), x.size());
        hessian.setZero();
    }

    void MacroStrainLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
    {
        lagr_mults_ += k_al * (x(indices_) - values_);
    }
}