#include "MacroStrainALForm.hpp"

namespace polyfem::solver
{
    MacroStrainALForm::MacroStrainALForm(const Eigen::VectorXi &indices, const Eigen::VectorXd &values) : indices_(indices), values_(values)
    {
        assert(indices_.size() == values_.size());
    }

    double MacroStrainALForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        return (x(indices_) - values_).squaredNorm() / 2.0;
    }

    void MacroStrainALForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        Eigen::VectorXd grad = (x(indices_) - values_);

        gradv.setZero(x.size());
        for (int i = 0; i < indices_.size(); i++)
            gradv(indices_(i)) = grad(i);
    }

    void MacroStrainALForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        hessian.resize(x.size(), x.size());
        hessian.setZero();
        for (int i = 0; i < indices_.size(); i++)
            hessian.coeffRef(indices_(i), indices_(i)) += 1;
    }
}