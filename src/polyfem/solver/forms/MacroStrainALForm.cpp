#include "MacroStrainALForm.hpp"

namespace polyfem::solver
{
    MacroStrainALForm::MacroStrainALForm(const int dim, const Eigen::VectorXi &indices, const Eigen::VectorXd &values) : dim_(dim), indices_(indices), values_(values)
    {
        assert(dim == 2 || dim == 3);
        assert(indices_.size() == values_.size());
    }

    double MacroStrainALForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd disp_grad = x.tail(dim_ * dim_);
        return (disp_grad(indices_) - values_).squaredNorm();
    }

    void MacroStrainALForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        Eigen::VectorXd disp_grad = x.tail(dim_ * dim_);
        Eigen::VectorXd grad = 2 * (disp_grad(indices_) - values_);

        gradv.setZero(x.size());
        for (int i = 0; i < indices_.size(); i++)
            gradv(gradv.size() - dim_ * dim_ + indices_(i)) = grad(i);
    }

    void MacroStrainALForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        hessian.resize(x.size(), x.size());
        hessian.setZero();
        for (int i = 0; i < indices_.size(); i++)
        {
            int id = hessian.rows() - dim_ * dim_ + indices_(i);
            hessian.coeffRef(id, id) += 1;
        }
    }
}