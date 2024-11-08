#include "MacroStrainALForm.hpp"
#include <polyfem/assembler/MacroStrain.hpp>

namespace polyfem::solver
{
    MacroStrainALForm::MacroStrainALForm(const assembler::MacroStrainValue &macro_strain_constraint)
     : macro_strain_constraint_(macro_strain_constraint)
    {
    }

    double MacroStrainALForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        const Eigen::VectorXi indices_ = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        return (x(indices_) - values).squaredNorm() / 2.0;
    }

    void MacroStrainALForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        gradv.setZero(x.size());
        gradv(indices) = x(indices) - values;
    }

    void MacroStrainALForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        hessian.resize(x.size(), x.size());
        hessian.setZero();
        for (int i = 0; i < indices.size(); i++)
            hessian.coeffRef(indices(i), indices(i)) += 1;
    }

	void MacroStrainALForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		values = utils::flatten(macro_strain_constraint_.eval(t));
        values = values(macro_strain_constraint_.get_fixed_entry().array()).eval();
	}

    double MacroStrainALForm::compute_error(const Eigen::VectorXd &x) const
    {
        const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        return (x(indices).array() - values.array()).matrix().squaredNorm();
    }
}