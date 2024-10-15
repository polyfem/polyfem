#include "MacroStrainLagrangianForm.hpp"
#include <polyfem/assembler/MacroStrain.hpp>

namespace polyfem::solver
{
    MacroStrainLagrangianForm::MacroStrainLagrangianForm(const assembler::MacroStrainValue &macro_strain_constraint)
     : macro_strain_constraint_(macro_strain_constraint)
    {
        lagr_mults_.setZero(macro_strain_constraint_.get_fixed_entry().size());
    }

    double MacroStrainLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        return lagr_mults_.transpose() * (x(indices) - values);
    }

    void MacroStrainLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        gradv.setZero(x.size());
        gradv(indices) = lagr_mults_;
    }

    void MacroStrainLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        hessian.resize(x.size(), x.size());
        hessian.setZero();
    }

	void MacroStrainLagrangianForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		values = utils::flatten(macro_strain_constraint_.eval(t));
        values = values(macro_strain_constraint_.get_fixed_entry().array()).eval();
	}

    void MacroStrainLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
    {
        const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
        lagr_mults_ += k_al * (x(indices) - values);
    }
}