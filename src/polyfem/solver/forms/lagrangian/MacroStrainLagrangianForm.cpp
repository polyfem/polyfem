#include "MacroStrainLagrangianForm.hpp"
#include <polyfem/assembler/MacroStrain.hpp>

namespace polyfem::solver
{
	MacroStrainLagrangianForm::MacroStrainLagrangianForm(const assembler::MacroStrainValue &macro_strain_constraint)
		: AugmentedLagrangianForm(std::vector<int>()),
		  macro_strain_constraint_(macro_strain_constraint)
	{
		lagr_mults_.setZero(macro_strain_constraint_.get_fixed_entry().size());
	}

	double MacroStrainLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
		const double larg = lagr_mults_.transpose() * (x(indices) - values);
		const double pen = (x(indices) - values).squaredNorm() / 2.0;

		return larg + k_al_ * pen;
	}

	void MacroStrainLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
		gradv.setZero(x.size());
		gradv(indices) = lagr_mults_ + k_al_ * (x(indices) - values);
	}

	void MacroStrainLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
		hessian.resize(x.size(), x.size());
		hessian.setZero();
		for (int i = 0; i < indices.size(); i++)
			hessian.coeffRef(indices(i), indices(i)) += k_al_;
	}

	void MacroStrainLagrangianForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		values = utils::flatten(macro_strain_constraint_.eval(t));
		values = values(macro_strain_constraint_.get_fixed_entry().array()).eval();
	}

	double MacroStrainLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
		return (x(indices).array() - values.array()).matrix().squaredNorm();
	}

	void MacroStrainLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		k_al_ = k_al;

		const Eigen::VectorXi indices = macro_strain_constraint_.get_fixed_entry().array() + (x.size() - macro_strain_constraint_.dim() * macro_strain_constraint_.dim());
		lagr_mults_ += k_al * (x(indices) - values);
	}
} // namespace polyfem::solver