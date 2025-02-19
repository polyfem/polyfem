#pragma once

#include "AugmentedLagrangianForm.hpp"

namespace polyfem::assembler
{
	class MacroStrainValue;
}

namespace polyfem::solver
{
	/// @brief Form of the lagrangian in augmented lagrangian for homogenization
	class MacroStrainLagrangianForm : public AugmentedLagrangianForm
	{
	public:
		/// @brief Construct a new MacroStrainLagrangianForm object
		MacroStrainLagrangianForm(const assembler::MacroStrainValue &macro_strain_constraint);

		std::string name() const override { return "strain-Lagrangian"; }

		void update_quantities(const double t, const Eigen::VectorXd &x) override;
		void update_lagrangian(const Eigen::VectorXd &x, const double k_al) override;
		double compute_error(const Eigen::VectorXd &x) const override;

	protected:
		/// @brief Compute the contact barrier potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		Eigen::VectorXd values;

		Eigen::VectorXd lagr_mults_; ///< vector of lagrange multipliers
		const assembler::MacroStrainValue &macro_strain_constraint_;
	};
} // namespace polyfem::solver