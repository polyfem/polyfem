#pragma once

#include "Form.hpp"

namespace polyfem::solver
{
    class MacroStrainLagrangianForm : public Form
    {
    public:
        MacroStrainLagrangianForm(const Eigen::VectorXi &indices, const Eigen::VectorXd &values);

		std::string name() const override { return "Macro_Lagrangian"; }

		void update_lagrangian(const Eigen::VectorXd &x, const double k_al);
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
        const Eigen::VectorXi indices_;
        const Eigen::VectorXd values_;
    
		Eigen::VectorXd lagr_mults_;              ///< vector of lagrange multipliers
	};
}