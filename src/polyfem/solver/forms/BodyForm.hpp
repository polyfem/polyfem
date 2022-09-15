#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	/// @brief Form representing body forces
	class BodyForm : public Form
	{
	public:
		/// @brief Construct a new Body Form object
		/// @param state Reference to the simulation state
		/// @param rhs_assembler Reference to the right hand side assembler
		/// @param apply_DBC If true, set the Dirichlet boundary conditions in the RHS
		BodyForm(const State &state, const assembler::RhsAssembler &rhs_assembler, const bool apply_DBC);

	protected:
		/// @brief Compute the value of the body force form
		/// @param x Current solution
		/// @return Value of the body force form
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Vector containing the current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override { hessian.resize(x.size(), x.size()); }

	public:
		/// @brief Update time dependent quantities
		/// @param t New time
		/// @param x Solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

		bool get_apply_DBC() { return apply_DBC_; }
		void set_apply_DBC(const bool val) override
		{
			if (val != apply_DBC_)
			{
				apply_DBC_ = val;
				update_current_rhs();
			}
		}

	private:
		const State &state_;                           ///< Reference to the simulation state
		const assembler::RhsAssembler &rhs_assembler_; ///< Reference to the RHS assembler
		bool is_formulation_mixed_;                    ///< True if the formulation is mixed

		double t_; ///< Current time
		int ndof_; ///< Number of degrees of freedom

		bool apply_DBC_; ///< If true, set the Dirichlet boundary conditions in the RHS

		Eigen::MatrixXd current_rhs_; ///< Cached RHS for the current time

		/// @brief Update current_rhs
		void update_current_rhs();
	};
} // namespace polyfem::solver
