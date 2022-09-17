#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	/// @brief Form of the damping potential and forces
	class DampingForm : public Form
	{
	public:
		/// @brief Construct a new Damping Form object
		/// @param state Reference to the simulation state
		DampingForm(const State &state, const double dt);

	protected:
		/// @brief Compute the damping potential value
		/// @param x Current solution
		/// @return Value of the damping potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

	public:
		/// @brief Initialize lagged fields
		/// @param x Current solution
		void init_lagging(const Eigen::VectorXd &x) override;

	private:
		const State &state_;                                  ///< Reference to the simulation state
		const assembler::ViscousDampingAssembler &assembler_; ///< Reference to the assembler
		utils::SpareMatrixCache mat_cache_;                   ///< Matrix cache
        
        const double dt_;
        Eigen::VectorXd x_prev;
	};
} // namespace polyfem::solver
