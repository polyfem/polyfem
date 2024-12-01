#pragma once

#include <polyfem/solver/forms/Form.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian
	class AugmentedLagrangianForm : public Form
	{

	public:
		AugmentedLagrangianForm() {}

		virtual ~AugmentedLagrangianForm() {}

		virtual void update_lagrangian(const Eigen::VectorXd &x, const double k_al) = 0;

		virtual double compute_error(const Eigen::VectorXd &x) const = 0;

		inline void set_initial_weight(const double k_al) { k_al_ = k_al; }

		inline const StiffnessMatrix &constraint_matrix() const { return A_; }
		inline const Eigen::MatrixXd &constraint_value() const { return b_; }

	protected:
		Eigen::VectorXd lagr_mults_; ///< vector of lagrange multipliers
		double k_al_;                ///< penalty parameter

		StiffnessMatrix A_; ///< Constraints matrix
		Eigen::MatrixXd b_; ///< Constraints value
	};
} // namespace polyfem::solver
