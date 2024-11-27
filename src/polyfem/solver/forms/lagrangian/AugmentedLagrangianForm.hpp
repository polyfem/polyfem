#pragma once

#include <polyfem/solver/forms/Form.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian
	class AugmentedLagrangianForm : public Form
	{

	public:
		AugmentedLagrangianForm(const int n_dofs,
								const std::vector<int> &constraint_nodes)
			: constraint_nodes_(constraint_nodes)
		{
			lagr_mults_.resize(n_dofs);
			lagr_mults_.setZero();
		}

		virtual ~AugmentedLagrangianForm() {}

		virtual void update_lagrangian(const Eigen::VectorXd &x, const double k_al) = 0;

		virtual double compute_error(const Eigen::VectorXd &x) const = 0;

		inline void set_initial_weight(const double k_al) { k_al_ = k_al; }

		inline const std::vector<int> &constraint_nodes() const { return constraint_nodes_; }

		inline const StiffnessMatrix &constraint_matrix() const { return A_; }
		inline const Eigen::MatrixXd &constraint_value() const { return b_; }

	protected:
		Eigen::VectorXd lagr_mults_;        ///< vector of lagrange multipliers
		double k_al_;                       ///< penalty parameter
		std::vector<int> constraint_nodes_; ///< constraint nodes

		StiffnessMatrix A_; ///< Constraints matrix
		Eigen::MatrixXd b_; ///< Constraints value
	};
} // namespace polyfem::solver
