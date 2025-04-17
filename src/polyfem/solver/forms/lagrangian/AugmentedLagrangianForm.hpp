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

		inline double lagrangian_weight() const { return k_al_; }

		inline const StiffnessMatrix &constraint_matrix() const { return A_; }
		inline const Eigen::MatrixXd &constraint_value() const { return b_; }

		inline const StiffnessMatrix &constraint_projection_matrix() const { return A_proj_; }
		inline const Eigen::MatrixXd &constraint_projection_vector() const { return b_proj_; }

		inline bool has_projection() const { return A_proj_.rows() > 0; }

		virtual bool can_project() const { return false; }
		virtual void project_gradient(Eigen::VectorXd &grad) const { assert(false); }
		virtual void project_hessian(StiffnessMatrix &hessian) const { assert(false); }

		/// @brief sets the scale for the form
		/// @param scale
		void set_scale(const double scale) override { k_scale_ = scale; }

	protected:
		inline double L_weight() const { return 1 / k_scale_; }
		inline double A_weight() const { return k_al_ / k_scale_; }

		double k_al_; ///< penalty parameter

		Eigen::VectorXd lagr_mults_; ///< vector of lagrange multipliers

		StiffnessMatrix A_; ///< Constraints matrix
		Eigen::MatrixXd b_; ///< Constraints value

		StiffnessMatrix A_proj_; ///< Constraints projection matrix
		Eigen::MatrixXd b_proj_; ///< Constraints projection value
	private:
		double k_scale_ = 1;
	};
} // namespace polyfem::solver
