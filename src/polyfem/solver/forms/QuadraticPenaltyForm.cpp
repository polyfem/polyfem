#include "QuadraticPenaltyForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	QuadraticPenaltyForm::QuadraticPenaltyForm(const int n_dofs,
											   const int dim,
											   const Eigen::MatrixXd &A,
											   const Eigen::MatrixXd &b,
											   const double weight,
											   const std::vector<int> &local_to_global)
		: penalty_weight_(weight)
	{
		utils::scatter_matrix(n_dofs, dim, A, b, local_to_global, A_, b_);

		AtA_ = A_.transpose() * A_;
		Atb_ = A_.transpose() * b_;
	}

	QuadraticPenaltyForm::QuadraticPenaltyForm(const int n_dofs,
											   const int dim,
											   const std::vector<int> &rows,
											   const std::vector<int> &cols,
											   const std::vector<double> &vals,
											   const Eigen::MatrixXd &b,
											   const double weight,
											   const std::vector<int> &local_to_global)
		: penalty_weight_(weight)
	{
		utils::scatter_matrix(n_dofs, dim, rows, cols, vals, b, local_to_global, A_, b_);

		AtA_ = A_.transpose() * A_;
		Atb_ = A_.transpose() * b_;
	}

	double QuadraticPenaltyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd val = A_ * x - b_;
		return val.squaredNorm() / 2;
	}

	void QuadraticPenaltyForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = (AtA_ * x - Atb_);
	}

	void QuadraticPenaltyForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = AtA_;
	}
} // namespace polyfem::solver