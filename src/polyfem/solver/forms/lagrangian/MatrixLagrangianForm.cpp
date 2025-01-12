#include "MatrixLagrangianForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	MatrixLagrangianForm::MatrixLagrangianForm(const int n_dofs,
											   const int dim,
											   const Eigen::MatrixXd &A,
											   const Eigen::MatrixXd &b,
											   const std::vector<int> &local_to_global)
	{
		utils::scatter_matrix(n_dofs, dim, A, b, local_to_global, A_, b_);

		AtA = A_.transpose() * A_;
		Atb = A_.transpose() * b_;

		lagr_mults_.resize(A_.rows());
		lagr_mults_.setZero();
	}

	MatrixLagrangianForm::MatrixLagrangianForm(const int n_dofs,
											   const int dim,
											   const std::vector<int> &rows,
											   const std::vector<int> &cols,
											   const std::vector<double> &vals,
											   const Eigen::MatrixXd &b,
											   const std::vector<int> &local_to_global)
	{
		utils::scatter_matrix(n_dofs, dim, rows, cols, vals, b, local_to_global, A_, b_);

		AtA = A_.transpose() * A_;
		Atb = A_.transpose() * b_;

		lagr_mults_.resize(A_.rows());
		lagr_mults_.setZero();
	}

	double MatrixLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A_ * x - b_;
		const double L_penalty = lagr_mults_.transpose() * res;
		const double A_penalty = res.squaredNorm() / 2;
		return L_weight() * L_penalty + A_weight() * A_penalty;
	}

	void MatrixLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = L_weight() * A_.transpose() * lagr_mults_ + A_weight() * (AtA * x - Atb);
	}

	void MatrixLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = A_weight() * AtA;
	}

	void MatrixLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		k_al_ = k_al;
		lagr_mults_ += k_al * (A_ * x - b_);
	}

	double MatrixLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A_ * x - b_;
		return res.squaredNorm();
	}

} // namespace polyfem::solver
