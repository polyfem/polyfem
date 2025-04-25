#include "MatrixLagrangianForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	MatrixLagrangianForm::MatrixLagrangianForm(const int n_dofs,
											   const int dim,
											   const Eigen::MatrixXd &A,
											   const Eigen::MatrixXd &b,
											   const std::vector<int> &local_to_global,
											   const Eigen::MatrixXd &A_proj,
											   const Eigen::MatrixXd &b_proj)
	{
		utils::scatter_matrix(n_dofs, dim, A, b, local_to_global, A_, b_current_);
		b_prev_ = b_current_;
		b_prev_.setZero();

		if (b_proj.size() > 0)
		{
			utils::scatter_matrix_col(n_dofs, dim, A_proj, b_proj, local_to_global, A_proj_, b_current_proj_);

			b_prev_proj_ = b_current_proj_;
			b_prev_proj_.setZero();
		}

		AtA = A_.transpose() * A_;
		Atb = A_.transpose() * b_current_;

		lagr_mults_.resize(A_.rows());
		lagr_mults_.setZero();

		set_incr_load(incr_load_);
	}

	MatrixLagrangianForm::MatrixLagrangianForm(const int n_dofs,
											   const int dim,
											   const std::vector<int> &rows,
											   const std::vector<int> &cols,
											   const std::vector<double> &vals,
											   const Eigen::MatrixXd &b,
											   const std::vector<int> &local_to_global,
											   const std::vector<int> &rows_proj,
											   const std::vector<int> &cols_proj,
											   const std::vector<double> &vals_proj,
											   const Eigen::MatrixXd &b_proj)
	{
		utils::scatter_matrix(n_dofs, dim, rows, cols, vals, b, local_to_global, A_, b_current_);
		b_prev_ = b_current_;
		b_prev_.setZero();

		if (b_proj.size() > 0)
		{
			utils::scatter_matrix_col(n_dofs, dim, rows_proj, cols_proj, vals_proj, b_proj, local_to_global, A_proj_, b_current_proj_);

			b_prev_proj_ = b_current_proj_;
			b_prev_proj_.setZero();
		}

		AtA = A_.transpose() * A_;
		Atb = A_.transpose() * b_current_;

		lagr_mults_.resize(A_.rows());
		lagr_mults_.setZero();

		set_incr_load(incr_load_);
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
