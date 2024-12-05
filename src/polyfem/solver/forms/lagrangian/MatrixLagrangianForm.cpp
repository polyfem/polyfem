#include "MatrixLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	MatrixLagrangianForm::MatrixLagrangianForm(const int n_dofs,
											   const int dim,
											   const std::vector<int> &constraint_nodes,
											   const Eigen::MatrixXd &A,
											   const Eigen::MatrixXd &b)
	{
		assert(A.rows() == b.rows());
		assert(b.cols() == dim);

		std::vector<Eigen::Triplet<double>> Ae;
		for (int i = 0; i < A.rows(); ++i)
		{
			for (int j = 0; j < A.cols(); ++j)
			{
				const auto global_j = dim * constraint_nodes[j];

				if (A(i, j) != 0)
				{
					for (int d = 0; d < dim; ++d)
					{
						Ae.push_back(Eigen::Triplet<double>(
							i * dim + d,
							global_j + d,
							A(i, j)));
					}
				}
			}
		}

		b_.resize(b.rows() * dim, 1);
		for (int i = 0; i < b.rows(); ++i)
		{
			for (int d = 0; d < dim; ++d)
			{
				b_(i * dim + d) = b(i, d);
			}
		}

		A_.resize(b_.size(), n_dofs);
		A_.setFromTriplets(Ae.begin(), Ae.end());
		A_.makeCompressed();

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
		return L_penalty + k_al_ * A_penalty;
	}

	void MatrixLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = A_.transpose() * lagr_mults_ + k_al_ * (AtA * x - Atb);
	}

	void MatrixLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = k_al_ * AtA;
	}

	void MatrixLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		lagr_mults_ += k_al * (A_ * x - b_);
	}

	double MatrixLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A_ * x - b_;
		return res.squaredNorm();
	}

} // namespace polyfem::solver
