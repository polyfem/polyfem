#include "QuadraticPenaltyForm.hpp"

namespace polyfem::solver
{
	QuadraticPenaltyForm::QuadraticPenaltyForm(const int n_dofs,
											   const int dim,
											   const Eigen::MatrixXd &A,
											   const Eigen::MatrixXd &b,
											   const std::vector<int> &local_to_global)
	{
		assert(A.rows() == b.rows());
		assert(b.cols() == dim);

		std::vector<Eigen::Triplet<double>> Ae;
		for (int i = 0; i < A.rows(); ++i)
		{
			for (int j = 0; j < A.cols(); ++j)
			{
				const auto global_j = (local_to_global.empty() ? j : local_to_global[j]) * dim;

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

		AtA_ = A_.transpose() * A_;
		Atb_ = A_.transpose() * b_;
	}

	QuadraticPenaltyForm::QuadraticPenaltyForm(const int n_dofs,
											   const int dim,
											   const std::vector<int> &rows,
											   const std::vector<int> &cols,
											   const std::vector<double> &vals,
											   const Eigen::MatrixXd &b,
											   const std::vector<int> &local_to_global)
	{
		assert(b.cols() == dim);
		assert(rows.size() == cols.size());
		assert(rows.size() == vals.size());

		std::vector<Eigen::Triplet<double>> Ae;
		for (int k = 0; k < rows.size(); ++k)
		{
			const auto i = rows[k];
			const auto j = cols[k];
			const auto val = vals[k];

			const auto global_j = (local_to_global.empty() ? j : local_to_global[j]) * dim;

			if (val != 0)
			{
				for (int d = 0; d < dim; ++d)
				{
					Ae.push_back(Eigen::Triplet<double>(
						i * dim + d,
						global_j + d,
						val));
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
		gradv = AtA_ * x - Atb_;
	}

	void QuadraticPenaltyForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = AtA_;
	}
} // namespace polyfem::solver