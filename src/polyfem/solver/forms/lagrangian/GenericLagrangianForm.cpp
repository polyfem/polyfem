#include "GenericLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	GenericLagrangianForm::GenericLagrangianForm(const int n_dofs,
												 const int dim,
												 const std::vector<int> &constraint_nodes,
												 const Eigen::MatrixXd &A,
												 const Eigen::MatrixXd &Ai,
												 const Eigen::MatrixXd &b)
		: AugmentedLagrangianForm(n_dofs, constraint_nodes)
	{
		assert(A.rows() == Ai.cols());
		assert(A.rows() == b.rows());
		assert(b.cols() == dim);
		assert(A.cols() == Ai.rows());

		std::vector<Eigen::Triplet<double>> Ae;
		std::vector<Eigen::Triplet<double>> Aie;

		for (int i = 0; i < A.rows(); ++i)
		{
			for (int j = 0; j < A.cols(); ++j)
			{
				const auto global_i = dim * constraint_nodes[i];
				const auto global_j = dim * constraint_nodes[j];

				if (A(i, j) != 0)
				{
					for (int d = 0; d < dim; ++d)
					{
						Ae.push_back(Eigen::Triplet<double>(
							global_i + d,
							global_j + d,
							A(i, j)));
					}
				}

				if (Ai(j, i) != 0)
				{
					for (int d = 0; d < dim; ++d)
					{
						Aie.push_back(Eigen::Triplet<double>(
							global_j + d,
							global_i + d,
							Ai(j, i)));
					}
				}
			}
		}

		this->b.resize(n_dofs);
		for (int i = 0; i < b.rows(); ++i)
		{
			for (int d = 0; d < dim; ++d)
			{
				this->b[dim * constraint_nodes[i] + d] = b(i, d);
			}
		}

		this->A.resize(n_dofs, n_dofs);
		this->A.setFromTriplets(Ae.begin(), Ae.end());
		this->A.makeCompressed();

		this->Ai.resize(n_dofs, n_dofs);
		this->Ai.setFromTriplets(Aie.begin(), Aie.end());
		this->Ai.makeCompressed();

		constraint_nodes_.clear();

		for (const auto v : constraint_nodes)
		{
			for (int d = 0; d < dim; ++d)
			{
				constraint_nodes_.push_back(dim * v + d);
			}
		}

		AtA = this->A.transpose() * this->A;
		Atb = this->A.transpose() * this->b;
	}

	double GenericLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A * x - b;
		const double L_penalty = lagr_mults_.transpose() * res;
		const double A_penalty = res.squaredNorm() / 2;
		return L_penalty + k_al_ * A_penalty;
	}

	void GenericLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = A * lagr_mults_ + k_al_ * (AtA * x - Atb);
	}

	void GenericLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = k_al_ * AtA;
	}

	void GenericLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		lagr_mults_ += k_al * (A * x - b);
	}

	double GenericLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A * x - b;
		return res.squaredNorm();
	}

	Eigen::VectorXd GenericLagrangianForm::target(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd delta = Ai * (b - A * x);
		return x + delta;
	}

} // namespace polyfem::solver
