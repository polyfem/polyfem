#include "PeriodicLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <igl/slice.h>

namespace polyfem::solver
{
	PeriodicLagrangianForm::PeriodicLagrangianForm(const int ndof,
												   const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc)
		: periodic_bc_(periodic_bc),
		  n_dofs_(ndof)
	{
		std::vector<Eigen::Triplet<double>> A_triplets;

		const int dim = periodic_bc_->dim();
		for (int i = 0; i < periodic_bc_->n_constraints(); ++i)
		{
			auto c = periodic_bc_->constraint(i);
			for (int d = 0; d < dim; d++)
			{
				A_triplets.emplace_back(d + i * dim, d + dim * c[0], 1.0);
				A_triplets.emplace_back(d + i * dim, d + dim * c[1], -1.0);
			}
		}
		A_.resize(periodic_bc_->n_constraints() * dim, n_dofs_);
		A_.setFromTriplets(A_triplets.begin(), A_triplets.end());
		A_.makeCompressed();

		b_.setZero(A_.rows(), 1);

		A_triplets.clear();
		{
			const auto &index_map = periodic_bc_->index_map();
			for (int i = 0; i < index_map.size(); i++)
				for (int d = 0; d < dim; d++)
					A_triplets.emplace_back(i * dim + d, index_map(i) * dim + d, 1.0);
			assert(index_map.size() * dim == n_dofs_);
		}

		A_proj_.resize(n_dofs_, periodic_bc_->n_periodic_dof());
		A_proj_.setFromTriplets(A_triplets.begin(), A_triplets.end());
		A_proj_.makeCompressed();

		b_proj_.setZero(A_proj_.rows(), 1);

		lagr_mults_.resize(A_.rows());
		lagr_mults_.setZero();
	}

	bool PeriodicLagrangianForm::can_project() const { return true; }

	void PeriodicLagrangianForm::project_gradient(Eigen::VectorXd &grad) const
	{
		grad = periodic_bc_->full_to_periodic(grad, true);
	}

	void PeriodicLagrangianForm::project_hessian(StiffnessMatrix &hessian) const
	{
		periodic_bc_->full_to_periodic(hessian);
	}

	double PeriodicLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd dist = A_ * x - b_;
		const double L_penalty = -lagr_mults_.transpose() * dist;
		const double A_penalty = 0.5 * dist.transpose() * dist;

		return L_weight() * L_penalty + A_weight() * A_penalty;
	}

	void PeriodicLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = L_weight() * A_.transpose() * (-lagr_mults_ + A_weight() * (A_ * x - b_));
	}

	void PeriodicLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = A_weight() * A_.transpose() * A_;
	}

	double PeriodicLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		// return (b_ - x).transpose() * A_ * (b_ - x);
		const Eigen::VectorXd res = A_ * x - b_;
		return res.squaredNorm();
	}

	void PeriodicLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		k_al_ = k_al;
		lagr_mults_ -= k_al * (A_ * x - b_);
	}
} // namespace polyfem::solver
