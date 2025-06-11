#include "QuadraticPenaltyForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{

	QuadraticPenaltyForm::QuadraticPenaltyForm(const StiffnessMatrix &A,
											   const Eigen::MatrixXd &b,
											   const double weight)
		: penalty_weight_(weight), A_(A), b_(b)
	{
		assert(A.rows() == b.rows());

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