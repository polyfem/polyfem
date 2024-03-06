
#include "L2ProjectionForm.hpp"

namespace polyfem::solver
{
	L2ProjectionForm::L2ProjectionForm(
		const StiffnessMatrix &M,
		const StiffnessMatrix &A,
		const Eigen::VectorXd &y)
		: M_(M), rhs_(A * y)
	{
	}

	double L2ProjectionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return x.transpose() * (0.5 * (M_ * x) - rhs_);
	}

	void L2ProjectionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = M_ * x - rhs_;
	}

	void L2ProjectionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = M_;
	}
} // namespace polyfem::solver