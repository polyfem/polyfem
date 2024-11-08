#include "GenericPenaltyForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	GenericPenaltyForm::GenericPenaltyForm(const StiffnessMatrix &A,
										   const Eigen::VectorXd &b)
		: A(A), b(b)
	{
	}

	double GenericPenaltyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A * x - b;
		return res.squaredNorm() / 2;
	}

	void GenericPenaltyForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = A * x - b;
	}

	void GenericPenaltyForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = A;
	}

	double GenericPenaltyForm::compute_error(const Eigen::VectorXd &x) const
	{
		return value_unweighted(x);
	}
} // namespace polyfem::solver
