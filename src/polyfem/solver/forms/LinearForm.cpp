#include "LinearForm.hpp"

namespace polyfem::solver
{
	LinearForm::LinearForm(const Eigen::VectorXd &coeffs)
		: coeffs_(coeffs)
	{
	}

	double LinearForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(coeffs_.size() == x.size());
		return coeffs_.dot(x);
	}

	void LinearForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = coeffs_;
	}

	void LinearForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		hessian.resize(x.size(), x.size());
		// hessian is zero
	}
} // namespace polyfem::solver