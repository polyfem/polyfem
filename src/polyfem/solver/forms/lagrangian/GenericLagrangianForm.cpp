#include "GenericLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	GenericLagrangianForm::GenericLagrangianForm(const StiffnessMatrix &A,
												 const Eigen::VectorXd &b)
		: A(A), b(b)
	{
	}

	double GenericLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A * x - b;
		const double AL_penalty = lagr_mults_.transpose() * res;
		return AL_penalty;
	}

	void GenericLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = A * lagr_mults_;
	}

	void GenericLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian.resize(A.rows(), A.cols());
		hessian.setZero();
	}

	void GenericLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		lagr_mults_ += k_al * (A * x - b);
	}
} // namespace polyfem::solver
