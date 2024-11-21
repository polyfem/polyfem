#include "GenericLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	GenericLagrangianForm::GenericLagrangianForm(const std::vector<int> &constraint_nodes,
												 const StiffnessMatrix &A,
												 const Eigen::VectorXd &b)
		: AugmentedLagrangianForm(constraint_nodes), A(A), b(b)
	{
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
		gradv = A * lagr_mults_ + k_al_ * (A * x - b);
	}

	void GenericLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = A;
	}

	void GenericLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		lagr_mults_ += k_al * (A * x - b);
	}

	double GenericLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd res = A * x - b;
		return res.norm();
	}
} // namespace polyfem::solver
