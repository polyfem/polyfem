#pragma once

#include "AugmentedLagrangianForm.hpp"

namespace polyfem::solver
{
	/// @brief Form of the lagrangian in augmented lagrangian
	class MatrixLagrangianForm : public AugmentedLagrangianForm
	{
	public:
		/// @brief Construct a new MatrixLagrangianForm object for the constraints Ax = b
		/// @param n_dofs Number of degrees of freedom
		/// @param dim Dimension of the problem
		/// @param A Constraints matrix, local or global
		/// @param b Constraints value
		/// @param local_to_global local to global nodes
		MatrixLagrangianForm(const int n_dofs,
							 const int dim,
							 const Eigen::MatrixXd &A,
							 const Eigen::MatrixXd &b,
							 const std::vector<int> &local_to_global = {});

		/// @brief Construct a new MatrixLagrangianForm object for the constraints Ax = b, where A is sparse
		/// @param n_dofs Number of degrees of freedom
		/// @param dim Dimension of the problem
		/// @param rows, cols, vals are the triplets of the constraints matrix, local or global
		/// @param b Constraints value
		/// @param local_to_global local to global nodes
		MatrixLagrangianForm(const int n_dofs,
							 const int dim,
							 const std::vector<int> &rows,
							 const std::vector<int> &cols,
							 const std::vector<double> &vals,
							 const Eigen::MatrixXd &b,
							 const std::vector<int> &local_to_global = {});

		std::string name() const override { return "generic-lagrangian"; }

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		void update_lagrangian(const Eigen::VectorXd &x, const double k_al) override;
		double compute_error(const Eigen::VectorXd &x) const override;

	private:
		StiffnessMatrix AtA;
		Eigen::VectorXd Atb;
	};
} // namespace polyfem::solver
