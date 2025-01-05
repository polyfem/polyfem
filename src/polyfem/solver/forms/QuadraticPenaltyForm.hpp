#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form for quadratic soft constraints
	class QuadraticPenaltyForm : public Form
	{
	public:
		/// @brief Construct a new QuadraticPenaltyForm object for the constraints Ax = b
		/// @param n_dofs Number of degrees of freedom
		/// @param dim Dimension of the problem
		/// @param A Constraints matrix, local or global
		/// @param b Constraints value
		/// @param local_to_global local to global nodes
		QuadraticPenaltyForm(const int n_dofs,
							 const int dim,
							 const Eigen::MatrixXd &A,
							 const Eigen::MatrixXd &b,
							 const double weight,
							 const std::vector<int> &local_to_global = {});

		/// @brief Construct a new QuadraticPenaltyForm object for the constraints Ax = b, where A is sparse
		/// @param n_dofs Number of degrees of freedom
		/// @param dim Dimension of the problem
		/// @param rows, cols, vals are the triplets of the constraints matrix, local or global
		/// @param b Constraints value
		/// @param local_to_global local to global nodes
		QuadraticPenaltyForm(const int n_dofs,
							 const int dim,
							 const std::vector<int> &rows,
							 const std::vector<int> &cols,
							 const std::vector<double> &vals,
							 const Eigen::MatrixXd &b,
							 const double weight,
							 const std::vector<int> &local_to_global = {});

		std::string name() const override { return "quadratic-penalty"; }

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

	private:
		StiffnessMatrix A_; ///< Constraints matrix
		Eigen::MatrixXd b_; ///< Constraints value

		StiffnessMatrix AtA_;
		Eigen::VectorXd Atb_;

		const double weight_;
	};
} // namespace polyfem::solver