
#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	class InversionBarrierForm : public polyfem::solver::Form
	{
	public:
		InversionBarrierForm(
			const Eigen::MatrixXd &rest_positions, const Eigen::MatrixXi &elements, const int dim, const double vhat);

		std::string name() const override { return "inversion_barrier"; }

	protected:
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
		/// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	protected:
		static double element_volume(const Eigen::MatrixXd &element_vertices);
		static Eigen::VectorXd element_volume_gradient(const Eigen::MatrixXd &element_vertices);
		static Eigen::MatrixXd element_volume_hessian(const Eigen::MatrixXd &element_vertices);

	private:
		Eigen::MatrixXd rest_positions_;
		Eigen::MatrixXi elements_;
		int dim_;
		double vhat_;
	};
} // namespace polyfem::solver