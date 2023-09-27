#pragma once

#include <polyfem/solver/forms/Form.hpp>

namespace polyfem
{
	namespace solver
	{
		class AMIPSForm : public Form
		{
		public:
			/// Assumes the variable to optimize over is X_rest.row(0)
			AMIPSForm(
				const Eigen::MatrixXd X_rest,
				const Eigen::MatrixXd X);

			bool is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const override;

			static double energy(
				const Eigen::Vector2d &x0_rest,
				const Eigen::Vector2d &x1_rest,
				const Eigen::Vector2d &x2_rest,
				const Eigen::Vector2d &x0,
				const Eigen::Vector2d &x1,
				const Eigen::Vector2d &x2);

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

		private:
			const Eigen::MatrixXd X_rest_;
			const Eigen::MatrixXd X_;
		};

	} // namespace solver

	namespace autogen
	{
		double AMIPS2D_energy(
			double x0_rest, double y0_rest,
			double x1_rest, double y1_rest,
			double x2_rest, double y2_rest,
			double x0, double y0,
			double x1, double y1,
			double x2, double y2);

		void AMIPS2D_gradient(
			double x0_rest, double y0_rest,
			double x1_rest, double y1_rest,
			double x2_rest, double y2_rest,
			double x0, double y0,
			double x1, double y1,
			double x2, double y2,
			double g[2]);

		void AMIPS2D_hessian(
			double x0_rest, double y0_rest,
			double x1_rest, double y1_rest,
			double x2_rest, double y2_rest,
			double x0, double y0,
			double x1, double y1,
			double x2, double y2,
			double H[4]);

		// TODO: 3D
	} // namespace autogen
} // namespace polyfem