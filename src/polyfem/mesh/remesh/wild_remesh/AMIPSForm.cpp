#include "AMIPSForm.hpp"

#include <igl/predicates/predicates.h>

namespace polyfem
{
	namespace solver
	{
		AMIPSForm::AMIPSForm(
			const Eigen::MatrixXd X_rest,
			const Eigen::MatrixXd X)
			: X_rest_(X_rest), X_(X)
		{
		}

		bool AMIPSForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const
		{
			igl::predicates::exactinit();

			// Use igl for checking orientation
			igl::predicates::Orientation res = igl::predicates::orient2d(
				Eigen::Vector2d(x1.head<2>()),
				Eigen::Vector2d(X_rest_.row(1).head<2>()),
				Eigen::Vector2d(X_rest_.row(2).head<2>()));

			// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
			return res == igl::predicates::Orientation::POSITIVE;
		}

		double AMIPSForm::energy(
			const Eigen::Vector2d &x0_rest,
			const Eigen::Vector2d &x1_rest,
			const Eigen::Vector2d &x2_rest,
			const Eigen::Vector2d &x0,
			const Eigen::Vector2d &x1,
			const Eigen::Vector2d &x2)
		{
			return autogen::AMIPS2D_energy(
				x0_rest.x(), x0_rest.y(),
				x1_rest.x(), x1_rest.y(),
				x2_rest.x(), x2_rest.y(),
				x0.x(), x0.y(),
				x1.x(), x1.y(),
				x2.x(), x2.y());
		}

		double AMIPSForm::value_unweighted(const Eigen::VectorXd &x) const
		{
			if (x.size() == 2)
			{
				return energy(
					x.head<2>(),
					X_rest_.row(1).head<2>(),
					X_rest_.row(2).head<2>(),
					X_.row(0).head<2>(),
					X_.row(1).head<2>(),
					X_.row(2).head<2>());
			}
			else
			{
				assert(x.size() == 3);
				throw std::runtime_error("AMIPSForm::first_derivative_unweighted not implemented for 3D");
			}
		}

		void AMIPSForm::first_derivative_unweighted(
			const Eigen::VectorXd &x, Eigen::VectorXd &grad) const
		{
			grad.resize(x.size());

			if (x.size() == 2)
			{
				autogen::AMIPS2D_gradient(
					x[0], x[1],
					X_rest_(1, 0), X_rest_(1, 1),
					X_rest_(2, 0), X_rest_(2, 1),
					X_(0, 0), X_(0, 1),
					X_(1, 0), X_(1, 1),
					X_(2, 0), X_(2, 1),
					grad.data());
			}
			else
			{
				assert(x.size() == 3);
				throw std::runtime_error("AMIPSForm::first_derivative_unweighted not implemented for 3D");
			}
		}

		void AMIPSForm::second_derivative_unweighted(
			const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
		{
			Eigen::MatrixXd H(x.size(), x.size());
			if (x.size() == 2)
			{
				// NOTE: it doesnt matter if H is column major or row major because the hessian is symmetric
				autogen::AMIPS2D_hessian(
					x[0], x[1],
					X_rest_(1, 0), X_rest_(1, 1),
					X_rest_(2, 0), X_rest_(2, 1),
					X_(0, 0), X_(0, 1),
					X_(1, 0), X_(1, 1),
					X_(2, 0), X_(2, 1),
					H.data());
			}
			else
			{
				assert(x.size() == 3);
				throw std::runtime_error("AMIPSForm::second_derivative_unweighted not implemented for 3D");
			}
			hessian = H.sparseView();
		}
	} // namespace solver

	namespace autogen
	{
		double AMIPS2D_energy(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2)
		{
			const auto t0 = x0 * y1 - x0 * y2 - x1 * y0 + x1 * y2 + x2 * y0 - x2 * y1;
			const auto t1 = 1.0 / (x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest);
			const auto t2 = x0 - x1;
			const auto t3 = x0_rest - x2_rest;
			const auto t4 = x0 - x2;
			const auto t5 = x0_rest - x1_rest;
			const auto t6 = y0_rest - y2_rest;
			const auto t7 = y0_rest - y1_rest;
			const auto t8 = y0 - y2;
			const auto t9 = y0 - y1;
			return ((t0 * t1 > 0) ? (
						t1 * (std::pow(t2 * t3 - t4 * t5, 2) + std::pow(t2 * t6 - t4 * t7, 2) + std::pow(-t3 * t9 + t5 * t8, 2) + std::pow(t6 * t9 - t7 * t8, 2)) / t0)
								  : (
									  std::numeric_limits<double>::infinity()));
		}

		void AMIPS2D_gradient(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2, double g[2])
		{
			const auto t0 = x0 - x1;
			const auto t1 = -y2_rest;
			const auto t2 = t1 + y0_rest;
			const auto t3 = -x2;
			const auto t4 = t3 + x0;
			const auto t5 = y0_rest - y1_rest;
			const auto t6 = t4 * t5;
			const auto t7 = t0 * t2 - t6;
			const auto t8 = std::pow(t7, 2);
			const auto t9 = 1.0 / (x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest);
			const auto t10 = t1 + y1_rest;
			const auto t11 = t10 * t9;
			const auto t12 = 2 * t11;
			const auto t13 = y0 - y1;
			const auto t14 = -y2;
			const auto t15 = t14 + y0;
			const auto t16 = t15 * t5;
			const auto t17 = t13 * t2 - t16;
			const auto t18 = std::pow(t17, 2);
			const auto t19 = -x2_rest;
			const auto t20 = t19 + x0_rest;
			const auto t21 = t0 * t20;
			const auto t22 = x0_rest - x1_rest;
			const auto t23 = t22 * t4;
			const auto t24 = t21 - t23;
			const auto t25 = t3 + x1;
			const auto t26 = t13 * t20;
			const auto t27 = t15 * t22 - t26;
			const auto t28 = t14 + y1;
			const auto t29 = std::pow(t27, 2);
			const auto t30 = t18 + t29 + t8;
			const auto t31 = x0 * y1 - x0 * y2 - x1 * y0 + x1 * y2 + x2 * y0 - x2 * y1;
			const auto t32 = t9 / t31;
			const auto t33 = t31 * t9 > 0;
			const auto t34 = std::pow(t24, 2);
			const auto t35 = t19 + x1_rest;
			const auto t36 = t35 * t9;
			const auto t37 = 2 * t36;
			if (t33)
			{
				g[0] = t32 * (t10 * t9 * (std::pow(t24, 2) + t30) - t12 * t18 - t12 * t8 - 2 * t24 * (t11 * t21 - t11 * t23 + t25) - 2 * t27 * (t10 * t15 * t22 * t9 - t11 * t26 - t28));
			}
			else
			{
				g[0] = 0;
			}
			if (t33)
			{
				g[1] = t32 * (2 * t17 * (t13 * t2 * t35 * t9 - t16 * t36 - t28) + t29 * t37 + t34 * t37 - t36 * (t30 + t34) + 2 * t7 * (t0 * t2 * t35 * t9 - t25 - t36 * t6));
			}
			else
			{
				g[1] = 0;
			}
		}

		void AMIPS2D_hessian(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2, double H[4])
		{
			const auto t0 = x0 - x1;
			const auto t1 = -y2_rest;
			const auto t2 = t1 + y0_rest;
			const auto t3 = -x2;
			const auto t4 = t3 + x0;
			const auto t5 = y0_rest - y1_rest;
			const auto t6 = t4 * t5;
			const auto t7 = t0 * t2 - t6;
			const auto t8 = std::pow(t7, 2);
			const auto t9 = t1 + y1_rest;
			const auto t10 = x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest;
			const auto t11 = std::pow(t10, -2);
			const auto t12 = 3 * t11;
			const auto t13 = t12 * std::pow(t9, 2);
			const auto t14 = y0 - y1;
			const auto t15 = -y2;
			const auto t16 = t15 + y0;
			const auto t17 = t16 * t5;
			const auto t18 = t14 * t2 - t17;
			const auto t19 = std::pow(t18, 2);
			const auto t20 = -x2_rest;
			const auto t21 = t20 + x0_rest;
			const auto t22 = t0 * t21;
			const auto t23 = 1.0 / t10;
			const auto t24 = t23 * t9;
			const auto t25 = t22 * t24;
			const auto t26 = x0_rest - x1_rest;
			const auto t27 = t26 * t4;
			const auto t28 = t24 * t27;
			const auto t29 = t3 + x1;
			const auto t30 = t25 - t28 + t29;
			const auto t31 = t14 * t21;
			const auto t32 = t24 * t31;
			const auto t33 = t15 + y1;
			const auto t34 = t16 * t23 * t26 * t9 - t32 - t33;
			const auto t35 = t22 - t27;
			const auto t36 = t30 * t35;
			const auto t37 = 2 * t24;
			const auto t38 = t16 * t26 - t31;
			const auto t39 = t34 * t38;
			const auto t40 = t19 * t24 + t24 * t8 + t36 + t39;
			const auto t41 = x0 * y1 - x0 * y2 - x1 * y0 + x1 * y2 + x2 * y0 - x2 * y1;
			const auto t42 = 2 / t41;
			const auto t43 = t23 * t42;
			const auto t44 = t23 * t41 > 0;
			const auto t45 = t20 + x1_rest;
			const auto t46 = t23 * t45;
			const auto t47 = t46 * t6;
			const auto t48 = t0 * t2 * t23 * t45 - t29 - t47;
			const auto t49 = t48 * t7;
			const auto t50 = t17 * t46;
			const auto t51 = t14 * t2 * t23 * t45 - t33 - t50;
			const auto t52 = t18 * t51;
			const auto t53 = std::pow(t38, 2);
			const auto t54 = t46 * t53 + t49 + t52;
			const auto t55 = ((t44) ? (
								  t11 * t42 * (-t18 * t9 * (2 * t14 * t2 * t23 * t45 - t33 - 2 * t50) - t35 * t45 * (2 * t25 - 2 * t28 + t29) - t36 * t45 - t38 * t45 * (2 * t16 * t23 * t26 * t9 - 2 * t32 - t33) - t39 * t45 + t40 * t45 - t49 * t9 - t52 * t9 - t7 * t9 * (2 * t0 * t2 * t23 * t45 - t29 - 2 * t47) + t9 * (std::pow(t35, 2) * t46 + t54)))
									: (
										0));
			const auto t56 = std::pow(t35, 2);
			const auto t57 = t12 * std::pow(t45, 2);
			const auto t58 = 2 * t46;
			if (t44)
			{
				H[0] = t43 * (t13 * t19 + t13 * t8 + std::pow(t30, 2) + std::pow(t34, 2) + t36 * t37 + t37 * t39 - t37 * t40);
			}
			else
			{
				H[0] = 0;
			}
			H[1] = t55;
			H[2] = t55;
			if (t44)
			{
				H[3] = t43 * (std::pow(t48, 2) + t49 * t58 + std::pow(t51, 2) + t52 * t58 + t53 * t57 + t56 * t57 - t58 * (t46 * t56 + t54));
			}
			else
			{
				H[3] = 0;
			}
		}
	} // namespace autogen

} // namespace polyfem
