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
			const Eigen::VectorXd &x, StiffnessMatrix &hessian)
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
			const auto t0 = x0 - x1;
			const auto t1 = x0_rest - x2_rest;
			const auto t2 = x0 - x2;
			const auto t3 = x0_rest - x1_rest;
			const auto t4 = t0 * t1 - t2 * t3;
			const auto t5 = y0 - y1;
			const auto t6 = y0_rest - y2_rest;
			const auto t7 = y0 - y2;
			const auto t8 = y0_rest - y1_rest;
			const auto t9 = t5 * t6 - t7 * t8;
			const auto t10 = t0 * t6 - t2 * t8;
			const auto t11 = -t1 * t5 + t3 * t7;
			const auto t12 = t10 * t11;
			return (((t12 + t4 * t9) / std::pow(x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest, 2) > 0)
						? (-(std::pow(t10, 2) + std::pow(t11, 2) + std::pow(t4, 2) + std::pow(t9, 2)) / (-t12 - t4 * t9))
						: (std::numeric_limits<double>::infinity()));
		}

		void AMIPS2D_gradient(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2, double g[2])
		{
			const auto t0 = -x1;
			const auto t1 = t0 + x0;
			const auto t2 = -t1;
			const auto t3 = -x2_rest;
			const auto t4 = t3 + x0_rest;
			const auto t5 = t2 * t4;
			const auto t6 = -x2;
			const auto t7 = t6 + x0;
			const auto t8 = -t7;
			const auto t9 = x0_rest - x1_rest;
			const auto t10 = -t9;
			const auto t11 = t10 * t8;
			const auto t12 = t11 + t5;
			const auto t13 = -y1;
			const auto t14 = t13 + y0;
			const auto t15 = -t14;
			const auto t16 = -y2_rest;
			const auto t17 = t16 + y0_rest;
			const auto t18 = -t17;
			const auto t19 = t15 * t18;
			const auto t20 = -y2;
			const auto t21 = t20 + y0;
			const auto t22 = -t21;
			const auto t23 = y0_rest - y1_rest;
			const auto t24 = t22 * t23;
			const auto t25 = t19 + t24;
			const auto t26 = t12 * t25;
			const auto t27 = t18 * t2;
			const auto t28 = t23 * t8;
			const auto t29 = t27 + t28;
			const auto t30 = t10 * t22;
			const auto t31 = t15 * t4;
			const auto t32 = t30 + t31;
			const auto t33 = t29 * t32;
			const auto t34 = 1.0 / (t26 - t33);
			const auto t35 = t23 * t7;
			const auto t36 = t1 * t17 - t35;
			const auto t37 = 2 * t36;
			const auto t38 = x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest;
			const auto t39 = 1.0 / t38;
			const auto t40 = t16 + y1_rest;
			const auto t41 = t39 * t40;
			const auto t42 = t21 * t23;
			const auto t43 = t14 * t17 - t42;
			const auto t44 = 2 * t43;
			const auto t45 = t1 * t4;
			const auto t46 = t45 - t7 * t9;
			const auto t47 = -t46;
			const auto t48 = t6 + x1;
			const auto t49 = t21 * t9;
			const auto t50 = t14 * t4;
			const auto t51 = t49 - t50;
			const auto t52 = t20 + y1;
			const auto t53 = -t39 * t40;
			const auto t54 = std::pow(t47, 2);
			const auto t55 = std::pow(t51, 2);
			const auto t56 = t34 * (std::pow(t36, 2) + std::pow(t43, 2) + t54 + t55);
			const auto t57 = (t36 * t51 + t43 * t46) / std::pow(t38, 2) > 0;
			const auto t58 = t3 + x1_rest;
			const auto t59 = t39 * t58;
			const auto t60 = 2 * t59;
			if (t57)
			{
				g[0] = -t34 * (-t36 * t37 * t41 - t41 * t43 * t44 + 2 * t47 * (t41 * t45 - t41 * t7 * t9 + t48) + 2 * t51 * (-t41 * t49 + t41 * t50 + t52) + t56 * (t25 * (-t11 * t53 - t48 - t5 * t53) - t26 * t53 + t29 * (t30 * t53 + t31 * t53 + t52) + t33 * t53));
			}
			else
			{
				g[0] = 0;
			}
			if (t57)
			{
				g[1] = -t34 * (t37 * (t1 * t17 * t39 * t58 - t35 * t59 - t48) + t44 * (t14 * t17 * t39 * t58 - t42 * t59 - t52) + t54 * t60 + t55 * t60 + t56 * (-t12 * (t13 + t19 * t59 + t24 * t59 + y2) - t26 * t59 + t29 * t32 * t39 * t58 + t32 * (t0 + t27 * t59 + t28 * t59 + x2)));
			}
			else
			{
				g[1] = 0;
			}
		}

		void AMIPS2D_hessian(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2, double H[4])
		{
			const auto t0 = -x1;
			const auto t1 = t0 + x0;
			const auto t2 = -t1;
			const auto t3 = -x2_rest;
			const auto t4 = t3 + x0_rest;
			const auto t5 = t2 * t4;
			const auto t6 = -x2;
			const auto t7 = t6 + x0;
			const auto t8 = -t7;
			const auto t9 = x0_rest - x1_rest;
			const auto t10 = -t9;
			const auto t11 = t10 * t8;
			const auto t12 = t11 + t5;
			const auto t13 = -y1;
			const auto t14 = t13 + y0;
			const auto t15 = -t14;
			const auto t16 = -y2_rest;
			const auto t17 = t16 + y0_rest;
			const auto t18 = -t17;
			const auto t19 = t15 * t18;
			const auto t20 = -y2;
			const auto t21 = t20 + y0;
			const auto t22 = -t21;
			const auto t23 = y0_rest - y1_rest;
			const auto t24 = t22 * t23;
			const auto t25 = t19 + t24;
			const auto t26 = t12 * t25;
			const auto t27 = t18 * t2;
			const auto t28 = t23 * t8;
			const auto t29 = t27 + t28;
			const auto t30 = t10 * t22;
			const auto t31 = t15 * t4;
			const auto t32 = t30 + t31;
			const auto t33 = t29 * t32;
			const auto t34 = t26 - t33;
			const auto t35 = 1.0 / t34;
			const auto t36 = t1 * t17;
			const auto t37 = t23 * t7;
			const auto t38 = t36 - t37;
			const auto t39 = -t38;
			const auto t40 = x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest;
			const auto t41 = std::pow(t40, -2);
			const auto t42 = t16 + y1_rest;
			const auto t43 = std::pow(t42, 2);
			const auto t44 = t41 * t43;
			const auto t45 = 2 * t44;
			const auto t46 = std::pow(t38, 2);
			const auto t47 = 4 * t44;
			const auto t48 = t14 * t17;
			const auto t49 = t21 * t23;
			const auto t50 = t48 - t49;
			const auto t51 = -t50;
			const auto t52 = std::pow(t50, 2);
			const auto t53 = t1 * t4;
			const auto t54 = 1.0 / t40;
			const auto t55 = t42 * t54;
			const auto t56 = t7 * t9;
			const auto t57 = t6 + x1;
			const auto t58 = t53 * t55 - t55 * t56 + t57;
			const auto t59 = t14 * t4;
			const auto t60 = t21 * t9;
			const auto t61 = t20 + y1;
			const auto t62 = t55 * t59 - t55 * t60 + t61;
			const auto t63 = t53 - t7 * t9;
			const auto t64 = -t63;
			const auto t65 = 4 * t55;
			const auto t66 = -t59 + t60;
			const auto t67 = -t62;
			const auto t68 = -t42;
			const auto t69 = t54 * t68;
			const auto t70 = t30 * t69 + t31 * t69 + t61;
			const auto t71 = t11 * t69 + t5 * t69 + t57;
			const auto t72 = -t25 * t71 - t26 * t69 + t29 * t70 + t33 * t69;
			const auto t73 = std::pow(t64, 2);
			const auto t74 = std::pow(t66, 2);
			const auto t75 = t46 + t52 + t73 + t74;
			const auto t76 = 2 * t75 / std::pow(t34, 2);
			const auto t77 = t39 * t42;
			const auto t78 = t42 * t51;
			const auto t79 = t58 * t64;
			const auto t80 = t62 * t66;
			const auto t81 = t38 * t54 * t77 + t50 * t54 * t78 + t79 + t80;
			const auto t82 = t35 * t72;
			const auto t83 = 2 * t54;
			const auto t84 = t29 * t68;
			const auto t85 = t41 * (t38 * t66 + t50 * t63) > 0;
			const auto t86 = t3 + x1_rest;
			const auto t87 = t54 * t86;
			const auto t88 = t14 * t17 * t54 * t86 - t49 * t87 - t61;
			const auto t89 = t1 * t17 * t54 * t86 - t37 * t87 - t57;
			const auto t90 = t41 * t86;
			const auto t91 = t68 * t83;
			const auto t92 = t83 * t86;
			const auto t93 = t42 * (-t48 * t92 + t49 * t92 + t61);
			const auto t94 = t42 * t83;
			const auto t95 = t86 * (t59 * t94 - t60 * t94 + t61);
			const auto t96 = t42 * (-t36 * t92 + t37 * t92 + t57);
			const auto t97 = t25 * t87 * (t11 * t91 + t5 * t91 + t57) - t29 * t54 * t95 - t32 * t54 * t96 + t54 * t64 * t93;
			const auto t98 = t64 * t86;
			const auto t99 = t38 * t89;
			const auto t100 = t50 * t88;
			const auto t101 = t100 + t73 * t87 + t74 * t87 + t99;
			const auto t102 = t13 + t19 * t87 + t24 * t87 + y2;
			const auto t103 = t102 * t12;
			const auto t104 = t0 + t27 * t87 + t28 * t87 + x2;
			const auto t105 = -t103 + t104 * t32 - t26 * t87 + t29 * t32 * t54 * t86;
			const auto t106 = t105 * t35;
			const auto t107 = 2 * t101 * t82 + t105 * t72 * t76 + 2 * t106 * t81 + t83 * (t38 * t96 + t50 * t93 + t66 * t95 + t77 * t89 + t78 * t88 + t79 * t86 + t80 * t86 + t98 * (t53 * t94 - t56 * t94 + t57));
			const auto t108 = 6 * t41 * std::pow(t86, 2);
			const auto t109 = 4 * t87;
			if (t85)
			{
				H[0] = t35 * (t35 * t54 * t75 * (3 * t25 * t68 * t71 - 2 * t29 * t42 * t67 - t32 * t38 * t43 * t83 + t42 * t51 * t58 + 2 * t43 * t50 * t54 * t64 - t62 * t84 - t70 * t77) - std::pow(t39, 2) * t45 - t45 * std::pow(t51, 2) - t46 * t47 - t47 * t52 - 2 * std::pow(t58, 2) + t58 * t64 * t65 - 2 * std::pow(t62, 2) - t65 * t66 * t67 - std::pow(t72, 2) * t76 - 4 * t81 * t82);
			}
			else
			{
				H[0] = 0;
			}
			if (t85)
			{
				H[1] = t35 * (-t107 + t35 * t75 * (t26 * t68 * t90 + t58 * t88 - t66 * t84 * t90 - t70 * t89 + t97));
			}
			else
			{
				H[1] = 0;
			}
			if (t85)
			{
				H[2] = t35 * (-t107 + t35 * t75 * (t102 * t71 - t104 * t62 - t32 * t77 * t90 + t41 * t78 * t98 + t97));
			}
			else
			{
				H[2] = 0;
			}
			if (t85)
			{
				H[3] = t35 * (-t100 * t109 - 4 * t101 * t106 - std::pow(t105, 2) * t76 - t108 * t73 - t108 * t74 - t109 * t99 + t35 * t54 * t75 * t86 * (t103 - t104 * t66 + t26 * t92 - t29 * t66 * t92 - 3 * t32 * t89 + 3 * t64 * t88) - 2 * std::pow(t88, 2) - 2 * std::pow(t89, 2));
			}
			else
			{
				H[3] = 0;
			}
		}
	} // namespace autogen

} // namespace polyfem
