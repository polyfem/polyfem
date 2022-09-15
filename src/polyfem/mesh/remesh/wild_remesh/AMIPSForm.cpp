#include "AMIPSForm.hpp"

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
			const auto t1 = y0_rest - y1_rest;
			const auto t2 = x0_rest - x1_rest;
			const auto t3 = y0 - y1;
			const auto t4 = t0 * t1 - t2 * t3;
			const auto t5 = x0 - x2;
			const auto t6 = y0_rest - y2_rest;
			const auto t7 = x0_rest - x2_rest;
			const auto t8 = y0 - y2;
			const auto t9 = t5 * t6 - t7 * t8;
			const auto t10 = t0 * t6 - t3 * t7;
			const auto t11 = t1 * t5 - t2 * t8;
			return -(std::pow(t10, 2) + std::pow(t11, 2) + std::pow(t4, 2) + std::pow(t9, 2)) / (t10 * t11 - t4 * t9);
		}

		void AMIPS2D_gradient(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2, double g[2])
		{
			const auto t0 = x0 - x1;
			const auto t1 = y0_rest - y1_rest;
			const auto t2 = t0 * t1;
			const auto t3 = x0_rest - x1_rest;
			const auto t4 = y0 - y1;
			const auto t5 = t2 - t3 * t4;
			const auto t6 = -t5;
			const auto t7 = x0 - x2;
			const auto t8 = -y2_rest;
			const auto t9 = t8 + y0_rest;
			const auto t10 = t7 * t9;
			const auto t11 = -x2_rest;
			const auto t12 = t11 + x0_rest;
			const auto t13 = y0 - y2;
			const auto t14 = t12 * t13;
			const auto t15 = t10 - t14;
			const auto t16 = t0 * t9;
			const auto t17 = t12 * t4;
			const auto t18 = t16 - t17;
			const auto t19 = t1 * t7;
			const auto t20 = -t13 * t3 + t19;
			const auto t21 = -t20;
			const auto t22 = 1.0 / (t15 * t6 - t18 * t21);
			const auto t23 = 1.0 / (x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest);
			const auto t24 = t8 + y1_rest;
			const auto t25 = t23 * t24;
			const auto t26 = t3 * t4;
			const auto t27 = t2 * t25 - t25 * t26 + t4;
			const auto t28 = 2 * t5;
			const auto t29 = t16 * t25 - t17 * t25 + t4;
			const auto t30 = 2 * t18;
			const auto t31 = t13 * t3;
			const auto t32 = t13 + t19 * t25 - t25 * t31;
			const auto t33 = 2 * t20;
			const auto t34 = 2 * t15;
			const auto t35 = t7 * t9;
			const auto t36 = -t12 * t13;
			const auto t37 = t35 + t36;
			const auto t38 = -t23 * t24;
			const auto t39 = t22 * (std::pow(t15, 2) + std::pow(t18, 2) + std::pow(t20, 2) + std::pow(t5, 2));
			const auto t40 = t23 * (t11 + x1_rest);
			const auto t41 = t0 + t2 * t40 - t26 * t40;
			const auto t42 = t0 + t16 * t40 - t17 * t40;
			const auto t43 = t19 * t40 - t31 * t40 + t7;
			g[0] = t22 * (t27 * t28 + t29 * t30 + t32 * t33 + t34 * (t10 * t25 + t13 - t14 * t25) - t39 * (t18 * t32 - t21 * t29 - t27 * t37 - t6 * (t35 * t38 + t36 * t38 - y0 + y2)));
			g[1] = -t22 * (t28 * t41 + t30 * t42 + t33 * t43 + t34 * (t10 * t40 - t14 * t40 + t7) + t39 * (-t18 * t43 + t21 * t42 + t37 * t41 - t6 * (t35 * t40 + t36 * t40 + t7)));
		}

		void AMIPS2D_hessian(double x0_rest, double y0_rest, double x1_rest, double y1_rest, double x2_rest, double y2_rest, double x0, double y0, double x1, double y1, double x2, double y2, double H[4])
		{
			const auto t0 = x0 - x1;
			const auto t1 = y0_rest - y1_rest;
			const auto t2 = t0 * t1;
			const auto t3 = x0_rest - x1_rest;
			const auto t4 = y0 - y1;
			const auto t5 = t2 - t3 * t4;
			const auto t6 = -t5;
			const auto t7 = x0 - x2;
			const auto t8 = -y2_rest;
			const auto t9 = t8 + y0_rest;
			const auto t10 = t7 * t9;
			const auto t11 = -x2_rest;
			const auto t12 = t11 + x0_rest;
			const auto t13 = y0 - y2;
			const auto t14 = t12 * t13;
			const auto t15 = t10 - t14;
			const auto t16 = t0 * t9;
			const auto t17 = t12 * t4;
			const auto t18 = t16 - t17;
			const auto t19 = t1 * t7;
			const auto t20 = -t13 * t3 + t19;
			const auto t21 = -t20;
			const auto t22 = t15 * t6 - t18 * t21;
			const auto t23 = 1.0 / t22;
			const auto t24 = t8 + y1_rest;
			const auto t25 = 1.0 / (x0_rest * y1_rest - x0_rest * y2_rest - x1_rest * y0_rest + x1_rest * y2_rest + x2_rest * y0_rest - x2_rest * y1_rest);
			const auto t26 = t24 * t25;
			const auto t27 = t2 * t26;
			const auto t28 = t3 * t4;
			const auto t29 = t26 * t28;
			const auto t30 = t27 - t29 + t4;
			const auto t31 = t16 * t26;
			const auto t32 = -t17 * t26 + t31 + t4;
			const auto t33 = t19 * t26;
			const auto t34 = t13 * t3;
			const auto t35 = t26 * t34;
			const auto t36 = t13 + t33 - t35;
			const auto t37 = t10 * t26;
			const auto t38 = t14 * t26;
			const auto t39 = t13 + t37 - t38;
			const auto t40 = t30 * t5;
			const auto t41 = 4 * t26;
			const auto t42 = t18 * t32;
			const auto t43 = t20 * t36;
			const auto t44 = t15 * t39;
			const auto t45 = -t7;
			const auto t46 = -t9;
			const auto t47 = t45 * t46;
			const auto t48 = -t13;
			const auto t49 = t12 * t48;
			const auto t50 = t47 + t49;
			const auto t51 = -y0;
			const auto t52 = -t24;
			const auto t53 = t25 * t52;
			const auto t54 = t47 * t53;
			const auto t55 = t49 * t53;
			const auto t56 = t51 + t54 + t55 + y2;
			const auto t57 = -t32;
			const auto t58 = t18 * t36 + t21 * t57 - t30 * t50 - t56 * t6;
			const auto t59 = std::pow(t15, 2) + std::pow(t18, 2) + std::pow(t20, 2) + std::pow(t5, 2);
			const auto t60 = 2 * t59 / std::pow(t22, 2);
			const auto t61 = t40 + t42 + t43 + t44;
			const auto t62 = -t0;
			const auto t63 = t1 * t62;
			const auto t64 = -t3;
			const auto t65 = -t4;
			const auto t66 = t64 * t65;
			const auto t67 = t25 * (t63 + t66);
			const auto t68 = 2 * t67;
			const auto t69 = t46 * t62;
			const auto t70 = t12 * t65;
			const auto t71 = t69 + t70;
			const auto t72 = t1 * t45;
			const auto t73 = t48 * t64;
			const auto t74 = t72 + t73;
			const auto t75 = t4 + t53 * t63 + t53 * t66;
			const auto t76 = t51 + t53 * t69 + t53 * t70 + y1;
			const auto t77 = t13 + t53 * t72 + t53 * t73;
			const auto t78 = -t39;
			const auto t79 = t23 * t59;
			const auto t80 = t11 + x1_rest;
			const auto t81 = t25 * t80;
			const auto t82 = t0 + t16 * t81 - t17 * t81;
			const auto t83 = t19 * t81 - t34 * t81 + t7;
			const auto t84 = -t83;
			const auto t85 = t47 * t81 + t49 * t81 + t7;
			const auto t86 = t0 + t2 * t81 - t28 * t81;
			const auto t87 = -t86;
			const auto t88 = 2 * t80;
			const auto t89 = t13 * t80 + t24 * t7;
			const auto t90 = t25 * (t33 * t88 - t35 * t88 + t89);
			const auto t91 = t0 * t24 + t4 * t80;
			const auto t92 = -2 * t12 * t24 * t25 * t4 * t80 + t31 * t88 + t91;
			const auto t93 = t25 * t74;
			const auto t94 = t25 * (t27 * t88 - t29 * t88 + t91);
			const auto t95 = -t15 * t94 - t67 * (-t45 * t52 + t48 * t80 + t54 * t88 + t55 * t88) + t71 * t90 - t92 * t93;
			const auto t96 = t10 * t81 - t14 * t81 + t7;
			const auto t97 = t5 * t86;
			const auto t98 = t18 * t82;
			const auto t99 = t20 * t83;
			const auto t100 = t15 * t96;
			const auto t101 = t100 + t97 + t98 + t99;
			const auto t102 = t18 * t84 + t21 * t82 - t50 * t87 - t6 * t85;
			const auto t103 = t102 * t23;
			const auto t104 = 2 * t25;
			const auto t105 = -2 * t101 * t23 * t58 - t102 * t58 * t60 + 2 * t103 * t61 + t104 * t15 * (t37 * t88 - t38 * t88 + t89) + t104 * t18 * t92 + 2 * t20 * t90 + 2 * t30 * t86 + 2 * t32 * t82 + 2 * t36 * t83 + 2 * t39 * t96 + 2 * t5 * t94;
			const auto t106 = t0 + t69 * t81 + t70 * t81;
			const auto t107 = -x0;
			const auto t108 = t107 + t72 * t81 + t73 * t81 + x2;
			const auto t109 = t107 + t63 * t81 + t66 * t81 + x1;
			const auto t110 = 4 * t81;
			const auto t111 = 2 * t81;
			H[0] = t23 * (4 * t23 * t58 * t61 - 2 * std::pow(t30, 2) - 2 * std::pow(t32, 2) - 2 * std::pow(t36, 2) - 2 * std::pow(t39, 2) - t40 * t41 - t41 * t42 - t41 * t43 - t41 * t44 - std::pow(t58, 2) * t60 - t79 * (2 * t15 * t26 * t30 + 2 * t24 * t25 * t32 * t74 - 2 * t24 * t25 * t36 * t71 - t30 * t78 + t36 * t76 - t52 * t56 * t68 - t56 * t75 + t57 * t77));
			H[1] = t23 * (t105 - t79 * (-t75 * t85 + t76 * t84 + t77 * t82 - t78 * t87 + t95));
			H[2] = t23 * (t105 - t79 * (t106 * t36 + t108 * t57 - t109 * t56 - t30 * t96 + t95));
			H[3] = -t23 * (t100 * t110 + 4 * t101 * t103 + std::pow(t102, 2) * t60 + t110 * t97 + t110 * t98 + t110 * t99 + t79 * (t106 * t84 + t108 * t82 - t109 * t85 - t111 * t15 * t87 + t111 * t71 * t84 - t68 * t80 * t85 + t82 * t88 * t93 - t87 * t96) + 2 * std::pow(t82, 2) + 2 * std::pow(t83, 2) + 2 * std::pow(t86, 2) + 2 * std::pow(t96, 2));
		}
	} // namespace autogen

} // namespace polyfem
