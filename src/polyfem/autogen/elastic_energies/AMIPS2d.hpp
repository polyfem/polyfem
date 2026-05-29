// Auto-generated code for AMIPS2d energy
#pragma once
#include <Eigen/Dense>

namespace polyfem
{
	namespace autogen
	{
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2d_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2, 2);
			const double F00 = F(0, 0);
			const double F01 = F(0, 1);
			const double F10 = F(1, 0);
			const double F11 = F(1, 1);
			std::array<double, 4> result_0;
			const auto helper_0 = F00 * F11 - F01 * F10;
			const auto helper_1 = pow(helper_0, -2);
			const auto helper_2 = 2 * F01;
			const auto helper_3 = F10 - 2 * F11;
			const auto helper_4 = (1.0 / 2.0) * (3 * pow(F00, 2) + 3 * pow(F10, 2) + pow(helper_3, 2) + pow(F00 - helper_2, 2)) / helper_0;
			result_0[0] = helper_1 * (2 * F00 - F01 - F11 * helper_4);
			result_0[1] = helper_1 * (-F00 + F10 * helper_4 + helper_2);
			result_0[2] = helper_1 * (F01 * helper_4 + 2 * F10 - F11);
			result_0[3] = -helper_1 * (F00 * helper_4 + helper_3);
			;
			grad(0, 0) = result_0[0];
			grad(0, 1) = result_0[1];
			grad(1, 0) = result_0[2];
			grad(1, 1) = result_0[3];
			return grad;
		}

		inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2d_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4, 4);
			std::array<double, 16> result_0;
			const double F00 = F(0, 0);
			const double F01 = F(0, 1);
			const double F10 = F(1, 0);
			const double F11 = F(1, 1);
			const auto helper_0 = F01 * F10;
			const auto helper_1 = F00 * F11 - helper_0;
			const auto helper_2 = pow(helper_1, -2);
			const auto helper_3 = 2 * F00;
			const auto helper_4 = -F01 + helper_3;
			const auto helper_5 = 1.0 / helper_1;
			const auto helper_6 = pow(F00, 2);
			const auto helper_7 = pow(F10, 2);
			const auto helper_8 = 2 * F01;
			const auto helper_9 = F00 - helper_8;
			const auto helper_10 = pow(helper_9, 2);
			const auto helper_11 = 2 * F11;
			const auto helper_12 = F10 - helper_11;
			const auto helper_13 = pow(helper_12, 2);
			const auto helper_14 = helper_10 + helper_13 + 3 * helper_6 + 3 * helper_7;
			const auto helper_15 = helper_14 * helper_2;
			const auto helper_16 = (3.0 / 2.0) * helper_15;
			const auto helper_17 = helper_2 * (-F10 * F11 * helper_16 + 2 * F10 * helper_4 * helper_5 + 2 * F11 * helper_5 * helper_9 - 1);
			const auto helper_18 = pow(helper_1, -3);
			const auto helper_19 = 2 * F10;
			const auto helper_20 = -F11 + helper_19;
			const auto helper_21 = (3.0 / 2.0) * helper_14 * helper_5;
			const auto helper_22 = helper_18 * (-F01 * F11 * helper_21 + 2 * F01 * helper_4 - helper_11 * helper_20);
			const auto helper_23 = (3.0 / 2.0) * helper_6;
			const auto helper_24 = (3.0 / 2.0) * helper_7;
			const auto helper_25 = (1.0 / 2.0) * helper_10 + (1.0 / 2.0) * helper_13 + helper_23 + helper_24;
			const auto helper_26 = helper_18 * ((3.0 / 2.0) * F00 * F11 * helper_14 * helper_5 + 2 * F11 * helper_12 - helper_25 - helper_3 * helper_4);
			const auto helper_27 = helper_18 * (helper_0 * helper_21 + helper_19 * helper_20 + helper_25 - helper_8 * helper_9);
			const auto helper_28 = helper_18 * (-F00 * F10 * helper_21 + 2 * F00 * helper_9 - helper_12 * helper_19);
			const auto helper_29 = helper_20 * helper_5;
			const auto helper_30 = helper_12 * helper_5;
			const auto helper_31 = -helper_2 * (F00 * F01 * helper_16 + helper_29 * helper_3 + helper_30 * helper_8 + 1);
			result_0[0] = helper_2 * (pow(F11, 2) * helper_16 - 4 * F11 * helper_4 * helper_5 + 2);
			result_0[1] = helper_17;
			result_0[2] = helper_22;
			result_0[3] = helper_26;
			result_0[4] = helper_17;
			result_0[5] = helper_2 * (-4 * F10 * helper_5 * helper_9 + helper_15 * helper_24 + 2);
			result_0[6] = helper_27;
			result_0[7] = helper_28;
			result_0[8] = helper_22;
			result_0[9] = helper_27;
			result_0[10] = helper_2 * (pow(F01, 2) * helper_16 + 4 * F01 * helper_29 + 2);
			result_0[11] = helper_31;
			result_0[12] = helper_26;
			result_0[13] = helper_28;
			result_0[14] = helper_31;
			result_0[15] = helper_2 * (4 * F00 * helper_30 + helper_15 * helper_23 + 2);
			;
			hess(0, 0) = result_0[0];
			hess(0, 1) = result_0[1];
			hess(0, 2) = result_0[2];
			hess(0, 3) = result_0[3];
			hess(1, 0) = result_0[4];
			hess(1, 1) = result_0[5];
			hess(1, 2) = result_0[6];
			hess(1, 3) = result_0[7];
			hess(2, 0) = result_0[8];
			hess(2, 1) = result_0[9];
			hess(2, 2) = result_0[10];
			hess(2, 3) = result_0[11];
			hess(3, 0) = result_0[12];
			hess(3, 1) = result_0[13];
			hess(3, 2) = result_0[14];
			hess(3, 3) = result_0[15];
			return hess;
		}
	} // namespace autogen
} // namespace polyfem
