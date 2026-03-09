// Auto-generated code for AMIPS2drest energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2drest_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F0 = F(0, 0);
            const double F1 = F(0, 1);
            const double F2 = F(1, 0);
            const double F3 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = 1.0/(F0*F3 - F1*F2);
const auto helper_1 = helper_0*(pow(F0, 2) + pow(F1, 2) + pow(F2, 2) + pow(F3, 2));
result_0[0] = helper_0*(2*F0 - F3*helper_1);
result_0[1] = helper_0*(2*F1 + F2*helper_1);
result_0[2] = helper_0*(F1*helper_1 + 2*F2);
result_0[3] = helper_0*(-F0*helper_1 + 2*F3);;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2drest_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F0 = F(0, 0);
            const double F1 = F(0, 1);
            const double F2 = F(1, 0);
            const double F3 = F(1, 1);
            const auto helper_0 = pow(F3, 2);
const auto helper_1 = F0*F3;
const auto helper_2 = F1*F2;
const auto helper_3 = helper_1 - helper_2;
const auto helper_4 = pow(helper_3, -2);
const auto helper_5 = pow(F1, 2);
const auto helper_6 = pow(F2, 2);
const auto helper_7 = helper_5 + helper_6;
const auto helper_8 = pow(F0, 2);
const auto helper_9 = helper_0 + helper_8;
const auto helper_10 = helper_7 + helper_9;
const auto helper_11 = helper_10*helper_4;
const auto helper_12 = 1.0/helper_3;
const auto helper_13 = 2*helper_12;
const auto helper_14 = -helper_1*helper_13 + 1;
const auto helper_15 = F1*F3;
const auto helper_16 = F0*F2;
const auto helper_17 = F2*F3;
const auto helper_18 = helper_10*helper_12;
const auto helper_19 = 2*helper_4;
const auto helper_20 = helper_19*(-helper_15 + helper_16 - helper_17*helper_18);
const auto helper_21 = F0*F1;
const auto helper_22 = helper_19*(-helper_15*helper_18 - helper_17 + helper_21);
const auto helper_23 = helper_4*(2*F0*F3*helper_10*helper_12 - 3*helper_0 - helper_7 - 3*helper_8);
const auto helper_24 = helper_13*helper_2;
const auto helper_25 = helper_24 + 1;
const auto helper_26 = helper_4*(helper_10*helper_24 + 3*helper_5 + 3*helper_6 + helper_9);
const auto helper_27 = helper_19*(-helper_16*helper_18 + helper_17 - helper_21);
const auto helper_28 = helper_19*(helper_15 - helper_16 - helper_18*helper_21);
result_0[0] = helper_13*(helper_0*helper_11 + helper_14);
result_0[1] = helper_20;
result_0[2] = helper_22;
result_0[3] = helper_23;
result_0[4] = helper_20;
result_0[5] = helper_13*(helper_11*helper_6 + helper_25);
result_0[6] = helper_26;
result_0[7] = helper_27;
result_0[8] = helper_22;
result_0[9] = helper_26;
result_0[10] = helper_13*(helper_11*helper_5 + helper_25);
result_0[11] = helper_28;
result_0[12] = helper_23;
result_0[13] = helper_27;
result_0[14] = helper_28;
result_0[15] = helper_13*(helper_11*helper_8 + helper_14);;
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
    }
}
