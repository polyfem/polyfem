// Auto-generated code for AMIPS2drest_p2 energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2drest_p2_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = F00*F11 - F01*F10;
const auto helper_1 = pow(F00, 2) + pow(F01, 2) + pow(F10, 2) + pow(F11, 2);
const auto helper_2 = helper_1/helper_0;
const auto helper_3 = 2*helper_1/pow(helper_0, 2);
result_0[0] = helper_3*(2*F00 - F11*helper_2);
result_0[1] = helper_3*(2*F01 + F10*helper_2);
result_0[2] = helper_3*(F01*helper_2 + 2*F10);
result_0[3] = helper_3*(-F00*helper_2 + 2*F11);;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2drest_p2_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            const auto helper_0 = pow(F00, 2);
const auto helper_1 = pow(F11, 2);
const auto helper_2 = pow(F01, 2);
const auto helper_3 = pow(F10, 2);
const auto helper_4 = helper_0 + helper_1 + helper_2 + helper_3;
const auto helper_5 = pow(helper_4, 2);
const auto helper_6 = F00*F11;
const auto helper_7 = F01*F10;
const auto helper_8 = helper_6 - helper_7;
const auto helper_9 = pow(helper_8, -2);
const auto helper_10 = 3*helper_5*helper_9;
const auto helper_11 = 2*helper_3;
const auto helper_12 = 2*helper_1;
const auto helper_13 = helper_11 + helper_12;
const auto helper_14 = 2*helper_2;
const auto helper_15 = 1.0/helper_8;
const auto helper_16 = helper_15*helper_4;
const auto helper_17 = 8*helper_16;
const auto helper_18 = helper_14 - helper_17*helper_6;
const auto helper_19 = 2*helper_9;
const auto helper_20 = 4*F00;
const auto helper_21 = F01*helper_20;
const auto helper_22 = F11*helper_10;
const auto helper_23 = 4*F11;
const auto helper_24 = F10*helper_20;
const auto helper_25 = -F01*helper_16*helper_23 + helper_16*helper_24;
const auto helper_26 = helper_19*(-F10*helper_22 + helper_21 + helper_25);
const auto helper_27 = -F10*helper_16*helper_23 + helper_16*helper_21;
const auto helper_28 = helper_19*(-F01*helper_22 + helper_24 + helper_27);
const auto helper_29 = helper_15*helper_5;
const auto helper_30 = 4*helper_16;
const auto helper_31 = helper_19*(3*F00*F11*helper_5*helper_9 + 4*F00*F11 - helper_0*helper_30 - helper_1*helper_30 - helper_29);
const auto helper_32 = 2*helper_0;
const auto helper_33 = helper_17*helper_7 + helper_32;
const auto helper_34 = helper_19*(helper_10*helper_7 + helper_2*helper_30 + helper_29 + helper_3*helper_30 + 4*helper_7);
const auto helper_35 = F00*helper_10;
const auto helper_36 = helper_19*(4*F01*F11 - F10*helper_35 - helper_27);
const auto helper_37 = helper_19*(-F01*helper_35 + 4*F10*F11 - helper_25);
result_0[0] = helper_19*(6*helper_0 + helper_1*helper_10 + helper_13 + helper_18);
result_0[1] = helper_26;
result_0[2] = helper_28;
result_0[3] = helper_31;
result_0[4] = helper_26;
result_0[5] = helper_19*(helper_10*helper_3 + helper_13 + 6*helper_2 + helper_33);
result_0[6] = helper_34;
result_0[7] = helper_36;
result_0[8] = helper_28;
result_0[9] = helper_34;
result_0[10] = helper_19*(helper_10*helper_2 + helper_12 + helper_14 + 6*helper_3 + helper_33);
result_0[11] = helper_37;
result_0[12] = helper_31;
result_0[13] = helper_36;
result_0[14] = helper_37;
result_0[15] = helper_19*(helper_0*helper_10 + 6*helper_1 + helper_11 + helper_18 + helper_32);;
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
