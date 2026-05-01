// Auto-generated code for AMIPS2d energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2d_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = 1.0/(F00*F11 - F01*F10);
const auto helper_1 = helper_0*(pow(F00, 2) - F00*F01 + pow(F01, 2) + pow(F10, 2) - F10*F11 + pow(F11, 2));
const auto helper_2 = (2.0/3.0)*sqrt(3)*helper_0;
result_0[0] = helper_2*(2*F00 - F01 - F11*helper_1);
result_0[1] = helper_2*(-F00 + 2*F01 + F10*helper_1);
result_0[2] = helper_2*(F01*helper_1 + 2*F10 - F11);
result_0[3] = -helper_2*(F00*helper_1 + F10 - 2*F11);;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2d_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            const auto helper_0 = 2*F00;
const auto helper_1 = -F01 + helper_0;
const auto helper_2 = F01*F10;
const auto helper_3 = F00*F11 - helper_2;
const auto helper_4 = 1.0/helper_3;
const auto helper_5 = pow(F11, 2);
const auto helper_6 = pow(helper_3, -2);
const auto helper_7 = pow(F00, 2);
const auto helper_8 = pow(F01, 2);
const auto helper_9 = pow(F10, 2);
const auto helper_10 = F00*F01;
const auto helper_11 = F10*F11;
const auto helper_12 = -helper_10 - helper_11 + helper_5 + helper_7 + helper_8 + helper_9;
const auto helper_13 = helper_12*helper_6;
const auto helper_14 = sqrt(3);
const auto helper_15 = helper_14*helper_4;
const auto helper_16 = (4.0/3.0)*helper_15;
const auto helper_17 = 2*F01;
const auto helper_18 = F00 - helper_17;
const auto helper_19 = 2*helper_13;
const auto helper_20 = (2.0/3.0)*helper_15;
const auto helper_21 = helper_20*(F10*helper_1*helper_4 + F11*helper_18*helper_4 - helper_11*helper_19 - 1);
const auto helper_22 = 2*F10 - F11;
const auto helper_23 = helper_12*helper_4;
const auto helper_24 = (2.0/3.0)*helper_14*helper_6;
const auto helper_25 = helper_24*(F01*helper_1 - F11*helper_17*helper_23 - F11*helper_22);
const auto helper_26 = F10 - 2*F11;
const auto helper_27 = helper_24*(2*F00*F11*helper_12*helper_4 - F00*helper_1 + F11*helper_26 - helper_12);
const auto helper_28 = helper_24*(-F01*helper_18 + F10*helper_22 + helper_12 + 2*helper_2*helper_23);
const auto helper_29 = helper_24*(F00*helper_18 - F10*helper_0*helper_23 - F10*helper_26);
const auto helper_30 = helper_22*helper_4;
const auto helper_31 = helper_26*helper_4;
const auto helper_32 = -helper_20*(F00*helper_30 + F01*helper_31 + helper_10*helper_19 + 1);
result_0[0] = helper_16*(-F11*helper_1*helper_4 + helper_13*helper_5 + 1);
result_0[1] = helper_21;
result_0[2] = helper_25;
result_0[3] = helper_27;
result_0[4] = helper_21;
result_0[5] = helper_16*(-F10*helper_18*helper_4 + helper_13*helper_9 + 1);
result_0[6] = helper_28;
result_0[7] = helper_29;
result_0[8] = helper_25;
result_0[9] = helper_28;
result_0[10] = helper_16*(F01*helper_30 + helper_13*helper_8 + 1);
result_0[11] = helper_32;
result_0[12] = helper_27;
result_0[13] = helper_29;
result_0[14] = helper_32;
result_0[15] = helper_16*(F00*helper_31 + helper_13*helper_7 + 1);;
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
