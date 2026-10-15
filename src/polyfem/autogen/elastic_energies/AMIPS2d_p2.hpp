// Auto-generated code for AMIPS2d_p2 energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2d_p2_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = F00*F11 - F01*F10;
const auto helper_1 = 3*pow(F00, 2) + 3*pow(F10, 2) + pow(F00 - 2*F01, 2) + pow(F10 - 2*F11, 2);
const auto helper_2 = helper_1/helper_0;
const auto helper_3 = (1.0/6.0)*helper_1/pow(helper_0, 2);
result_0[0] = helper_3*(8*F00 - 4*F01 - F11*helper_2);
result_0[1] = helper_3*(-4*F00 + 8*F01 + F10*helper_2);
result_0[2] = helper_3*(F01*helper_2 + 8*F10 - 4*F11);
result_0[3] = -helper_3*(F00*helper_2 + 4*F10 - 8*F11);;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2d_p2_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            const auto helper_0 = 2*F00 - F01;
const auto helper_1 = pow(F00, 2);
const auto helper_2 = 3*helper_1;
const auto helper_3 = pow(F10, 2);
const auto helper_4 = 3*helper_3;
const auto helper_5 = F00 - 2*F01;
const auto helper_6 = pow(helper_5, 2);
const auto helper_7 = F10 - 2*F11;
const auto helper_8 = pow(helper_7, 2);
const auto helper_9 = helper_2 + helper_4 + helper_6 + helper_8;
const auto helper_10 = pow(helper_9, 2);
const auto helper_11 = F01*F10;
const auto helper_12 = F00*F11 - helper_11;
const auto helper_13 = pow(helper_12, -2);
const auto helper_14 = helper_10*helper_13;
const auto helper_15 = 3*helper_14;
const auto helper_16 = 16*helper_0;
const auto helper_17 = 1.0/helper_12;
const auto helper_18 = helper_17*helper_9;
const auto helper_19 = F11*helper_18;
const auto helper_20 = 8*helper_6;
const auto helper_21 = 24*helper_1 + 24*helper_3;
const auto helper_22 = helper_21 + 8*helper_8;
const auto helper_23 = helper_20 + helper_22;
const auto helper_24 = (1.0/6.0)*helper_13;
const auto helper_25 = F11*helper_15;
const auto helper_26 = 12*helper_1 + 12*helper_3 + 4*helper_6 + 4*helper_8;
const auto helper_27 = helper_24*(8*F10*helper_0*helper_17*helper_9 - F10*helper_25 + 8*F11*helper_17*helper_5*helper_9 - helper_16*helper_5 - helper_26);
const auto helper_28 = 2*F10 - F11;
const auto helper_29 = helper_24*(8*F01*helper_0*helper_17*helper_9 - F01*helper_25 + 16*helper_0*helper_28 - 8*helper_19*helper_28);
const auto helper_30 = helper_10*helper_17;
const auto helper_31 = helper_24*(3*F00*F11*helper_10*helper_13 - 8*F00*helper_0*helper_18 + 8*F11*helper_17*helper_7*helper_9 - helper_16*helper_7 - helper_30);
const auto helper_32 = 16*helper_5;
const auto helper_33 = F10*helper_18;
const auto helper_34 = F01*helper_18;
const auto helper_35 = 8*helper_33;
const auto helper_36 = helper_24*(helper_11*helper_15 - helper_28*helper_32 + helper_28*helper_35 + helper_30 - 8*helper_34*helper_5);
const auto helper_37 = F00*helper_15;
const auto helper_38 = helper_24*(8*F00*helper_17*helper_5*helper_9 - F10*helper_37 - helper_35*helper_7 + 16*helper_5*helper_7);
const auto helper_39 = 16*helper_28;
const auto helper_40 = F00*helper_18;
const auto helper_41 = -helper_24*(F01*helper_37 + helper_26 + 8*helper_28*helper_40 + 8*helper_34*helper_7 + helper_39*helper_7);
result_0[0] = helper_24*(pow(F11, 2)*helper_15 + 16*pow(helper_0, 2) - helper_16*helper_19 + helper_23);
result_0[1] = helper_27;
result_0[2] = helper_29;
result_0[3] = helper_31;
result_0[4] = helper_27;
result_0[5] = helper_24*(helper_14*helper_4 + helper_22 - helper_32*helper_33 + 24*helper_6);
result_0[6] = helper_36;
result_0[7] = helper_38;
result_0[8] = helper_29;
result_0[9] = helper_36;
result_0[10] = helper_24*(pow(F01, 2)*helper_15 + helper_23 + 16*pow(helper_28, 2) + helper_34*helper_39);
result_0[11] = helper_41;
result_0[12] = helper_31;
result_0[13] = helper_38;
result_0[14] = helper_41;
result_0[15] = helper_24*(helper_14*helper_2 + helper_20 + helper_21 + 16*helper_40*helper_7 + 24*helper_8);;
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
