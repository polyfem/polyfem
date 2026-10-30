// Auto-generated code for AMIPS2drest energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2drest_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F, const double power) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = pow(F00, 2) + pow(F01, 2) + pow(F10, 2) + pow(F11, 2);
const auto helper_1 = helper_0/(F00*F11 - F01*F10);
const auto helper_2 = pow(helper_1, power)*power/helper_0;
result_0[0] = helper_2*(2*F00 - F11*helper_1);
result_0[1] = helper_2*(2*F01 + F10*helper_1);
result_0[2] = helper_2*(F01*helper_1 + 2*F10);
result_0[3] = -helper_2*(F00*helper_1 - 2*F11);;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2drest_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F, const double power) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            const auto helper_0 = F00*F11;
const auto helper_1 = F01*F10;
const auto helper_2 = helper_0 - helper_1;
const auto helper_3 = 1.0/helper_2;
const auto helper_4 = 2*F00;
const auto helper_5 = pow(F01, 2);
const auto helper_6 = pow(F10, 2);
const auto helper_7 = helper_5 + helper_6;
const auto helper_8 = pow(F00, 2);
const auto helper_9 = pow(F11, 2);
const auto helper_10 = helper_8 + helper_9;
const auto helper_11 = helper_10 + helper_7;
const auto helper_12 = helper_11*helper_3;
const auto helper_13 = -F11*helper_12 + helper_4;
const auto helper_14 = helper_13*helper_3;
const auto helper_15 = 1.0/helper_11;
const auto helper_16 = helper_15*power;
const auto helper_17 = helper_13*helper_15;
const auto helper_18 = 2*helper_11/pow(helper_2, 2);
const auto helper_19 = 4*helper_3;
const auto helper_20 = -helper_0*helper_19 + 2;
const auto helper_21 = pow(helper_12, power)*helper_16;
const auto helper_22 = 2*F01;
const auto helper_23 = F01*F11;
const auto helper_24 = F00*F10;
const auto helper_25 = F10*F11;
const auto helper_26 = 2*helper_3;
const auto helper_27 = F10*helper_12 + helper_22;
const auto helper_28 = -helper_13*helper_15*helper_27*power + helper_26*(helper_12*helper_25 + helper_23 - helper_24);
const auto helper_29 = 2*F10;
const auto helper_30 = F00*F01;
const auto helper_31 = F01*helper_12 + helper_29;
const auto helper_32 = -helper_13*helper_15*helper_31*power + helper_26*(helper_12*helper_23 + helper_25 - helper_30);
const auto helper_33 = 2*F11;
const auto helper_34 = 2*helper_12;
const auto helper_35 = F00*helper_12 - helper_33;
const auto helper_36 = helper_13*helper_16*helper_35 + helper_3*(-helper_0*helper_34 + helper_7 + 3*helper_8 + 3*helper_9);
const auto helper_37 = helper_15*helper_27;
const auto helper_38 = helper_27*helper_3;
const auto helper_39 = helper_1*helper_19 + 2;
const auto helper_40 = -helper_15*helper_27*helper_31*power - helper_3*(helper_1*helper_34 + helper_10 + 3*helper_5 + 3*helper_6);
const auto helper_41 = helper_26*(helper_12*helper_24 - helper_25 + helper_30);
const auto helper_42 = helper_16*helper_27*helper_35;
const auto helper_43 = helper_15*helper_31;
const auto helper_44 = helper_3*helper_31;
const auto helper_45 = helper_26*(helper_12*helper_30 - helper_23 + helper_24);
const auto helper_46 = helper_16*helper_31*helper_35;
const auto helper_47 = helper_3*helper_35;
const auto helper_48 = helper_15*helper_35;
result_0[0] = helper_21*(F11*helper_14 + pow(helper_13, 2)*helper_16 - helper_17*helper_4 + helper_18*helper_9 + helper_20);
result_0[1] = helper_21*(-F10*helper_14 - helper_17*helper_22 - helper_28);
result_0[2] = helper_21*(-F01*helper_14 - helper_17*helper_29 - helper_32);
result_0[3] = helper_21*(F00*helper_13*helper_3 - helper_17*helper_33 - helper_36);
result_0[4] = helper_21*(F11*helper_27*helper_3 - helper_28 - helper_37*helper_4);
result_0[5] = helper_21*(-F10*helper_38 + helper_16*pow(helper_27, 2) + helper_18*helper_6 - helper_22*helper_37 + helper_39);
result_0[6] = helper_21*(-F01*helper_38 - helper_29*helper_37 - helper_40);
result_0[7] = helper_21*(F00*helper_27*helper_3 - helper_33*helper_37 - helper_41 - helper_42);
result_0[8] = helper_21*(F11*helper_3*helper_31 - helper_32 - helper_4*helper_43);
result_0[9] = helper_21*(-F10*helper_44 - helper_22*helper_43 - helper_40);
result_0[10] = helper_21*(-F01*helper_44 + helper_16*pow(helper_31, 2) + helper_18*helper_5 - helper_29*helper_43 + helper_39);
result_0[11] = helper_21*(F00*helper_3*helper_31 - helper_33*helper_43 - helper_45 - helper_46);
result_0[12] = helper_21*(2*F00*helper_15*helper_35 - F11*helper_47 - helper_36);
result_0[13] = helper_21*(F10*helper_47 + helper_22*helper_48 - helper_41 - helper_42);
result_0[14] = helper_21*(F01*helper_47 + helper_29*helper_48 - helper_45 - helper_46);
result_0[15] = helper_21*(-F00*helper_47 + helper_16*pow(helper_35, 2) + helper_18*helper_8 + helper_20 + helper_33*helper_48);;
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
