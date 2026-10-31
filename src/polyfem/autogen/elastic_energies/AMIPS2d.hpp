// Auto-generated code for AMIPS2d energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPS2d_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F, const double power) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = pow(F00, 2) - F00*F01 + pow(F01, 2) + pow(F10, 2) - F10*F11 + pow(F11, 2);
const auto helper_1 = helper_0/(F00*F11 - F01*F10);
const auto helper_2 = power*pow((2.0/3.0)*sqrt(3)*helper_1, power)/helper_0;
result_0[0] = -helper_2*(-2*F00 + F01 + F11*helper_1);
result_0[1] = helper_2*(-F00 + 2*F01 + F10*helper_1);
result_0[2] = helper_2*(F01*helper_1 + 2*F10 - F11);
result_0[3] = -helper_2*(F00*helper_1 + F10 - 2*F11);;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPS2d_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F, const double power) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            const auto helper_0 = 2*F00;
const auto helper_1 = -F01 + helper_0;
const auto helper_2 = F00*F11;
const auto helper_3 = F01*F10;
const auto helper_4 = helper_2 - helper_3;
const auto helper_5 = 1.0/helper_4;
const auto helper_6 = 2*F11;
const auto helper_7 = pow(F11, 2);
const auto helper_8 = pow(F00, 2);
const auto helper_9 = pow(F01, 2);
const auto helper_10 = pow(F10, 2);
const auto helper_11 = F00*F01;
const auto helper_12 = F10*F11;
const auto helper_13 = helper_10 - helper_11 - helper_12 + helper_7 + helper_8 + helper_9;
const auto helper_14 = 2*helper_13/pow(helper_4, 2);
const auto helper_15 = F11*helper_5;
const auto helper_16 = helper_13*helper_15;
const auto helper_17 = F01 - helper_0 + helper_16;
const auto helper_18 = 1.0/helper_13;
const auto helper_19 = helper_18*power;
const auto helper_20 = helper_17*helper_18;
const auto helper_21 = helper_13*helper_5;
const auto helper_22 = helper_19*pow((2.0/3.0)*sqrt(3)*helper_21, power);
const auto helper_23 = F00 - 2*F01;
const auto helper_24 = -helper_23;
const auto helper_25 = 2*F01;
const auto helper_26 = F10*helper_5;
const auto helper_27 = helper_13*helper_26;
const auto helper_28 = -F00 + helper_25 + helper_27;
const auto helper_29 = helper_17*helper_19;
const auto helper_30 = -F10*helper_1*helper_5 + helper_12*helper_14 + helper_15*helper_24 + helper_28*helper_29 + 1;
const auto helper_31 = F01*helper_5;
const auto helper_32 = 2*F10;
const auto helper_33 = -F11 + helper_32;
const auto helper_34 = helper_5*(-F01*helper_1 + F11*helper_33 + helper_16*helper_25);
const auto helper_35 = helper_13*helper_31 + helper_33;
const auto helper_36 = helper_29*helper_35;
const auto helper_37 = F00*helper_5;
const auto helper_38 = F10 - helper_6;
const auto helper_39 = -helper_38;
const auto helper_40 = 2*helper_21;
const auto helper_41 = helper_13*helper_37 + helper_38;
const auto helper_42 = -helper_17*helper_18*helper_41*power + helper_5*(F00*helper_1 + F11*helper_39 + helper_13 - helper_2*helper_40);
const auto helper_43 = helper_18*helper_28;
const auto helper_44 = -F01*helper_23 + F10*helper_33 + helper_13 + helper_3*helper_40;
const auto helper_45 = helper_0*helper_27;
const auto helper_46 = helper_19*helper_28;
const auto helper_47 = helper_41*helper_46;
const auto helper_48 = helper_18*helper_35;
const auto helper_49 = helper_11*helper_14 + helper_19*helper_35*helper_41 + helper_31*helper_38 + helper_33*helper_37 + 1;
const auto helper_50 = helper_18*helper_41;
result_0[0] = helper_22*(helper_1*helper_20 - helper_1*helper_5*helper_6 + helper_14*helper_7 - helper_15*helper_17 + pow(helper_17, 2)*helper_19 + 2);
result_0[1] = helper_22*(F10*helper_17*helper_5 - helper_20*helper_23 - helper_30);
result_0[2] = helper_22*(helper_17*helper_31 + helper_20*helper_33 - helper_34 - helper_36);
result_0[3] = helper_22*(-helper_17*helper_37 - helper_20*helper_38 - helper_42);
result_0[4] = helper_22*(F11*helper_28*helper_5 - helper_1*helper_43 - helper_30);
result_0[5] = helper_22*(helper_10*helper_14 + helper_19*pow(helper_28, 2) - helper_23*helper_32*helper_5 + helper_23*helper_43 - helper_26*helper_28 + 2);
result_0[6] = helper_22*(helper_18*helper_28*helper_35*power - helper_28*helper_31 - helper_33*helper_43 + helper_44*helper_5);
result_0[7] = helper_22*(helper_28*helper_37 + helper_38*helper_43 - helper_47 - helper_5*(-F00*helper_23 + F10*helper_38 + helper_45));
result_0[8] = helper_22*(F11*helper_35*helper_5 - helper_1*helper_48 - helper_34 - helper_36);
result_0[9] = helper_22*(helper_23*helper_48 - helper_26*helper_35 + helper_35*helper_46 + helper_44*helper_5);
result_0[10] = helper_22*(helper_14*helper_9 + helper_19*pow(helper_35, 2) + helper_25*helper_33*helper_5 - helper_31*helper_35 - helper_33*helper_48 + 2);
result_0[11] = helper_22*(F00*helper_35*helper_5 + helper_18*helper_35*helper_38 - helper_49);
result_0[12] = helper_22*(helper_1*helper_18*helper_41 - helper_15*helper_41 - helper_42);
result_0[13] = helper_22*(F10*helper_41*helper_5 - helper_23*helper_50 - helper_47 - helper_5*(F00*helper_24 - F10*helper_39 + helper_45));
result_0[14] = helper_22*(F01*helper_41*helper_5 + helper_18*helper_33*helper_41 - helper_49);
result_0[15] = helper_22*(helper_0*helper_38*helper_5 + helper_14*helper_8 + helper_19*pow(helper_41, 2) - helper_37*helper_41 - helper_38*helper_50 + 2);;
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
