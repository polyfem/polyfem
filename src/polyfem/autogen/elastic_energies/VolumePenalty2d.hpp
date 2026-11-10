// Auto-generated code for VolumePenalty2d energy
#pragma once
#include <Eigen/Dense>

namespace polyfem {
    namespace autogen {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> VolumePenalty2d_gradient(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F, const double k) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(2,2);
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            std::array<double, 4> result_0;
            const auto helper_0 = F01*F10;
const auto helper_1 = 0.5*k*(1.0*F00*F11 - 1.0*helper_0 - 1/(F00*F11 - helper_0));
result_0[0] = F11*helper_1;
result_0[1] = -F10*helper_1;
result_0[2] = -F01*helper_1;
result_0[3] = F00*helper_1;;
            grad(0, 0) = result_0[0];
            grad(0, 1) = result_0[1];
            grad(1, 0) = result_0[2];
            grad(1, 1) = result_0[3];
            return grad;
        }

        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> VolumePenalty2d_hessian(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F, const double k) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess(4,4);
            std::array<double, 16> result_0;
            const double F00 = F(0, 0);
            const double F01 = F(0, 1);
            const double F10 = F(1, 0);
            const double F11 = F(1, 1);
            const auto helper_0 = F00*F11;
const auto helper_1 = F01*F10;
const auto helper_2 = helper_0 - helper_1;
const auto helper_3 = pow(helper_2, -2);
const auto helper_4 = 0.5*k;
const auto helper_5 = helper_4*(helper_3 + 1.0);
const auto helper_6 = F11*helper_5;
const auto helper_7 = -F10*helper_6;
const auto helper_8 = -F01*helper_6;
const auto helper_9 = 1.0/helper_2;
const auto helper_10 = helper_4*(helper_0*helper_3 + 2.0*helper_0 - 1.0*helper_1 - helper_9);
const auto helper_11 = helper_4*(-1.0*helper_0 + helper_1*helper_3 + 2.0*helper_1 + helper_9);
const auto helper_12 = F00*helper_5;
const auto helper_13 = -F10*helper_12;
const auto helper_14 = -F01*helper_12;
result_0[0] = pow(F11, 2)*helper_5;
result_0[1] = helper_7;
result_0[2] = helper_8;
result_0[3] = helper_10;
result_0[4] = helper_7;
result_0[5] = pow(F10, 2)*helper_5;
result_0[6] = helper_11;
result_0[7] = helper_13;
result_0[8] = helper_8;
result_0[9] = helper_11;
result_0[10] = pow(F01, 2)*helper_5;
result_0[11] = helper_14;
result_0[12] = helper_10;
result_0[13] = helper_13;
result_0[14] = helper_14;
result_0[15] = pow(F00, 2)*helper_5;;
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
