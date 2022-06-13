#pragma once

#include <Eigen/Dense>

namespace polyfem {
namespace autogen {
template<typename T>
T int_pow(T val, int exp) { T res = exp <=0 ? T(0.): val; for(int i = 1; i < exp; ++i) res = res*val; return res; }

template<typename T>
void eigs_2d(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &m, Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> &res) {
res.resize(2);
const auto helper_0 = m(0,0) + m(1,1);
const auto helper_1 = (1.0/2.0)*helper_0;
const auto helper_2 = 4*pow(m(0,1), 2) + pow(m(0,0) - m(1,1), 2);
const auto helper_3 = helper_2 < 1.0e-10;
const auto helper_4 = sqrt(helper_2);
if (helper_3) {
   res(0) = helper_1;
}
else {
   res(0) = 0.5*helper_0 - 0.5*helper_4;
}
if (helper_3) {
   res(1) = helper_1;
}
else {
   res(1) = 0.5*helper_0 + 0.5*helper_4;
}
}

template<typename T>
void eigs_3d(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &m, Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> &res) {
res.resize(3);
const auto helper_0 = m(0,0) + m(1,1);
const auto helper_1 = helper_0 + m(2,2);
const auto helper_2 = 0.33333333333333331*helper_1;
const auto helper_3 = pow(m(0,1), 2);
const auto helper_4 = pow(m(0,2), 2);
const auto helper_5 = pow(m(1,2), 2);
const auto helper_6 = -m(2,2);
const auto helper_7 = 3.0*helper_3 + 3.0*helper_4 + 3.0*helper_5 + 0.5*pow(helper_6 + m(0,0), 2) + 0.5*pow(helper_6 + m(1,1), 2) + 0.5*pow(m(0,0) - m(1,1), 2);
const auto helper_8 = sqrt(helper_7);
const auto helper_9 = 2.0*helper_8 < 1.0e-10;
const auto helper_10 = (-1.5*helper_0*(m(0,0) + m(2,2))*(m(1,1) + m(2,2)) + 4.5*helper_1*(helper_3 + helper_4 + helper_5) - 13.5*helper_3*m(2,2) - 13.5*helper_4*m(1,1) - 13.5*helper_5*m(0,0) + 1.0*pow(m(0,0), 3) + 9.0*m(0,0)*m(1,1)*m(2,2) + 27.0*m(0,1)*m(0,2)*m(1,2) + 1.0*pow(m(1,1), 3) + 1.0*pow(m(2,2), 3))/pow(helper_7, 3.0/2.0);
const auto helper_11 = 0.33333333333333331*((helper_10 >= 1.0) ? (
   T(0)
)
: ((helper_10 <= -1.0) ? (
   T(M_PI)
)
: (
   acos(helper_10)
)));
const auto helper_12 = 0.66666666666666663*helper_8;
const auto helper_13 = 0.33333333333333331*m(0,0) + 0.33333333333333331*m(1,1) + 0.33333333333333331*m(2,2);
const auto helper_14 = 0.66666666666666663*M_PI;
if (helper_9) {
   res(0) = helper_2;
}
else {
   res(0) = helper_12*cos(helper_11) + helper_13;
}
if (helper_9) {
   res(1) = helper_2;
}
else {
   res(1) = helper_12*cos(helper_11 + helper_14) + helper_13;
}
if (helper_9) {
   res(2) = helper_2;
}
else {
   res(2) = helper_12*cos(helper_11 - helper_14) + helper_13;
}
}



}}
