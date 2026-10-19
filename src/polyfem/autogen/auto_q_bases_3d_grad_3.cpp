#include <Eigen/Dense>
#include <cassert>

namespace polyfem {
namespace autogen {
namespace {
void q_3_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*x;
double helper_3 = helper_2 - 1.9999999999999996;
val[0] = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*x;
double helper_3 = helper_2 - 1.9999999999999996;
val[0] = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = 1.4999999999999998*y;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*y;
double helper_3 = helper_2 - 1.9999999999999996;
val[1] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*y;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*y;
double helper_3 = helper_2 - 1.9999999999999996;
val[1] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*z;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*z;
double helper_3 = helper_2 - 1.9999999999999996;
val[2] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
}

void q_3_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*x;
double helper_3 = helper_2 - 1.9999999999999996;
val[0] = -z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*z;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*z;
double helper_3 = helper_2 - 1.9999999999999996;
val[2] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
}

void q_3_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*x;
double helper_3 = helper_2 - 1.9999999999999996;
val[0] = y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = 1.4999999999999998*y;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*y;
double helper_3 = helper_2 - 1.9999999999999996;
val[1] = x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = 1.4999999999999998*z;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*z;
double helper_3 = helper_2 - 1.9999999999999996;
val[2] = x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
}

void q_3_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*y;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*y;
double helper_3 = helper_2 - 1.9999999999999996;
val[1] = -z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = 1.4999999999999998*z;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*z;
double helper_3 = helper_2 - 1.9999999999999996;
val[2] = -y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
}

void q_3_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*(x - 1)*(3.0*x - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*(x - 1)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*(x - 1)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*(x - 1)*(3.0*x - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 8.9999999999999982;
val[2] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 4.4999999999999991;
val[2] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 4.4999999999999991;
val[2] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 8.9999999999999982;
val[2] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 4.4999999999999991;
val[2] = -x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 8.9999999999999982;
val[2] = x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 4.4999999999999991;
val[2] = y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 13.499999999999996*z - 8.9999999999999982;
val[2] = -y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = -z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = -x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*z*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*z*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = -x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = -x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = -y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = -x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_30(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*z*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_31(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*z*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = -z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = -y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*z + 6.7499999999999973*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_32(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = -z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = -y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_33(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_34(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_35(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = -z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = -y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_36(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*x + 30.374999999999986*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_37(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*x + 30.374999999999986*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = -x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = -x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_38(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*x + 30.374999999999986*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = -x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = -x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_39(double x, double y, double z, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*x + 30.374999999999986*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_40(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = -z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 2.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = -x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_41(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 1.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_42(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_43(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = -z*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = -x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_44(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 2.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*y + 30.374999999999986*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_45(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = -y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 1.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*y + 30.374999999999986*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = -x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_46(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = -y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*y + 30.374999999999986*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 40.499999999999986;
val[2] = -x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_47(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = y*z*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*y + 30.374999999999986*helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 60.749999999999979*z - 20.249999999999993;
val[2] = x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_48(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = -y*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = -x*(x - 1)*(3.0*x - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_49(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = y*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = x*(x - 1)*(3.0*x - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_50(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = y*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = x*(x - 1)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_51(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = -y*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = -x*(x - 1)*(3.0*x - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 1.5*z - 1.0;
double helper_2 = 3.0*z - 1.0;
val[2] = -x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(60.749999999999979*helper_0*helper_1 + 30.374999999999989*helper_0*helper_2 + 20.249999999999993*helper_1*helper_2);}
}

void q_3_basis_grad_value_3d_single_52(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = y*z*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = x*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*z + 30.374999999999986*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_53(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = -y*z*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = -x*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = -x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*z + 30.374999999999986*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_54(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = -y*z*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = -x*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = -x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*z + 30.374999999999986*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_55(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = y*z*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = x*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
{double helper_0 = 1.4999999999999998*z - 0.49999999999999989;
double helper_1 = 2.9999999999999996*z - 1.9999999999999996;
val[2] = x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(20.249999999999993*helper_0*helper_1 + 60.749999999999972*helper_0*z + 30.374999999999986*helper_1*z);}
}

void q_3_basis_grad_value_3d_single_56(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 182.24999999999991;
val[0] = y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 182.24999999999991;
val[1] = x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 182.24999999999991;
val[2] = x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_57(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 182.24999999999991;
val[0] = -y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 182.24999999999991;
val[1] = -x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 91.124999999999957;
val[2] = -x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_58(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 182.24999999999991;
val[0] = -y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 91.124999999999957;
val[1] = -x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 182.24999999999991;
val[2] = -x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_59(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 182.24999999999991;
val[0] = y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 91.124999999999957;
val[1] = x*z*(x - 1)*(3.0*x - 2.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 91.124999999999957;
val[2] = x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_60(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 91.124999999999957;
val[0] = -y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 182.24999999999991;
val[1] = -x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 182.24999999999991;
val[2] = -x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_61(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 91.124999999999957;
val[0] = y*z*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 182.24999999999991;
val[1] = x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 91.124999999999957;
val[2] = x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_62(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 91.124999999999957;
val[0] = y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 91.124999999999957;
val[1] = x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 2.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 182.24999999999991;
val[2] = x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}

void q_3_basis_grad_value_3d_single_63(double x, double y, double z, double *val) {
{double helper_0 = x - 1;
double helper_1 = 273.37499999999989*x - 91.124999999999957;
val[0] = -y*z*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 273.37499999999989*y - 91.124999999999957;
val[1] = -x*z*(x - 1)*(3.0*x - 1.0)*(z - 1)*(3.0*z - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*y + helper_1*y);}
{double helper_0 = z - 1;
double helper_1 = 273.37499999999989*z - 91.124999999999957;
val[2] = -x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 273.37499999999989*helper_0*z + helper_1*z);}
}



}

void q_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 20:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_20(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 21:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_21(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 22:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_22(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 23:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_23(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 24:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_24(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 25:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_25(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 26:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_26(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 27:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_27(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 28:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_28(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 29:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_29(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 30:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_30(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 31:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_31(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 32:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_32(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 33:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_33(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 34:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_34(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 35:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_35(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 36:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_36(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 37:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_37(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 38:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_38(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 39:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_39(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 40:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_40(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 41:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_41(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 42:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_42(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 43:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_43(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 44:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_44(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 45:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_45(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 46:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_46(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 47:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_47(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 48:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_48(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 49:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_49(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 50:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_50(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 51:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_51(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 52:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_52(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 53:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_53(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 54:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_54(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 55:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_55(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 56:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_56(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 57:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_57(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 58:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_58(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 59:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_59(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 60:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_60(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 61:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_61(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 62:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_62(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 63:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_3d_single_63(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}

}}