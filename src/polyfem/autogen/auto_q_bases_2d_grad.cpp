#include "auto_q_bases_2d_grad.hpp"


namespace polyfem {
namespace autogen {
namespace {
void q_0_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 0;}
}



void q_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_0_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}

void q_1_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 1.0*(y - 1);}
{val[1] = 1.0*(x - 1);}
}

void q_1_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 1.0*(1 - y);}
{val[1] = -1.0*x;}
}

void q_1_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 1.0*y;}
{val[1] = 1.0*x;}
}

void q_1_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = -1.0*y;}
{val[1] = 1.0*(1 - x);}
}



void q_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_1_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}

void q_2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = (4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0);}
}

void q_2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = (4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 3.0);}
}

void q_2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = y*(4.0*x - 1.0)*(2.0*y - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 1.0);}
}

void q_2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = y*(4.0*x - 3.0)*(2.0*y - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0);}
}

void q_2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = -4.0*(2*x - 1)*(y - 1)*(2.0*y - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 12.0);}
}

void q_2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -y*(16.0*x - 4.0)*(y - 1);}
{val[1] = -4.0*x*(2.0*x - 1.0)*(2*y - 1);}
}

void q_2_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = -4.0*y*(2*x - 1)*(2.0*y - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 4.0);}
}

void q_2_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = -y*(16.0*x - 12.0)*(y - 1);}
{val[1] = -4.0*(x - 1)*(2.0*x - 1.0)*(2*y - 1);}
}

void q_2_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = 16.0*y*(2*x - 1)*(y - 1);}
{val[1] = 16.0*x*(x - 1)*(2*y - 1);}
}



void q_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_4(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_5(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_6(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_7(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_2_basis_grad_value_2d_single_8(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}

void q_3_basis_grad_value_2d_single_0(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

void q_3_basis_grad_value_2d_single_1(double x, double y, double *val) {
{double helper_0 = 1.4999999999999998*x;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*x;
double helper_3 = helper_2 - 1.9999999999999996;
val[0] = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

void q_3_basis_grad_value_2d_single_2(double x, double y, double *val) {
{double helper_0 = 1.4999999999999998*x;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*x;
double helper_3 = helper_2 - 1.9999999999999996;
val[0] = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
{double helper_0 = 1.4999999999999998*y;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*y;
double helper_3 = helper_2 - 1.9999999999999996;
val[1] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
}

void q_3_basis_grad_value_2d_single_3(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = 1.4999999999999998*y;
double helper_1 = helper_0 - 0.49999999999999989;
double helper_2 = 2.9999999999999996*y;
double helper_3 = helper_2 - 1.9999999999999996;
val[1] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_3 + helper_1*helper_2 + 1.0*helper_1*helper_3);}
}

void q_3_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*(x - 1)*(3.0*x - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_2d_single_5(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*(x - 1)*(3.0*x - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

void q_3_basis_grad_value_2d_single_6(double x, double y, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*(y - 1)*(3.0*y - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_7(double x, double y, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*(y - 1)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_8(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*(x - 1)*(3.0*x - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
}

void q_3_basis_grad_value_2d_single_9(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*(x - 1)*(3.0*x - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
}

void q_3_basis_grad_value_2d_single_10(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*(y - 1)*(3.0*y - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_11(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*(y - 1)*(3.0*y - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_12(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = y*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = x*(x - 1)*(3.0*x - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_13(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = -y*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = -x*(x - 1)*(3.0*x - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_14(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = -y*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = -x*(x - 1)*(3.0*x - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}

void q_3_basis_grad_value_2d_single_15(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = y*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = x*(x - 1)*(3.0*x - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}



void q_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_4(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_5(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_6(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_7(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_8(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_9(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_10(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_11(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_12(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_13(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_14(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_3_basis_grad_value_2d_single_15(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}

void q_m2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -(y - 1)*(4.0*x + 2.0*y - 3.0);}
{val[1] = -(x - 1)*(2.0*x + 4.0*y - 3.0);}
}

void q_m2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = (y - 1)*(-4.0*x + 2.0*y + 1.0);}
{val[1] = -x*(2.0*x - 4.0*y + 1.0);}
}

void q_m2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = y*(4.0*x + 2.0*y - 3.0);}
{val[1] = x*(2.0*x + 4.0*y - 3.0);}
}

void q_m2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = -y*(-4.0*x + 2.0*y + 1.0);}
{val[1] = (x - 1)*(2.0*x - 4.0*y + 1.0);}
}

void q_m2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = 4*(2*x - 1)*(y - 1);}
{val[1] = 4*x*(x - 1);}
}

void q_m2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -4*y*(y - 1);}
{val[1] = -4*x*(2*y - 1);}
}

void q_m2_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = -4*y*(2*x - 1);}
{val[1] = -4*x*(x - 1);}
}

void q_m2_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = 4*y*(y - 1);}
{val[1] = 4*(x - 1)*(2*y - 1);}
}



void q_m2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_4(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_5(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_6(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			q_m2_basis_grad_value_2d_single_7(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}

}

void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_grad_value_2d(local_index, uv, val); break;
	case 1: q_1_basis_grad_value_2d(local_index, uv, val); break;
	case 2: q_2_basis_grad_value_2d(local_index, uv, val); break;
	case 3: q_3_basis_grad_value_2d(local_index, uv, val); break;
	case -2: q_m2_basis_grad_value_2d(local_index, uv, val); break;
	default: assert(false);
}}
}}
