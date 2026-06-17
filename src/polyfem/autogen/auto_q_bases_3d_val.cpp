#include "auto_q_bases_3d_val.hpp"


namespace polyfem {
namespace autogen {
namespace {
double q_0_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1;
return result;
}



void q_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_0_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
double q_1_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -1.0*(x - 1)*(y - 1)*(z - 1);
return result;
}

double q_1_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = 1.0*x*(y - 1)*(z - 1);
return result;
}

double q_1_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = -1.0*x*y*(z - 1);
return result;
}

double q_1_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = 1.0*y*(x - 1)*(z - 1);
return result;
}

double q_1_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = 1.0*z*(x - 1)*(y - 1);
return result;
}

double q_1_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = -1.0*x*z*(y - 1);
return result;
}

double q_1_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = 1.0*x*y*z;
return result;
}

double q_1_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = -1.0*y*z*(x - 1);
return result;
}



void q_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
double q_2_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = 1.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = 1.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = 1.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = 1.0*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = 1.0*x*z*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = 1.0*x*y*z*(2.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = 1.0*y*z*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = -4.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = -4.0*x*y*(2.0*x - 1.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = -4.0*x*y*(x - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = -4.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = -4.0*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = -4.0*x*z*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = -4.0*x*y*z*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = -4.0*y*z*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = -4.0*x*z*(x - 1)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = -4.0*x*y*z*(2.0*x - 1.0)*(y - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = -4.0*x*y*z*(x - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = -4.0*y*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_20(double x, double y, double z) {
double result;
result = 16.0*y*z*(x - 1)*(2.0*x - 1.0)*(y - 1)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_21(double x, double y, double z) {
double result;
result = 16.0*x*y*z*(2.0*x - 1.0)*(y - 1)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_22(double x, double y, double z) {
double result;
result = 16.0*x*z*(x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_23(double x, double y, double z) {
double result;
result = 16.0*x*y*z*(x - 1)*(2.0*y - 1.0)*(z - 1);
return result;
}

double q_2_basis_value_3d_single_24(double x, double y, double z) {
double result;
result = 16.0*x*y*(x - 1)*(y - 1)*(z - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_25(double x, double y, double z) {
double result;
result = 16.0*x*y*z*(x - 1)*(y - 1)*(2.0*z - 1.0);
return result;
}

double q_2_basis_value_3d_single_26(double x, double y, double z) {
double result;
result = -64.0*x*y*z*(x - 1)*(y - 1)*(z - 1);
return result;
}



void q_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 20:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_20(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 21:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_21(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 22:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_22(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 23:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_23(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 24:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_24(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 25:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_25(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 26:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_3d_single_26(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
double q_3_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -1.0*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = 1.0*x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = -1.0*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = 1.0*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = 1.0*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = -1.0*x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = 1.0*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = -1.0*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = -4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = 4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = 4.4999999999999991*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = -4.4999999999999991*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_20(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_21(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_22(double x, double y, double z) {
double result;
result = 4.4999999999999991*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_23(double x, double y, double z) {
double result;
result = -4.4999999999999991*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_24(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_25(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_26(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_27(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_28(double x, double y, double z) {
double result;
result = -4.4999999999999991*x*y*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_29(double x, double y, double z) {
double result;
result = 4.4999999999999991*x*y*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_30(double x, double y, double z) {
double result;
result = 4.4999999999999991*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_31(double x, double y, double z) {
double result;
result = -4.4999999999999991*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_32(double x, double y, double z) {
double result;
result = -20.249999999999993*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_33(double x, double y, double z) {
double result;
result = 20.249999999999993*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_34(double x, double y, double z) {
double result;
result = 20.249999999999993*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_35(double x, double y, double z) {
double result;
result = -20.249999999999993*y*z*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_36(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_37(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_38(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_39(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*z*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_40(double x, double y, double z) {
double result;
result = -20.249999999999993*x*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_41(double x, double y, double z) {
double result;
result = 20.249999999999993*x*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_42(double x, double y, double z) {
double result;
result = 20.249999999999993*x*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_43(double x, double y, double z) {
double result;
result = -20.249999999999993*x*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_44(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_45(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*z*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_46(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_47(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*z*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_48(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_49(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_50(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_51(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(1.5*z - 1.0)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_52(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_53(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_54(double x, double y, double z) {
double result;
result = -20.249999999999993*x*y*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_55(double x, double y, double z) {
double result;
result = 20.249999999999993*x*y*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(1.4999999999999998*z - 0.49999999999999989)*(2.9999999999999996*z - 1.9999999999999996);
return result;
}

double q_3_basis_value_3d_single_56(double x, double y, double z) {
double result;
result = 91.124999999999957*x*y*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_57(double x, double y, double z) {
double result;
result = -91.124999999999957*x*y*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_58(double x, double y, double z) {
double result;
result = -91.124999999999957*x*y*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_59(double x, double y, double z) {
double result;
result = 91.124999999999957*x*y*z*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_60(double x, double y, double z) {
double result;
result = -91.124999999999957*x*y*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_61(double x, double y, double z) {
double result;
result = 91.124999999999957*x*y*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0)*(z - 1)*(3.0*z - 1.0);
return result;
}

double q_3_basis_value_3d_single_62(double x, double y, double z) {
double result;
result = 91.124999999999957*x*y*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 2.0);
return result;
}

double q_3_basis_value_3d_single_63(double x, double y, double z) {
double result;
result = -91.124999999999957*x*y*z*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0)*(z - 1)*(3.0*z - 1.0);
return result;
}



void q_3_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 20:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_20(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 21:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_21(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 22:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_22(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 23:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_23(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 24:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_24(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 25:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_25(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 26:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_26(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 27:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_27(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 28:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_28(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 29:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_29(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 30:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_30(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 31:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_31(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 32:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_32(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 33:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_33(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 34:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_34(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 35:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_35(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 36:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_36(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 37:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_37(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 38:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_38(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 39:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_39(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 40:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_40(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 41:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_41(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 42:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_42(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 43:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_43(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 44:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_44(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 45:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_45(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 46:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_46(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 47:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_47(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 48:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_48(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 49:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_49(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 50:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_50(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 51:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_51(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 52:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_52(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 53:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_53(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 54:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_54(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 55:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_55(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 56:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_56(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 57:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_57(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 58:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_58(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 59:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_59(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 60:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_60(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 61:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_61(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 62:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_62(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 63:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_3d_single_63(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
double q_m2_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1.0*(x - 1)*(y - 1)*(z - 1)*(2*x + 2*y + 2*z - 1);
return result;
}

double q_m2_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = -1.0*x*(y - 1)*(z - 1)*(-2*x + 2*y + 2*z + 1);
return result;
}

double q_m2_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = -1.0*x*y*(z - 1)*(2*x + 2*y - 2*z - 3);
return result;
}

double q_m2_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = -1.0*y*(x - 1)*(z - 1)*(2*x - 2*y + 2*z + 1);
return result;
}

double q_m2_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = -1.0*z*(x - 1)*(y - 1)*(2*x + 2*y - 2*z + 1);
return result;
}

double q_m2_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = -1.0*x*z*(y - 1)*(2*x - 2*y + 2*z - 3);
return result;
}

double q_m2_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = x*y*z*(2.0*x + 2.0*y + 2.0*z - 5.0);
return result;
}

double q_m2_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = 1.0*y*z*(x - 1)*(2*x - 2*y - 2*z + 3);
return result;
}

double q_m2_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = -4*x*(x - 1)*(y - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 4*x*y*(y - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = 4*x*y*(x - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = -4*y*(x - 1)*(y - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = -4*z*(x - 1)*(y - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = 4*x*z*(y - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = -4*x*y*z*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = 4*y*z*(x - 1)*(z - 1);
return result;
}

double q_m2_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = 4*x*z*(x - 1)*(y - 1);
return result;
}

double q_m2_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = -4*x*y*z*(y - 1);
return result;
}

double q_m2_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = -4*x*y*z*(x - 1);
return result;
}

double q_m2_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = 4*y*z*(x - 1)*(y - 1);
return result;
}



void q_m2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
}

void q_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_value_3d(local_index, uv, val); break;
	case 1: q_1_basis_value_3d(local_index, uv, val); break;
	case 2: q_2_basis_value_3d(local_index, uv, val); break;
	case 3: q_3_basis_value_3d(local_index, uv, val); break;
	case -2: q_m2_basis_value_3d(local_index, uv, val); break;
	default: assert(false);
}}
}}
