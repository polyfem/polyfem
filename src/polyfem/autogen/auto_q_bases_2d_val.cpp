#include "auto_q_bases_2d_val.hpp"


namespace polyfem {
namespace autogen {
namespace {
double q_0_basis_value_2d_single_0(double x, double y) {
double result;
result = 1;
return result;
}



void q_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_0_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
double q_1_basis_value_2d_single_0(double x, double y) {
double result;
result = 1.0*(x - 1)*(y - 1);
return result;
}

double q_1_basis_value_2d_single_1(double x, double y) {
double result;
result = -1.0*x*(y - 1);
return result;
}

double q_1_basis_value_2d_single_2(double x, double y) {
double result;
result = 1.0*x*y;
return result;
}

double q_1_basis_value_2d_single_3(double x, double y) {
double result;
result = -1.0*y*(x - 1);
return result;
}



void q_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_1_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
double q_2_basis_value_2d_single_0(double x, double y) {
double result;
result = 1.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);
return result;
}

double q_2_basis_value_2d_single_1(double x, double y) {
double result;
result = 1.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);
return result;
}

double q_2_basis_value_2d_single_2(double x, double y) {
double result;
result = 1.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0);
return result;
}

double q_2_basis_value_2d_single_3(double x, double y) {
double result;
result = 1.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0);
return result;
}

double q_2_basis_value_2d_single_4(double x, double y) {
double result;
result = -4.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0);
return result;
}

double q_2_basis_value_2d_single_5(double x, double y) {
double result;
result = -4.0*x*y*(2.0*x - 1.0)*(y - 1);
return result;
}

double q_2_basis_value_2d_single_6(double x, double y) {
double result;
result = -4.0*x*y*(x - 1)*(2.0*y - 1.0);
return result;
}

double q_2_basis_value_2d_single_7(double x, double y) {
double result;
result = -4.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1);
return result;
}

double q_2_basis_value_2d_single_8(double x, double y) {
double result;
result = 16.0*x*y*(x - 1)*(y - 1);
return result;
}



void q_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_4(uv(i, 0), uv(i, 1));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_5(uv(i, 0), uv(i, 1));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_6(uv(i, 0), uv(i, 1));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_7(uv(i, 0), uv(i, 1));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_2_basis_value_2d_single_8(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
double q_3_basis_value_2d_single_0(double x, double y) {
double result;
result = 1.0*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_1(double x, double y) {
double result;
result = -1.0*x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_2(double x, double y) {
double result;
result = 1.0*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

double q_3_basis_value_2d_single_3(double x, double y) {
double result;
result = -1.0*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

double q_3_basis_value_2d_single_4(double x, double y) {
double result;
result = -4.4999999999999991*x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_5(double x, double y) {
double result;
result = 4.4999999999999991*x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_6(double x, double y) {
double result;
result = 4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0);
return result;
}

double q_3_basis_value_2d_single_7(double x, double y) {
double result;
result = -4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_8(double x, double y) {
double result;
result = -4.4999999999999991*x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

double q_3_basis_value_2d_single_9(double x, double y) {
double result;
result = 4.4999999999999991*x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

double q_3_basis_value_2d_single_10(double x, double y) {
double result;
result = 4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_11(double x, double y) {
double result;
result = -4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0);
return result;
}

double q_3_basis_value_2d_single_12(double x, double y) {
double result;
result = 20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0);
return result;
}

double q_3_basis_value_2d_single_13(double x, double y) {
double result;
result = -20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0);
return result;
}

double q_3_basis_value_2d_single_14(double x, double y) {
double result;
result = -20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0);
return result;
}

double q_3_basis_value_2d_single_15(double x, double y) {
double result;
result = 20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0);
return result;
}



void q_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_4(uv(i, 0), uv(i, 1));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_5(uv(i, 0), uv(i, 1));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_6(uv(i, 0), uv(i, 1));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_7(uv(i, 0), uv(i, 1));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_8(uv(i, 0), uv(i, 1));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_9(uv(i, 0), uv(i, 1));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_10(uv(i, 0), uv(i, 1));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_11(uv(i, 0), uv(i, 1));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_12(uv(i, 0), uv(i, 1));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_13(uv(i, 0), uv(i, 1));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_14(uv(i, 0), uv(i, 1));
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_3_basis_value_2d_single_15(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
double q_m2_basis_value_2d_single_0(double x, double y) {
double result;
result = -1.0*(x - 1)*(y - 1)*(2*x + 2*y - 1);
return result;
}

double q_m2_basis_value_2d_single_1(double x, double y) {
double result;
result = 1.0*x*(y - 1)*(-2*x + 2*y + 1);
return result;
}

double q_m2_basis_value_2d_single_2(double x, double y) {
double result;
result = x*y*(2.0*x + 2.0*y - 3.0);
return result;
}

double q_m2_basis_value_2d_single_3(double x, double y) {
double result;
result = 1.0*y*(x - 1)*(2*x - 2*y + 1);
return result;
}

double q_m2_basis_value_2d_single_4(double x, double y) {
double result;
result = 4*x*(x - 1)*(y - 1);
return result;
}

double q_m2_basis_value_2d_single_5(double x, double y) {
double result;
result = -4*x*y*(y - 1);
return result;
}

double q_m2_basis_value_2d_single_6(double x, double y) {
double result;
result = -4*x*y*(x - 1);
return result;
}

double q_m2_basis_value_2d_single_7(double x, double y) {
double result;
result = 4*y*(x - 1)*(y - 1);
return result;
}



void q_m2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_4(uv(i, 0), uv(i, 1));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_5(uv(i, 0), uv(i, 1));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_6(uv(i, 0), uv(i, 1));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = q_m2_basis_value_2d_single_7(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
}

void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
switch(q){
	case 0: q_0_basis_value_2d(local_index, uv, val); break;
	case 1: q_1_basis_value_2d(local_index, uv, val); break;
	case 2: q_2_basis_value_2d(local_index, uv, val); break;
	case 3: q_3_basis_value_2d(local_index, uv, val); break;
	case -2: q_m2_basis_value_2d(local_index, uv, val); break;
	default: assert(false);
}}
}}
