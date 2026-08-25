#include "auto_b_bases.hpp"


namespace polyfem {
namespace autogen {
namespace {
double b_0_basis_value_2d_single_0(double x, double y) {
double result;
result = 1;
return result;
}



void b_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_0_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
void b_0_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 0;}
}



void b_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_0_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}


double b_1_basis_value_2d_single_0(double x, double y) {
double result;
result = -x - y + 1;
return result;
}

double b_1_basis_value_2d_single_1(double x, double y) {
double result;
result = x;
return result;
}

double b_1_basis_value_2d_single_2(double x, double y) {
double result;
result = y;
return result;
}



void b_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
void b_1_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -1;}
{val[1] = -1;}
}

void b_1_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 1;}
{val[1] = 0;}
}

void b_1_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 1;}
}



void b_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}


double b_2_basis_value_2d_single_0(double x, double y) {
double result;
result = pow(x + y - 1, 2);
return result;
}

double b_2_basis_value_2d_single_1(double x, double y) {
double result;
result = pow(x, 2);
return result;
}

double b_2_basis_value_2d_single_2(double x, double y) {
double result;
result = pow(y, 2);
return result;
}

double b_2_basis_value_2d_single_3(double x, double y) {
double result;
result = -2*x*(x + y - 1);
return result;
}

double b_2_basis_value_2d_single_4(double x, double y) {
double result;
result = 2*x*y;
return result;
}

double b_2_basis_value_2d_single_5(double x, double y) {
double result;
result = -2*y*(x + y - 1);
return result;
}



void b_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_2d_single_4(uv(i, 0), uv(i, 1));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_2d_single_5(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
void b_2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 2*(x + y - 1);}
{val[1] = 2*(x + y - 1);}
}

void b_2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 2*x;}
{val[1] = 0;}
}

void b_2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 2*y;}
}

void b_2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = 2*(-2*x - y + 1);}
{val[1] = -2*x;}
}

void b_2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = 2*y;}
{val[1] = 2*x;}
}

void b_2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -2*y;}
{val[1] = 2*(-x - 2*y + 1);}
}



void b_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_2d_single_4(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_2d_single_5(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}


double b_3_basis_value_2d_single_0(double x, double y) {
double result;
result = -pow(x + y - 1, 3);
return result;
}

double b_3_basis_value_2d_single_1(double x, double y) {
double result;
result = pow(x, 3);
return result;
}

double b_3_basis_value_2d_single_2(double x, double y) {
double result;
result = pow(y, 3);
return result;
}

double b_3_basis_value_2d_single_3(double x, double y) {
double result;
result = 3*x*pow(x + y - 1, 2);
return result;
}

double b_3_basis_value_2d_single_4(double x, double y) {
double result;
result = -3*pow(x, 2)*(x + y - 1);
return result;
}

double b_3_basis_value_2d_single_5(double x, double y) {
double result;
result = 3*pow(x, 2)*y;
return result;
}

double b_3_basis_value_2d_single_6(double x, double y) {
double result;
result = 3*x*pow(y, 2);
return result;
}

double b_3_basis_value_2d_single_7(double x, double y) {
double result;
result = -3*pow(y, 2)*(x + y - 1);
return result;
}

double b_3_basis_value_2d_single_8(double x, double y) {
double result;
result = 3*y*pow(x + y - 1, 2);
return result;
}

double b_3_basis_value_2d_single_9(double x, double y) {
double result;
result = -6*x*y*(x + y - 1);
return result;
}



void b_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_4(uv(i, 0), uv(i, 1));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_5(uv(i, 0), uv(i, 1));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_6(uv(i, 0), uv(i, 1));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_7(uv(i, 0), uv(i, 1));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_8(uv(i, 0), uv(i, 1));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_2d_single_9(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
void b_3_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -3*pow(x + y - 1, 2);}
{val[1] = -3*pow(x + y - 1, 2);}
}

void b_3_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 3*pow(x, 2);}
{val[1] = 0;}
}

void b_3_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 3*pow(y, 2);}
}

void b_3_basis_grad_value_2d_single_3(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = 3*(helper_0 + x)*(helper_0 + 3*x);}
{val[1] = 6*x*(x + y - 1);}
}

void b_3_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = 3*x;
val[0] = -helper_0*(helper_0 + 2*y - 2);}
{val[1] = -3*pow(x, 2);}
}

void b_3_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = 6*x*y;}
{val[1] = 3*pow(x, 2);}
}

void b_3_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = 3*pow(y, 2);}
{val[1] = 6*x*y;}
}

void b_3_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = -3*pow(y, 2);}
{double helper_0 = 3*y;
val[1] = -helper_0*(helper_0 + 2*x - 2);}
}

void b_3_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = 6*y*(x + y - 1);}
{double helper_0 = x - 1;
val[1] = 3*(helper_0 + y)*(helper_0 + 3*y);}
}

void b_3_basis_grad_value_2d_single_9(double x, double y, double *val) {
{val[0] = -6*y*(2*x + y - 1);}
{val[1] = -6*x*(x + 2*y - 1);}
}



void b_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_4(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_5(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_6(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_7(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_8(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_2d_single_9(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}


double b_4_basis_value_2d_single_0(double x, double y) {
double result;
result = pow(x + y - 1, 4);
return result;
}

double b_4_basis_value_2d_single_1(double x, double y) {
double result;
result = pow(x, 4);
return result;
}

double b_4_basis_value_2d_single_2(double x, double y) {
double result;
result = pow(y, 4);
return result;
}

double b_4_basis_value_2d_single_3(double x, double y) {
double result;
result = -4*x*pow(x + y - 1, 3);
return result;
}

double b_4_basis_value_2d_single_4(double x, double y) {
double result;
result = 6*pow(x, 2)*pow(x + y - 1, 2);
return result;
}

double b_4_basis_value_2d_single_5(double x, double y) {
double result;
result = -4*pow(x, 3)*(x + y - 1);
return result;
}

double b_4_basis_value_2d_single_6(double x, double y) {
double result;
result = 4*pow(x, 3)*y;
return result;
}

double b_4_basis_value_2d_single_7(double x, double y) {
double result;
result = 6*pow(x, 2)*pow(y, 2);
return result;
}

double b_4_basis_value_2d_single_8(double x, double y) {
double result;
result = 4*x*pow(y, 3);
return result;
}

double b_4_basis_value_2d_single_9(double x, double y) {
double result;
result = -4*pow(y, 3)*(x + y - 1);
return result;
}

double b_4_basis_value_2d_single_10(double x, double y) {
double result;
result = 6*pow(y, 2)*pow(x + y - 1, 2);
return result;
}

double b_4_basis_value_2d_single_11(double x, double y) {
double result;
result = -4*y*pow(x + y - 1, 3);
return result;
}

double b_4_basis_value_2d_single_12(double x, double y) {
double result;
result = 12*x*y*pow(x + y - 1, 2);
return result;
}

double b_4_basis_value_2d_single_13(double x, double y) {
double result;
result = -12*x*pow(y, 2)*(x + y - 1);
return result;
}

double b_4_basis_value_2d_single_14(double x, double y) {
double result;
result = -12*pow(x, 2)*y*(x + y - 1);
return result;
}



void b_4_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_0(uv(i, 0), uv(i, 1));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_1(uv(i, 0), uv(i, 1));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_2(uv(i, 0), uv(i, 1));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_3(uv(i, 0), uv(i, 1));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_4(uv(i, 0), uv(i, 1));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_5(uv(i, 0), uv(i, 1));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_6(uv(i, 0), uv(i, 1));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_7(uv(i, 0), uv(i, 1));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_8(uv(i, 0), uv(i, 1));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_9(uv(i, 0), uv(i, 1));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_10(uv(i, 0), uv(i, 1));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_11(uv(i, 0), uv(i, 1));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_12(uv(i, 0), uv(i, 1));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_13(uv(i, 0), uv(i, 1));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_2d_single_14(uv(i, 0), uv(i, 1));
		break;
	default: assert(false);
}
}
void b_4_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 4*pow(x + y - 1, 3);}
{val[1] = 4*pow(x + y - 1, 3);}
}

void b_4_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 4*pow(x, 3);}
{val[1] = 0;}
}

void b_4_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 4*pow(y, 3);}
}

void b_4_basis_grad_value_2d_single_3(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = -4*pow(helper_0 + x, 2)*(helper_0 + 4*x);}
{val[1] = -12*x*pow(x + y - 1, 2);}
}

void b_4_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = 12*x*(helper_0 + x)*(helper_0 + 2*x);}
{val[1] = 12*pow(x, 2)*(x + y - 1);}
}

void b_4_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -4*pow(x, 2)*(4*x + 3*y - 3);}
{val[1] = -4*pow(x, 3);}
}

void b_4_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = 12*pow(x, 2)*y;}
{val[1] = 4*pow(x, 3);}
}

void b_4_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = 12*x*pow(y, 2);}
{val[1] = 12*pow(x, 2)*y;}
}

void b_4_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = 4*pow(y, 3);}
{val[1] = 12*x*pow(y, 2);}
}

void b_4_basis_grad_value_2d_single_9(double x, double y, double *val) {
{val[0] = -4*pow(y, 3);}
{val[1] = -4*pow(y, 2)*(3*x + 4*y - 3);}
}

void b_4_basis_grad_value_2d_single_10(double x, double y, double *val) {
{val[0] = 12*pow(y, 2)*(x + y - 1);}
{double helper_0 = x - 1;
val[1] = 12*y*(helper_0 + y)*(helper_0 + 2*y);}
}

void b_4_basis_grad_value_2d_single_11(double x, double y, double *val) {
{val[0] = -12*y*pow(x + y - 1, 2);}
{double helper_0 = x - 1;
val[1] = -4*pow(helper_0 + y, 2)*(helper_0 + 4*y);}
}

void b_4_basis_grad_value_2d_single_12(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = 12*y*(helper_0 + x)*(helper_0 + 3*x);}
{double helper_0 = x - 1;
val[1] = 12*x*(helper_0 + y)*(helper_0 + 3*y);}
}

void b_4_basis_grad_value_2d_single_13(double x, double y, double *val) {
{val[0] = -12*pow(y, 2)*(2*x + y - 1);}
{val[1] = -12*x*y*(2*x + 3*y - 2);}
}

void b_4_basis_grad_value_2d_single_14(double x, double y, double *val) {
{val[0] = -12*x*y*(3*x + 2*y - 2);}
{val[1] = -12*pow(x, 2)*(x + 2*y - 1);}
}



void b_4_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 2);
double gradient[2];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_0(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_1(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_2(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_3(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_4(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_5(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_6(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_7(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_8(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_9(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_10(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_11(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_12(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_13(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_2d_single_14(uv(i, 0), uv(i, 1), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
		}
		break;
	default: assert(false);
}
}


}


void b_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_value_2d(local_index, uv, val); break;
	case 1: b_1_basis_value_2d(local_index, uv, val); break;
	case 2: b_2_basis_value_2d(local_index, uv, val); break;
	case 3: b_3_basis_value_2d(local_index, uv, val); break;
	case 4: b_4_basis_value_2d(local_index, uv, val); break;
	default: assert(false); 
}}

void b_grad_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_grad_value_2d(local_index, uv, val); break;
	case 1: b_1_basis_grad_value_2d(local_index, uv, val); break;
	case 2: b_2_basis_grad_value_2d(local_index, uv, val); break;
	case 3: b_3_basis_grad_value_2d(local_index, uv, val); break;
	case 4: b_4_basis_grad_value_2d(local_index, uv, val); break;
	default: assert(false); 
}}

namespace {
double b_0_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1;
return result;
}



void b_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_0_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
void b_0_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 0;}
}



void b_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_0_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}


double b_1_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -x - y - z + 1;
return result;
}

double b_1_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = x;
return result;
}

double b_1_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = y;
return result;
}

double b_1_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = z;
return result;
}



void b_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_1_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
void b_1_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -1;}
{val[1] = -1;}
{val[2] = -1;}
}

void b_1_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 1;}
{val[1] = 0;}
{val[2] = 0;}
}

void b_1_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 1;}
{val[2] = 0;}
}

void b_1_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 1;}
}



void b_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_1_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}


double b_2_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = pow(x + y + z - 1, 2);
return result;
}

double b_2_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = pow(x, 2);
return result;
}

double b_2_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = pow(y, 2);
return result;
}

double b_2_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = pow(z, 2);
return result;
}

double b_2_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = -2*x*(x + y + z - 1);
return result;
}

double b_2_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = 2*x*y;
return result;
}

double b_2_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = -2*y*(x + y + z - 1);
return result;
}

double b_2_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = -2*z*(x + y + z - 1);
return result;
}

double b_2_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = 2*x*z;
return result;
}

double b_2_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 2*y*z;
return result;
}



void b_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_2_basis_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
void b_2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 2*(x + y + z - 1);}
{val[1] = 2*(x + y + z - 1);}
{val[2] = 2*(x + y + z - 1);}
}

void b_2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 2*x;}
{val[1] = 0;}
{val[2] = 0;}
}

void b_2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 2*y;}
{val[2] = 0;}
}

void b_2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 2*z;}
}

void b_2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 2*(-2*x - y - z + 1);}
{val[1] = -2*x;}
{val[2] = -2*x;}
}

void b_2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = 2*y;}
{val[1] = 2*x;}
{val[2] = 0;}
}

void b_2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = -2*y;}
{val[1] = 2*(-x - 2*y - z + 1);}
{val[2] = -2*y;}
}

void b_2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = -2*z;}
{val[1] = -2*z;}
{val[2] = 2*(-x - y - 2*z + 1);}
}

void b_2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = 2*z;}
{val[1] = 0;}
{val[2] = 2*x;}
}

void b_2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 2*z;}
{val[2] = 2*y;}
}



void b_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_2_basis_grad_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}


double b_3_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -pow(x + y + z - 1, 3);
return result;
}

double b_3_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = pow(x, 3);
return result;
}

double b_3_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = pow(y, 3);
return result;
}

double b_3_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = pow(z, 3);
return result;
}

double b_3_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = 3*x*pow(x + y + z - 1, 2);
return result;
}

double b_3_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = -3*pow(x, 2)*(x + y + z - 1);
return result;
}

double b_3_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = 3*pow(x, 2)*y;
return result;
}

double b_3_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = 3*x*pow(y, 2);
return result;
}

double b_3_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = -3*pow(y, 2)*(x + y + z - 1);
return result;
}

double b_3_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 3*y*pow(x + y + z - 1, 2);
return result;
}

double b_3_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = 3*z*pow(x + y + z - 1, 2);
return result;
}

double b_3_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = -3*pow(z, 2)*(x + y + z - 1);
return result;
}

double b_3_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = 3*pow(x, 2)*z;
return result;
}

double b_3_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = 3*x*pow(z, 2);
return result;
}

double b_3_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = 3*pow(y, 2)*z;
return result;
}

double b_3_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = 3*y*pow(z, 2);
return result;
}

double b_3_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = -6*x*y*(x + y + z - 1);
return result;
}

double b_3_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = -6*x*z*(x + y + z - 1);
return result;
}

double b_3_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = 6*x*y*z;
return result;
}

double b_3_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = -6*y*z*(x + y + z - 1);
return result;
}



void b_3_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_3_basis_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
void b_3_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -3*pow(x + y + z - 1, 2);}
{val[1] = -3*pow(x + y + z - 1, 2);}
{val[2] = -3*pow(x + y + z - 1, 2);}
}

void b_3_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 3*pow(x, 2);}
{val[1] = 0;}
{val[2] = 0;}
}

void b_3_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 3*pow(y, 2);}
{val[2] = 0;}
}

void b_3_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 3*pow(z, 2);}
}

void b_3_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 3*(helper_0 + x)*(helper_0 + 3*x);}
{val[1] = 6*x*(x + y + z - 1);}
{val[2] = 6*x*(x + y + z - 1);}
}

void b_3_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = 3*x;
val[0] = -helper_0*(helper_0 + 2*y + 2*z - 2);}
{val[1] = -3*pow(x, 2);}
{val[2] = -3*pow(x, 2);}
}

void b_3_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = 6*x*y;}
{val[1] = 3*pow(x, 2);}
{val[2] = 0;}
}

void b_3_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = 3*pow(y, 2);}
{val[1] = 6*x*y;}
{val[2] = 0;}
}

void b_3_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = -3*pow(y, 2);}
{double helper_0 = 3*y;
val[1] = -helper_0*(helper_0 + 2*x + 2*z - 2);}
{val[2] = -3*pow(y, 2);}
}

void b_3_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 6*y*(x + y + z - 1);}
{double helper_0 = x + z - 1;
val[1] = 3*(helper_0 + y)*(helper_0 + 3*y);}
{val[2] = 6*y*(x + y + z - 1);}
}

void b_3_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = 6*z*(x + y + z - 1);}
{val[1] = 6*z*(x + y + z - 1);}
{double helper_0 = x + y - 1;
val[2] = 3*(helper_0 + z)*(helper_0 + 3*z);}
}

void b_3_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -3*pow(z, 2);}
{val[1] = -3*pow(z, 2);}
{double helper_0 = 3*z;
val[2] = -helper_0*(helper_0 + 2*x + 2*y - 2);}
}

void b_3_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = 6*x*z;}
{val[1] = 0;}
{val[2] = 3*pow(x, 2);}
}

void b_3_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = 3*pow(z, 2);}
{val[1] = 0;}
{val[2] = 6*x*z;}
}

void b_3_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 6*y*z;}
{val[2] = 3*pow(y, 2);}
}

void b_3_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 3*pow(z, 2);}
{val[2] = 6*y*z;}
}

void b_3_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = -6*y*(2*x + y + z - 1);}
{val[1] = -6*x*(x + 2*y + z - 1);}
{val[2] = -6*x*y;}
}

void b_3_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = -6*z*(2*x + y + z - 1);}
{val[1] = -6*x*z;}
{val[2] = -6*x*(x + y + 2*z - 1);}
}

void b_3_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = 6*y*z;}
{val[1] = 6*x*z;}
{val[2] = 6*x*y;}
}

void b_3_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = -6*y*z;}
{val[1] = -6*z*(x + 2*y + z - 1);}
{val[2] = -6*y*(x + y + 2*z - 1);}
}



void b_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_3_basis_grad_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}


double b_4_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = pow(x + y + z - 1, 4);
return result;
}

double b_4_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = pow(x, 4);
return result;
}

double b_4_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = pow(y, 4);
return result;
}

double b_4_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = pow(z, 4);
return result;
}

double b_4_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = -4*x*pow(x + y + z - 1, 3);
return result;
}

double b_4_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = 6*pow(x, 2)*pow(x + y + z - 1, 2);
return result;
}

double b_4_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = -4*pow(x, 3)*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = 4*pow(x, 3)*y;
return result;
}

double b_4_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = 6*pow(x, 2)*pow(y, 2);
return result;
}

double b_4_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 4*x*pow(y, 3);
return result;
}

double b_4_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = -4*pow(y, 3)*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = 6*pow(y, 2)*pow(x + y + z - 1, 2);
return result;
}

double b_4_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = -4*y*pow(x + y + z - 1, 3);
return result;
}

double b_4_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = -4*z*pow(x + y + z - 1, 3);
return result;
}

double b_4_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = 6*pow(z, 2)*pow(x + y + z - 1, 2);
return result;
}

double b_4_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = -4*pow(z, 3)*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = 4*pow(x, 3)*z;
return result;
}

double b_4_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = 6*pow(x, 2)*pow(z, 2);
return result;
}

double b_4_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = 4*x*pow(z, 3);
return result;
}

double b_4_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = 4*pow(y, 3)*z;
return result;
}

double b_4_basis_value_3d_single_20(double x, double y, double z) {
double result;
result = 6*pow(y, 2)*pow(z, 2);
return result;
}

double b_4_basis_value_3d_single_21(double x, double y, double z) {
double result;
result = 4*y*pow(z, 3);
return result;
}

double b_4_basis_value_3d_single_22(double x, double y, double z) {
double result;
result = 12*x*y*pow(x + y + z - 1, 2);
return result;
}

double b_4_basis_value_3d_single_23(double x, double y, double z) {
double result;
result = -12*x*pow(y, 2)*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_24(double x, double y, double z) {
double result;
result = -12*pow(x, 2)*y*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_25(double x, double y, double z) {
double result;
result = 12*x*z*pow(x + y + z - 1, 2);
return result;
}

double b_4_basis_value_3d_single_26(double x, double y, double z) {
double result;
result = -12*x*pow(z, 2)*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_27(double x, double y, double z) {
double result;
result = -12*pow(x, 2)*z*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_28(double x, double y, double z) {
double result;
result = 12*pow(x, 2)*y*z;
return result;
}

double b_4_basis_value_3d_single_29(double x, double y, double z) {
double result;
result = 12*x*y*pow(z, 2);
return result;
}

double b_4_basis_value_3d_single_30(double x, double y, double z) {
double result;
result = 12*x*pow(y, 2)*z;
return result;
}

double b_4_basis_value_3d_single_31(double x, double y, double z) {
double result;
result = -12*pow(y, 2)*z*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_32(double x, double y, double z) {
double result;
result = -12*y*pow(z, 2)*(x + y + z - 1);
return result;
}

double b_4_basis_value_3d_single_33(double x, double y, double z) {
double result;
result = 12*y*z*pow(x + y + z - 1, 2);
return result;
}

double b_4_basis_value_3d_single_34(double x, double y, double z) {
double result;
result = -24*x*y*z*(x + y + z - 1);
return result;
}



void b_4_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0){
result_0.resize(uv.rows(), 1);
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 20:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_20(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 21:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_21(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 22:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_22(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 23:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_23(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 24:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_24(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 25:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_25(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 26:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_26(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 27:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_27(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 28:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_28(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 29:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_29(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 30:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_30(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 31:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_31(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 32:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_32(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 33:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_33(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	case 34:
		for (Eigen::Index i = 0; i < uv.rows(); ++i)
			result_0(i, 0) = b_4_basis_value_3d_single_34(uv(i, 0), uv(i, 1), uv(i, 2));
		break;
	default: assert(false);
}
}
void b_4_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 4*pow(x + y + z - 1, 3);}
{val[1] = 4*pow(x + y + z - 1, 3);}
{val[2] = 4*pow(x + y + z - 1, 3);}
}

void b_4_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 4*pow(x, 3);}
{val[1] = 0;}
{val[2] = 0;}
}

void b_4_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 4*pow(y, 3);}
{val[2] = 0;}
}

void b_4_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 4*pow(z, 3);}
}

void b_4_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = -4*pow(helper_0 + x, 2)*(helper_0 + 4*x);}
{val[1] = -12*x*pow(x + y + z - 1, 2);}
{val[2] = -12*x*pow(x + y + z - 1, 2);}
}

void b_4_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 12*x*(helper_0 + x)*(helper_0 + 2*x);}
{val[1] = 12*pow(x, 2)*(x + y + z - 1);}
{val[2] = 12*pow(x, 2)*(x + y + z - 1);}
}

void b_4_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = -4*pow(x, 2)*(4*x + 3*y + 3*z - 3);}
{val[1] = -4*pow(x, 3);}
{val[2] = -4*pow(x, 3);}
}

void b_4_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = 12*pow(x, 2)*y;}
{val[1] = 4*pow(x, 3);}
{val[2] = 0;}
}

void b_4_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = 12*x*pow(y, 2);}
{val[1] = 12*pow(x, 2)*y;}
{val[2] = 0;}
}

void b_4_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 4*pow(y, 3);}
{val[1] = 12*x*pow(y, 2);}
{val[2] = 0;}
}

void b_4_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = -4*pow(y, 3);}
{val[1] = -4*pow(y, 2)*(3*x + 4*y + 3*z - 3);}
{val[2] = -4*pow(y, 3);}
}

void b_4_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = 12*pow(y, 2)*(x + y + z - 1);}
{double helper_0 = x + z - 1;
val[1] = 12*y*(helper_0 + y)*(helper_0 + 2*y);}
{val[2] = 12*pow(y, 2)*(x + y + z - 1);}
}

void b_4_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = -12*y*pow(x + y + z - 1, 2);}
{double helper_0 = x + z - 1;
val[1] = -4*pow(helper_0 + y, 2)*(helper_0 + 4*y);}
{val[2] = -12*y*pow(x + y + z - 1, 2);}
}

void b_4_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = -12*z*pow(x + y + z - 1, 2);}
{val[1] = -12*z*pow(x + y + z - 1, 2);}
{double helper_0 = x + y - 1;
val[2] = -4*pow(helper_0 + z, 2)*(helper_0 + 4*z);}
}

void b_4_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = 12*pow(z, 2)*(x + y + z - 1);}
{val[1] = 12*pow(z, 2)*(x + y + z - 1);}
{double helper_0 = x + y - 1;
val[2] = 12*z*(helper_0 + z)*(helper_0 + 2*z);}
}

void b_4_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = -4*pow(z, 3);}
{val[1] = -4*pow(z, 3);}
{val[2] = -4*pow(z, 2)*(3*x + 3*y + 4*z - 3);}
}

void b_4_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = 12*pow(x, 2)*z;}
{val[1] = 0;}
{val[2] = 4*pow(x, 3);}
}

void b_4_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = 12*x*pow(z, 2);}
{val[1] = 0;}
{val[2] = 12*pow(x, 2)*z;}
}

void b_4_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = 4*pow(z, 3);}
{val[1] = 0;}
{val[2] = 12*x*pow(z, 2);}
}

void b_4_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 12*pow(y, 2)*z;}
{val[2] = 4*pow(y, 3);}
}

void b_4_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 12*y*pow(z, 2);}
{val[2] = 12*pow(y, 2)*z;}
}

void b_4_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 4*pow(z, 3);}
{val[2] = 12*y*pow(z, 2);}
}

void b_4_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 12*y*(helper_0 + x)*(helper_0 + 3*x);}
{double helper_0 = x + z - 1;
val[1] = 12*x*(helper_0 + y)*(helper_0 + 3*y);}
{val[2] = 24*x*y*(x + y + z - 1);}
}

void b_4_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{val[0] = -12*pow(y, 2)*(2*x + y + z - 1);}
{val[1] = -12*x*y*(2*x + 3*y + 2*z - 2);}
{val[2] = -12*x*pow(y, 2);}
}

void b_4_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{val[0] = -12*x*y*(3*x + 2*y + 2*z - 2);}
{val[1] = -12*pow(x, 2)*(x + 2*y + z - 1);}
{val[2] = -12*pow(x, 2)*y;}
}

void b_4_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 12*z*(helper_0 + x)*(helper_0 + 3*x);}
{val[1] = 24*x*z*(x + y + z - 1);}
{double helper_0 = x + y - 1;
val[2] = 12*x*(helper_0 + z)*(helper_0 + 3*z);}
}

void b_4_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{val[0] = -12*pow(z, 2)*(2*x + y + z - 1);}
{val[1] = -12*x*pow(z, 2);}
{val[2] = -12*x*z*(2*x + 2*y + 3*z - 2);}
}

void b_4_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
{val[0] = -12*x*z*(3*x + 2*y + 2*z - 2);}
{val[1] = -12*pow(x, 2)*z;}
{val[2] = -12*pow(x, 2)*(x + y + 2*z - 1);}
}

void b_4_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
{val[0] = 24*x*y*z;}
{val[1] = 12*pow(x, 2)*z;}
{val[2] = 12*pow(x, 2)*y;}
}

void b_4_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
{val[0] = 12*y*pow(z, 2);}
{val[1] = 12*x*pow(z, 2);}
{val[2] = 24*x*y*z;}
}

void b_4_basis_grad_value_3d_single_30(double x, double y, double z, double *val) {
{val[0] = 12*pow(y, 2)*z;}
{val[1] = 24*x*y*z;}
{val[2] = 12*x*pow(y, 2);}
}

void b_4_basis_grad_value_3d_single_31(double x, double y, double z, double *val) {
{val[0] = -12*pow(y, 2)*z;}
{val[1] = -12*y*z*(2*x + 3*y + 2*z - 2);}
{val[2] = -12*pow(y, 2)*(x + y + 2*z - 1);}
}

void b_4_basis_grad_value_3d_single_32(double x, double y, double z, double *val) {
{val[0] = -12*y*pow(z, 2);}
{val[1] = -12*pow(z, 2)*(x + 2*y + z - 1);}
{val[2] = -12*y*z*(2*x + 2*y + 3*z - 2);}
}

void b_4_basis_grad_value_3d_single_33(double x, double y, double z, double *val) {
{val[0] = 24*y*z*(x + y + z - 1);}
{double helper_0 = x + z - 1;
val[1] = 12*z*(helper_0 + y)*(helper_0 + 3*y);}
{double helper_0 = x + y - 1;
val[2] = 12*y*(helper_0 + z)*(helper_0 + 3*z);}
}

void b_4_basis_grad_value_3d_single_34(double x, double y, double z, double *val) {
{val[0] = -24*y*z*(2*x + y + z - 1);}
{val[1] = -24*x*z*(x + 2*y + z - 1);}
{val[2] = -24*x*y*(x + y + 2*z - 1);}
}



void b_4_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){
val.resize(uv.rows(), 3);
double gradient[3];
switch(local_index){
	case 0:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_0(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 1:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_1(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 2:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_2(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 3:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_3(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 4:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_4(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 5:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_5(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 6:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_6(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 7:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_7(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 8:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_8(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 9:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_9(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 10:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_10(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 11:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_11(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 12:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_12(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 13:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_13(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 14:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_14(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 15:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_15(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 16:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_16(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 17:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_17(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 18:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_18(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 19:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_19(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 20:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_20(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 21:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_21(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 22:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_22(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 23:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_23(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 24:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_24(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 25:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_25(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 26:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_26(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 27:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_27(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 28:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_28(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 29:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_29(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 30:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_30(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 31:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_31(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 32:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_32(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 33:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_33(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	case 34:
		for (Eigen::Index i = 0; i < uv.rows(); ++i) {
			b_4_basis_grad_value_3d_single_34(uv(i, 0), uv(i, 1), uv(i, 2), gradient);
			val(i, 0) = gradient[0];
			val(i, 1) = gradient[1];
			val(i, 2) = gradient[2];
		}
		break;
	default: assert(false);
}
}


}


void b_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_value_3d(local_index, uv, val); break;
	case 1: b_1_basis_value_3d(local_index, uv, val); break;
	case 2: b_2_basis_value_3d(local_index, uv, val); break;
	case 3: b_3_basis_value_3d(local_index, uv, val); break;
	case 4: b_4_basis_value_3d(local_index, uv, val); break;
	default: assert(false); 
}}

void b_grad_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val){

switch(b){
	case 0: b_0_basis_grad_value_3d(local_index, uv, val); break;
	case 1: b_1_basis_grad_value_3d(local_index, uv, val); break;
	case 2: b_2_basis_grad_value_3d(local_index, uv, val); break;
	case 3: b_3_basis_grad_value_3d(local_index, uv, val); break;
	case 4: b_4_basis_grad_value_3d(local_index, uv, val); break;
	default: assert(false); 
}}

namespace {

}}}
