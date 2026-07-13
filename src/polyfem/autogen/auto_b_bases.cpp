#include "auto_b_bases.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH double b_0_basis_value_2d_single_0(double x, double y) {
double result;
result = 1;
return result;
}



POLYFEM_BOTH void b_0_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_0_basis_value_2d_single_0(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_0_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 0;}
}



POLYFEM_BOTH void b_0_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_0_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_1_basis_value_2d_single_0(double x, double y) {
double result;
result = -x - y + 1;
return result;
}

inline POLYFEM_BOTH double b_1_basis_value_2d_single_1(double x, double y) {
double result;
result = x;
return result;
}

inline POLYFEM_BOTH double b_1_basis_value_2d_single_2(double x, double y) {
double result;
result = y;
return result;
}



POLYFEM_BOTH void b_1_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_2d_single_2(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_1_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -1;}
{val[1] = -1;}
}

inline POLYFEM_BOTH void b_1_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 1;}
{val[1] = 0;}
}

inline POLYFEM_BOTH void b_1_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 1;}
}



POLYFEM_BOTH void b_1_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_2_basis_value_2d_single_0(double x, double y) {
double result;
result = pow(x + y - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_2d_single_1(double x, double y) {
double result;
result = pow(x, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_2d_single_2(double x, double y) {
double result;
result = pow(y, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_2d_single_3(double x, double y) {
double result;
result = -2*x*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_2d_single_4(double x, double y) {
double result;
result = 2*x*y;
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_2d_single_5(double x, double y) {
double result;
result = -2*y*(x + y - 1);
return result;
}



POLYFEM_BOTH void b_2_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_2d_single_5(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 2*(x + y - 1);}
{val[1] = 2*(x + y - 1);}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 2*x;}
{val[1] = 0;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 2*y;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = 2*(-2*x - y + 1);}
{val[1] = -2*x;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = 2*y;}
{val[1] = 2*x;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -2*y;}
{val[1] = 2*(-x - 2*y + 1);}
}



POLYFEM_BOTH void b_2_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_3_basis_value_2d_single_0(double x, double y) {
double result;
result = -pow(x + y - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_1(double x, double y) {
double result;
result = pow(x, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_2(double x, double y) {
double result;
result = pow(y, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_3(double x, double y) {
double result;
result = 3*x*pow(x + y - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_4(double x, double y) {
double result;
result = -3*pow(x, 2)*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_5(double x, double y) {
double result;
result = 3*pow(x, 2)*y;
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_6(double x, double y) {
double result;
result = 3*x*pow(y, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_7(double x, double y) {
double result;
result = -3*pow(y, 2)*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_8(double x, double y) {
double result;
result = 3*y*pow(x + y - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_2d_single_9(double x, double y) {
double result;
result = -6*x*y*(x + y - 1);
return result;
}



POLYFEM_BOTH void b_3_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_7(x[i], y[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_8(x[i], y[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_2d_single_9(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -3*pow(x + y - 1, 2);}
{val[1] = -3*pow(x + y - 1, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 3*pow(x, 2);}
{val[1] = 0;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 3*pow(y, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_3(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = 3*(helper_0 + x)*(helper_0 + 3*x);}
{val[1] = 6*x*(x + y - 1);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = 3*x;
val[0] = -helper_0*(helper_0 + 2*y - 2);}
{val[1] = -3*pow(x, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = 6*x*y;}
{val[1] = 3*pow(x, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = 3*pow(y, 2);}
{val[1] = 6*x*y;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = -3*pow(y, 2);}
{double helper_0 = 3*y;
val[1] = -helper_0*(helper_0 + 2*x - 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = 6*y*(x + y - 1);}
{double helper_0 = x - 1;
val[1] = 3*(helper_0 + y)*(helper_0 + 3*y);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_2d_single_9(double x, double y, double *val) {
{val[0] = -6*y*(2*x + y - 1);}
{val[1] = -6*x*(x + 2*y - 1);}
}



POLYFEM_BOTH void b_3_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_8(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_2d_single_9(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_4_basis_value_2d_single_0(double x, double y) {
double result;
result = pow(x + y - 1, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_1(double x, double y) {
double result;
result = pow(x, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_2(double x, double y) {
double result;
result = pow(y, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_3(double x, double y) {
double result;
result = -4*x*pow(x + y - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_4(double x, double y) {
double result;
result = 6*pow(x, 2)*pow(x + y - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_5(double x, double y) {
double result;
result = -4*pow(x, 3)*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_6(double x, double y) {
double result;
result = 4*pow(x, 3)*y;
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_7(double x, double y) {
double result;
result = 6*pow(x, 2)*pow(y, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_8(double x, double y) {
double result;
result = 4*x*pow(y, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_9(double x, double y) {
double result;
result = -4*pow(y, 3)*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_10(double x, double y) {
double result;
result = 6*pow(y, 2)*pow(x + y - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_11(double x, double y) {
double result;
result = -4*y*pow(x + y - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_12(double x, double y) {
double result;
result = 12*x*y*pow(x + y - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_13(double x, double y) {
double result;
result = -12*x*pow(y, 2)*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_2d_single_14(double x, double y) {
double result;
result = -12*pow(x, 2)*y*(x + y - 1);
return result;
}



POLYFEM_BOTH void b_4_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_7(x[i], y[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_8(x[i], y[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_9(x[i], y[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_10(x[i], y[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_11(x[i], y[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_12(x[i], y[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_13(x[i], y[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_2d_single_14(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 4*pow(x + y - 1, 3);}
{val[1] = 4*pow(x + y - 1, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 4*pow(x, 3);}
{val[1] = 0;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 4*pow(y, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_3(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = -4*pow(helper_0 + x, 2)*(helper_0 + 4*x);}
{val[1] = -12*x*pow(x + y - 1, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = 12*x*(helper_0 + x)*(helper_0 + 2*x);}
{val[1] = 12*pow(x, 2)*(x + y - 1);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -4*pow(x, 2)*(4*x + 3*y - 3);}
{val[1] = -4*pow(x, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = 12*pow(x, 2)*y;}
{val[1] = 4*pow(x, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = 12*x*pow(y, 2);}
{val[1] = 12*pow(x, 2)*y;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = 4*pow(y, 3);}
{val[1] = 12*x*pow(y, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_9(double x, double y, double *val) {
{val[0] = -4*pow(y, 3);}
{val[1] = -4*pow(y, 2)*(3*x + 4*y - 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_10(double x, double y, double *val) {
{val[0] = 12*pow(y, 2)*(x + y - 1);}
{double helper_0 = x - 1;
val[1] = 12*y*(helper_0 + y)*(helper_0 + 2*y);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_11(double x, double y, double *val) {
{val[0] = -12*y*pow(x + y - 1, 2);}
{double helper_0 = x - 1;
val[1] = -4*pow(helper_0 + y, 2)*(helper_0 + 4*y);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_12(double x, double y, double *val) {
{double helper_0 = y - 1;
val[0] = 12*y*(helper_0 + x)*(helper_0 + 3*x);}
{double helper_0 = x - 1;
val[1] = 12*x*(helper_0 + y)*(helper_0 + 3*y);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_13(double x, double y, double *val) {
{val[0] = -12*pow(y, 2)*(2*x + y - 1);}
{val[1] = -12*x*y*(2*x + 3*y - 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_2d_single_14(double x, double y, double *val) {
{val[0] = -12*x*y*(3*x + 2*y - 2);}
{val[1] = -12*pow(x, 2)*(x + 2*y - 1);}
}



POLYFEM_BOTH void b_4_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_8(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_9(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_10(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_11(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_12(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_13(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_2d_single_14(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


POLYFEM_BOTH int b_basis_count_2d(const int b) {
switch(b) {
	case 0: return 1;
	case 1: return 3;
	case 2: return 6;
	case 3: return 10;
	case 4: return 15;
	default: assert(false); return 0;
}}

POLYFEM_BOTH void b_basis_value_2d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){

switch(b){
	case 0: b_0_basis_value_2d(local_index, x, y, z, val); break;
	case 1: b_1_basis_value_2d(local_index, x, y, z, val); break;
	case 2: b_2_basis_value_2d(local_index, x, y, z, val); break;
	case 3: b_3_basis_value_2d(local_index, x, y, z, val); break;
	case 4: b_4_basis_value_2d(local_index, x, y, z, val); break;
	default: assert(false); 
}}

POLYFEM_BOTH void b_grad_basis_value_2d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){

switch(b){
	case 0: b_0_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: b_1_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: b_2_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: b_3_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 4: b_4_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default: assert(false); 
}}

inline POLYFEM_BOTH double b_0_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1;
return result;
}



POLYFEM_BOTH void b_0_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_0_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_0_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 0;}
}



POLYFEM_BOTH void b_0_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_0_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_1_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -x - y - z + 1;
return result;
}

inline POLYFEM_BOTH double b_1_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = x;
return result;
}

inline POLYFEM_BOTH double b_1_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = y;
return result;
}

inline POLYFEM_BOTH double b_1_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = z;
return result;
}



POLYFEM_BOTH void b_1_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_1_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_1_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -1;}
{val[1] = -1;}
{val[2] = -1;}
}

inline POLYFEM_BOTH void b_1_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 1;}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_1_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 1;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_1_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 1;}
}



POLYFEM_BOTH void b_1_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_1_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_2_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = pow(x, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = pow(y, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = pow(z, 2);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = -2*x*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = 2*x*y;
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = -2*y*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = -2*z*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = 2*x*z;
return result;
}

inline POLYFEM_BOTH double b_2_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 2*y*z;
return result;
}



POLYFEM_BOTH void b_2_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_2_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 2*(x + y + z - 1);}
{val[1] = 2*(x + y + z - 1);}
{val[2] = 2*(x + y + z - 1);}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 2*x;}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 2*y;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 2*z;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 2*(-2*x - y - z + 1);}
{val[1] = -2*x;}
{val[2] = -2*x;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = 2*y;}
{val[1] = 2*x;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = -2*y;}
{val[1] = 2*(-x - 2*y - z + 1);}
{val[2] = -2*y;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = -2*z;}
{val[1] = -2*z;}
{val[2] = 2*(-x - y - 2*z + 1);}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = 2*z;}
{val[1] = 0;}
{val[2] = 2*x;}
}

inline POLYFEM_BOTH void b_2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 2*z;}
{val[2] = 2*y;}
}



POLYFEM_BOTH void b_2_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_2_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_3_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -pow(x + y + z - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = pow(x, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = pow(y, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = pow(z, 3);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = 3*x*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = -3*pow(x, 2)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = 3*pow(x, 2)*y;
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = 3*x*pow(y, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = -3*pow(y, 2)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 3*y*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = 3*z*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = -3*pow(z, 2)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = 3*pow(x, 2)*z;
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = 3*x*pow(z, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = 3*pow(y, 2)*z;
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = 3*y*pow(z, 2);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = -6*x*y*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = -6*x*z*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = 6*x*y*z;
return result;
}

inline POLYFEM_BOTH double b_3_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = -6*y*z*(x + y + z - 1);
return result;
}



POLYFEM_BOTH void b_3_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_14(x[i], y[i], z[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_15(x[i], y[i], z[i]);
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_16(x[i], y[i], z[i]);
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_17(x[i], y[i], z[i]);
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_18(x[i], y[i], z[i]);
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_3_basis_value_3d_single_19(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -3*pow(x + y + z - 1, 2);}
{val[1] = -3*pow(x + y + z - 1, 2);}
{val[2] = -3*pow(x + y + z - 1, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 3*pow(x, 2);}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 3*pow(y, 2);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 3*pow(z, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 3*(helper_0 + x)*(helper_0 + 3*x);}
{val[1] = 6*x*(x + y + z - 1);}
{val[2] = 6*x*(x + y + z - 1);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = 3*x;
val[0] = -helper_0*(helper_0 + 2*y + 2*z - 2);}
{val[1] = -3*pow(x, 2);}
{val[2] = -3*pow(x, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = 6*x*y;}
{val[1] = 3*pow(x, 2);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = 3*pow(y, 2);}
{val[1] = 6*x*y;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = -3*pow(y, 2);}
{double helper_0 = 3*y;
val[1] = -helper_0*(helper_0 + 2*x + 2*z - 2);}
{val[2] = -3*pow(y, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 6*y*(x + y + z - 1);}
{double helper_0 = x + z - 1;
val[1] = 3*(helper_0 + y)*(helper_0 + 3*y);}
{val[2] = 6*y*(x + y + z - 1);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = 6*z*(x + y + z - 1);}
{val[1] = 6*z*(x + y + z - 1);}
{double helper_0 = x + y - 1;
val[2] = 3*(helper_0 + z)*(helper_0 + 3*z);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -3*pow(z, 2);}
{val[1] = -3*pow(z, 2);}
{double helper_0 = 3*z;
val[2] = -helper_0*(helper_0 + 2*x + 2*y - 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = 6*x*z;}
{val[1] = 0;}
{val[2] = 3*pow(x, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = 3*pow(z, 2);}
{val[1] = 0;}
{val[2] = 6*x*z;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 6*y*z;}
{val[2] = 3*pow(y, 2);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 3*pow(z, 2);}
{val[2] = 6*y*z;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = -6*y*(2*x + y + z - 1);}
{val[1] = -6*x*(x + 2*y + z - 1);}
{val[2] = -6*x*y;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = -6*z*(2*x + y + z - 1);}
{val[1] = -6*x*z;}
{val[2] = -6*x*(x + y + 2*z - 1);}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = 6*y*z;}
{val[1] = 6*x*z;}
{val[2] = 6*x*y;}
}

inline POLYFEM_BOTH void b_3_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = -6*y*z;}
{val[1] = -6*z*(x + 2*y + z - 1);}
{val[2] = -6*y*(x + y + 2*z - 1);}
}



POLYFEM_BOTH void b_3_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_3_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double b_4_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = pow(x + y + z - 1, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = pow(x, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = pow(y, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = pow(z, 4);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = -4*x*pow(x + y + z - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = 6*pow(x, 2)*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = -4*pow(x, 3)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = 4*pow(x, 3)*y;
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = 6*pow(x, 2)*pow(y, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 4*x*pow(y, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = -4*pow(y, 3)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = 6*pow(y, 2)*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = -4*y*pow(x + y + z - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = -4*z*pow(x + y + z - 1, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = 6*pow(z, 2)*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = -4*pow(z, 3)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = 4*pow(x, 3)*z;
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = 6*pow(x, 2)*pow(z, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = 4*x*pow(z, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = 4*pow(y, 3)*z;
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_20(double x, double y, double z) {
double result;
result = 6*pow(y, 2)*pow(z, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_21(double x, double y, double z) {
double result;
result = 4*y*pow(z, 3);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_22(double x, double y, double z) {
double result;
result = 12*x*y*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_23(double x, double y, double z) {
double result;
result = -12*x*pow(y, 2)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_24(double x, double y, double z) {
double result;
result = -12*pow(x, 2)*y*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_25(double x, double y, double z) {
double result;
result = 12*x*z*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_26(double x, double y, double z) {
double result;
result = -12*x*pow(z, 2)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_27(double x, double y, double z) {
double result;
result = -12*pow(x, 2)*z*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_28(double x, double y, double z) {
double result;
result = 12*pow(x, 2)*y*z;
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_29(double x, double y, double z) {
double result;
result = 12*x*y*pow(z, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_30(double x, double y, double z) {
double result;
result = 12*x*pow(y, 2)*z;
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_31(double x, double y, double z) {
double result;
result = -12*pow(y, 2)*z*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_32(double x, double y, double z) {
double result;
result = -12*y*pow(z, 2)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_33(double x, double y, double z) {
double result;
result = 12*y*z*pow(x + y + z - 1, 2);
return result;
}

inline POLYFEM_BOTH double b_4_basis_value_3d_single_34(double x, double y, double z) {
double result;
result = -24*x*y*z*(x + y + z - 1);
return result;
}



POLYFEM_BOTH void b_4_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_14(x[i], y[i], z[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_15(x[i], y[i], z[i]);
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_16(x[i], y[i], z[i]);
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_17(x[i], y[i], z[i]);
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_18(x[i], y[i], z[i]);
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_19(x[i], y[i], z[i]);
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_20(x[i], y[i], z[i]);
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_21(x[i], y[i], z[i]);
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_22(x[i], y[i], z[i]);
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_23(x[i], y[i], z[i]);
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_24(x[i], y[i], z[i]);
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_25(x[i], y[i], z[i]);
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_26(x[i], y[i], z[i]);
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_27(x[i], y[i], z[i]);
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_28(x[i], y[i], z[i]);
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_29(x[i], y[i], z[i]);
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_30(x[i], y[i], z[i]);
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_31(x[i], y[i], z[i]);
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_32(x[i], y[i], z[i]);
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_33(x[i], y[i], z[i]);
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = b_4_basis_value_3d_single_34(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 4*pow(x + y + z - 1, 3);}
{val[1] = 4*pow(x + y + z - 1, 3);}
{val[2] = 4*pow(x + y + z - 1, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 4*pow(x, 3);}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 4*pow(y, 3);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 4*pow(z, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = -4*pow(helper_0 + x, 2)*(helper_0 + 4*x);}
{val[1] = -12*x*pow(x + y + z - 1, 2);}
{val[2] = -12*x*pow(x + y + z - 1, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 12*x*(helper_0 + x)*(helper_0 + 2*x);}
{val[1] = 12*pow(x, 2)*(x + y + z - 1);}
{val[2] = 12*pow(x, 2)*(x + y + z - 1);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = -4*pow(x, 2)*(4*x + 3*y + 3*z - 3);}
{val[1] = -4*pow(x, 3);}
{val[2] = -4*pow(x, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = 12*pow(x, 2)*y;}
{val[1] = 4*pow(x, 3);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = 12*x*pow(y, 2);}
{val[1] = 12*pow(x, 2)*y;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 4*pow(y, 3);}
{val[1] = 12*x*pow(y, 2);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = -4*pow(y, 3);}
{val[1] = -4*pow(y, 2)*(3*x + 4*y + 3*z - 3);}
{val[2] = -4*pow(y, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = 12*pow(y, 2)*(x + y + z - 1);}
{double helper_0 = x + z - 1;
val[1] = 12*y*(helper_0 + y)*(helper_0 + 2*y);}
{val[2] = 12*pow(y, 2)*(x + y + z - 1);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = -12*y*pow(x + y + z - 1, 2);}
{double helper_0 = x + z - 1;
val[1] = -4*pow(helper_0 + y, 2)*(helper_0 + 4*y);}
{val[2] = -12*y*pow(x + y + z - 1, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = -12*z*pow(x + y + z - 1, 2);}
{val[1] = -12*z*pow(x + y + z - 1, 2);}
{double helper_0 = x + y - 1;
val[2] = -4*pow(helper_0 + z, 2)*(helper_0 + 4*z);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = 12*pow(z, 2)*(x + y + z - 1);}
{val[1] = 12*pow(z, 2)*(x + y + z - 1);}
{double helper_0 = x + y - 1;
val[2] = 12*z*(helper_0 + z)*(helper_0 + 2*z);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = -4*pow(z, 3);}
{val[1] = -4*pow(z, 3);}
{val[2] = -4*pow(z, 2)*(3*x + 3*y + 4*z - 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = 12*pow(x, 2)*z;}
{val[1] = 0;}
{val[2] = 4*pow(x, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = 12*x*pow(z, 2);}
{val[1] = 0;}
{val[2] = 12*pow(x, 2)*z;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = 4*pow(z, 3);}
{val[1] = 0;}
{val[2] = 12*x*pow(z, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 12*pow(y, 2)*z;}
{val[2] = 4*pow(y, 3);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 12*y*pow(z, 2);}
{val[2] = 12*pow(y, 2)*z;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 4*pow(z, 3);}
{val[2] = 12*y*pow(z, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 12*y*(helper_0 + x)*(helper_0 + 3*x);}
{double helper_0 = x + z - 1;
val[1] = 12*x*(helper_0 + y)*(helper_0 + 3*y);}
{val[2] = 24*x*y*(x + y + z - 1);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{val[0] = -12*pow(y, 2)*(2*x + y + z - 1);}
{val[1] = -12*x*y*(2*x + 3*y + 2*z - 2);}
{val[2] = -12*x*pow(y, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{val[0] = -12*x*y*(3*x + 2*y + 2*z - 2);}
{val[1] = -12*pow(x, 2)*(x + 2*y + z - 1);}
{val[2] = -12*pow(x, 2)*y;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{double helper_0 = y + z - 1;
val[0] = 12*z*(helper_0 + x)*(helper_0 + 3*x);}
{val[1] = 24*x*z*(x + y + z - 1);}
{double helper_0 = x + y - 1;
val[2] = 12*x*(helper_0 + z)*(helper_0 + 3*z);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{val[0] = -12*pow(z, 2)*(2*x + y + z - 1);}
{val[1] = -12*x*pow(z, 2);}
{val[2] = -12*x*z*(2*x + 2*y + 3*z - 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
{val[0] = -12*x*z*(3*x + 2*y + 2*z - 2);}
{val[1] = -12*pow(x, 2)*z;}
{val[2] = -12*pow(x, 2)*(x + y + 2*z - 1);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
{val[0] = 24*x*y*z;}
{val[1] = 12*pow(x, 2)*z;}
{val[2] = 12*pow(x, 2)*y;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
{val[0] = 12*y*pow(z, 2);}
{val[1] = 12*x*pow(z, 2);}
{val[2] = 24*x*y*z;}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_30(double x, double y, double z, double *val) {
{val[0] = 12*pow(y, 2)*z;}
{val[1] = 24*x*y*z;}
{val[2] = 12*x*pow(y, 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_31(double x, double y, double z, double *val) {
{val[0] = -12*pow(y, 2)*z;}
{val[1] = -12*y*z*(2*x + 3*y + 2*z - 2);}
{val[2] = -12*pow(y, 2)*(x + y + 2*z - 1);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_32(double x, double y, double z, double *val) {
{val[0] = -12*y*pow(z, 2);}
{val[1] = -12*pow(z, 2)*(x + 2*y + z - 1);}
{val[2] = -12*y*z*(2*x + 2*y + 3*z - 2);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_33(double x, double y, double z, double *val) {
{val[0] = 24*y*z*(x + y + z - 1);}
{double helper_0 = x + z - 1;
val[1] = 12*z*(helper_0 + y)*(helper_0 + 3*y);}
{double helper_0 = x + y - 1;
val[2] = 12*y*(helper_0 + z)*(helper_0 + 3*z);}
}

inline POLYFEM_BOTH void b_4_basis_grad_value_3d_single_34(double x, double y, double z, double *val) {
{val[0] = -24*y*z*(2*x + y + z - 1);}
{val[1] = -24*x*z*(x + 2*y + z - 1);}
{val[2] = -24*x*y*(x + y + 2*z - 1);}
}



POLYFEM_BOTH void b_4_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_20(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_21(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_22(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_23(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_24(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_25(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_26(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_27(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_28(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_29(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_30(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_31(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_32(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_33(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i) {
			b_4_basis_grad_value_3d_single_34(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


POLYFEM_BOTH int b_basis_count_3d(const int b) {
switch(b) {
	case 0: return 1;
	case 1: return 4;
	case 2: return 10;
	case 3: return 20;
	case 4: return 35;
	default: assert(false); return 0;
}}

POLYFEM_BOTH void b_basis_value_3d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){

switch(b){
	case 0: b_0_basis_value_3d(local_index, x, y, z, val); break;
	case 1: b_1_basis_value_3d(local_index, x, y, z, val); break;
	case 2: b_2_basis_value_3d(local_index, x, y, z, val); break;
	case 3: b_3_basis_value_3d(local_index, x, y, z, val); break;
	case 4: b_4_basis_value_3d(local_index, x, y, z, val); break;
	default: assert(false); 
}}

POLYFEM_BOTH void b_grad_basis_value_3d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){

switch(b){
	case 0: b_0_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: b_1_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: b_2_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: b_3_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 4: b_4_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default: assert(false); 
}}


}}
