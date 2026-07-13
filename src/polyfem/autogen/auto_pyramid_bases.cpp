#include "auto_pyramid_bases.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH double pyramid_0_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1;
return result;
}



POLYFEM_BOTH void pyramid_0_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_0_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void pyramid_0_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 0;}
}



POLYFEM_BOTH void pyramid_0_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_0_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double pyramid_1_basis_value_3d_single_0(double x, double y, double z) {
double result;
double helper_0 = z - 1;
result = -(helper_0*(helper_0 + x + y) + x*y)/helper_0;
return result;
}

inline POLYFEM_BOTH double pyramid_1_basis_value_3d_single_1(double x, double y, double z) {
double result;
double helper_0 = z - 1;
result = x*(helper_0 + y)/helper_0;
return result;
}

inline POLYFEM_BOTH double pyramid_1_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = -x*y/(z - 1);
return result;
}

inline POLYFEM_BOTH double pyramid_1_basis_value_3d_single_3(double x, double y, double z) {
double result;
double helper_0 = z - 1;
result = y*(helper_0 + x)/helper_0;
return result;
}

inline POLYFEM_BOTH double pyramid_1_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = z;
return result;
}



POLYFEM_BOTH void pyramid_1_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_1_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_1_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_1_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_1_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_1_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void pyramid_1_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
val[0] = -(helper_0 + y)/helper_0;}
{double helper_0 = z - 1;
val[1] = -(helper_0 + x)/helper_0;}
{double helper_0 = 2*z;
double helper_1 = pow(z, 2);
val[2] = (helper_0 - helper_1 + x*y - 1)/(-helper_0 + helper_1 + 1);}
}

inline POLYFEM_BOTH void pyramid_1_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
val[0] = (helper_0 + y)/helper_0;}
{val[1] = x/(z - 1);}
{val[2] = -x*y/pow(z - 1, 2);}
}

inline POLYFEM_BOTH void pyramid_1_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = -y/(z - 1);}
{val[1] = -x/(z - 1);}
{val[2] = x*y/pow(z - 1, 2);}
}

inline POLYFEM_BOTH void pyramid_1_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = y/(z - 1);}
{double helper_0 = z - 1;
val[1] = (helper_0 + x)/helper_0;}
{val[2] = -x*y/pow(z - 1, 2);}
}

inline POLYFEM_BOTH void pyramid_1_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 1;}
}



POLYFEM_BOTH void pyramid_1_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_1_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_1_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_1_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_1_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_1_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_0(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = 4*helper_0;
result = (helper_0*helper_4*(6*x + 6*y + 1) + helper_1*(helper_0 + 2*helper_1 + 2*helper_2 + 2*helper_3 + 10*helper_4 + helper_5*x + helper_5*y + x + y) + 4*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_1(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*y - 1;
result = x*(helper_0*y*(helper_2 + 6*x) + helper_1*(helper_2 + 2*x) + 4*x*pow(y, 2))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_2(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = x*y;
result = helper_2*(helper_0*(2*x + 2*y + 1) + 2*helper_1 + 4*helper_2)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_3(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*x - 1;
result = y*(helper_0*x*(helper_2 + 6*y) + helper_1*(helper_2 + 2*y) + 4*pow(x, 2)*y)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = z*(2*z - 1);
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_5(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
result = -4*x*(helper_0*y*(3*x + 2*y) + helper_1*(helper_0 + x + 3*y) + 2*x*pow(y, 2))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_6(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*x;
result = -4*x*y*(helper_0*(helper_2 + y) + helper_1 + helper_2*y)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_7(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*y;
result = -4*x*y*(helper_0*(helper_2 + x) + helper_1 + helper_2*x)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_8(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
result = -4*y*(helper_0*x*(2*x + 3*y) + helper_1*(helper_0 + 3*x + y) + 2*pow(x, 2)*y)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = -4*z*(x*y + x*z - x + y*z - y + pow(z, 2) - 2*z + 1)/(z - 1);
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_10(double x, double y, double z) {
double result;
double helper_0 = z - 1;
result = 4*x*z*(helper_0 + y)/helper_0;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_11(double x, double y, double z) {
double result;
result = -4*x*y*z/(z - 1);
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_12(double x, double y, double z) {
double result;
double helper_0 = z - 1;
result = 4*y*z*(helper_0 + x)/helper_0;
return result;
}

inline POLYFEM_BOTH double pyramid_2_basis_value_3d_single_13(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = x*y;
result = 16*helper_2*(helper_0*(x + y) + helper_1 + helper_2)/helper_1;
return result;
}



POLYFEM_BOTH void pyramid_2_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_2_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 6*x;
double helper_3 = helper_0*y;
val[0] = (helper_1*(4*x + 10*y + 4*z - 3) + helper_2*helper_3 + helper_3*(helper_2 + 6*y + 1) + 8*x*pow(y, 2))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 6*x;
val[1] = (helper_0*helper_2*y + helper_0*x*(helper_2 + 6*y + 1) + helper_1*(10*x + 4*y + 4*z - 3) + 8*pow(x, 2)*y)/helper_1;}
{double helper_0 = pow(z, 3);
double helper_1 = pow(z, 2);
double helper_2 = 4*x;
double helper_3 = 4*y;
double helper_4 = x*y;
double helper_5 = 12*z;
double helper_6 = pow(y, 2);
double helper_7 = 6*helper_6*x;
double helper_8 = 12*helper_1;
double helper_9 = pow(x, 2);
double helper_10 = 6*helper_9*y;
val[2] = -(-helper_0*helper_2 - helper_0*helper_3 + 15*helper_0 - 21*helper_1 + helper_10*z - helper_10 + helper_2 + helper_3 + helper_4*z - helper_4 - helper_5*x - helper_5*y + 8*helper_6*helper_9 + helper_7*z - helper_7 + helper_8*x + helper_8*y - 4*pow(z, 4) + 13*z - 3)/(helper_0 - 3*helper_1 + 3*z - 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 2*y - 1;
double helper_4 = helper_0*y;
double helper_5 = 2*x;
val[0] = (helper_1*(helper_3 + helper_5) + 4*helper_2*x + helper_4*(helper_3 + 6*x) + helper_5*(helper_1 + 2*helper_2 + 3*helper_4))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*y;
val[1] = x*(helper_0*helper_2 + helper_0*(helper_2 + 6*x - 1) + 2*helper_1 + 8*x*y)/helper_1;}
{double helper_0 = 6*x;
double helper_1 = 2*y;
double helper_2 = x*y;
val[2] = -helper_2*(helper_0*z - helper_0 + helper_1*z - helper_1 + 8*helper_2 - z + 1)/(pow(z, 3) - 3*pow(z, 2) + 3*z - 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*y;
double helper_3 = 2*x;
val[0] = y*(helper_0*(helper_2 + helper_3 + 1) + 2*helper_1 + helper_3*(helper_0 + helper_2) + 4*x*y)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*x;
double helper_3 = 2*y;
val[1] = x*(helper_0*(helper_2 + helper_3 + 1) + 2*helper_1 + helper_3*(helper_0 + helper_2) + 4*x*y)/helper_1;}
{double helper_0 = 2*x;
double helper_1 = 2*y;
double helper_2 = x*y;
val[2] = -helper_2*(helper_0*z - helper_0 + helper_1*z - helper_1 + 8*helper_2 + z - 1)/(pow(z, 3) - 3*pow(z, 2) + 3*z - 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*x;
val[0] = y*(helper_0*helper_2 + helper_0*(helper_2 + 6*y - 1) + 2*helper_1 + 8*x*y)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 2*x - 1;
double helper_4 = helper_0*x;
double helper_5 = 2*y;
val[1] = (helper_1*(helper_3 + helper_5) + 4*helper_2*y + helper_4*(helper_3 + 6*y) + helper_5*(helper_1 + 2*helper_2 + 3*helper_4))/helper_1;}
{double helper_0 = 2*x;
double helper_1 = 6*y;
double helper_2 = x*y;
val[2] = -helper_2*(helper_0*z - helper_0 + helper_1*z - helper_1 + 8*helper_2 - z + 1)/(pow(z, 3) - 3*pow(z, 2) + 3*z - 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 4*z - 1;}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*pow(y, 2);
double helper_3 = 3*y;
val[0] = -4*(helper_0*y*(3*x + 2*y) + helper_1*(helper_0 + helper_3 + x) + helper_2*x + x*(helper_0*helper_3 + helper_1 + helper_2))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 4*x;
double helper_3 = 2*y;
val[1] = -helper_2*(helper_0*helper_3 + helper_0*(helper_3 + 3*x) + 3*helper_1 + helper_2*y)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = 4*x;
val[2] = helper_2*(helper_0*y*(3*x + 2*y) - helper_1 + helper_2*pow(y, 2))/helper_1;}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*x;
val[0] = -4*y*(helper_0*(helper_2 + y) + helper_1 + helper_2*y + helper_2*(helper_0 + y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*x;
val[1] = -4*x*(helper_0*(helper_2 + y) + helper_1 + helper_2*y + y*(helper_0 + helper_2))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = 4*x*y;
val[2] = helper_1*(helper_0*(2*x + y) + helper_1)/pow(helper_0, 3);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*y;
val[0] = -4*y*(helper_0*(helper_2 + x) + helper_1 + helper_2*x + x*(helper_0 + helper_2))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*y;
val[1] = -4*x*(helper_0*(helper_2 + x) + helper_1 + helper_2*x + helper_2*(helper_0 + x))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = 4*x*y;
val[2] = helper_1*(helper_0*(x + 2*y) + helper_1)/pow(helper_0, 3);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 4*y;
double helper_3 = 2*x;
val[0] = -helper_2*(helper_0*helper_3 + helper_0*(helper_3 + 3*y) + 3*helper_1 + helper_2*x)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = 2*pow(x, 2);
double helper_3 = 3*x;
val[1] = -4*(helper_0*x*(2*x + 3*y) + helper_1*(helper_0 + helper_3 + y) + helper_2*y + y*(helper_0*helper_3 + helper_1 + helper_2))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = 4*y;
val[2] = helper_2*(helper_0*x*(2*x + 3*y) - helper_1 + helper_2*pow(x, 2))/helper_1;}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
val[0] = -4*z*(helper_0 + y)/helper_0;}
{double helper_0 = z - 1;
val[1] = -4*z*(helper_0 + x)/helper_0;}
{double helper_0 = 2*z;
double helper_1 = pow(z, 2);
val[2] = -4*(-helper_0*x - helper_0*y + helper_1*x + helper_1*y - 5*helper_1 - x*y + x + y + 2*pow(z, 3) + 4*z - 1)/(-helper_0 + helper_1 + 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
val[0] = 4*z*(helper_0 + y)/helper_0;}
{val[1] = 4*x*z/(z - 1);}
{double helper_0 = 2*z;
double helper_1 = pow(z, 2);
val[2] = -4*x*(helper_0 - helper_1 + y - 1)/(-helper_0 + helper_1 + 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -4*y*z/(z - 1);}
{val[1] = -4*x*z/(z - 1);}
{val[2] = 4*x*y/pow(z - 1, 2);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = 4*y*z/(z - 1);}
{double helper_0 = z - 1;
val[1] = 4*z*(helper_0 + x)/helper_0;}
{double helper_0 = 2*z;
double helper_1 = pow(z, 2);
val[2] = -4*y*(helper_0 - helper_1 + x - 1)/(-helper_0 + helper_1 + 1);}
}

inline POLYFEM_BOTH void pyramid_2_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
val[0] = 16*y*(helper_0*(x + y) + helper_1 + x*y + x*(helper_0 + y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
val[1] = 16*x*(helper_0*(x + y) + helper_1 + x*y + y*(helper_0 + x))/helper_1;}
{double helper_0 = x*y;
val[2] = -16*helper_0*(2*helper_0 + x*z - x + y*z - y)/(pow(z, 3) - 3*pow(z, 2) + 3*z - 1);}
}



POLYFEM_BOTH void pyramid_2_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_2_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_0(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = -helper_0;
result = -(helper_0*helper_4*helper_5*(40.499999999999936*x + 40.499999999999886*y + 12.062499999999872) + helper_1*(54.812499999999595*helper_0*helper_7 + 8.9999999999999449*helper_0*x + 8.9999999999996945*helper_0*y + 4.4999999999999165*helper_1 + 4.5000000000000098*helper_2 + 4.5000000000000053*helper_3 - 13.499999999999972*helper_4*helper_8 + 66.062499999999744*helper_4*y + 4.4999999999999574*helper_4 - 13.500000000000004*helper_5*helper_8 + 66.062499999999773*helper_5*x + 4.4999999999999929*helper_5 + 13.499999999999956*helper_6*x + 13.499999999999801*helper_6*y + 4.4999999999999192*helper_6 + 25.562499999999488*helper_7 + 0.99999999999999223*x + 0.99999999999987943*y + 0.99999999999995881*z - 0.99999999999995881) + 20.249999999999982*helper_2*helper_3 + helper_6*helper_7*(24.749999999999986*helper_4 + 24.749999999999901*helper_5 + 93.062499999999659*helper_7 + 16.562499999999797*x + 16.562499999999961*y + 0.99999999999986067))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_1(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = helper_0*x;
result = (-helper_1*(-1.2795320358804834e-14*helper_0*helper_4 + 9.4979579756681612e-14*helper_0*y + 4.5907722068249983e-14*helper_1 - 4.5000000000000089*helper_2 - 4.6369158512859959e-15*helper_3 - 1.4374999999999076*helper_4*x - 7.8548278992229257e-15*helper_4 - 8.1874999999998632*helper_5*y + 4.5000000000000258*helper_5 + 1.2823075934420435e-14*helper_6 + 3.0473887302484431e-14*helper_7*x + 7.6619266486943136e-14*helper_7*y + 7.2913897142256815e-14*helper_7 + 7.5625000000001279*helper_8 + 3.0625000000001013*helper_9*y + 4.0412118096355496e-14*helper_9 - 0.99999999999998823*x + 4.1473768863653157e-14*y + 2.2318952241917474e-14*z - 2.2318952241917474e-14) + 20.249999999999932*helper_2*helper_3 + helper_4*helper_6*(40.499999999999844*x + 20.249999999999851*y - 12.062499999999959) + helper_7*helper_8*(4.4999999999999201*helper_4 + 24.74999999999995*helper_5 + 28.437499999999769*helper_8 - 16.562500000000046*x - 7.5624999999999183*y + 0.99999999999996503))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_2(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*y;
result = -(-helper_1*(-1.6930901125533511e-14*helper_0*helper_4 + 4.0564773762241386e-14*helper_0*x + 4.518607710224348e-14*helper_1 - 7.562499999999833*helper_10*x + 1.0613732115416391e-13*helper_10 - 8.9928064994637159e-15*helper_2 - 8.4307560932471799e-16*helper_3 - 12.062499999999844*helper_4*x - 1.110223024625149e-14*helper_4 - 12.062499999999808*helper_5*y + 3.2696068075210633e-14*helper_5 + 1.9650947535865129e-14*helper_6 - 7.5624999999998135*helper_7 + 6.7293393080091581e-14*helper_8*y + 6.2755356466936456e-14*helper_8 + 3.3037808599978051e-14*helper_9 + 5.3949900102878085e-15*x + 4.8135107011403926e-14*y + 2.403285903618248e-14*z - 2.403285903618248e-14) + 20.249999999999883*helper_2*helper_3 + helper_4*helper_6*(20.249999999999801*x + 20.249999999999787*y + 12.062499999999956) + helper_9*y*(4.4999999999999094*helper_4 + 4.4999999999999432*helper_5 + 32.312499999999645*helper_7 + 7.5624999999998925*x + 7.562499999999992*y + 0.99999999999996236))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_3(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
result = (-helper_1*(-3.4722225095151544e-14*helper_0*helper_4 + 3.0625000000000551*helper_0*helper_8 + 3.6567970873590721e-14*helper_0*x + 7.1331829332165386e-14*helper_0*y + 3.508304757815453e-14*helper_1 - 8.1046280797635717e-15*helper_2 - 4.5000000000000018*helper_3 - 8.1874999999999183*helper_4*x + 4.4999999999999787*helper_4 - 1.4374999999998757*helper_5*y + 1.7152945730458366e-14*helper_5 + 4.551914400962943e-15*helper_6 + 2.868365267527624e-14*helper_7*x + 3.8288816561759417e-14*helper_7*y + 6.0479399266454759e-14*helper_7 + 7.5625000000000391*helper_8 + 1.0044048925905609e-14*x - 0.99999999999996236*y + 2.162506285152665e-14*z - 2.162506285152665e-14) + 20.24999999999984*helper_2*helper_3 + helper_4*helper_6*(20.249999999999726*x + 40.499999999999702*y - 12.062499999999913) + helper_7*helper_8*(24.749999999999865*helper_4 + 4.4999999999999094*helper_5 + 28.437499999999623*helper_8 - 7.5624999999999849*x - 16.562499999999858*y + 0.9999999999999708))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_4(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = 8.0000000000000284*helper_3;
double helper_5 = 8.0000000000000284*x;
double helper_6 = helper_0*y;
double helper_7 = helper_6*x;
double helper_8 = 6.4392935428259324e-15*helper_3;
double helper_9 = 3.5527136788005136e-15*helper_2;
result = (-helper_1*(-4.5000000000000044*pow(helper_0, 3) + helper_0*helper_8 + helper_0*helper_9 + 5.9952043329758556e-15*helper_0*x + 5.5511151231257961e-15*helper_1*x + 2.3980817331903451e-14*helper_1*y - 9.0000000000000142*helper_1 - 8.0000000000000302*helper_2*x - helper_4*y + 3.8191672047105499e-14*helper_6 - 8.0000000000000089*helper_7 + helper_8 + helper_9 - 7.9999999999999964*x*y - 2.7755575615630215e-16*x + 1.3142265054000328e-14*y - 5.5000000000000036*z + 4.5000000000000036) + helper_2*helper_4 + helper_7*(helper_5*y + helper_5 + 8.0000000000000302*y - 1.3808398868775429e-14))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_5(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*x;
result = (helper_1*(-3.6914915568787465e-15*helper_0*helper_4 - 6.7568173278686765e-13*helper_0*y - 2.220446049250305e-13*helper_1 + 63.562499999999091*helper_10*y + 4.4999999999998437*helper_10 + 13.500000000000027*helper_2 + 1.3910747553858e-14*helper_3 + 90.562499999999375*helper_4*x - 3.0669911055269949e-14*helper_4 + 137.81249999999929*helper_5*y + 4.4999999999998987*helper_5 + 26.999999999999936*helper_6 + 14.062499999998879*helper_7 - 4.651140583788995e-13*helper_8*y - 2.8047009159592926e-13*helper_8 + 13.499999999999876*helper_9 - 3.2761987567297834e-14*x - 2.6108282202841206e-13*y - 1.1421072421136057e-13*z + 1.1421072421136057e-13) + 60.749999999999893*helper_2*helper_3 + helper_4*helper_6*(121.4999999999997*x + 101.24999999999962*y + 9.5624999999997566) + helper_9*y*(40.499999999999716*helper_4 + 74.249999999999915*helper_5 + 212.06249999999906*helper_7 + 14.062499999999527*x + 9.5624999999999272*y - 2.7017971193643449e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_6(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*y;
double helper_10 = helper_0*x;
result = -(-helper_1*(-13.49999999999995*helper_0*helper_4 + 4.9085735476239521e-13*helper_0*y + 1.8635093468333172e-13*helper_1 - 10.687499999999392*helper_10*y + 4.5000000000001386*helper_10 - 13.500000000000027*helper_2 - 1.3910747553857994e-14*helper_3 - 84.937499999999446*helper_4*y + 4.500000000000087*helper_4 - 30.937499999999524*helper_5*x + 1.8984813721090272e-14*helper_5 - 1.2490009027030992e-15*helper_6 + 14.062500000000743*helper_7 + 1.0527516358660403e-13*helper_8*x + 2.5954238758174501e-13*helper_8 + 3.6108616097152263e-13*helper_9 + 3.4441199892043364e-14*x + 1.8806831092454835e-13*y + 9.0333990288015823e-14*z - 9.0333990288015823e-14) + 60.749999999999837*helper_2*helper_3 + helper_4*helper_6*(121.4999999999996*x + 80.999999999999574*y - 9.5625000000000657) + helper_9*x*(74.249999999999872*helper_4 + 20.24999999999973*helper_5 + 152.43749999999915*helper_7 - 14.062500000000297*x - 9.5624999999999396*y - 1.7845100397373306e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_7(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = helper_0*x;
result = -(-helper_1*(-5.0043302834978628e-14*helper_0*helper_4 + 3.0139779561011266e-13*helper_0*y + 1.4055423491754411e-13*helper_1 - 2.6978419498391212e-14*helper_2 - 7.5876804839225337e-15*helper_3 - 17.43749999999973*helper_4*x - 3.4777736246382839e-14*helper_4 - 30.937499999999623*helper_5*y + 8.7437002083134682e-14*helper_5 + 4.9217574460413819e-14*helper_6 + 9.5331728622305005e-14*helper_7*x + 2.1661838989217452e-13*helper_7*y + 2.1568857810905128e-13*helper_7 + 5.0625000000004032*helper_8 - 3.9374999999996732*helper_9*y + 1.2079920397312048e-13*helper_9 + 2.2915697117653442e-14*x + 1.3970682249952916e-13*y + 6.8198918734551803e-14*z - 6.8198918734551803e-14) + 60.749999999999794*helper_2*helper_3 + helper_4*helper_6*(101.24999999999959*x + 60.749999999999545*y - 9.5624999999999289) + helper_7*helper_8*(13.499999999999751*helper_4 + 40.499999999999886*helper_5 + 91.687499999999346*helper_8 - 9.5625000000001705*x - 5.062499999999801*y - 1.0769423547385332e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_8(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*y;
result = (-helper_1*(-5.814793091474222e-14*helper_0*helper_4 + 1.1056780491180905e-13*helper_0*x + 1.3739009929736221e-13*helper_1 - 9.5624999999996199*helper_10*x + 2.9853897132170217e-13*helper_10 - 2.6978419498391178e-14*helper_2 - 3.7938402419612511e-15*helper_3 - 23.062499999999638*helper_4*x - 4.0217829067046075e-14*helper_4 - 29.812499999999545*helper_5*y + 8.9091928279216278e-14*helper_5 + 4.9748399844062476e-14*helper_6 - 5.0624999999995692*helper_7 + 1.9649559757084313e-13*helper_8*y + 2.0508594822388073e-13*helper_8 + 8.9263665903337982e-14*helper_9 + 1.708008734446746e-14*x + 1.4061408287746476e-13*y + 7.0412425889898095e-14*z - 7.0412425889898095e-14) + 60.749999999999716*helper_2*helper_3 + helper_4*helper_6*(80.999999999999488*x + 60.749999999999446*y + 9.5624999999999876) + helper_9*y*(13.499999999999723*helper_4 + 20.249999999999854*helper_5 + 90.562499999999119*helper_7 + 9.5624999999997744*x + 5.0625000000001075*y - 1.0161402969055439e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_9(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*x;
result = (-helper_1*(-6.5752958633424404e-14*helper_0*helper_4 + 2.7555735471196067e-13*helper_0*y + 1.2534417948017896e-13*helper_1 - 9.5624999999996128*helper_10*y + 1.0703937736167197e-13*helper_10 - 2.6978419498391127e-14*helper_2 - 2.5292268279741469e-15*helper_3 - 29.812499999999574*helper_4*x - 4.5519144009631103e-14*helper_4 - 23.062499999999488*helper_5*y + 8.3433260300579757e-14*helper_5 + 4.2965631052993078e-14*helper_6 - 5.0624999999996083*helper_7 + 1.6849022177467317e-13*helper_8*y + 1.899869150889782e-13*helper_8 + 8.8167320666520463e-14*helper_9 + 1.7267437479872911e-14*x + 1.3232383938577132e-13*y + 7.304920557338269e-14*z - 7.304920557338269e-14) + 60.749999999999602*helper_2*helper_3 + helper_4*helper_6*(60.749999999999318*x + 80.999999999999247*y + 9.5625) + helper_9*y*(20.249999999999652*helper_4 + 13.499999999999787*helper_5 + 90.562499999998877*helper_7 + 5.0624999999998028*x + 9.5625000000001137*y - 9.6460166243427047e-14))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_10(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
result = -(-helper_1*(-8.3683060481120581e-14*helper_0*helper_4 - 3.9374999999996909*helper_0*helper_8 + 1.0615119894197174e-13*helper_0*x + 2.5854318685958025e-13*helper_0*y + 1.1890488593735303e-13*helper_1 - 2.6978419498391108e-14*helper_2 - 2.5292268279741421e-15*helper_3 - 30.937499999999616*helper_4*x - 5.6704640982729473e-14*helper_4 - 17.437499999999545*helper_5*y + 7.0110584005077802e-14*helper_5 + 3.0531133177191275e-14*helper_6 + 8.597809963983767e-14*helper_7*x + 1.5050460877574564e-13*helper_7*y + 1.9059753775252193e-13*helper_7 + 5.0625000000002842*helper_8 + 2.2638141361497064e-14*x + 1.2858464293330331e-13*y + 7.3472478101521006e-14*z - 7.3472478101521006e-14) + 60.749999999999559*helper_2*helper_3 + helper_4*helper_6*(60.749999999999247*x + 101.24999999999916*y - 9.5624999999998934) + helper_7*helper_8*(40.499999999999609*helper_4 + 13.499999999999758*helper_5 + 91.687499999998849*helper_8 - 5.062500000000095*x - 9.5624999999997566*y - 9.7234720275450716e-14))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_11(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = helper_0*x;
result = -(-helper_1*(-13.500000000000105*helper_0*helper_4 + 4.5000000000003775*helper_0*y + 1.4072076837123753e-13*helper_1 - 2.6978419498391156e-14*helper_2 - 13.500000000000004*helper_3 - 84.937499999999659*helper_4*x + 4.4999999999999316*helper_4 - 30.937499999999503*helper_5*y + 9.7429009304760763e-14*helper_5 + 5.8085480869607005e-14*helper_6 + 1.0474780764990686e-13*helper_7*x + 2.2258583864953413e-13*helper_7*y + 1.9559354136333539e-13*helper_7 + 14.062500000000487*helper_8 - 10.687499999999563*helper_9*y + 1.2091022627558265e-13*helper_9 + 1.9467066847411464e-14*x + 1.7930709000912734e-13*y + 8.0564027671314294e-14*z - 8.0564027671314294e-14) + 60.749999999999666*helper_2*helper_3 + helper_4*helper_6*(80.999999999999403*x + 121.49999999999928*y - 9.5624999999999254) + helper_7*helper_8*(74.249999999999602*helper_4 + 20.249999999999819*helper_5 + 152.43749999999898*helper_8 - 9.5625000000001865*x - 14.062499999999691*y - 1.6214373593781301e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_12(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*y;
double helper_10 = helper_0*y;
result = (helper_1*(-9.5010804779249371e-14*helper_0*helper_4 - 1.3537435061827762e-13*helper_0*x - 1.8385293287792512e-13*helper_1 + 63.562499999999218*helper_10*x + 4.4999999999994156*helper_10 + 2.6978419498391241e-14*helper_2 + 13.500000000000007*helper_3 + 90.562499999999375*helper_4*y - 1.3323023240197028e-13*helper_4 + 137.81249999999949*helper_5*x + 4.500000000000048*helper_5 + 27.000000000000082*helper_6 + 14.062499999999037*helper_7 - 1.2442998020833992e-13*helper_8*x - 2.170763568898327e-13*helper_8 + 13.499999999999634*helper_9 - 1.277103423014106e-14*x - 2.56607235460392e-13*y - 9.6641444846667456e-14*z + 9.6641444846667456e-14) + 60.749999999999858*helper_2*helper_3 + helper_4*helper_6*(101.2499999999997*x + 121.49999999999952*y + 9.5624999999998188) + helper_9*x*(40.499999999999929*helper_4 + 74.249999999999659*helper_5 + 212.06249999999909*helper_7 + 9.562499999999547*x + 14.062500000000103*y - 2.6097873861985355e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_13(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = 49.499999999999936*x;
double helper_5 = 49.499999999999936*y;
double helper_6 = x*y;
double helper_7 = helper_0*helper_6;
double helper_8 = 13.499999999999989*helper_2;
double helper_9 = 13.499999999999991*helper_3;
result = (helper_1*(13.499999999999979*pow(helper_0, 3) + helper_0*helper_8 + helper_0*helper_9 + 31.499999999999982*helper_0*x + 31.499999999999812*helper_0*y + 26.999999999999989*helper_1*x + 26.999999999999886*helper_1*y + 18.000000000000018*helper_1 + helper_2*helper_5 + helper_3*helper_4 + 80.999999999999702*helper_6 + 76.499999999999801*helper_7 + helper_8 + helper_9 + 4.5000000000000036*x + 4.4999999999999289*y + 4.4999999999999929*z - 4.4999999999999929) + 35.999999999999915*helper_2*helper_3 + helper_7*(helper_4 + helper_5 + 35.999999999999915*helper_6 + 4.4999999999999067))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_14(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = 18.000000000000068*helper_3;
double helper_5 = 18.000000000000068*x;
double helper_6 = x*y;
double helper_7 = helper_0*helper_6;
double helper_8 = 9.2703622556201139e-15*helper_3;
double helper_9 = 8.8817841970009486e-16*helper_2;
result = -(helper_1*(13.500000000000011*pow(helper_0, 3) - helper_0*helper_8 + helper_0*helper_9 + 22.499999999999986*helper_0*x + 22.499999999999918*helper_0*y + 13.499999999999988*helper_1*x + 13.499999999999948*helper_1*y + 22.500000000000043*helper_1 + 18.000000000000082*helper_2*x + helper_4*y + 40.499999999999972*helper_6 + 31.500000000000007*helper_7 - helper_8 + helper_9 + 8.9999999999999982*x + 8.999999999999968*y + 9.0000000000000107*z - 9.0000000000000107) + helper_2*helper_4 + helper_7*(helper_5*y + helper_5 + 18.000000000000082*y + 8.9999999999999591))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_15(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = helper_5*x;
double helper_7 = 13.499999999999984*helper_2;
double helper_8 = 4.0079051188968075e-14*helper_3;
result = (-helper_1*(4.09672296086681e-14*pow(helper_0, 3) - helper_0*helper_7 + helper_0*helper_8 + 4.5000000000000169*helper_0*x + 1.1657341758564043e-14*helper_1*x + 1.085798118083398e-13*helper_1*y + 3.2862601528904533e-14*helper_1 - 49.499999999999801*helper_2*y - 22.49999999999978*helper_3*x - 17.999999999999659*helper_4 + 1.6614487563515387e-13*helper_5 - 22.499999999999712*helper_6 - helper_7 + helper_8 + 4.4999999999999938*x + 5.0699028308897874e-14*y + 1.5959455978986537e-14*z - 1.5959455978986537e-14) + 35.999999999999787*helper_2*helper_3 + helper_6*(35.999999999999787*helper_4 + 49.499999999999801*x + 22.49999999999978*y - 4.5000000000000524))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_16(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = 17.999999999999865*helper_3;
double helper_5 = 17.999999999999865*x;
double helper_6 = x*y;
double helper_7 = helper_0*helper_6;
double helper_8 = 2.0816681711721568e-14*helper_3;
double helper_9 = 1.5099033134902072e-14*helper_2;
result = -(-helper_1*(1.4488410471358173e-14*pow(helper_0, 3) + helper_0*helper_8 + helper_0*helper_9 + 22.500000000000004*helper_0*x + 7.7271522513910365e-14*helper_0*y + 13.500000000000009*helper_1*x + 4.9293902293356622e-14*helper_1*y + 1.0991207943788971e-14*helper_1 - 17.999999999999869*helper_2*x - helper_4*y + 4.5000000000001812*helper_6 - 4.4999999999998419*helper_7 + helper_8 + helper_9 + 8.9999999999999947*x + 2.4202861936828226e-14*y + 4.7462034302724858e-15*z - 4.7462034302724858e-15) + helper_2*helper_4 + helper_7*(helper_5*y + helper_5 + 17.999999999999869*y - 9.0000000000000231))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_17(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*x;
double helper_6 = helper_5*y;
double helper_7 = 2.609024107869104e-14*helper_2;
double helper_8 = 3.4416913763379064e-15*helper_3;
result = (-helper_1*(2.1205259770340335e-14*pow(helper_0, 3) + helper_0*helper_7 + helper_0*helper_8 + 1.029731855339826e-13*helper_0*y + 1.6542323066914757e-14*helper_1*x + 6.2838623193783444e-14*helper_1*y + 8.8817841970011419e-15*helper_1 - 22.499999999999826*helper_2*y - 22.499999999999861*helper_3*x - 26.999999999999737*helper_4 + 1.5210055437364588e-14*helper_5 - 22.49999999999978*helper_6 + helper_7 + helper_8 - 4.1841530240560461e-15*x + 3.7997383017795755e-14*y + 9.9642516460107042e-15*z - 9.9642516460107042e-15) + 35.999999999999837*helper_2*helper_3 + helper_6*(35.999999999999837*helper_4 + 22.499999999999826*x + 22.499999999999861*y + 4.4999999999999583))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_18(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = helper_5*x;
double helper_7 = 1.6375789613220964e-14*helper_2;
double helper_8 = 8.881784197001773e-16*helper_3;
result = -(-helper_1*(1.3565537582138529e-14*pow(helper_0, 3) + helper_0*helper_7 - helper_0*helper_8 + 7.3829831137572516e-15*helper_0*x + 8.4932061383823955e-15*helper_1*x + 3.863576125695517e-14*helper_1*y + 5.1417203827951829e-15*helper_1 - 17.999999999999886*helper_2*y - 17.999999999999904*helper_3*x - 40.499999999999829*helper_4 + 6.6613381477508938e-14*helper_5 - 31.499999999999851*helper_6 + helper_7 - helper_8 - 2.9212743335449447e-15*x + 2.8643754035328887e-14*y + 8.2017725944182935e-15*z - 8.2017725944182935e-15) + 17.99999999999989*helper_2*helper_3 + helper_6*(17.99999999999989*helper_4 + 17.999999999999886*x + 17.999999999999904*y + 8.9999999999999698))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_19(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*x;
double helper_6 = helper_5*y;
double helper_7 = 3.6304292905242436e-14*helper_2;
double helper_8 = 13.499999999999996*helper_3;
result = (-helper_1*(2.2981616609740545e-14*pow(helper_0, 3) + helper_0*helper_7 - helper_0*helper_8 + 4.5000000000001199*helper_0*y + 1.4765966227514481e-14*helper_1*x + 6.9055872131684207e-14*helper_1*y + 3.9968028886504231e-15*helper_1 - 22.499999999999783*helper_2*y - 49.499999999999815*helper_3*x - 17.999999999999673*helper_4 + 5.4400928206631897e-15*helper_5 - 22.49999999999973*helper_6 + helper_7 - helper_8 - 1.1927958620816513e-14*x + 4.5000000000000435*y + 1.1740608485410934e-14*z - 1.1740608485410934e-14) + 35.999999999999787*helper_2*helper_3 + helper_6*(35.999999999999787*helper_4 + 22.499999999999783*x + 49.499999999999815*y - 4.5000000000000524))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_20(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*x;
double helper_6 = helper_5*y;
double helper_7 = 1.8152146452621212e-14*helper_2;
double helper_8 = 9.7699626167013255e-15*helper_3;
result = -(-helper_1*(1.598721155460215e-14*pow(helper_0, 3) + helper_0*helper_7 + helper_0*helper_8 + 22.500000000000068*helper_0*y + 1.0269562977782644e-14*helper_1*x + 13.500000000000041*helper_1*y + 1.509903313490206e-14*helper_1 - 17.999999999999883*helper_2*y - 17.99999999999989*helper_3*x + 4.5000000000001652*helper_4 + 5.6066262743569995e-15*helper_5 - 4.4999999999998579*helper_6 + helper_7 + helper_8 - 5.0584536559483569e-15*x + 9.0000000000000231*y + 8.0213613529167055e-15*z - 8.0213613529167055e-15) + 17.999999999999886*helper_2*helper_3 + helper_6*(17.999999999999886*helper_4 + 17.999999999999883*x + 17.99999999999989*y - 9.0000000000000231))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_21(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = 53.999999999999687*helper_3;
double helper_5 = 53.999999999999687*y;
double helper_6 = x*y;
double helper_7 = helper_0*helper_6;
double helper_8 = 26.999999999999982*helper_2;
double helper_9 = 4.1966430330830791e-14*helper_3;
result = -(helper_1*(-6.8944849829221956e-14*pow(helper_0, 3) + helper_0*helper_8 - helper_0*helper_9 + 26.999999999999975*helper_0*x - 2.7278179715039975e-13*helper_0*y + 26.999999999999982*helper_1*x - 1.6786572132332294e-13*helper_1*y - 5.3956838996782469e-14*helper_1 + 80.999999999999716*helper_2*y + helper_4*x + 80.999999999999432*helper_6 + 80.999999999999517*helper_7 + helper_8 - helper_9 + 3.747002708109871e-15*x - 9.2738317025719678e-14*y - 3.4472424914610972e-14*z + 3.4472424914610972e-14) + helper_2*helper_4 + helper_7*(helper_5*x + helper_5 + 80.999999999999716*x - 1.0922512894140318e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_22(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = 53.999999999999687*helper_3;
double helper_5 = 53.999999999999687*x;
double helper_6 = helper_0*x;
double helper_7 = helper_6*y;
double helper_8 = 3.7581049383561278e-14*helper_3;
double helper_9 = 3.1086244689504244e-14*helper_2;
result = -(-helper_1*(4.7961634663806485e-14*pow(helper_0, 3) + helper_0*helper_8 + helper_0*helper_9 + 2.0516921495072769e-13*helper_0*y + 2.5479618415147194e-14*helper_1*x + 1.3544720900426834e-13*helper_1*y + 3.5971225997854895e-14*helper_1 - 26.999999999999705*helper_2*x - helper_4*y + 2.6589841439772386e-14*helper_6 - 26.999999999999595*helper_7 + helper_8 + helper_9 - 26.999999999999527*x*y - 5.9952043329758122e-15*x + 6.5947247662733857e-14*y + 1.9484414082171358e-14*z - 1.9484414082171358e-14) + helper_2*helper_4 + helper_7*(helper_5*y + helper_5 + 26.999999999999705*y - 6.7446048745977742e-14))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_23(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = helper_5*x;
double helper_7 = 4.6462833580562568e-14*helper_2;
double helper_8 = 5.9952043329757144e-15*helper_3;
result = -(-helper_1*(2.9976021664878968e-14*pow(helper_0, 3) + helper_0*helper_7 + helper_0*helper_8 + 1.9484414082171399e-14*helper_0*x + 2.5479618415147216e-14*helper_1*x + 9.5923269327612844e-14*helper_1*y + 1.1990408665951508e-14*helper_1 - 26.999999999999712*helper_2*y - 53.999999999999766*helper_3*x - 26.999999999999574*helper_4 + 1.5887291482385879e-13*helper_5 - 26.999999999999641*helper_6 + helper_7 + helper_8 - 1.0491607582707712e-14*x + 5.8453242246514113e-14*y + 1.6486811915683448e-14*z - 1.6486811915683448e-14) + 53.99999999999973*helper_2*helper_3 + helper_6*(53.99999999999973*helper_4 + 26.999999999999712*x + 53.999999999999766*y - 6.5947247662733857e-14))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_24(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*helper_4;
double helper_6 = 5.3568260938163475e-14*helper_2;
double helper_7 = 27.0*helper_3;
result = -(helper_1*(-4.7961634663806422e-14*pow(helper_0, 3) - helper_0*helper_6 + helper_0*helper_7 - 1.0602629885170106e-14*helper_0*x + 26.999999999999766*helper_0*y - 2.1926904736346662e-14*helper_1*x + 26.999999999999861*helper_1*y - 1.1990408665951448e-14*helper_1 + 53.999999999999623*helper_2*y + 80.999999999999673*helper_3*x + 80.999999999999403*helper_4 + 80.999999999999503*helper_5 - helper_6 + helper_7 + 1.7236212457305536e-14*x - 8.9928064994637162e-14*y - 2.248201624865925e-14*z + 2.248201624865925e-14) + 53.999999999999616*helper_2*helper_3 + helper_5*(53.999999999999616*helper_4 + 53.999999999999623*x + 80.999999999999673*y - 1.1241008124329651e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_25(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*y;
result = -(-helper_1*(-1.9109713811360436e-13*helper_0*helper_4 + 4.1479319978776499e-13*helper_0*x + 5.2158277696889632e-13*helper_1 - 86.062499999998252*helper_10*x + 1.4366008382893305e-12*helper_10 - 8.0935258495173697e-14*helper_2 - 2.2763041451767622e-14*helper_3 - 207.56249999999881*helper_4*x - 1.1016187961843079e-13*helper_4 - 207.56249999999847*helper_5*y + 3.3048563885529236e-13*helper_5 + 2.1582735598712967e-13*helper_6 - 5.0624999999978639*helper_7 + 9.3862417838152614e-13*helper_8*y + 7.1267991508250068e-13*helper_8 + 3.491738148619903e-13*helper_9 + 6.772707394908596e-14*x + 6.3005850536867691e-13*y + 2.7624777465540136e-13*z - 2.7624777465540136e-13) + 182.24999999999952*helper_2*helper_3 + helper_4*helper_6*(303.74999999999903*x + 303.74999999999858*y + 5.0624999999998135) + helper_9*y*(121.49999999999909*helper_4 + 121.49999999999976*helper_5 + 511.31249999999761*helper_7 + 5.0624999999990248*x + 5.0625000000004343*y - 5.6823296068486381e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_26(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*y;
double helper_10 = helper_0*y;
result = (-helper_1*(-2.5404678360985013e-13*helper_0*helper_4 + 3.6111738599408966e-13*helper_0*x + 4.3615111522398977e-13*helper_1 - 35.437499999998778*helper_10*x + 1.0656475701864488e-12*helper_10 - 8.0935258495173495e-14*helper_2 - 1.1381520725883746e-14*helper_3 - 156.93749999999892*helper_4*x - 1.7311152511467673e-13*helper_4 - 96.187499999998565*helper_5*y + 2.7849597628026675e-13*helper_5 + 1.604653909748054e-13*helper_6 + 5.0625000000013829*helper_7 + 2.9999440431804727e-13*helper_8*x + 6.3624105983705734e-13*helper_8 + 6.531025720235507e-13*helper_9 + 6.3792721105570397e-14*x + 5.064074160010495e-13*y + 2.4983140556322614e-13*z - 2.4983140556322614e-13) + 182.24999999999906*helper_2*helper_3 + helper_4*helper_6*(242.99999999999832*x + 303.74999999999801*y - 5.0624999999998597) + helper_9*x*(121.49999999999895*helper_4 + 60.74999999999951*helper_5 + 399.9374999999971*helper_7 - 5.0625000000005933*x - 5.0624999999993054*y - 4.1310704856911351e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_27(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*y;
result = (-helper_1*(-1.489641743290796e-13*helper_0*helper_4 + 3.8126099499713335e-13*helper_0*x + 4.8550052866857883e-13*helper_1 - 35.437499999998636*helper_10*x + 1.1813883205036234e-12*helper_10 - 8.0935258495173685e-14*helper_2 - 2.2763041451767615e-14*helper_3 - 96.187499999998963*helper_4*x - 8.8262730457699516e-14*helper_4 - 156.93749999999869*helper_5*y + 2.9065985729381698e-13*helper_5 + 1.7600157442565429e-13*helper_6 + 5.0625000000016547*helper_7 + 8.1250284278410876e-13*helper_8*y + 6.9375061251264187e-13*helper_8 + 3.1011131162994466e-13*helper_9 + 6.8226674310167306e-14*x + 5.1939615802742675e-13*y + 2.4308680068862882e-13*z - 2.4308680068862882e-13) + 182.24999999999949*helper_2*helper_3 + helper_4*helper_6*(303.74999999999898*x + 242.99999999999869*y - 5.0625000000000062) + helper_9*y*(60.749999999999176*helper_4 + 121.49999999999973*helper_5 + 399.9374999999979*helper_7 - 5.0625000000007505*x - 5.0624999999995346*y - 4.271470330219324e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_28(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = x*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = helper_8*x;
double helper_10 = helper_0*y;
result = -(-helper_1*(-2.0458634786279948e-13*helper_0*helper_4 + 3.4397484860448695e-13*helper_0*x + 4.271583087245261e-13*helper_1 - 25.312499999998877*helper_10*x + 9.5098928731828619e-13*helper_10 - 8.0935258495173508e-14*helper_2 - 1.1381520725883749e-14*helper_3 - 86.062499999998906*helper_4*x - 1.4388490399141953e-13*helper_4 - 86.062499999998622*helper_5*y + 2.6641189254661236e-13*helper_5 + 1.4838130724115101e-13*helper_6 - 5.0624999999987423*helper_7 + 6.0813853952623206e-13*helper_8*y + 6.3174465658732566e-13*helper_8 + 2.8004161489736208e-13*helper_9 + 5.9296317855838589e-14*x + 4.5132647619183402e-13*y + 2.3240784297051514e-13*z - 2.3240784297051514e-13) + 182.24999999999909*helper_2*helper_3 + helper_4*helper_6*(242.99999999999838*x + 242.99999999999815*y + 5.0625000000000924) + helper_9*y*(60.749999999999062*helper_4 + 60.749999999999538*helper_5 + 329.06249999999727*helper_7 + 5.0624999999994031*x + 5.0625000000005231*y - 3.3554409251123861e-13))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_3_basis_value_3d_single_29(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = x*y;
double helper_5 = helper_0*x;
double helper_6 = helper_5*y;
double helper_7 = 1.0658141036401431e-13*helper_2;
double helper_8 = 4.2632564145605632e-14*helper_3;
result = (-helper_1*(1.1368683772161527e-13*pow(helper_0, 3) + helper_0*helper_7 + helper_0*helper_8 + 5.2580162446247081e-13*helper_0*y + 6.0396132539608125e-14*helper_1*x + 3.1263880373444201e-13*helper_1*y + 7.1054273576009514e-14*helper_1 - 127.99999999999915*helper_2*y - 127.99999999999925*helper_3*x - 127.99999999999872*helper_4 + 4.973799150320671e-14*helper_5 - 127.99999999999891*helper_6 + helper_7 + helper_8 - 2.4424906541753368e-14*x + 1.8540724511240001e-13*y + 5.6843418860807636e-14*z - 5.6843418860807636e-14) + 127.99999999999916*helper_2*helper_3 + helper_6*(127.99999999999916*helper_4 + 127.99999999999915*x + 127.99999999999925*y - 2.045030811359525e-13))/helper_1;
return result;
}



POLYFEM_BOTH void pyramid_3_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_14(x[i], y[i], z[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_15(x[i], y[i], z[i]);
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_16(x[i], y[i], z[i]);
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_17(x[i], y[i], z[i]);
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_18(x[i], y[i], z[i]);
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_19(x[i], y[i], z[i]);
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_20(x[i], y[i], z[i]);
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_21(x[i], y[i], z[i]);
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_22(x[i], y[i], z[i]);
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_23(x[i], y[i], z[i]);
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_24(x[i], y[i], z[i]);
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_25(x[i], y[i], z[i]);
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_26(x[i], y[i], z[i]);
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_27(x[i], y[i], z[i]);
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_28(x[i], y[i], z[i]);
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_3_basis_value_3d_single_29(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 93.062499999999659*y;
double helper_7 = x*y;
val[0] = -(helper_1*(26.999999999999943*helper_0*x + 54.812499999999595*helper_0*y + 13.500000000000028*helper_2 + 66.062499999999773*helper_3 + 13.499999999999956*helper_5 + 132.12499999999949*helper_7 + 8.9999999999999147*x + 25.562499999999488*y + 8.9999999999999449*z - 7.9999999999999529) + 40.499999999999936*helper_2*helper_4 + 60.749999999999943*helper_2*pow(y, 3) + 2*helper_4*x*(40.499999999999936*x + 40.499999999999886*y + 12.062499999999872) + helper_5*helper_7*(helper_6 + 49.499999999999972*x + 16.562499999999797) + helper_5*y*(24.749999999999986*helper_2 + 24.749999999999901*helper_3 + helper_6*x + 16.562499999999797*x + 16.562499999999961*y + 0.99999999999986067))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 93.062499999999659*x;
double helper_7 = x*y;
val[1] = -(helper_1*(54.812499999999595*helper_0*x + 27.000000000000007*helper_0*y + 13.500000000000016*helper_2 + 66.062499999999744*helper_3 + 13.499999999999801*helper_5 + 132.12499999999955*helper_7 + 25.562499999999488*x + 8.9999999999999858*y + 8.9999999999996945*z - 7.9999999999998153) + 40.499999999999886*helper_2*helper_4 + 60.749999999999943*helper_2*pow(x, 3) + 2*helper_4*y*(40.499999999999936*x + 40.499999999999886*y + 12.062499999999872) + helper_5*helper_7*(helper_6 + 49.499999999999801*y + 16.562499999999961) + helper_5*x*(24.749999999999901*helper_2 + 24.749999999999986*helper_3 + helper_6*y + 16.562499999999797*x + 16.562499999999961*y + 0.99999999999986067))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*z;
double helper_4 = pow(x, 2);
double helper_5 = 13.499999999999972*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(z, 5);
double helper_8 = helper_6*x;
double helper_9 = pow(y, 3);
double helper_10 = helper_9*x;
double helper_11 = 24.749999999999901*helper_10;
double helper_12 = helper_1*x;
double helper_13 = helper_2*x;
double helper_14 = helper_4*y;
double helper_15 = helper_4*z;
double helper_16 = pow(x, 3);
double helper_17 = helper_16*y;
double helper_18 = 24.749999999999986*helper_17;
double helper_19 = helper_0*y;
double helper_20 = helper_6*z;
double helper_21 = helper_4*helper_6;
double helper_22 = 80.999999999999773*helper_4;
double helper_23 = 80.999999999999858*helper_16;
val[2] = (helper_0*helper_11 + 16.56249999999973*helper_0*helper_14 + helper_0*helper_18 + 93.062499999999659*helper_0*helper_21 - helper_0*helper_22 - 81.0*helper_0*helper_6 + 16.562499999999886*helper_0*helper_8 + 215.99999999999946*helper_0*x - 118.4999999999976*helper_0 + 53.999999999999886*helper_1*helper_4 + 53.999999999999972*helper_1*helper_6 - 233.99999999999716*helper_1*y + 183.99999999999639*helper_1 - 49.499999999999801*helper_10*z + helper_11 + 219.24999999999829*helper_12*y - 233.99999999999932*helper_12 - 54.812499999999602*helper_13*y + 125.9999999999996*helper_13 - 33.124999999999574*helper_14*z + 16.562499999999787*helper_14 - 161.9999999999996*helper_15*helper_6 + 80.999999999999773*helper_15*helper_9 + 53.999999999999901*helper_15 + 60.749999999999943*helper_16*helper_9 - 49.499999999999972*helper_17*z + helper_18 - 327.87499999999767*helper_19*x + 215.99999999999778*helper_19 - helper_2*helper_5 - 13.5*helper_2*helper_6 + 125.99999999999832*helper_2*y - 158.49999999999699*helper_2 + helper_20*helper_23 + 54.000000000000043*helper_20 + 68.937499999999929*helper_21 - helper_22*helper_9 - helper_23*helper_6 + 217.24999999999866*helper_3*y - 98.999999999999773*helper_3 - helper_5 - 13.500000000000011*helper_6 - 26.999999999999908*helper_7*x - 26.999999999999602*helper_7*y + 71.99999999999865*helper_7 - 33.124999999999886*helper_8*z + 16.562499999999972*helper_8 - 53.812499999999737*x*y + 17.999999999999964*x - 98.999999999999204*y*z + 17.999999999999908*y - 13.49999999999975*pow(z, 6) + 39.999999999999147*z - 5.4999999999998703)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 28.437499999999769*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (40.499999999999844*helper_0*helper_2*helper_3 - helper_1*(3.0625000000001013*helper_0*y - 13.500000000000027*helper_2 - 1.4374999999999076*helper_3 + 3.0473887302484431e-14*helper_4 - 16.374999999999726*helper_6 + 2.564615186884087e-14*helper_7 + 9.0000000000000515*x + 7.5625000000001279*y + 4.0412118096355496e-14*z - 1.0000000000000286) + 60.749999999999801*helper_2*pow(y, 3) + 2*helper_3*helper_7*(40.499999999999844*x + 20.249999999999851*y - 12.062499999999959) + helper_4*helper_6*(helper_5 + 49.499999999999901*x - 16.562500000000046) + helper_4*y*(24.74999999999995*helper_2 + 4.4999999999999201*helper_3 + helper_5*x - 16.562500000000046*x - 7.5624999999999183*y + 0.99999999999996503))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 28.437499999999769*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (20.249999999999851*helper_0*helper_2*helper_3 + helper_1*(-3.0625000000001013*helper_0*x + 1.3910747553857988e-14*helper_2 + 8.1874999999998632*helper_3 - 7.6619266486943136e-14*helper_4 + 2.8749999999998153*helper_6 + 2.5590640717609669e-14*helper_7 - 7.5625000000001279*x + 1.5709655798445851e-14*y - 9.4979579756681612e-14*z + 5.3505810893028455e-14) + 60.749999999999801*helper_2*pow(x, 3) + 2*helper_3*helper_7*(40.499999999999844*x + 20.249999999999851*y - 12.062499999999959) + helper_4*helper_6*(helper_5 + 8.9999999999998401*y - 7.5624999999999183) + helper_4*x*(4.4999999999999201*helper_2 + 24.74999999999995*helper_3 + helper_5*y - 16.562500000000046*x - 7.5624999999999183*y + 0.99999999999996503))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(y, 2);
double helper_8 = 1.2795320358804834e-14*helper_7;
double helper_9 = pow(z, 5);
double helper_10 = 7.5624999999999183*helper_7;
double helper_11 = pow(y, 3);
double helper_12 = 4.4999999999999201*helper_11*x;
double helper_13 = helper_0*x;
double helper_14 = helper_1*x;
double helper_15 = helper_6*z;
double helper_16 = pow(x, 3);
double helper_17 = helper_16*y;
double helper_18 = 24.749999999999957*helper_17;
double helper_19 = 5.1181281435219319e-14*helper_7;
double helper_20 = helper_6*helper_7;
double helper_21 = 40.499999999999702*helper_11;
double helper_22 = helper_0*helper_6;
double helper_23 = 80.999999999999687*helper_16*helper_7;
val[2] = -(helper_0*helper_12 + helper_0*helper_18 + 19.375000000000576*helper_0*helper_3 - 7.6771922152828969e-14*helper_0*helper_7 - 9.6250785119877421e-13*helper_0*y + 7.4148326367761767e-13*helper_0 + helper_1*helper_19 - 5.1514348342607491e-14*helper_1*helper_6 + 1.1524670107121361e-12*helper_1*y - 1.3854611902175322e-12*helper_1 - helper_10*helper_13 - helper_10*x + 60.749999999999801*helper_11*helper_16 - 8.9999999999998401*helper_11*helper_4 + helper_12 - 3.6770586575585023e-13*helper_13 - 12.250000000000407*helper_14*y + 4.471978343190179e-13*helper_14 + helper_15*helper_21 - 4.973799150320719e-14*helper_15 - 49.499999999999915*helper_17*z + helper_18 + helper_19*z + 3.0625000000001013*helper_2*helper_3 + 1.2823075934420432e-14*helper_2*helper_6 - helper_2*helper_8 - 2.6432675492848935e-13*helper_2*x - 6.7121308511275058e-13*helper_2*y + 1.3590274738905982e-12*helper_2 - 80.99999999999946*helper_20*z + 52.56249999999968*helper_20 - helper_21*helper_6 + 28.437499999999766*helper_22*helper_7 - 16.562500000000057*helper_22*y + 7.8159700933610831e-14*helper_22 + helper_23*z - helper_23 + 4.0625000000000657*helper_3 + 15.124999999999837*helper_4*helper_7 - 14.25000000000033*helper_4*y + 1.4210854715202004e-13*helper_4 + 33.125000000000099*helper_5*helper_6 + 3.8627434584270209e-13*helper_5 - 16.56250000000005*helper_6*y + 1.2434497875801753e-14*helper_6 - helper_8 + 6.0947774604969128e-14*helper_9*x + 1.5323853297388597e-13*helper_9*y - 6.8051120294398586e-13*helper_9 - 2.042810365310288e-14*x - 5.8258953217205052e-14*y + 1.3772316620475016e-13*pow(z, 6) - 1.86475834773602e-13*z + 1.4214324162153806e-14)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 32.312499999999645*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = -(20.249999999999801*helper_0*helper_2*helper_3 + helper_1*(7.562499999999833*helper_0*y + 2.6978419498391146e-14*helper_2 + 12.062499999999844*helper_3 - 3.3037808599978051e-14*helper_4 + 24.124999999999616*helper_6 - 3.9301895071730258e-14*helper_7 - 6.5392136150421266e-14*x + 7.5624999999998135*y - 4.0564773762241386e-14*z + 3.5169783751953579e-14) + 60.749999999999645*helper_2*pow(y, 3) + 2*helper_3*helper_7*(20.249999999999801*x + 20.249999999999787*y + 12.062499999999956) + helper_4*helper_6*(helper_5 + 8.9999999999998863*x + 7.5624999999998925) + helper_4*y*(4.4999999999999432*helper_2 + 4.4999999999999094*helper_3 + helper_5*x + 7.5624999999998925*x + 7.562499999999992*y + 0.99999999999996236))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 32.312499999999645*x;
double helper_7 = x*y;
val[1] = -(helper_1*(7.562499999999833*helper_0*x + 3.3861802251067022e-14*helper_0*y + 2.529226827974154e-15*helper_2 + 12.062499999999808*helper_3 - 6.7293393080091581e-14*helper_5 + 24.124999999999687*helper_7 + 7.5624999999998135*x + 2.2204460492502979e-14*y - 1.0613732115416391e-13*z + 5.8002214142759979e-14) + 20.249999999999787*helper_2*helper_4 + 60.749999999999645*helper_2*pow(x, 3) + 2*helper_4*y*(20.249999999999801*x + 20.249999999999787*y + 12.062499999999956) + helper_5*helper_7*(helper_6 + 8.9999999999998188*y + 7.562499999999992) + helper_5*x*(4.4999999999999094*helper_2 + 4.4999999999999432*helper_3 + helper_6*y + 7.5624999999998925*x + 7.562499999999992*y + 0.99999999999996236))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = 1.6930901125533511e-14*helper_5;
double helper_7 = pow(z, 5);
double helper_8 = helper_5*x;
double helper_9 = 7.5624999999999893*helper_8;
double helper_10 = pow(y, 3);
double helper_11 = helper_10*x;
double helper_12 = 4.4999999999999094*helper_11;
double helper_13 = helper_4*y;
double helper_14 = 7.8603790143460263e-14*helper_4;
double helper_15 = pow(x, 3);
double helper_16 = helper_15*y;
double helper_17 = 4.4999999999999432*helper_16;
double helper_18 = helper_0*y;
double helper_19 = helper_1*y;
double helper_20 = helper_2*y;
double helper_21 = 6.7723604502133867e-14*helper_5;
double helper_22 = helper_4*helper_5;
double helper_23 = 40.499999999999574*helper_10*helper_4;
double helper_24 = 40.499999999999602*helper_15*helper_5;
val[2] = (helper_0*helper_12 + 7.5624999999999041*helper_0*helper_13 + helper_0*helper_17 + 32.312499999999645*helper_0*helper_22 + 1.1790568521519039e-13*helper_0*helper_4 - 1.0158540675320223e-13*helper_0*helper_5 + helper_0*helper_9 - 4.1736752942611038e-13*helper_0*x + 9.2246349447932297e-13*helper_0 - helper_1*helper_14 + helper_1*helper_21 + 4.9849707695059526e-13*helper_1*x - 1.55218893294061e-12*helper_1 + 60.749999999999645*helper_10*helper_15 - 8.9999999999998188*helper_11*z + helper_12 - 15.12499999999978*helper_13*z + 7.5624999999998899*helper_13 - helper_14*z - 8.9999999999998863*helper_16*z + helper_17 - 44.374999999999048*helper_18*x - 7.0904393467684895e-13*helper_18 + 30.249999999999346*helper_19*x + 9.2131857698517505e-13*helper_19 + 1.9650947535865132e-14*helper_2*helper_4 - helper_2*helper_6 - 2.8981331223753888e-13*helper_2*x + 1.4298527639677748e-12*helper_2 - 7.562499999999833*helper_20*x - 5.6679660964675168e-13*helper_20 + helper_21*z - 40.499999999999382*helper_22*z + 8.1874999999997371*helper_22 + helper_23*z - helper_23 + helper_24*z - helper_24 + 28.24999999999941*helper_3*x + 2.4838464618426037e-13*helper_3 + 1.9650947535865129e-14*helper_4 - helper_6 + 6.6075617199955358e-14*helper_7*x + 1.3458678616018289e-13*helper_7*y - 6.8783867490651109e-13*helper_7 - 15.124999999999979*helper_8*z + helper_9 - 6.5624999999998712*x*y + 1.6811899095081459e-13*x*z - 2.5510843437714458e-14*x - 2.8449465006019611e-14*y + 1.355582313067305e-13*pow(z, 6) - 2.8192725931574832e-13*z + 3.4080377409040567e-14)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 28.437499999999623*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (20.249999999999726*helper_0*helper_2*helper_3 - helper_1*(3.0625000000000551*helper_0*y - 2.4313884239290714e-14*helper_2 - 8.1874999999999183*helper_3 + 2.868365267527624e-14*helper_4 - 2.8749999999997513*helper_6 + 9.103828801925886e-15*helper_7 + 3.4305891460916731e-14*x + 7.5625000000000391*y + 3.6567970873590721e-14*z - 2.6523921947685112e-14) + 60.749999999999517*helper_2*pow(y, 3) + 2*helper_3*helper_7*(20.249999999999726*x + 40.499999999999702*y - 12.062499999999913) + helper_4*helper_6*(helper_5 + 8.9999999999998188*x - 7.5624999999999849) + helper_4*y*(4.4999999999999094*helper_2 + 24.749999999999865*helper_3 + helper_5*x - 7.5624999999999849*x - 16.562499999999858*y + 0.9999999999999708))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 28.437499999999623*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (40.499999999999702*helper_0*helper_2*helper_3 + helper_1*(-3.0625000000000551*helper_0*x + 13.500000000000005*helper_2 + 1.4374999999998757*helper_3 - 3.8288816561759417e-14*helper_4 + 16.374999999999837*helper_6 + 6.9444450190303087e-14*helper_7 - 7.5625000000000391*x - 8.9999999999999574*y - 7.1331829332165386e-14*z + 1.0000000000000338) + 60.749999999999517*helper_2*pow(x, 3) + 2*helper_3*helper_7*(20.249999999999726*x + 40.499999999999702*y - 12.062499999999913) + helper_4*helper_6*(helper_5 + 49.49999999999973*y - 16.562499999999858) + helper_4*x*(24.749999999999865*helper_2 + 4.4999999999999094*helper_3 + helper_5*y - 7.5624999999999849*x - 16.562499999999858*y + 0.9999999999999708))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(z, 5);
double helper_8 = helper_6*x;
double helper_9 = pow(y, 3);
double helper_10 = helper_9*x;
double helper_11 = 24.749999999999872*helper_10;
double helper_12 = helper_1*x;
double helper_13 = helper_5*z;
double helper_14 = pow(x, 3);
double helper_15 = helper_14*y;
double helper_16 = 4.4999999999999094*helper_15;
double helper_17 = helper_6*z;
double helper_18 = 80.999999999999403*helper_9;
double helper_19 = helper_0*helper_5;
double helper_20 = 40.499999999999446*helper_14;
val[2] = -(helper_0*helper_11 + helper_0*helper_16 + 19.375000000000302*helper_0*helper_3 - 2.0605739337042938e-13*helper_0*helper_6 - 16.562499999999844*helper_0*helper_8 - 3.5426522826397996e-13*helper_0*x - 3.3750779948604814e-13*helper_0*y + 4.9889953279701864e-13*helper_0 - 1.8207657603851775e-14*helper_1*helper_5 + 1.3855583347322004e-13*helper_1*helper_6 + 4.809486142676169e-13*helper_1*y - 9.8189512076628111e-13*helper_1 - 49.499999999999744*helper_10*z + helper_11 - 12.250000000000218*helper_12*y + 4.2740117001116169e-13*helper_12 + helper_13*helper_18 - 1.8207657603851772e-14*helper_13 + 60.749999999999517*helper_14*helper_9 - 8.9999999999998188*helper_15*z + helper_16 + helper_17*helper_20 - 80.999999999999062*helper_17*helper_5 + 33.124999999999716*helper_17*x + 1.3500311979441883e-13*helper_17 - helper_18*helper_5 + 28.437499999999616*helper_19*helper_6 - 7.5624999999999822*helper_19*y + 2.7311486405777582e-14*helper_19 + 3.0625000000000551*helper_2*helper_3 + 4.551914400962943e-15*helper_2*helper_5 - 3.4722225095151184e-14*helper_2*helper_6 - 2.5026855587917179e-13*helper_2*x - 3.1155633628542845e-13*helper_2*y + 9.9556821120393414e-13*helper_2 - helper_20*helper_6 - 14.250000000000167*helper_3*z + 4.0625000000000266*helper_3 + 15.124999999999968*helper_4*helper_5 + 9.7699626167013548e-14*helper_4 + 52.562499999999439*helper_5*helper_6 - 7.5624999999999858*helper_5*y + 4.5519144009629399e-15*helper_5 - 3.5527136788004504e-14*helper_6 + 5.7367305350551685e-14*helper_7*x + 7.6577633123518216e-14*helper_7*y - 5.1053605787387177e-13*helper_7 - 16.562499999999858*helper_8 + 1.4056464325839919e-13*x*z - 2.0799334476961759e-14*x - 5.3290705182007514e-15*y + 1.0524914273446335e-13*pow(z, 6) - 1.1320111514834019e-13*z + 5.9154070530807229e-15)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 16.000000000000057*x;
double helper_4 = 8.0000000000000284*x;
double helper_5 = helper_0*y;
double helper_6 = 1.2878587085651865e-14*x;
val[0] = (helper_1*(-helper_0*helper_6 - 5.5511151231257961e-15*helper_1 + 8.0000000000000302*helper_2 + helper_3*y + 8.0000000000000089*helper_5 - helper_6 + 7.9999999999999964*y - 5.9952043329758556e-15*z + 6.2727600891321581e-15) + helper_2*helper_3 + helper_4*helper_5*(y + 1) + helper_5*(helper_4*y + helper_4 + 8.0000000000000302*y - 1.3808398868775429e-14))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 8.0000000000000284*x;
double helper_4 = helper_0*x;
double helper_5 = 7.1054273576010271e-15*y;
val[1] = (helper_1*(-helper_0*helper_5 - 2.3980817331903451e-14*helper_1 + 8.0000000000000284*helper_2 + 8.0000000000000089*helper_4 - helper_5 + 16.00000000000006*x*y + 7.9999999999999964*x - 3.8191672047105499e-14*z + 2.504940699310517e-14) + 16.000000000000057*helper_2*y + helper_4*y*(helper_3 + 8.0000000000000302) + helper_4*(helper_3*y + helper_3 + 8.0000000000000302*y - 1.3808398868775429e-14))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = 3.5527136788005136e-15*helper_5;
double helper_7 = pow(z, 4);
double helper_8 = 8.0000000000000284*helper_4;
double helper_9 = helper_8*y;
double helper_10 = helper_0*y;
double helper_11 = 1.0658141036401541e-14*helper_5;
double helper_12 = helper_5*helper_8;
val[2] = -(-helper_0*helper_11 - 1.9317880628477796e-14*helper_0*helper_4 + 4.862776847858245e-14*helper_0*x + 43.499999999999979*helper_0 - 8.0*helper_1*helper_2 + 6.4392935428259316e-15*helper_1*helper_4 + helper_1*helper_6 - 3.8413716652030069e-14*helper_1*x - 1.5365486660812225e-13*helper_1*y - 68.500000000000028*helper_1 + 24.000000000000028*helper_10*x + 1.7319479184152503e-13*helper_10 + helper_11*z + helper_12*z + helper_12 - 24.000000000000039*helper_2*z + 8.0000000000000231*helper_2 + 8.000000000000032*helper_3*helper_5 - 2.642330798607882e-14*helper_3 + 1.9317880628477793e-14*helper_4*z - 6.4392935428259324e-15*helper_4 - 8.0000000000000302*helper_5*x - helper_6 + 1.1102230246251592e-14*helper_7*x + 4.7961634663806548e-14*helper_7*y + 49.500000000000043*helper_7 + helper_9*z - helper_9 + 5.1070259132757367e-15*x - 7.7271522513911236e-14*y*z + 9.7699626167014028e-15*y - 13.500000000000012*pow(z, 5) - 11.999999999999968*z + 1.0)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 212.06249999999906*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (121.4999999999997*helper_0*helper_2*helper_3 + helper_1*(63.562499999999091*helper_0*y + 40.500000000000078*helper_2 + 90.562499999999375*helper_3 + 13.499999999999876*helper_4 + 275.62499999999858*helper_6 + 53.999999999999872*helper_7 + 8.9999999999997975*x + 14.062499999998879*y + 4.4999999999998437*z - 4.4999999999998765) + 182.24999999999969*helper_2*pow(y, 3) + 2*helper_3*helper_7*(121.4999999999997*x + 101.24999999999962*y + 9.5624999999997566) + helper_4*helper_6*(helper_5 + 148.49999999999983*x + 14.062499999999527) + helper_4*y*(74.249999999999915*helper_2 + 40.499999999999716*helper_3 + helper_5*x + 14.062499999999527*x + 9.5624999999999272*y - 2.7017971193643449e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 212.06249999999906*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (101.24999999999962*helper_0*helper_2*helper_3 + helper_1*(63.562499999999091*helper_0*x + 4.1732242661573998e-14*helper_2 + 137.81249999999929*helper_3 - 4.651140583788995e-13*helper_4 + 181.12499999999875*helper_6 - 7.3829831137574929e-15*helper_7 + 14.062499999998879*x - 6.1339822110539899e-14*y - 6.7568173278686765e-13*z + 4.1459891075845558e-13) + 182.24999999999969*helper_2*pow(x, 3) + 2*helper_3*helper_7*(121.4999999999997*x + 101.24999999999962*y + 9.5624999999997566) + helper_4*helper_6*(helper_5 + 80.999999999999432*y + 9.5624999999999272) + helper_4*x*(40.499999999999716*helper_2 + 74.249999999999915*helper_3 + helper_5*y + 14.062499999999527*x + 9.5624999999999272*y - 2.7017971193643449e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = 3.6914915568787465e-15*helper_5;
double helper_7 = pow(z, 5);
double helper_8 = helper_5*x;
double helper_9 = pow(y, 3);
double helper_10 = helper_9*x;
double helper_11 = 40.499999999999716*helper_10;
double helper_12 = helper_0*x;
double helper_13 = helper_2*x;
double helper_14 = helper_4*y;
double helper_15 = helper_4*z;
double helper_16 = pow(x, 3);
double helper_17 = 74.249999999999915*helper_16*y;
double helper_18 = helper_1*y;
double helper_19 = 1.4765966227514986e-14*helper_5;
double helper_20 = helper_4*helper_5;
double helper_21 = 202.49999999999926*helper_9;
double helper_22 = 242.99999999999937*helper_16*helper_5;
double helper_23 = helper_0*helper_5;
val[2] = -(helper_0*helper_11 + 14.062499999999488*helper_0*helper_14 + helper_0*helper_17 - 161.99999999999966*helper_0*helper_4 - 5.2481907708567845e-12*helper_0*y + 5.0678697349759506e-12*helper_0 - helper_1*helper_19 + 107.99999999999983*helper_1*helper_4 - 251.99999999999818*helper_1*x - 8.1701173604286869e-12*helper_1 - 80.999999999999432*helper_10*z + helper_11 - 381.37499999999488*helper_12*y + 242.99999999999841*helper_12 - 63.562499999999091*helper_13*y + 130.49999999999886*helper_13 - 28.124999999998977*helper_14*z + 14.062499999999545*helper_14 + helper_15*helper_21 + 107.99999999999966*helper_15 - 148.49999999999983*helper_16*helper_3 + 182.24999999999969*helper_16*helper_9 + helper_17 + 254.24999999999636*helper_18*x + 6.5995542364305194e-12*helper_18 - helper_19*z - 26.999999999999943*helper_2*helper_4 + helper_2*helper_6 - 3.9754588510021263e-12*helper_2*y + 7.3015170298784391e-12*helper_2 - 404.99999999999858*helper_20*z + 192.93749999999955*helper_20 - helper_21*helper_4 + helper_22*z - helper_22 + 212.06249999999903*helper_23*helper_4 + 9.5625000000000071*helper_23*x + 2.2148949341272479e-14*helper_23 + 254.24999999999696*helper_3*x + 1.9484136526415244e-12*helper_3 - 26.999999999999929*helper_4 + helper_6 - 26.999999999999744*helper_7*x + 9.3022811675779819e-13*helper_7*y - 3.4358627054586911e-12*helper_7 - 19.124999999999901*helper_8*z + 9.5624999999999218*helper_8 - 63.562499999999361*x*y - 116.99999999999937*x*z + 22.499999999999908*x - 2.5454638397093014e-13*y + 6.6613381477509079e-13*pow(z, 6) - 1.6489448695366983e-12*z + 2.1940435579459307e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 152.43749999999915*y;
double helper_7 = x*y;
val[0] = -(helper_1*(26.999999999999901*helper_0*x + 10.687499999999392*helper_0*y + 40.500000000000078*helper_2 + 30.937499999999524*helper_3 - 1.0527516358660403e-13*helper_5 + 169.87499999999889*helper_7 - 9.0000000000001741*x - 14.062500000000743*y - 4.5000000000001386*z + 4.5000000000001039) + 121.4999999999996*helper_2*helper_4 + 182.24999999999952*helper_2*pow(y, 3) + 2*helper_4*x*(121.4999999999996*x + 80.999999999999574*y - 9.5625000000000657) + helper_5*helper_7*(helper_6 + 148.49999999999974*x - 14.062500000000297) + helper_5*y*(74.249999999999872*helper_2 + 20.24999999999973*helper_3 + helper_6*x - 14.062500000000297*x - 9.5624999999999396*y - 1.7845100397373306e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 152.43749999999915*x;
double helper_7 = x*y;
val[1] = -(helper_1*(10.687499999999392*helper_0*x + 2.4980018054061983e-15*helper_0*y + 4.1732242661573985e-14*helper_2 + 84.937499999999446*helper_3 - 3.6108616097152263e-13*helper_5 + 61.874999999999048*helper_7 - 14.062500000000743*x - 3.7969627442180543e-14*y - 4.9085735476239521e-13*z + 3.0278904383784686e-13) + 80.999999999999574*helper_2*helper_4 + 182.24999999999952*helper_2*pow(x, 3) + 2*helper_4*y*(121.4999999999996*x + 80.999999999999574*y - 9.5625000000000657) + helper_5*helper_7*(helper_6 + 40.49999999999946*y - 9.5624999999999396) + helper_5*x*(20.24999999999973*helper_2 + 74.249999999999872*helper_3 + helper_6*y - 14.062500000000297*x - 9.5624999999999396*y - 1.7845100397373306e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = 1.2490009027030992e-15*helper_6;
double helper_8 = pow(z, 5);
double helper_9 = 9.5624999999999325*helper_6;
double helper_10 = pow(y, 3);
double helper_11 = 20.24999999999973*helper_10;
double helper_12 = helper_0*x;
double helper_13 = helper_2*x;
double helper_14 = helper_5*z;
double helper_15 = pow(x, 3);
double helper_16 = 74.249999999999858*helper_15*y;
double helper_17 = helper_1*y;
double helper_18 = 4.9960036108123966e-15*helper_6;
double helper_19 = helper_5*helper_6;
double helper_20 = 161.99999999999915*helper_10;
double helper_21 = helper_0*helper_5;
double helper_22 = 242.9999999999992*helper_15*helper_6;
val[2] = (helper_0*helper_16 + 152.43749999999915*helper_0*helper_19 - 7.494005416218595e-15*helper_0*helper_6 - 4.2765790908560804e-12*helper_0*y + 3.7369482508431232e-12*helper_0 + helper_1*helper_18 + 53.999999999999829*helper_1*helper_5 - 17.999999999998444*helper_1*x - 6.3515442905170672e-12*helper_1 + 182.24999999999952*helper_10*helper_15 - 40.49999999999946*helper_10*helper_3 + helper_11*helper_12 + helper_11*x - helper_12*helper_9 - 64.124999999996504*helper_12*y + 26.999999999998721*helper_12 - 10.687499999999389*helper_13*y + 4.4999999999990852*helper_13 + helper_14*helper_20 - 323.99999999999852*helper_14*helper_6 + 53.999999999999801*helper_14 - 148.49999999999972*helper_15*helper_4 + helper_16 + 42.749999999997556*helper_17*x + 5.258293800380871e-12*helper_17 + helper_18*z + 171.56249999999929*helper_19 - 13.49999999999995*helper_2*helper_5 - helper_2*helper_7 - 3.1200042549528308e-12*helper_2*y + 5.8807021752204935e-12*helper_2 - helper_20*helper_5 - 14.062500000000206*helper_21*y - 80.999999999999773*helper_21 + helper_22*z - helper_22 + 19.124999999999865*helper_3*helper_6 - 17.999999999999503*helper_3 + 28.125000000000639*helper_4*helper_5 + 42.74999999999789*helper_4*x + 1.6474321906656461e-12*helper_4 - 14.06250000000032*helper_5*y - 13.499999999999943*helper_5 - helper_7 + 2.1055032717320901e-13*helper_8*x + 7.2217232194304456e-13*helper_8*y - 2.835232049136481e-12*helper_8 - helper_9*x - 10.687499999999568*x*y + 4.4999999999999289*x - 2.3131496718065e-13*y + 5.5905280404999576e-13*pow(z, 6) - 1.1202289096345845e-12*z + 1.3030201917452128e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 91.687499999999346*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = -(101.24999999999959*helper_0*helper_2*helper_3 + helper_1*(3.9374999999996732*helper_0*y + 8.0935258495173634e-14*helper_2 + 17.43749999999973*helper_3 - 9.5331728622305005e-14*helper_4 + 61.874999999999247*helper_6 - 9.8435148920827639e-14*helper_7 - 1.7487400416626936e-13*x - 5.0625000000004032*y - 1.2079920397312048e-13*z + 9.7883506855467039e-14) + 182.24999999999937*helper_2*pow(y, 3) + 2*helper_3*helper_7*(101.24999999999959*x + 60.749999999999545*y - 9.5624999999999289) + helper_4*helper_6*(helper_5 + 80.999999999999773*x - 9.5625000000001705) + helper_4*y*(40.499999999999886*helper_2 + 13.499999999999751*helper_3 + helper_5*x - 9.5625000000001705*x - 5.062499999999801*y - 1.0769423547385332e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 91.687499999999346*x;
double helper_7 = x*y;
val[1] = -(helper_1*(3.9374999999996732*helper_0*x + 1.0008660566995726e-13*helper_0*y + 2.27630414517676e-14*helper_2 + 30.937499999999623*helper_3 - 2.1661838989217452e-13*helper_5 + 34.87499999999946*helper_7 - 5.0625000000004032*x + 6.9555472492765679e-14*y - 3.0139779561011266e-13*z + 1.6169097311058349e-13) + 60.749999999999545*helper_2*helper_4 + 182.24999999999937*helper_2*pow(x, 3) + 2*helper_4*y*(101.24999999999959*x + 60.749999999999545*y - 9.5624999999999289) + helper_5*helper_7*(helper_6 + 26.999999999999503*y - 5.062499999999801) + helper_5*x*(13.499999999999751*helper_2 + 40.499999999999886*helper_3 + helper_6*y - 9.5625000000001705*x - 5.062499999999801*y - 1.0769423547385332e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = 4.9217574460413756e-14*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = 5.0043302834978975e-14*helper_8;
double helper_10 = pow(z, 5);
double helper_11 = 5.062499999999801*helper_8*x;
double helper_12 = pow(y, 3);
double helper_13 = helper_12*x;
double helper_14 = 13.499999999999751*helper_13;
double helper_15 = 9.5625000000001705*y;
double helper_16 = 1.9687029784165503e-13*helper_6;
double helper_17 = pow(x, 3);
double helper_18 = helper_17*y;
double helper_19 = 40.499999999999886*helper_18;
double helper_20 = helper_1*y;
double helper_21 = 2.001732113399159e-13*helper_8;
double helper_22 = helper_6*helper_8;
double helper_23 = 121.49999999999909*helper_12*helper_6;
double helper_24 = helper_0*helper_6;
double helper_25 = 202.49999999999915*helper_17*helper_8;
val[2] = (-helper_0*helper_11 + helper_0*helper_14 + helper_0*helper_19 - 23.624999999998149*helper_0*helper_3 - 3.0025981700987111e-13*helper_0*helper_8 - 1.1818393486073778e-12*helper_0*x - 2.5239810241828139e-12*helper_0*y + 2.4203625215157701e-12*helper_0 - helper_1*helper_16 + helper_1*helper_21 + 1.4234377565536187e-12*helper_1*x - 4.3922782078098279e-12*helper_1 + 1.9066345724461009e-13*helper_10*x + 4.3323677978434909e-13*helper_10*y - 2.0985990722976908e-12*helper_10 - helper_11 + 182.24999999999937*helper_12*helper_17 - 26.999999999999503*helper_13*z + helper_14 - helper_15*helper_24 - helper_15*helper_6 - helper_16*z - 80.999999999999773*helper_18*z + helper_19 - 3.9374999999996732*helper_2*helper_3 + helper_2*helper_7 - helper_2*helper_9 - 8.3251808224993007e-13*helper_2*x - 1.8647861033116317e-12*helper_2*y + 4.2362537089335251e-12*helper_2 + 15.749999999998693*helper_20*x + 3.1267766154030394e-12*helper_20 + helper_21*z - 202.49999999999858*helper_22*z + 110.81249999999922*helper_22 + helper_23*z - helper_23 + 91.687499999999375*helper_24*helper_8 + 2.9530544676248256e-13*helper_24 + helper_25*z - helper_25 - 3.9374999999997797*helper_3 + 10.124999999999602*helper_4*helper_8 + 4.7012047033056826e-13*helper_4 + 19.125000000000341*helper_5*helper_6 + 15.749999999998913*helper_5*x + 9.605927164812938e-13*helper_5 + helper_7 - helper_9 - 6.9864253271489348e-14*x - 1.31838984174236e-13*y + 4.2166270475263208e-13*pow(z, 6) - 6.4588612236348859e-13*z + 5.8484467269081721e-14)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 90.562499999999119*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (80.999999999999488*helper_0*helper_2*helper_3 + helper_1*(9.5624999999996199*helper_0*y + 8.0935258495173533e-14*helper_2 + 23.062499999999638*helper_3 - 8.9263665903337982e-14*helper_4 + 59.624999999999091*helper_6 - 9.9496799688124953e-14*helper_7 - 1.7818385655843256e-13*x + 5.0624999999995692*y - 1.1056780491180905e-13*z + 9.3487717567341583e-14) + 182.24999999999915*helper_2*pow(y, 3) + 2*helper_3*helper_7*(80.999999999999488*x + 60.749999999999446*y + 9.5624999999999876) + helper_4*helper_6*(helper_5 + 40.499999999999709*x + 9.5624999999997744) + helper_4*y*(20.249999999999854*helper_2 + 13.499999999999723*helper_3 + helper_5*x + 9.5624999999997744*x + 5.0625000000001075*y - 1.0161402969055439e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 90.562499999999119*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (60.749999999999446*helper_0*helper_2*helper_3 + helper_1*(9.5624999999996199*helper_0*x + 1.1381520725883752e-14*helper_2 + 29.812499999999545*helper_3 - 1.9649559757084313e-13*helper_4 + 46.124999999999275*helper_6 + 1.1629586182948444e-13*helper_7 + 5.0624999999995692*x + 8.043565813409215e-14*y - 2.9853897132170217e-13*z + 1.5792488844423741e-13) + 182.24999999999915*helper_2*pow(x, 3) + 2*helper_3*helper_7*(80.999999999999488*x + 60.749999999999446*y + 9.5624999999999876) + helper_4*helper_6*(helper_5 + 26.999999999999446*y + 5.0625000000001075) + helper_4*x*(13.499999999999723*helper_2 + 20.249999999999854*helper_3 + helper_5*y + 9.5624999999997744*x + 5.0625000000001075*y - 1.0161402969055439e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 4.9748399844062016e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 5.8147930914742384e-14*helper_7;
double helper_9 = pow(z, 5);
double helper_10 = helper_7*x;
double helper_11 = pow(y, 3);
double helper_12 = helper_11*x;
double helper_13 = 13.499999999999723*helper_12;
double helper_14 = helper_1*x;
double helper_15 = helper_5*y;
double helper_16 = 9.5624999999997797*helper_15;
double helper_17 = 1.9899359937624973e-13*helper_5;
double helper_18 = pow(x, 3);
double helper_19 = 20.249999999999851*helper_18*y;
double helper_20 = 2.3259172365896858e-13*helper_7;
double helper_21 = helper_5*helper_7;
double helper_22 = 121.49999999999891*helper_11*helper_5;
double helper_23 = 161.99999999999898*helper_18*helper_7;
val[2] = -(5.0625000000000959*helper_0*helper_10 + helper_0*helper_13 + helper_0*helper_16 + helper_0*helper_19 + 90.562499999999147*helper_0*helper_21 - 57.374999999997819*helper_0*helper_3 + 2.9849039906437457e-13*helper_0*helper_5 - 3.4888758548845428e-13*helper_0*helper_7 - 1.1218664885958947e-12*helper_0*x - 2.1386781234866497e-12*helper_0*y + 2.5033100592430735e-12*helper_0 - helper_1*helper_17 + helper_1*helper_20 + 2.7357560661300528e-12*helper_1*y - 4.4233366969237102e-12*helper_1 - 10.125000000000192*helper_10*z + 5.0625000000001101*helper_10 + 182.24999999999915*helper_11*helper_18 - 26.999999999999446*helper_12*z + helper_13 + 38.249999999998494*helper_14*y + 1.3430020984195259e-12*helper_14 - 19.124999999999559*helper_15*z + helper_16 - helper_17*z - 40.499999999999702*helper_18*helper_4 + helper_19 - 9.5624999999996199*helper_2*helper_3 + helper_2*helper_6 - helper_2*helper_8 - 7.8206885412157196e-13*helper_2*x - 1.6664170043867292e-12*helper_2*y + 4.2021074120323912e-12*helper_2 + helper_20*z - 161.99999999999832*helper_21*z + 71.437499999999176*helper_21 + helper_22*z - helper_22 + helper_23*z - helper_23 - 9.5624999999997229*helper_3 + 38.249999999998693*helper_4*x + 7.7080009042162208e-13*helper_4 + helper_6 - helper_8 + 1.7852733180667624e-13*helper_9*x + 3.9299119514168686e-13*helper_9*y - 2.0628498909047584e-12*helper_9 + 4.5036543938614327e-13*x*z - 6.7959526894867016e-14*x - 9.4452223819983097e-14*y + 4.1217029789208669e-13*pow(z, 6) - 7.0381200867330412e-13*z + 7.2410827334223011e-14)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 90.562499999998877*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (60.749999999999318*helper_0*helper_2*helper_3 + helper_1*(9.5624999999996128*helper_0*y + 8.0935258495173382e-14*helper_2 + 29.812499999999574*helper_3 - 8.8167320666520463e-14*helper_4 + 46.124999999998977*helper_6 - 8.5931262105986157e-14*helper_7 - 1.6686652060115951e-13*x + 5.0624999999996083*y - 1.0703937736167197e-13*z + 8.977193988179906e-14) + 182.24999999999881*helper_2*pow(y, 3) + 2*helper_3*helper_7*(60.749999999999318*x + 80.999999999999247*y + 9.5625) + helper_4*helper_6*(helper_5 + 26.999999999999574*x + 5.0624999999998028) + helper_4*y*(13.499999999999787*helper_2 + 20.249999999999652*helper_3 + helper_5*x + 5.0624999999998028*x + 9.5625000000001137*y - 9.6460166243427047e-14))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 90.562499999998877*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (80.999999999999247*helper_0*helper_2*helper_3 + helper_1*(9.5624999999996128*helper_0*x + 7.5876804839224406e-15*helper_2 + 23.062499999999488*helper_3 - 1.6849022177467317e-13*helper_4 + 59.624999999999147*helper_6 + 1.3150591726684881e-13*helper_7 + 5.0624999999996083*x + 9.1038288019262205e-14*y - 2.7555735471196067e-13*z + 1.4323351532618936e-13) + 182.24999999999881*helper_2*pow(x, 3) + 2*helper_3*helper_7*(60.749999999999318*x + 80.999999999999247*y + 9.5625) + helper_4*helper_6*(helper_5 + 40.499999999999304*y + 9.5625000000001137) + helper_4*x*(20.249999999999652*helper_2 + 13.499999999999787*helper_3 + helper_5*y + 5.0624999999998028*x + 9.5625000000001137*y - 9.6460166243427047e-14))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = 4.2965631052992252e-14*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = 6.5752958633424076e-14*helper_6;
double helper_8 = pow(z, 5);
double helper_9 = helper_6*x;
double helper_10 = pow(y, 3);
double helper_11 = helper_10*x;
double helper_12 = 20.249999999999652*helper_11;
double helper_13 = helper_0*x;
double helper_14 = helper_1*x;
double helper_15 = 5.0624999999998064*y;
double helper_16 = 1.7186252421197196e-13*helper_4;
double helper_17 = pow(x, 3);
double helper_18 = helper_17*y;
double helper_19 = 13.499999999999787*helper_18;
double helper_20 = helper_2*y;
double helper_21 = 2.6301183453369756e-13*helper_6;
double helper_22 = helper_4*helper_6;
double helper_23 = 161.99999999999847*helper_10*helper_4;
double helper_24 = helper_0*helper_4;
double helper_25 = 121.49999999999864*helper_17*helper_6;
val[2] = -(helper_0*helper_12 + helper_0*helper_19 + 90.562499999998892*helper_0*helper_22 - 3.9451775180054635e-13*helper_0*helper_6 + 9.5625000000001421*helper_0*helper_9 - 1.7164603072216996e-12*helper_0*y + 2.2790450082687858e-12*helper_0 - helper_1*helper_16 + helper_1*helper_21 + 2.2675750166456215e-12*helper_1*y - 4.0131092893247051e-12*helper_1 + 182.24999999999881*helper_10*helper_17 - 40.499999999999304*helper_11*z + helper_12 - 57.374999999997762*helper_13*y - 1.1211101491603773e-12*helper_13 + 38.249999999998465*helper_14*y + 1.3351889038837211e-12*helper_14 + helper_15*helper_24 + helper_15*helper_4 - helper_16*z - 26.999999999999574*helper_18*z + helper_19 + helper_2*helper_5 - helper_2*helper_7 - 7.7463382930353254e-13*helper_2*x + 3.8136681312916532e-12*helper_2 - 9.5624999999996128*helper_20*x - 1.4093448630347708e-12*helper_20 + helper_21*z - 161.99999999999778*helper_22*z + 71.437499999998892*helper_22 + helper_23*z - helper_23 + 2.5779378631795903e-13*helper_24 + helper_25*z - helper_25 - 10.124999999999613*helper_3*helper_4 + 38.249999999998629*helper_3*x + 5.8267279889888879e-13*helper_3 + helper_5 - helper_7 + 1.7633464133304125e-13*helper_8*x + 3.369804435493462e-13*helper_8*y - 1.8762214004652627e-12*helper_8 - 19.125000000000256*helper_9*z + 9.5625000000001137*helper_9 - 9.5624999999997105*x*y + 4.5351569721851266e-13*x*z - 6.9295263971368867e-14*x - 6.1423088837385321e-14*y + 3.7603253844053613e-13*pow(z, 6) - 6.4852290204696455e-13*z + 6.9107913835963801e-14)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 91.687499999998849*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = -(60.749999999999247*helper_0*helper_2*helper_3 + helper_1*(3.9374999999996909*helper_0*y + 8.0935258495173331e-14*helper_2 + 30.937499999999616*helper_3 - 8.597809963983767e-14*helper_4 + 34.874999999999091*helper_6 - 6.1062266354382549e-14*helper_7 - 1.402211680101556e-13*x - 5.0625000000002842*y - 1.0615119894197174e-13*z + 8.3513057580474677e-14) + 182.24999999999869*helper_2*pow(y, 3) + 2*helper_3*helper_7*(60.749999999999247*x + 101.24999999999916*y - 9.5624999999998934) + helper_4*helper_6*(helper_5 + 26.999999999999517*x - 5.062500000000095) + helper_4*y*(13.499999999999758*helper_2 + 40.499999999999609*helper_3 + helper_5*x - 5.062500000000095*x - 9.5624999999997566*y - 9.7234720275450716e-14))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 91.687499999998849*x;
double helper_7 = x*y;
val[1] = -(helper_1*(3.9374999999996909*helper_0*x + 1.6736612096224116e-13*helper_0*y + 7.5876804839224264e-15*helper_2 + 17.437499999999545*helper_3 - 1.5050460877574564e-13*helper_5 + 61.874999999999233*helper_7 - 5.0625000000002842*x + 1.1340928196545895e-13*y - 2.5854318685958025e-13*z + 1.2995854392627694e-13) + 101.24999999999916*helper_2*helper_4 + 182.24999999999869*helper_2*pow(x, 3) + 2*helper_4*y*(60.749999999999247*x + 101.24999999999916*y - 9.5624999999998934) + helper_5*helper_7*(helper_6 + 80.999999999999218*y - 9.5624999999997566) + helper_5*x*(40.499999999999609*helper_2 + 13.499999999999758*helper_3 + helper_6*y - 5.062500000000095*x - 9.5624999999997566*y - 9.7234720275450716e-14))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 3.0531133177191609e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 8.3683060481121086e-14*helper_7;
double helper_9 = pow(z, 5);
double helper_10 = pow(y, 3);
double helper_11 = helper_10*x;
double helper_12 = 40.499999999999616*helper_11;
double helper_13 = helper_0*x;
double helper_14 = 1.2212453270876495e-13*helper_5;
double helper_15 = pow(x, 3);
double helper_16 = helper_15*y;
double helper_17 = 13.499999999999758*helper_16;
double helper_18 = helper_1*y;
double helper_19 = helper_2*y;
double helper_20 = 3.3473224192448303e-13*helper_7;
double helper_21 = helper_5*helper_7;
double helper_22 = 202.49999999999835*helper_10*helper_5;
double helper_23 = helper_0*helper_5;
double helper_24 = 121.49999999999848*helper_15*helper_7;
double helper_25 = helper_0*helper_7;
val[2] = (helper_0*helper_12 + helper_0*helper_17 + 91.687499999998835*helper_0*helper_21 - 1.4588330543574353e-12*helper_0*y + 1.9796039807395737e-12*helper_0 - helper_1*helper_14 + helper_1*helper_20 + 1.2949571970288684e-12*helper_1*x - 3.6162323135968265e-12*helper_1 + 182.24999999999869*helper_10*helper_15 - 80.999999999999233*helper_11*z + helper_12 - 23.624999999998245*helper_13*y - 1.0826547991449224e-12*helper_13 - helper_14*z - 26.999999999999517*helper_16*z + helper_17 + 15.749999999998764*helper_18*x + 1.9759194280765917e-12*helper_18 - 3.9374999999996909*helper_19*x - 1.2465029008978759e-12*helper_19 + helper_2*helper_6 - helper_2*helper_8 - 7.5362979745640445e-13*helper_2*x + 3.518216967757189e-12*helper_2 + helper_20*z - 202.49999999999747*helper_21*z + 110.81249999999864*helper_21 + helper_22*z - helper_22 - 5.0625000000001137*helper_23*y + 1.8318679906314964e-13*helper_23 + helper_24*z - helper_24 - 9.5624999999997229*helper_25*x - 5.0209836288672301e-13*helper_25 + 19.124999999999503*helper_3*helper_7 + 4.3517620063048888e-13*helper_3 + 10.125000000000199*helper_4*helper_5 + 15.749999999998948*helper_4*x + 4.7087334031913604e-13*helper_4 - 5.0625000000000924*helper_5*y + helper_6 - 9.5624999999997513*helper_7*x - helper_8 + 1.7195619927967663e-13*helper_9*x + 3.0100921755148966e-13*helper_9*y - 1.7590928713673069e-12*helper_9 - 3.9374999999997873*x*y - 6.5805000337704592e-14*x - 4.2466030691910584e-14*y + 3.5671465781205982e-13*pow(z, 6) - 5.2820248175321584e-13*z + 4.8992060408535686e-14)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 152.43749999999898*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = -(80.999999999999403*helper_0*helper_2*helper_3 + helper_1*(10.687499999999563*helper_0*y + 8.093525849517347e-14*helper_2 + 84.937499999999659*helper_3 - 1.0474780764990686e-13*helper_4 + 61.874999999999005*helper_6 - 1.1617096173921401e-13*helper_7 - 1.9485801860952153e-13*x - 14.062500000000487*y - 1.2091022627558265e-13*z + 1.0144315942817118e-13) + 182.24999999999901*helper_2*pow(y, 3) + 2*helper_3*helper_7*(80.999999999999403*x + 121.49999999999928*y - 9.5624999999999254) + helper_4*helper_6*(helper_5 + 40.499999999999638*x - 9.5625000000001865) + helper_4*y*(20.249999999999819*helper_2 + 74.249999999999602*helper_3 + helper_5*x - 9.5625000000001865*x - 14.062499999999691*y - 1.6214373593781301e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 152.43749999999898*x;
double helper_7 = x*y;
val[1] = -(helper_1*(10.687499999999563*helper_0*x + 27.00000000000021*helper_0*y + 40.500000000000014*helper_2 + 30.937499999999503*helper_3 - 2.2258583864953413e-13*helper_5 + 169.87499999999932*helper_7 - 14.062500000000487*x - 8.9999999999998632*y - 4.5000000000003775*z + 4.5000000000001981) + 121.49999999999928*helper_2*helper_4 + 182.24999999999901*helper_2*pow(x, 3) + 2*helper_4*y*(80.999999999999403*x + 121.49999999999928*y - 9.5624999999999254) + helper_5*helper_7*(helper_6 + 148.4999999999992*y - 14.062499999999691) + helper_5*x*(74.249999999999602*helper_2 + 20.249999999999819*helper_3 + helper_6*y - 9.5625000000001865*x - 14.062499999999691*y - 1.6214373593781301e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 5.8085480869606929e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 13.500000000000107*helper_7;
double helper_9 = pow(z, 5);
double helper_10 = helper_7*x;
double helper_11 = pow(y, 3);
double helper_12 = helper_11*x;
double helper_13 = 74.249999999999602*helper_12;
double helper_14 = helper_1*x;
double helper_15 = 2.3234192347842772e-13*helper_5;
double helper_16 = pow(x, 3);
double helper_17 = 20.249999999999822*helper_16;
double helper_18 = helper_0*y;
double helper_19 = helper_2*y;
double helper_20 = helper_7*z;
double helper_21 = helper_5*helper_7;
double helper_22 = 242.99999999999858*helper_11*helper_5;
double helper_23 = 161.99999999999881*helper_16;
val[2] = (-14.062499999999645*helper_0*helper_10 + helper_0*helper_13 + 152.43749999999898*helper_0*helper_21 + 3.4851288521764157e-13*helper_0*helper_5 - 81.000000000000625*helper_0*helper_7 - 1.3694947953446427e-12*helper_0*x + 2.9039479154668678e-12*helper_0 - helper_1*helper_15 + 54.000000000000426*helper_1*helper_7 - 17.999999999997058*helper_1*y - 4.8536313856928004e-12*helper_1 - 14.062499999999702*helper_10 + 182.24999999999901*helper_11*helper_16 - 148.4999999999992*helper_12*z + helper_13 + 42.749999999998238*helper_14*y + 1.6113152478958069e-12*helper_14 - helper_15*z - 40.499999999999645*helper_16*helper_4 + helper_17*helper_18 + helper_17*y - 9.5625000000001634*helper_18*helper_5 - 64.124999999997584*helper_18*x + 26.999999999997826*helper_18 - 10.687499999999559*helper_19*x + 4.4999999999981526*helper_19 + helper_2*helper_6 - helper_2*helper_8 - 9.2656785022348773e-13*helper_2*x + 4.4570631907436496e-12*helper_2 + helper_20*helper_23 - 323.99999999999773*helper_20*helper_5 + 28.124999999999403*helper_20*x + 54.000000000000398*helper_20 + 171.56249999999881*helper_21 + helper_22*z - helper_22 - helper_23*helper_7 + 42.749999999998586*helper_3*y + 5.6383717139673868e-13*helper_3 + 19.125000000000384*helper_4*helper_5 - 17.999999999999289*helper_4 - 9.5625000000001918*helper_5*y + helper_6 - helper_8 + 2.094956152998141e-13*helper_9*x + 4.451716772990676e-13*helper_9*y - 2.1417867479556046e-12*helper_9 - 10.687499999999719*x*y - 8.8585389024231669e-14*x + 4.4999999999999307*y + 4.2216230511371214e-13*pow(z, 6) - 8.9929452773417876e-13*z + 1.1153925005835568e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 212.06249999999909*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (101.2499999999997*helper_0*helper_2*helper_3 + helper_1*(63.562499999999218*helper_0*y + 8.0935258495173722e-14*helper_2 + 137.81249999999949*helper_3 - 1.2442998020833992e-13*helper_4 + 181.12499999999875*helper_6 - 1.9002160955849874e-13*helper_7 - 2.6646046480394057e-13*x + 14.062499999999037*y - 1.3537435061827762e-13*z + 1.2260331638813657e-13) + 182.24999999999957*helper_2*pow(y, 3) + 2*helper_3*helper_7*(101.2499999999997*x + 121.49999999999952*y + 9.5624999999998188) + helper_4*helper_6*(helper_5 + 80.999999999999858*x + 9.562499999999547) + helper_4*y*(40.499999999999929*helper_2 + 74.249999999999659*helper_3 + helper_5*x + 9.562499999999547*x + 14.062500000000103*y - 2.6097873861985355e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 212.06249999999909*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (121.49999999999952*helper_0*helper_2*helper_3 + helper_1*(63.562499999999218*helper_0*x + 40.500000000000021*helper_2 + 90.562499999999375*helper_3 + 13.499999999999634*helper_4 + 275.62499999999898*helper_6 + 54.000000000000163*helper_7 + 14.062499999999037*x + 9.0000000000000959*y + 4.4999999999994156*z - 4.4999999999996723) + 182.24999999999957*helper_2*pow(x, 3) + 2*helper_3*helper_7*(101.2499999999997*x + 121.49999999999952*y + 9.5624999999998188) + helper_4*helper_6*(helper_5 + 148.49999999999932*y + 14.062500000000103) + helper_4*x*(74.249999999999659*helper_2 + 40.499999999999929*helper_3 + helper_5*y + 9.562499999999547*x + 14.062500000000103*y - 2.6097873861985355e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 9.501080477924975e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 27.000000000000085*helper_7;
double helper_9 = pow(z, 5);
double helper_10 = helper_7*x;
double helper_11 = 14.062500000000085*helper_10;
double helper_12 = pow(y, 3);
double helper_13 = helper_12*x;
double helper_14 = 74.249999999999659*helper_13;
double helper_15 = helper_1*x;
double helper_16 = 9.5624999999995524*helper_5;
double helper_17 = 3.8004321911699718e-13*helper_5;
double helper_18 = pow(x, 3);
double helper_19 = 40.499999999999929*helper_18;
double helper_20 = helper_0*y;
double helper_21 = helper_2*y;
double helper_22 = helper_7*z;
double helper_23 = 242.99999999999901*helper_12*helper_5;
double helper_24 = helper_0*helper_5;
double helper_25 = 202.49999999999937*helper_18;
val[2] = -(helper_0*helper_11 + helper_0*helper_14 - 162.00000000000045*helper_0*helper_7 - 1.6763535004571318e-12*helper_0*x + 4.5117035107899817e-12*helper_0 - helper_1*helper_17 + 108.00000000000028*helper_1*helper_7 - 251.99999999999505*helper_1*y - 7.076214614265525e-12*helper_1 - 28.125000000000171*helper_10*z + helper_11 + 182.24999999999957*helper_12*helper_18 - 148.49999999999932*helper_13*z + helper_14 + 254.24999999999693*helper_15*y + 1.947102201693688e-12*helper_15 + helper_16*helper_20 + helper_16*y - helper_17*z - 80.999999999999858*helper_18*helper_4 + helper_19*helper_20 + helper_19*y + helper_2*helper_6 - helper_2*helper_8 - 1.1089254514651214e-12*helper_2*x + 6.1992598554549705e-12*helper_2 - 381.37499999999568*helper_20*x + 242.99999999999613*helper_20 - 63.562499999999204*helper_21*x + 130.49999999999696*helper_21 + helper_22*helper_25 - 404.99999999999852*helper_22*helper_5 + 108.00000000000034*helper_22 + helper_23*z - helper_23 + 212.06249999999909*helper_24*helper_7 + 5.7006482867549572e-13*helper_24 - helper_25*helper_7 + 254.24999999999747*helper_3*y + 7.0280239961028885e-13*helper_3 - 19.124999999999105*helper_4*helper_5 - 116.99999999999864*helper_4 + 192.93749999999946*helper_5*helper_7 + helper_6 - helper_8 + 2.4885996041667903e-13*helper_9*x - 26.999999999999275*helper_9*y - 2.8752000780229866e-12*helper_9 - 63.562499999999474*x*y - 1.1348560979840261e-13*x + 22.499999999999851*y + 5.5155879863377535e-13*pow(z, 6) - 1.5251550022909953e-12*z + 2.1404752970077502e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 35.999999999999915*y;
double helper_4 = helper_0*y;
double helper_5 = 26.999999999999979*x;
val[0] = (helper_1*(helper_0*helper_5 + 26.999999999999989*helper_1 + 49.499999999999936*helper_2 + 76.499999999999801*helper_4 + helper_5 + 98.999999999999872*x*y + 80.999999999999702*y + 31.499999999999982*z - 26.999999999999979) + 71.999999999999829*helper_2*x + helper_4*x*(helper_3 + 49.499999999999936) + helper_4*(helper_3*x + 49.499999999999936*x + 49.499999999999936*y + 4.4999999999999067))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 35.999999999999915*x;
double helper_4 = helper_0*x;
double helper_5 = 26.999999999999982*y;
val[1] = (helper_1*(helper_0*helper_5 + 26.999999999999886*helper_1 + 49.499999999999936*helper_2 + 76.499999999999801*helper_4 + helper_5 + 98.999999999999872*x*y + 80.999999999999702*x + 31.499999999999812*z - 26.999999999999883) + 71.999999999999829*helper_2*y + helper_4*y*(helper_3 + 49.499999999999936) + helper_4*(helper_3*y + 49.499999999999936*x + 49.499999999999936*y + 4.4999999999999067))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 13.499999999999989*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 13.499999999999989*helper_7;
double helper_9 = pow(z, 4);
double helper_10 = 35.999999999999915*helper_5*helper_7;
val[2] = -(229.49999999999943*helper_0*helper_2 + 40.499999999999972*helper_0*helper_5 + 40.499999999999979*helper_0*helper_7 - 229.49999999999989*helper_0*x - 229.49999999999918*helper_0*y + 202.49999999999909*helper_0 - 76.499999999999801*helper_1*helper_2 - helper_1*helper_6 - helper_1*helper_8 + 184.49999999999994*helper_1*x + 184.49999999999929*helper_1*y - 265.4999999999992*helper_1 + helper_10*z + helper_10 + 71.999999999999886*helper_2 + 49.499999999999929*helper_3*helper_7 - 224.99999999999952*helper_3*y + 121.5*helper_3 + 49.499999999999929*helper_4*helper_5 + 121.49999999999963*helper_4 - 49.499999999999936*helper_5*y - 40.499999999999964*helper_5*z + helper_6 - 49.499999999999936*helper_7*x - 40.499999999999972*helper_7*z + helper_8 - 53.999999999999979*helper_9*x - 53.999999999999773*helper_9*y + 166.49999999999969*helper_9 - 22.5*x - 22.499999999999961*y - 40.499999999999929*pow(z, 5) - 71.999999999999517*z + 8.9999999999998934)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 36.000000000000135*x;
double helper_4 = 18.000000000000068*x;
double helper_5 = helper_0*y;
double helper_6 = 1.8540724511240228e-14*x;
val[0] = -(helper_1*(-helper_0*helper_6 + 13.499999999999988*helper_1 + 18.000000000000082*helper_2 + helper_3*y + 31.500000000000007*helper_5 - helper_6 + 40.499999999999972*y + 22.499999999999986*z - 13.499999999999988) + helper_2*helper_3 + helper_4*helper_5*(y + 1) + helper_5*(helper_4*y + helper_4 + 18.000000000000082*y + 8.9999999999999591))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 18.000000000000068*x;
double helper_4 = helper_0*x;
double helper_5 = 1.7763568394001897e-15*y;
val[1] = -(helper_1*(helper_0*helper_5 + 13.499999999999948*helper_1 + 18.000000000000068*helper_2 + 31.500000000000007*helper_4 + helper_5 + 36.000000000000163*x*y + 40.499999999999972*x + 22.499999999999918*z - 13.49999999999995) + 36.000000000000135*helper_2*y + helper_4*y*(helper_3 + 18.000000000000082) + helper_4*(helper_3*y + helper_3 + 18.000000000000082*y + 8.9999999999999591))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 9.2703622556201139e-15*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 8.8817841970009486e-16*helper_7;
double helper_9 = pow(z, 4);
double helper_10 = 18.000000000000082*helper_7;
double helper_11 = 18.000000000000068*helper_5;
double helper_12 = 2.7811086766860209e-14*helper_5;
double helper_13 = helper_1*y;
double helper_14 = 2.6645352591002846e-15*helper_7;
double helper_15 = helper_5*helper_7;
val[2] = (-helper_0*helper_12 + helper_0*helper_14 + 94.500000000000028*helper_0*helper_2 - 94.499999999999901*helper_0*x - 94.499999999999631*helper_0*y + 161.99999999999983*helper_0 + helper_1*helper_6 - helper_1*helper_8 + 85.499999999999915*helper_1*x - 233.99999999999994*helper_1 + helper_10*helper_3 - helper_10*x + helper_11*helper_4 - helper_11*y + helper_12*z - 31.500000000000011*helper_13*x + 85.499999999999659*helper_13 - helper_14*z + 18.000000000000068*helper_15*z + 18.00000000000006*helper_15 + 22.50000000000005*helper_2 + 40.499999999999943*helper_3 - 85.500000000000071*helper_4*x + 40.499999999999829*helper_4 - helper_6 + helper_8 - 26.999999999999975*helper_9*x - 26.999999999999897*helper_9*y + 157.50000000000006*helper_9 - 4.4999999999999885*x - 4.4999999999999787*y - 40.500000000000028*pow(z, 5) - 49.499999999999844*z + 4.4999999999999574)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 35.999999999999787*y;
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = 26.999999999999968*x;
val[0] = (helper_0*helper_4*(helper_3 + 49.499999999999801) + helper_1*(helper_0*helper_6 - 1.1657341758564043e-14*helper_1 + 22.49999999999978*helper_2 + 98.999999999999602*helper_4 + 22.499999999999712*helper_5 + helper_6 + 17.999999999999659*y - 4.5000000000000169*z + 2.3092638912203256e-14) + 71.999999999999574*helper_2*x + helper_5*(helper_3*x + 49.499999999999801*x + 22.49999999999978*y - 4.5000000000000524))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 35.999999999999787*x;
double helper_4 = x*y;
double helper_5 = helper_0*x;
double helper_6 = 8.0158102377936151e-14*y;
val[1] = (helper_0*helper_4*(helper_3 + 22.49999999999978) + helper_1*(-helper_0*helper_6 - 1.085798118083398e-13*helper_1 + 49.499999999999801*helper_2 + 44.999999999999559*helper_4 + 22.499999999999712*helper_5 - helper_6 + 17.999999999999659*x - 1.6614487563515387e-13*z + 1.1544584732625601e-13) + 71.999999999999574*helper_2*y + helper_5*(helper_3*y + 49.499999999999801*x + 22.49999999999978*y - 4.5000000000000524))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = 4.0079051188968536e-14*helper_5;
double helper_7 = pow(z, 4);
double helper_8 = helper_1*x;
double helper_9 = 49.499999999999801*y;
double helper_10 = helper_4*z;
double helper_11 = 1.2023715356690559e-13*helper_5;
double helper_12 = 35.999999999999787*helper_4*helper_5;
val[2] = -(-helper_0*helper_11 + 67.499999999999119*helper_0*helper_2 + 40.49999999999995*helper_0*helper_4 - 13.499999999999911*helper_0*x + 8.0452311479461575e-13*helper_0*y - 8.8254403785014792e-13*helper_0 - 13.499999999999982*helper_1*helper_4 + helper_1*helper_6 - 7.0249361883156518e-13*helper_1*y + 9.8207553200779361e-13*helper_1 + helper_10*helper_9 - 40.499999999999943*helper_10 + helper_11*z + helper_12*z + helper_12 - 71.99999999999919*helper_2*z + 26.999999999999766*helper_2 + 22.499999999999787*helper_3*helper_5 + 13.499999999999959*helper_3 - helper_4*helper_9 + 13.499999999999984*helper_4 - 22.49999999999978*helper_5*x - helper_6 + 2.3314683517127827e-14*helper_7*x + 2.1715962361667991e-13*helper_7*y - 5.4878324107221266e-13*helper_7 - 22.499999999999709*helper_8*y + 4.4999999999999236*helper_8 - 4.4999999999999885*x - 3.7020386756125779e-13*y*z + 5.1014747981525817e-14*y + 1.2290168882600417e-13*pow(z, 5) + 3.9948599983574463e-13*z - 7.3135941747181846e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{double helper_0 = pow(z, 2);
double helper_1 = 4.1633363423442758e-14*x;
double helper_2 = 35.99999999999973*x;
double helper_3 = y*z;
double helper_4 = pow(y, 2);
double helper_5 = 17.999999999999869*helper_4;
val[0] = -z*(-helper_0*helper_1 + 4.4999999999998419*helper_0*y + 31.500000000000032*helper_0 - helper_1 + helper_2*helper_3 + helper_2*helper_4 - helper_2*y - 17.999999999999709*helper_3 + helper_5*z - helper_5 + 8.3266726846886665e-14*x*z + 13.499999999999863*y - 13.500000000000011*pow(z, 3) - 22.500000000000036*z + 4.5000000000000142)/(1.0*helper_0 - 2.0*z + 1.0);}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 17.999999999999865*x;
double helper_4 = helper_0*x;
double helper_5 = 3.0198066269804144e-14*y;
val[1] = -(-helper_1*(helper_0*helper_5 + 4.9293902293356622e-14*helper_1 - 17.999999999999865*helper_2 - 4.4999999999998419*helper_4 + helper_5 - 35.999999999999737*x*y + 4.5000000000001812*x + 7.7271522513910365e-14*z - 5.3068660577082142e-14) + 35.99999999999973*helper_2*y + helper_4*y*(helper_3 + 17.999999999999869) + helper_4*(helper_3*y + helper_3 + 17.999999999999869*y - 9.0000000000000231))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = pow(x, 2);
double helper_5 = pow(y, 2);
double helper_6 = 1.5099033134902072e-14*helper_5;
double helper_7 = pow(z, 4);
double helper_8 = 17.999999999999869*helper_5;
double helper_9 = 17.999999999999865*helper_4;
double helper_10 = helper_9*y;
double helper_11 = 6.2450045135165434e-14*helper_4;
double helper_12 = helper_0*y;
double helper_13 = 4.5297099404706551e-14*helper_5;
double helper_14 = helper_5*helper_9;
val[2] = (-helper_0*helper_11 - helper_0*helper_13 + 94.500000000000114*helper_0*x - 3.1699642910609565e-13*helper_0 - 4.4999999999998419*helper_1*helper_2 + 2.0816681711721572e-14*helper_1*helper_4 + helper_1*helper_6 - 85.500000000000071*helper_1*x - 3.1707969583294582e-13*helper_1*y + 3.5146885402070464e-13*helper_1 + helper_10*z - helper_10 + helper_11*z + 13.499999999999531*helper_12*x + 3.597122599785491e-13*helper_12 + helper_13*z + helper_14*z + helper_14 - 22.499999999999552*helper_2*z + 13.499999999999867*helper_2 + helper_3*helper_8 - 40.500000000000057*helper_3 - 2.0816681711721568e-14*helper_4 - helper_6 + 27.000000000000021*helper_7*x + 9.8587804586713282e-14*helper_7*y - 1.9534374118279506e-13*helper_7 - helper_8*x + 4.5000000000000142*x - 1.6253665080512143e-13*y*z + 2.1316282072802879e-14*y + 4.3465231414075144e-14*pow(z, 5) + 1.4363510381087816e-13*z - 2.6229018956769188e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 35.999999999999837*y;
double helper_4 = helper_0*y;
double helper_5 = 5.218048215738208e-14*x;
val[0] = (helper_1*(-helper_0*helper_5 - 1.6542323066914757e-14*helper_1 + 22.499999999999861*helper_2 + 22.49999999999978*helper_4 - helper_5 + 44.999999999999652*x*y + 26.999999999999737*y - 1.5210055437364588e-14*z + 1.9394208461420634e-14) + 71.999999999999673*helper_2*x + helper_4*x*(helper_3 + 22.499999999999826) + helper_4*(helper_3*x + 22.499999999999826*x + 22.499999999999861*y + 4.4999999999999583))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 35.999999999999837*x;
double helper_4 = helper_0*x;
double helper_5 = 6.8833827526758128e-15*y;
val[1] = (helper_1*(-helper_0*helper_5 - 6.2838623193783444e-14*helper_1 + 22.499999999999826*helper_2 + 22.49999999999978*helper_4 - helper_5 + 44.999999999999723*x*y + 26.999999999999737*x - 1.029731855339826e-13*z + 6.4975802516186845e-14) + 71.999999999999673*helper_2*y + helper_4*y*(helper_3 + 22.499999999999861) + helper_4*(helper_3*y + 22.499999999999826*x + 22.499999999999861*y + 4.4999999999999583))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = 2.6090241078691141e-14*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = 3.4416913763379064e-15*helper_6;
double helper_8 = pow(z, 4);
double helper_9 = helper_6*x;
double helper_10 = 7.8270723236073423e-14*helper_4;
double helper_11 = 1.0325074129013719e-14*helper_6;
double helper_12 = helper_4*helper_6;
val[2] = -(-helper_0*helper_10 - helper_0*helper_11 + 67.499999999999346*helper_0*helper_2 + 1.5287771049088201e-13*helper_0*x + 4.4514392172345384e-13*helper_0*y - 5.5946913768422787e-13*helper_0 - 22.499999999999787*helper_1*helper_2 + helper_1*helper_5 + helper_1*helper_7 - 1.171285290979528e-13*helper_1*x - 3.9973580001628458e-13*helper_1*y + 5.7506777118021112e-13*helper_1 + helper_10*z + helper_11*z + 35.999999999999837*helper_12*z + 35.999999999999844*helper_12 - 62.999999999999389*helper_2*z + 17.999999999999822*helper_2 + 22.499999999999822*helper_3*helper_4 - 1.9378942894831916e-13*helper_3 - 22.499999999999826*helper_4*y - helper_5 - helper_7 + 3.3084646133829166e-14*helper_8*x + 1.2567724638756684e-13*helper_8*y - 3.0031532816110267e-13*helper_8 + 22.499999999999865*helper_9*z - 22.499999999999861*helper_9 - 8.6708418223225294e-14*x*z + 1.7874590696464926e-14*x + 2.2704060853584347e-14*y + 6.3615779311020056e-14*pow(z, 5) + 2.769173779171277e-13*z - 5.5816462563029202e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 17.99999999999989*y;
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = 3.2751579226441929e-14*x;
val[0] = -(helper_0*helper_4*(helper_3 + 17.999999999999886) + helper_1*(-helper_0*helper_6 - 8.4932061383823955e-15*helper_1 + 17.999999999999904*helper_2 + 35.999999999999773*helper_4 + 31.499999999999851*helper_5 - helper_6 + 40.499999999999829*y - 7.3829831137572516e-15*z + 1.0304257447302196e-14) + 35.99999999999978*helper_2*x + helper_5*(helper_3*x + 17.999999999999886*x + 17.999999999999904*y + 8.9999999999999698))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 17.99999999999989*x;
double helper_4 = helper_0*x;
double helper_5 = 1.7763568394003546e-15*y;
val[1] = -(helper_1*(helper_0*helper_5 - 3.863576125695517e-14*helper_1 + 17.999999999999886*helper_2 + 31.499999999999851*helper_4 + helper_5 + 35.999999999999808*x*y + 40.499999999999829*x - 6.6613381477508938e-14*z + 3.7969627442180051e-14) + 35.99999999999978*helper_2*y + helper_4*y*(helper_3 + 17.999999999999904) + helper_4*(helper_3*y + 17.999999999999886*x + 17.999999999999904*y + 8.9999999999999698))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = pow(x, 2);
double helper_4 = 1.6375789613220964e-14*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = 8.881784197001773e-16*helper_5;
double helper_7 = pow(z, 4);
double helper_8 = 17.999999999999904*helper_5*x;
double helper_9 = helper_0*x;
double helper_10 = 17.999999999999886*helper_3*y;
double helper_11 = 4.9127368839663145e-14*helper_3;
double helper_12 = 2.6645352591005319e-15*helper_5;
double helper_13 = helper_3*helper_5;
val[2] = (-helper_0*helper_11 + helper_0*helper_12 + 2.6378899065093603e-13*helper_0*y - 3.6987080065386838e-13*helper_0 - 31.499999999999851*helper_1*helper_2 + helper_1*helper_4 - helper_1*helper_6 - 6.0562665993302706e-14*helper_1*x - 2.4247270857813505e-13*helper_1*y + 3.7403413699621322e-13*helper_1 + helper_10*z - helper_10 + helper_11*z - helper_12*z + 17.99999999999989*helper_13*z + 17.999999999999897*helper_13 - 85.499999999999574*helper_2*z + 22.499999999999879*helper_2 - helper_4 + helper_6 + 1.6986412276764791e-14*helper_7*x + 7.7271522513910239e-14*helper_7*y - 1.9319962296648732e-13*helper_7 + helper_8*z - helper_8 + 94.499999999999545*helper_9*y + 7.9769524319316008e-14*helper_9 - 4.5796699765787411e-14*x*z + 9.6034291630075394e-15*x - 1.092459456231153e-13*y*z + 1.0658141036401402e-14*y + 4.0696612746415964e-14*pow(z, 5) + 1.8695461845297311e-13*z - 3.8614944575243947e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 35.999999999999787*y;
double helper_4 = helper_0*y;
double helper_5 = 7.2608585810484872e-14*x;
val[0] = (helper_1*(-helper_0*helper_5 - 1.4765966227514481e-14*helper_1 + 49.499999999999815*helper_2 + 22.49999999999973*helper_4 - helper_5 + 44.999999999999567*x*y + 17.999999999999673*y - 5.4400928206631897e-15*z + 1.7368051441479704e-14) + 71.999999999999574*helper_2*x + helper_4*x*(helper_3 + 22.499999999999783) + helper_4*(helper_3*x + 22.499999999999783*x + 49.499999999999815*y - 4.5000000000000524))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 35.999999999999787*x;
double helper_4 = helper_0*x;
double helper_5 = 26.999999999999993*y;
val[1] = (helper_1*(helper_0*helper_5 - 6.9055872131684207e-14*helper_1 + 22.499999999999783*helper_2 + 22.49999999999973*helper_4 + helper_5 + 98.999999999999631*x*y + 17.999999999999673*x - 4.5000000000001199*z + 7.638334409421077e-14) + 71.999999999999574*helper_2*y + helper_4*y*(helper_3 + 49.499999999999815) + helper_4*(helper_3*y + 22.499999999999783*x + 49.499999999999815*y - 4.5000000000000524))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = y*z;
double helper_3 = pow(x, 2);
double helper_4 = 3.6304292905242398e-14*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(z, 4);
double helper_7 = helper_5*x;
double helper_8 = helper_0*x;
double helper_9 = 1.0891287871572719e-13*helper_3;
double helper_10 = helper_1*y;
double helper_11 = 40.499999999999986*helper_5;
double helper_12 = 35.999999999999787*helper_3*helper_5;
val[2] = -(helper_0*helper_11 - helper_0*helper_9 - 13.499999999999531*helper_0*y - 6.767086890846447e-13*helper_0 + helper_1*helper_4 - 13.499999999999989*helper_1*helper_5 - 1.1268763699945225e-13*helper_1*x + 6.6921468366842308e-13*helper_1 - 22.49999999999973*helper_10*x + 4.4999999999995666*helper_10 - helper_11*z + helper_12*z + helper_12 + 22.49999999999978*helper_2*helper_3 - 71.999999999999233*helper_2*x + 13.499999999999808*helper_2 - 22.499999999999783*helper_3*y - helper_4 + 13.5*helper_5 + 2.9531932455028949e-14*helper_6*x + 1.3811174426336894e-13*helper_6*y - 3.3673064336880715e-13*helper_6 + 49.499999999999801*helper_7*z - 49.499999999999815*helper_7 + 67.499999999999204*helper_8*y + 1.6087131626818448e-13*helper_8 + helper_9*z + 26.99999999999978*x*y - 1.0180745135812558e-13*x*z + 2.4091839634366001e-14*x - 4.4999999999999822*y + 6.8944849829221918e-14*pow(z, 5) + 3.4797165149313626e-13*z - 7.2691852537331506e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 17.999999999999886*y;
double helper_4 = helper_0*y;
double helper_5 = 3.6304292905242423e-14*x;
val[0] = -(-helper_1*(helper_0*helper_5 + 1.0269562977782644e-14*helper_1 - 17.99999999999989*helper_2 - 4.4999999999998579*helper_4 + helper_5 - 35.999999999999766*x*y + 4.5000000000001652*y + 5.6066262743569995e-15*z - 1.0665079930305356e-14) + 35.999999999999773*helper_2*x + helper_4*x*(helper_3 + 17.999999999999883) + helper_4*(helper_3*x + 17.999999999999883*x + 17.99999999999989*y - 9.0000000000000231))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 17.999999999999886*x;
double helper_4 = helper_0*x;
double helper_5 = 1.9539925233402651e-14*y;
val[1] = -(-helper_1*(helper_0*helper_5 + 13.500000000000041*helper_1 - 17.999999999999883*helper_2 - 4.4999999999998579*helper_4 + helper_5 - 35.99999999999978*x*y + 4.5000000000001652*x + 22.500000000000068*z - 13.500000000000044) + 35.999999999999773*helper_2*y + helper_4*y*(helper_3 + 17.99999999999989) + helper_4*(helper_3*y + 17.999999999999883*x + 17.99999999999989*y - 9.0000000000000231))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(z, 4);
double helper_8 = 17.99999999999989*helper_6;
double helper_9 = 17.999999999999883*helper_5;
double helper_10 = 5.4456439357863102e-14*helper_5;
double helper_11 = helper_0*y;
double helper_12 = 2.9309887850104158e-14*helper_6;
double helper_13 = 17.999999999999886*helper_5*helper_6;
val[2] = (-helper_0*helper_10 - helper_0*helper_12 + 1.0641487691032066e-13*helper_0*x - 3.2249203307798911e-13*helper_0 - 4.4999999999998579*helper_1*helper_2 + 1.8152146452621208e-14*helper_1*helper_5 + 9.7699626167013271e-15*helper_1*helper_6 - 7.6549877547903471e-14*helper_1*x - 85.500000000000256*helper_1*y + 3.6684544291176513e-13*helper_1 + helper_10*z + 13.499999999999574*helper_11*x + 94.500000000000284*helper_11 + helper_12*z + helper_13*z + helper_13 - 22.499999999999595*helper_2*z + 13.499999999999881*helper_2 + helper_3*helper_8 - 6.533662499919026e-14*helper_3 + helper_4*helper_9 - 40.500000000000128*helper_4 - 1.8152146452621212e-14*helper_5 - 9.7699626167013255e-15*helper_6 + 2.0539125955565289e-14*helper_7*x + 27.000000000000078*helper_7*y - 2.0961010704922791e-13*helper_7 - helper_8*x - helper_9*y + 1.4932499681208289e-14*x + 4.5000000000000142*y + 4.7961634663806548e-14*pow(z, 5) + 1.4307999229856505e-13*z - 2.5784929746919119e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 53.999999999999687*y;
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = 53.999999999999964*x;
val[0] = -(helper_0*helper_4*(helper_3 + 80.999999999999716) + helper_1*(helper_0*helper_6 + 26.999999999999982*helper_1 + 53.999999999999687*helper_2 + 161.99999999999943*helper_4 + 80.999999999999517*helper_5 + helper_6 + 80.999999999999432*y + 26.999999999999975*z - 26.999999999999972) + 107.99999999999937*helper_2*x + helper_5*(helper_3*x + helper_3 + 80.999999999999716*x - 1.0922512894140318e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 107.99999999999937*y;
double helper_4 = 53.999999999999687*y;
double helper_5 = helper_0*x;
double helper_6 = 8.3932860661661582e-14*y;
val[1] = -(helper_1*(-helper_0*helper_6 - 1.6786572132332294e-13*helper_1 + 80.999999999999716*helper_2 + helper_3*x + 80.999999999999517*helper_5 - helper_6 + 80.999999999999432*x - 2.7278179715039975e-13*z + 1.8004348012468007e-13) + helper_2*helper_3 + helper_4*helper_5*(x + 1) + helper_5*(helper_4*x + helper_4 + 80.999999999999716*x - 1.0922512894140318e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*z;
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = 26.999999999999979*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = 4.1966430330830482e-14*helper_6;
double helper_8 = pow(z, 4);
double helper_9 = 53.999999999999687*helper_6;
double helper_10 = helper_1*x;
double helper_11 = 80.999999999999716*helper_4;
double helper_12 = 80.999999999999943*helper_4;
double helper_13 = helper_0*y;
double helper_14 = 1.2589929099249275e-13*helper_6;
double helper_15 = helper_4*helper_9;
val[2] = (helper_0*helper_12 - helper_0*helper_14 - 242.99999999999986*helper_0*x - 1.5242807016591016e-12*helper_0 - helper_1*helper_5 + helper_1*helper_7 - 1.0701439734361841e-12*helper_1*y + 1.671163207817009e-12*helper_1 - 80.999999999999517*helper_10*y + 188.99999999999989*helper_10 + helper_11*helper_3 - helper_11*y - helper_12*z + 242.99999999999855*helper_13*x + 1.1960432644286811e-12*helper_13 + helper_14*z + helper_15*z + helper_15 + helper_2*helper_9 - 242.99999999999866*helper_2*y + 134.99999999999994*helper_2 - 5.2458037913538445e-13*helper_3 + helper_5 - helper_7 - 53.999999999999964*helper_8*x + 3.3573144264664582e-13*helper_8*y - 9.262590694447683e-13*helper_8 - helper_9*x + 80.999999999999631*x*y - 27.0*x + 6.2949645496246376e-14*y + 2.0683454948766575e-13*pow(z, 5) + 7.0593531020790276e-13*z - 1.3339329640871223e-13)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 107.99999999999937*x;
double helper_4 = 53.999999999999687*x;
double helper_5 = helper_0*y;
double helper_6 = 7.5162098767122555e-14*x;
val[0] = -(helper_1*(-helper_0*helper_6 - 2.5479618415147194e-14*helper_1 + 26.999999999999705*helper_2 + helper_3*y + 26.999999999999595*helper_5 - helper_6 + 26.999999999999527*y - 2.6589841439772386e-14*z + 3.2585045772748199e-14) + helper_2*helper_3 + helper_4*helper_5*(y + 1) + helper_5*(helper_4*y + helper_4 + 26.999999999999705*y - 6.7446048745977742e-14))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 53.999999999999687*x;
double helper_4 = helper_0*x;
double helper_5 = 6.2172489379008489e-14*y;
val[1] = -(helper_1*(-helper_0*helper_5 - 1.3544720900426834e-13*helper_1 + 53.999999999999687*helper_2 + 26.999999999999595*helper_4 - helper_5 + 53.99999999999941*x*y + 26.999999999999527*x - 2.0516921495072769e-13*z + 1.3922196728799382e-13) + 107.99999999999937*helper_2*y + helper_4*y*(helper_3 + 26.999999999999705) + helper_4*(helper_3*y + helper_3 + 26.999999999999705*y - 6.7446048745977742e-14))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 3.7581049383561745e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 3.1086244689504497e-14*helper_7;
double helper_9 = pow(z, 4);
double helper_10 = 26.999999999999705*helper_7;
double helper_11 = 53.999999999999687*helper_5;
double helper_12 = 1.1274314815068269e-13*helper_5;
double helper_13 = helper_0*y;
double helper_14 = 9.3258734068513478e-14*helper_7;
double helper_15 = helper_11*helper_7;
val[2] = (-helper_0*helper_12 - helper_0*helper_14 + 2.2598589666245049e-13*helper_0*x - 1.0656475701864513e-12*helper_0 - 26.999999999999602*helper_1*helper_2 + helper_1*helper_6 + helper_1*helper_8 - 1.7724710588140556e-13*helper_1*x - 8.7840845708341911e-13*helper_1*y + 1.1705636460135273e-12*helper_1 + helper_10*helper_3 - helper_10*x + helper_11*helper_4 - helper_11*y + helper_12*z + 80.999999999998778*helper_13*x + 1.0098588631990367e-12*helper_13 + helper_14*z + helper_15*z + helper_15 - 80.999999999998849*helper_2*z + 26.999999999999662*helper_2 - 1.2406742300186092e-13*helper_3 - 4.6807002718196398e-13*helper_4 - helper_6 - helper_8 + 5.0959236830294004e-14*helper_9*x + 2.7089441800853547e-13*helper_9*y - 6.4748206796138655e-13*helper_9 + 2.4369395390522211e-14*x + 6.5725203057808863e-14*y + 1.4388490399141966e-13*pow(z, 5) + 4.9010795422077283e-13*z - 9.1426866077880379e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 53.99999999999973*y;
double helper_4 = helper_0*y;
double helper_5 = 9.2925667161125135e-14*x;
val[0] = -(helper_1*(-helper_0*helper_5 - 2.5479618415147216e-14*helper_1 + 53.999999999999766*helper_2 + 26.999999999999641*helper_4 - helper_5 + 53.999999999999424*x*y + 26.999999999999574*y - 1.9484414082171399e-14*z + 2.9976021664879113e-14) + 107.99999999999946*helper_2*x + helper_4*x*(helper_3 + 26.999999999999712) + helper_4*(helper_3*x + 26.999999999999712*x + 53.999999999999766*y - 6.5947247662733857e-14))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 53.99999999999973*x;
double helper_4 = helper_0*x;
double helper_5 = 1.1990408665951429e-14*y;
val[1] = -(helper_1*(-helper_0*helper_5 - 9.5923269327612844e-14*helper_1 + 26.999999999999712*helper_2 + 26.999999999999641*helper_4 - helper_5 + 107.99999999999953*x*y + 26.999999999999574*x - 1.5887291482385879e-13*z + 1.0041967257734468e-13) + 107.99999999999946*helper_2*y + helper_4*y*(helper_3 + 53.999999999999766) + helper_4*(helper_3*y + 26.999999999999712*x + 53.999999999999766*y - 6.5947247662733857e-14))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 4.6462833580563085e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = pow(z, 4);
double helper_9 = 53.999999999999766*helper_7;
double helper_10 = helper_0*x;
double helper_11 = 26.999999999999712*helper_5;
double helper_12 = 1.3938850074168762e-13*helper_5;
double helper_13 = 1.7985612998927145e-14*helper_7;
double helper_14 = helper_5*helper_7;
val[2] = (-helper_0*helper_12 - helper_0*helper_13 + 6.7446048745977826e-13*helper_0*y - 8.0485618170200087e-13*helper_0 - 26.999999999999645*helper_1*helper_2 + helper_1*helper_6 + 5.9952043329757128e-15*helper_1*helper_7 - 1.8435253323900679e-13*helper_1*x - 6.0851323979704416e-13*helper_1*y + 8.1984419253443968e-13*helper_1 + 80.99999999999892*helper_10*y + 2.47302178735252e-13*helper_10 + helper_11*helper_4 - helper_11*y + helper_12*z + helper_13*z + 53.99999999999973*helper_14*z + 53.999999999999744*helper_14 - 80.999999999998991*helper_2*z + 26.999999999999709*helper_2 + helper_3*helper_9 - 1.4538370507466301e-13*helper_3 - 2.9076741014932602e-13*helper_4 - helper_6 - 5.9952043329757144e-15*helper_7 + 5.0959236830294004e-14*helper_8*x + 1.9184653865522619e-13*helper_8*y - 4.2565950764128204e-13*helper_8 - helper_9*x + 3.1474822748123188e-14*x + 3.2973623831367301e-14*y + 8.9928064994636872e-14*pow(z, 5) + 4.0317749139262287e-13*z - 8.2434059578417343e-14)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 53.999999999999616*y;
double helper_4 = helper_0*y;
double helper_5 = 1.0713652187632695e-13*x;
val[0] = -(helper_1*(-helper_0*helper_5 - 2.1926904736346662e-14*helper_1 + 80.999999999999673*helper_2 + 80.999999999999503*helper_4 - helper_5 + 107.99999999999925*x*y + 80.999999999999403*y - 1.0602629885170106e-14*z + 2.7838842342475642e-14) + 107.99999999999923*helper_2*x + helper_4*x*(helper_3 + 53.999999999999623) + helper_4*(helper_3*x + 53.999999999999623*x + 80.999999999999673*y - 1.1241008124329651e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 53.999999999999616*x;
double helper_4 = helper_0*x;
double helper_5 = 54.0*y;
val[1] = -(helper_1*(helper_0*helper_5 + 26.999999999999861*helper_1 + 53.999999999999623*helper_2 + 80.999999999999503*helper_4 + helper_5 + 161.99999999999935*x*y + 80.999999999999403*x + 26.999999999999766*z - 26.999999999999854) + 107.99999999999923*helper_2*y + helper_4*y*(helper_3 + 80.999999999999673) + helper_4*(helper_3*y + 53.999999999999623*x + 80.999999999999673*y - 1.1241008124329651e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*z;
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = 5.3568260938164188e-14*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = 27.0*helper_6;
double helper_8 = pow(z, 4);
double helper_9 = 80.999999999999673*helper_6;
double helper_10 = helper_0*x;
double helper_11 = 53.999999999999623*helper_4;
double helper_12 = 1.607047828144904e-13*helper_4;
double helper_13 = helper_1*y;
double helper_14 = 81.0*helper_6;
double helper_15 = helper_4*helper_6;
val[2] = (-helper_0*helper_12 + helper_0*helper_14 - 242.99999999999903*helper_0*y - 1.362410184668753e-12*helper_0 + helper_1*helper_5 - helper_1*helper_7 - 1.6481260800560406e-13*helper_1*x + 1.3654077868352411e-12*helper_1 + 242.99999999999852*helper_10*y + 2.3131496718065e-13*helper_10 + helper_11*helper_3 - helper_11*y + helper_12*z - 80.999999999999503*helper_13*x + 188.99999999999915*helper_13 - helper_14*z + 53.999999999999616*helper_15*z + 53.999999999999631*helper_15 + helper_2*helper_9 - 1.4360734823526299e-13*helper_2 - 242.99999999999864*helper_3*x + 134.9999999999996*helper_3 - helper_5 + helper_7 + 4.385380947269345e-14*helper_8*x - 53.999999999999723*helper_8*y - 6.954437026251922e-13*helper_8 - helper_9*x + 80.999999999999616*x*y + 3.3251179587523571e-14*x - 26.999999999999957*y + 1.4388490399141966e-13*pow(z, 5) + 6.9094729937546294e-13*z - 1.4238610290817532e-13)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 511.31249999999761*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = -(303.74999999999903*helper_0*helper_2*helper_3 + helper_1*(86.062499999998252*helper_0*y + 2.4280577548552108e-13*helper_2 + 207.56249999999881*helper_3 - 3.491738148619903e-13*helper_4 + 415.12499999999693*helper_6 - 4.3165471197425935e-13*helper_7 - 6.6097127771058472e-13*x + 5.0624999999978639*y - 4.1479319978776499e-13*z + 3.4706612583867904e-13) + 546.74999999999852*helper_2*pow(y, 3) + 2*helper_3*helper_7*(303.74999999999903*x + 303.74999999999858*y + 5.0624999999998135) + helper_4*helper_6*(helper_5 + 242.99999999999952*x + 5.0624999999990248) + helper_4*y*(121.49999999999976*helper_2 + 121.49999999999909*helper_3 + helper_5*x + 5.0624999999990248*x + 5.0625000000004343*y - 5.6823296068486381e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 511.31249999999761*x;
double helper_7 = x*y;
val[1] = -(helper_1*(86.062499999998252*helper_0*x + 3.8219427622720873e-13*helper_0*y + 6.8289124355302862e-14*helper_2 + 207.56249999999847*helper_3 - 9.3862417838152614e-13*helper_5 + 415.12499999999761*helper_7 + 5.0624999999978639*x + 2.2032375923686158e-13*y - 1.4366008382893305e-12*z + 8.0654233292065356e-13) + 303.74999999999858*helper_2*helper_4 + 546.74999999999852*helper_2*pow(x, 3) + 2*helper_4*y*(303.74999999999903*x + 303.74999999999858*y + 5.0624999999998135) + helper_5*helper_7*(helper_6 + 242.99999999999818*y + 5.0625000000004343) + helper_5*x*(121.49999999999909*helper_2 + 121.49999999999976*helper_3 + helper_6*y + 5.0624999999990248*x + 5.0625000000004343*y - 5.6823296068486381e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = 2.1582735598712947e-13*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = 1.9109713811360459e-13*helper_8;
double helper_10 = pow(z, 5);
double helper_11 = 5.0625000000003944*helper_8*x;
double helper_12 = pow(y, 3);
double helper_13 = helper_12*x;
double helper_14 = 121.49999999999909*helper_13;
double helper_15 = helper_1*x;
double helper_16 = helper_6*y;
double helper_17 = 8.6330942394851789e-13*helper_6;
double helper_18 = pow(x, 3);
double helper_19 = 121.49999999999974*helper_18*y;
double helper_20 = helper_0*y;
double helper_21 = helper_8*z;
double helper_22 = helper_6*helper_8;
double helper_23 = 607.49999999999704*helper_12*helper_6;
double helper_24 = 607.49999999999795*helper_18;
val[2] = (helper_0*helper_11 + helper_0*helper_14 + 5.0624999999991207*helper_0*helper_16 + helper_0*helper_19 + 511.3124999999975*helper_0*helper_22 + 1.2949641359227788e-12*helper_0*helper_6 - 1.1465828286816276e-12*helper_0*helper_8 - 4.4947170985132176e-12*helper_0*x + 1.0875113309882729e-11*helper_0 - helper_1*helper_17 + 7.6438855245441705e-13*helper_1*helper_8 + 1.3026080214473205e-11*helper_1*y - 1.814635941510537e-11*helper_1 + 6.9834762972398081e-13*helper_10*x + 1.8772483567630511e-12*helper_10*y - 7.9631301552751342e-12*helper_10 + helper_11 + 546.74999999999852*helper_12*helper_18 - 242.99999999999818*helper_13*z + helper_14 + 344.24999999999295*helper_15*y + 5.3243034980887461e-12*helper_15 + 5.062499999999007*helper_16 - helper_17*z - 242.99999999999949*helper_18*helper_5 + helper_19 - 86.062499999998238*helper_2*helper_3 + helper_2*helper_7 - helper_2*helper_9 - 3.0769449488321373e-12*helper_2*x - 7.9496409455259305e-12*helper_2*y + 1.6620673587430733e-11*helper_2 - 516.37499999999011*helper_20*x - 1.0152878537894542e-11*helper_20 + helper_21*helper_24 + 7.6438855245441836e-13*helper_21 - 1012.4999999999953*helper_22*z + 501.18749999999784*helper_22 + helper_23*z - helper_23 - helper_24*helper_8 - 86.062499999998806*helper_3 - 10.125000000000789*helper_4*helper_8 + 344.24999999999409*helper_4*y + 1.8325653494688413e-12*helper_4 - 10.124999999998241*helper_5*helper_6 + 3.6398384306579391e-12*helper_5 + helper_7 - helper_9 - 2.8355442993621527e-13*x - 4.4064751847372236e-13*y + 1.5647483309066892e-12*pow(z, 6) - 3.3666819332367316e-12*z + 4.1563627539708815e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 399.9374999999971*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (242.99999999999832*helper_0*helper_2*helper_3 + helper_1*(35.437499999998778*helper_0*y + 2.4280577548552047e-13*helper_2 + 156.93749999999892*helper_3 - 2.9999440431804727e-13*helper_4 + 192.37499999999713*helper_6 - 3.209307819496108e-13*helper_7 - 5.5699195256053349e-13*x - 5.0625000000013829*y - 3.6111738599408966e-13*z + 2.9732466488851926e-13) + 546.74999999999716*helper_2*pow(y, 3) + 2*helper_3*helper_7*(242.99999999999832*x + 303.74999999999801*y - 5.0624999999998597) + helper_4*helper_6*(helper_5 + 121.49999999999902*x - 5.0625000000005933) + helper_4*y*(60.74999999999951*helper_2 + 121.49999999999895*helper_3 + helper_5*x - 5.0625000000005933*x - 5.0624999999993054*y - 4.1310704856911351e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 399.9374999999971*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (303.74999999999801*helper_0*helper_2*helper_3 + helper_1*(35.437499999998778*helper_0*x + 3.4144562177651242e-14*helper_2 + 96.187499999998565*helper_3 - 6.531025720235507e-13*helper_4 + 313.87499999999784*helper_6 + 5.0809356721970027e-13*helper_7 - 5.0625000000013829*x + 3.4622305022935345e-13*y - 1.0656475701864488e-12*z + 5.5924015418539934e-13) + 546.74999999999716*helper_2*pow(x, 3) + 2*helper_3*helper_7*(242.99999999999832*x + 303.74999999999801*y - 5.0624999999998597) + helper_4*helper_6*(helper_5 + 242.9999999999979*y - 5.0624999999993054) + helper_4*x*(121.49999999999895*helper_2 + 60.74999999999951*helper_3 + helper_5*y - 5.0625000000005933*x - 5.0624999999993054*y - 4.1310704856911351e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*z;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = 1.6046539097480547e-13*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = 2.5404678360985023e-13*helper_7;
double helper_9 = pow(z, 5);
double helper_10 = 5.0624999999992806*helper_7;
double helper_11 = pow(y, 3);
double helper_12 = helper_11*x;
double helper_13 = 121.49999999999892*helper_12;
double helper_14 = helper_0*x;
double helper_15 = helper_2*x;
double helper_16 = 6.418615638992219e-13*helper_5;
double helper_17 = pow(x, 3);
double helper_18 = helper_17*y;
double helper_19 = 60.749999999999503*helper_18;
double helper_20 = helper_0*y;
double helper_21 = helper_1*y;
double helper_22 = 1.0161871344394001e-12*helper_7;
double helper_23 = helper_5*helper_7;
double helper_24 = 607.49999999999613*helper_11*helper_5;
double helper_25 = helper_0*helper_5;
double helper_26 = 485.9999999999967*helper_17*helper_7;
val[2] = -(helper_0*helper_13 + helper_0*helper_19 - 1.5242807016591008e-12*helper_0*helper_7 + 8.4009674217177522e-12*helper_0 - helper_1*helper_16 + helper_1*helper_22 + 4.5554185423845859e-12*helper_1*x - 1.4443571338951147e-11*helper_1 - helper_10*helper_14 - helper_10*x + 546.74999999999716*helper_11*helper_17 - 242.99999999999784*helper_12*z + helper_13 - 3.8331837703964069e-12*helper_14 - 35.437499999998778*helper_15*y - 2.6388266571863816e-12*helper_15 - helper_16*z - 121.49999999999901*helper_18*z + helper_19 + helper_2*helper_6 - helper_2*helper_8 - 5.4653781500490581e-12*helper_2*y + 1.3514220992272193e-11*helper_2 - 212.62499999999307*helper_20*x - 6.6681660193523226e-12*helper_20 + 141.74999999999523*helper_21*x + 8.7994611597252194e-12*helper_21 + helper_22*z - 809.99999999999409*helper_23*z + 410.06249999999687*helper_23 + helper_24*z - helper_24 + 399.93749999999716*helper_25*helper_7 - 5.062500000000643*helper_25*y + 9.6279234584883279e-13*helper_25 + helper_26*z - helper_26 + 10.124999999998561*helper_3*helper_7 + 141.74999999999588*helper_3*y + 1.5554744992041127e-12*helper_3 + 10.125000000001286*helper_4*helper_5 + 2.2684354394897108e-12*helper_4 - 5.0625000000005862*helper_5*y + helper_6 - helper_8 + 5.9998880863609494e-13*helper_9*x + 1.3062051440471014e-12*helper_9*y - 6.578237954357701e-12*helper_9 - 35.43749999999919*x*y - 2.3887142264200513e-13*x - 2.405575738606515e-13*y + 1.3084533456719687e-12*pow(z, 6) - 2.4876350979141264e-12*z + 2.8580263156108131e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 399.9374999999979*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = (303.74999999999898*helper_0*helper_2*helper_3 + helper_1*(35.437499999998636*helper_0*y + 2.4280577548552103e-13*helper_2 + 96.187499999998963*helper_3 - 3.1011131162994466e-13*helper_4 + 313.87499999999739*helper_6 - 3.5200314885130858e-13*helper_7 - 5.8131971458763395e-13*x - 5.0625000000016547*y - 3.8126099499713335e-13*z + 3.1303432068696604e-13) + 546.74999999999841*helper_2*pow(y, 3) + 2*helper_3*helper_7*(303.74999999999898*x + 242.99999999999869*y - 5.0625000000000062) + helper_4*helper_6*(helper_5 + 242.99999999999946*x - 5.0625000000007505) + helper_4*y*(121.49999999999973*helper_2 + 60.749999999999176*helper_3 + helper_5*x - 5.0625000000007505*x - 5.0624999999995346*y - 4.271470330219324e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 399.9374999999979*x;
double helper_6 = x*y;
double helper_7 = helper_0*y;
val[1] = (242.99999999999869*helper_0*helper_2*helper_3 + helper_1*(35.437499999998636*helper_0*x + 6.8289124355302849e-14*helper_2 + 156.93749999999869*helper_3 - 8.1250284278410876e-13*helper_4 + 192.37499999999793*helper_6 + 2.9792834865815919e-13*helper_7 - 5.0625000000016547*x + 1.7652546091539903e-13*y - 1.1813883205036234e-12*z + 6.6199216247619667e-13) + 546.74999999999841*helper_2*pow(x, 3) + 2*helper_3*helper_7*(303.74999999999898*x + 242.99999999999869*y - 5.0625000000000062) + helper_4*helper_6*(helper_5 + 121.49999999999835*y - 5.0624999999995346) + helper_4*x*(60.749999999999176*helper_2 + 121.49999999999973*helper_3 + helper_5*y - 5.0625000000007505*x - 5.0624999999995346*y - 4.271470330219324e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = 1.7600157442565361e-13*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = 1.4896417432907939e-13*helper_8;
double helper_10 = pow(z, 5);
double helper_11 = 5.0624999999995133*helper_8;
double helper_12 = pow(y, 3);
double helper_13 = helper_12*x;
double helper_14 = 60.749999999999176*helper_13;
double helper_15 = helper_0*x;
double helper_16 = helper_1*x;
double helper_17 = 7.0400629770261816e-13*helper_6;
double helper_18 = pow(x, 3);
double helper_19 = helper_18*y;
double helper_20 = 121.49999999999974*helper_19;
double helper_21 = 5.9585669731631818e-13*helper_8;
double helper_22 = helper_6*helper_8;
double helper_23 = 485.99999999999744*helper_12*helper_6;
double helper_24 = helper_0*helper_6;
double helper_25 = 607.49999999999795*helper_18*helper_8;
val[2] = -(helper_0*helper_14 + helper_0*helper_20 + 399.93749999999784*helper_0*helper_22 - 8.9378504597447803e-13*helper_0*helper_8 - 9.1617269326604371e-12*helper_0*y + 9.4310323439649805e-12*helper_0 - helper_1*helper_17 + helper_1*helper_21 + 1.1524503573667677e-11*helper_1*y - 1.6227366672616406e-11*helper_1 + 6.2022262325988993e-13*helper_10*x + 1.6250056855682177e-12*helper_10*y - 7.3515082910091355e-12*helper_10 - helper_11*helper_15 - helper_11*x + 546.74999999999841*helper_12*helper_18 - 121.49999999999835*helper_13*z + helper_14 - 212.62499999999218*helper_15*y - 3.9146602626160936e-12*helper_15 + 141.7499999999946*helper_16*y + 4.6771822526103596e-12*helper_16 - helper_17*z - 242.99999999999949*helper_19*z - 35.437499999998636*helper_2*helper_3 + helper_2*helper_7 - helper_2*helper_9 - 2.7198521213023124e-12*helper_2*x - 6.9436401073374662e-12*helper_2*y + 1.5153104465648255e-11*helper_2 + helper_20 + helper_21*z - 809.99999999999568*helper_22*z + 410.06249999999784*helper_22 + helper_23*z - helper_23 - 5.0625000000006608*helper_24*y + 1.0560094465539263e-12*helper_24 + helper_25*z - helper_25 - 35.437499999999048*helper_3 + 10.124999999999027*helper_4*helper_8 + 141.7499999999954*helper_4*y + 1.5760691363109132e-12*helper_4 + 10.125000000001549*helper_5*helper_6 + 3.3994751458265931e-12*helper_5 - 5.0625000000007745*helper_6*y + helper_7 - helper_9 - 2.3896162826275501e-13*x - 4.4361736506459597e-13*y + 1.4565015860057356e-12*pow(z, 6) - 2.7738505936625152e-12*z + 3.1208716166908119e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(helper_0, 2);
double helper_5 = 329.06249999999727*y;
double helper_6 = x*y;
double helper_7 = helper_0*x;
val[0] = -(242.99999999999838*helper_0*helper_2*helper_3 + helper_1*(25.312499999998877*helper_0*y + 2.4280577548552052e-13*helper_2 + 86.062499999998906*helper_3 - 2.8004161489736208e-13*helper_4 + 172.12499999999724*helper_6 - 2.9676261448230202e-13*helper_7 - 5.3282378509322472e-13*x + 5.0624999999987423*y - 3.4397484860448695e-13*z + 2.8467853074864839e-13) + 546.74999999999727*helper_2*pow(y, 3) + 2*helper_3*helper_7*(242.99999999999838*x + 242.99999999999815*y + 5.0625000000000924) + helper_4*helper_6*(helper_5 + 121.49999999999908*x + 5.0624999999994031) + helper_4*y*(60.749999999999538*helper_2 + 60.749999999999062*helper_3 + helper_5*x + 5.0624999999994031*x + 5.0625000000005231*y - 3.3554409251123861e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(x, 2);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(helper_0, 2);
double helper_6 = 329.06249999999727*x;
double helper_7 = x*y;
val[1] = -(helper_1*(25.312499999998877*helper_0*x + 4.0917269572559897e-13*helper_0*y + 3.4144562177651248e-14*helper_2 + 86.062499999998622*helper_3 - 6.0813853952623206e-13*helper_5 + 172.12499999999781*helper_7 + 5.0624999999987423*x + 2.8776980798283906e-13*y - 9.5098928731828619e-13*z + 4.9966281112645218e-13) + 242.99999999999815*helper_2*helper_4 + 546.74999999999727*helper_2*pow(x, 3) + 2*helper_4*y*(242.99999999999838*x + 242.99999999999815*y + 5.0625000000000924) + helper_5*helper_7*(helper_6 + 121.49999999999812*y + 5.0625000000005231) + helper_5*x*(60.749999999999062*helper_2 + 60.749999999999538*helper_3 + helper_6*y + 5.0624999999994031*x + 5.0625000000005231*y - 3.3554409251123861e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = 1.4838130724114917e-13*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = 2.0458634786279913e-13*helper_6;
double helper_8 = pow(z, 5);
double helper_9 = helper_6*x;
double helper_10 = pow(y, 3);
double helper_11 = helper_10*x;
double helper_12 = 60.749999999999062*helper_11;
double helper_13 = helper_0*x;
double helper_14 = helper_1*x;
double helper_15 = helper_2*x;
double helper_16 = helper_4*y;
double helper_17 = 5.9352522896460434e-13*helper_4;
double helper_18 = pow(x, 3);
double helper_19 = 60.749999999999531*helper_18*y;
double helper_20 = helper_6*z;
double helper_21 = helper_4*helper_6;
double helper_22 = 485.99999999999625*helper_10*helper_4;
double helper_23 = 485.9999999999967*helper_18;
val[2] = (helper_0*helper_12 + 5.06249999999946*helper_0*helper_16 + helper_0*helper_19 + 329.06249999999727*helper_0*helper_21 + 8.9028784344690647e-13*helper_0*helper_4 - 1.2275180871767971e-12*helper_0*helper_6 + 5.0625000000005471*helper_0*helper_9 - 6.4568350666149256e-12*helper_0*y + 7.9816778186802508e-12*helper_0 - helper_1*helper_17 + 8.1834539145119905e-13*helper_1*helper_6 + 8.358813641251498e-12*helper_1*y - 1.3924236763607115e-11*helper_1 + 546.74999999999727*helper_10*helper_18 - 121.49999999999812*helper_11*z + helper_12 - 151.87499999999349*helper_13*y - 3.536983206320319e-12*helper_13 + 101.24999999999551*helper_14*y + 4.2249329035292924e-12*helper_14 - 25.312499999998877*helper_15*y - 2.4564413003691337e-12*helper_15 - 10.12499999999892*helper_16*z + 5.0624999999994031*helper_16 - helper_17*z - 121.49999999999906*helper_18*helper_3 + helper_19 + helper_2*helper_5 - helper_2*helper_7 - 5.1303961079440344e-12*helper_2*y + 1.3137085169700937e-11*helper_2 + helper_20*helper_23 + 8.1834539145119652e-13*helper_20 - 647.99999999999432*helper_21*z + 318.9374999999971*helper_21 + helper_22*z - helper_22 - helper_23*helper_6 + 101.24999999999622*helper_3*x + 2.2774282459891758e-12*helper_3 + helper_5 - helper_7 + 5.6008322979472396e-13*helper_8*x + 1.2162770790524633e-12*helper_8*y - 6.4253602438668154e-12*helper_8 - 10.125000000001094*helper_9*z + 5.0625000000005187*helper_9 - 25.312499999999222*x*y + 1.4245167545556748e-12*x*z - 2.1610838119023739e-13*x - 2.6528779173417838e-13*y + 1.2814749261735778e-12*pow(z, 6) - 2.3010343630502726e-12*z + 2.5039345596944166e-13)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_3_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(y, 2);
double helper_3 = 127.99999999999916*y;
double helper_4 = x*y;
double helper_5 = helper_0*y;
double helper_6 = 2.1316282072802862e-13*x;
val[0] = (helper_0*helper_4*(helper_3 + 127.99999999999915) + helper_1*(-helper_0*helper_6 - 6.0396132539608125e-14*helper_1 + 127.99999999999925*helper_2 + 255.99999999999829*helper_4 + 127.99999999999891*helper_5 - helper_6 + 127.99999999999872*y - 4.973799150320671e-14*z + 7.4162898044960078e-14) + 255.99999999999832*helper_2*x + helper_5*(helper_3*x + 127.99999999999915*x + 127.99999999999925*y - 2.045030811359525e-13))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 2);
double helper_2 = pow(x, 2);
double helper_3 = 127.99999999999916*x;
double helper_4 = helper_0*x;
double helper_5 = 8.5265128291211265e-14*y;
val[1] = (helper_1*(-helper_0*helper_5 - 3.1263880373444201e-13*helper_1 + 127.99999999999915*helper_2 + 127.99999999999891*helper_4 - helper_5 + 255.99999999999849*x*y + 127.99999999999872*x - 5.2580162446247081e-13*z + 3.4039437935007077e-13) + 255.99999999999832*helper_2*y + helper_4*y*(helper_3 + 127.99999999999925) + helper_4*(helper_3*y + 127.99999999999915*x + 127.99999999999925*y - 2.045030811359525e-13))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = x*y;
double helper_3 = x*z;
double helper_4 = pow(x, 2);
double helper_5 = 1.0658141036401351e-13*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = 4.2632564145604799e-14*helper_6;
double helper_8 = pow(z, 4);
double helper_9 = 127.99999999999915*helper_4*y;
double helper_10 = 3.1974423109204231e-13*helper_4;
double helper_11 = helper_0*y;
double helper_12 = 1.2789769243681768e-13*helper_6;
double helper_13 = helper_4*helper_6;
val[2] = -(-helper_0*helper_10 - helper_0*helper_12 + 5.755396159656764e-13*helper_0*x - 2.7284841053187645e-12*helper_0 - 127.99999999999889*helper_1*helper_2 + helper_1*helper_5 + helper_1*helper_7 - 4.3343106881365824e-13*helper_1*x - 1.9753088054130648e-12*helper_1*y + 2.8990143619011914e-12*helper_1 + helper_10*z + 383.9999999999967*helper_11*x + 2.174260771425892e-12*helper_11 + helper_12*z + 127.99999999999916*helper_13*z + 127.99999999999918*helper_13 - 383.99999999999699*helper_2*z + 127.99999999999912*helper_2 + 127.99999999999923*helper_3*helper_6 - 3.3395508580724441e-13*helper_3 - helper_5 - 127.99999999999925*helper_6*x - helper_7 + 1.20792265079216e-13*helper_8*x + 6.2527760746888433e-13*helper_8*y - 1.5631940186722097e-12*helper_8 + helper_9*z - helper_9 + 7.1054273576009009e-14*x - 9.2370555648812338e-13*y*z + 9.9475983006413559e-14*y + 3.4106051316484511e-13*pow(z, 5) + 1.3073986337985753e-12*z - 2.5579538487363475e-13)/(-3.0*helper_0 + 1.0*helper_1 + 3.0*z - 1.0);}
}



POLYFEM_BOTH void pyramid_3_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_20(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_21(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_22(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_23(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_24(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_25(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_26(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_27(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_28(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_3_basis_grad_value_3d_single_29(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_0(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_10*x;
double helper_14 = helper_9*y;
double helper_15 = helper_0*y;
result = (helper_1*(47.999999999998813*helper_0*helper_10 + 428.20629715364714*helper_0*helper_13 + 42.666666666667389*helper_0*helper_4 + 209.94102223365456*helper_0*helper_8 + 48.000000000000668*helper_0*helper_9 + 14.666666666666524*helper_0*x + 10.666666666666666*helper_1 + 63.999999999999396*helper_10*helper_7 + 7.3333333333332984*helper_10 + 919.32512431419673*helper_11 + 42.666666666666671*helper_12*x + 42.666666666666195*helper_12*y + 16.0*helper_12 + 263.27512717193429*helper_13 + 263.27512717192877*helper_14 + 428.20629715364294*helper_15*helper_9 + 14.666666666666892*helper_15 + 10.666666666666622*helper_2 + 10.66666666666665*helper_3 + 323.55632716050559*helper_4*y + 16.000000000000686*helper_4 + 323.55632716050701*helper_5*x + 15.999999999999854*helper_5 + 42.666666666666487*helper_6 + 236.20552554869772*helper_7*helper_8 + 64.000000000000853*helper_7*helper_9 + 47.999999999999886*helper_7*x + 47.999999999998963*helper_7*y + 7.3333333333333321*helper_7 + 44.179941129402209*helper_8 + 7.3333333333331918*helper_9 + 1.0*x + 1.0*y + 1.0*z - 1.0) + helper_11*helper_7*(248.8888888888961*helper_10 + 807.56250000007196*helper_8 + 248.88888888890159*helper_9 + 181.78549382720408*x + 181.78549382721039*y + 22.179941129401165) + helper_12*helper_8*(101.3341049382815*helper_10 + 804.0077160494543*helper_13 + 804.00771604945442*helper_14 + 88.888888888892623*helper_4 + 88.888888888889767*helper_5 + 397.06062099913544*helper_8 + 101.3341049382753*helper_9 + 29.51327446273574*x + 29.513274462735268*y + 1.0) + 113.7777777777808*helper_2*helper_3 + helper_4*helper_6*(284.44444444445674*x + 284.4444444444531*y + 96.451388888930992))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_1(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_10*x;
double helper_14 = helper_9*y;
double helper_15 = helper_0*x;
double helper_16 = helper_7*x;
double helper_17 = helper_0*helper_9;
result = (helper_1*(1.3033310856030047e-13*helper_0*helper_10 + 3.2899746979470883e-13*helper_0*helper_4 - 1.0068469025672859e-13*helper_0*y - 6.0236411179779417*helper_10*helper_15 + 2.6732128741971971e-13*helper_10*helper_7 - 2.1777654268409629e-14*helper_10 + 0.63530949926462199*helper_11 - 7.647915603665759e-14*helper_12*y - 14.510366655242242*helper_13 - 40.72718764290012*helper_14 + 12.606917295384916*helper_15*y + 3.4540271877191976e-15*helper_15 + 5.0936428326485643*helper_16*y + 3.0592812234082594e-15*helper_16 - 9.1293509945225182*helper_17*y + 2.2816316737187219e-13*helper_17 + 10.666666666666623*helper_2 - 9.4739031434680256e-15*helper_3 - 0.45061728395837974*helper_4*x + 3.5209010870691113e-13*helper_4 + 31.99922839505389*helper_5*y - 15.999999999999909*helper_5 + 1.1684480543612302e-13*helper_6 + 2.3605808665809883e-13*helper_7*helper_9 - 3.1605443618517131e-13*helper_7*y + 14.84660779607055*helper_8 + 7.3333333333333357*helper_9 - 1.0*x) + helper_11*helper_7*(78.222222222226705*helper_10 + 330.21527777776021*helper_8 + 248.88888888889772*helper_9 - 181.78549382721192*x - 107.56867283955444*y + 22.179941129403684) + helper_12*helper_8*(-27.117283950627051*helper_10 + 87.986882716015629*helper_13 + 191.54783950614075*helper_14 + 10.6666666666671*helper_4 + 88.888888888891515*helper_5 - 148.29586048245488*helper_8 - 101.33410493828517*helper_9 + 29.51327446273752*x + 14.846607796070508*y - 1.0) + 113.77777777777922*helper_2*helper_3 + helper_4*helper_6*(284.44444444445236*x + 170.66666666667143*y - 96.451388888925962))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_2(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(x, 2);
double helper_8 = x*y;
double helper_9 = pow(y, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_9*x;
double helper_14 = helper_7*y;
double helper_15 = helper_0*y;
result = (helper_1*(59.328210733877711*helper_0*helper_13 + 59.328210733873618*helper_0*helper_14 + 7.1843765504638054e-14*helper_0*helper_4 + 1.0342344264953093e-13*helper_0*helper_7 - 2.7227177417081992e-14*helper_0*helper_9 - 1.1842378929348466e-15*helper_0*x + 1.5474041800998336e-13*helper_10*helper_7 + 26.877593449929265*helper_10*helper_8 - 2.8619082412572395e-15*helper_10*x - 1.9076957996458532e-13*helper_10*y + 1.4615449340038917e-13*helper_11 - 5.3290705182007514e-15*helper_12*x - 2.6525030838138875e-14*helper_12*y + 66.841485196611174*helper_13 + 66.841485196606754*helper_14 + 39.724201245997087*helper_15*x + 5.2924131222120068e-14*helper_15 - 1.5000346643824735e-14*helper_2 - 1.9283266572069033e-29*helper_3 + 43.11728395061229*helper_4*y + 4.776426168165703e-14*helper_4 + 43.117283950616155*helper_5*x + 1.7149528634559406e-13*helper_5 + 1.7149528634559446e-13*helper_6 + 206.00799468448565*helper_7*helper_9 - 1.1842378929348466e-15*helper_7 + 14.846607796068016*helper_8 + 2.9005904331359546e-14*helper_9) + helper_11*helper_7*(78.222222222224332*helper_7 + 352.45138888888886*helper_8 + 78.222222222223422*helper_9 + 107.5686728394939*x + 107.5686728394987*y + 22.17994112940077) + helper_12*helper_8*(224.90200617283739*helper_13 + 224.90200617283313*helper_14 + 10.666666666667503*helper_4 + 10.666666666666888*helper_5 + 27.11728395061024*helper_7 + 174.41015803610387*helper_8 + 27.117283950615121*helper_9 + 14.846607796068218*x + 14.846607796067751*y + 1.0) + 113.77777777777705*helper_2*helper_3 + helper_4*helper_6*(170.66666666666742*x + 170.66666666666657*y + 96.451388888884509))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_3(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_10*x;
double helper_13 = helper_9*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = helper_0*x;
double helper_17 = helper_7*x;
double helper_18 = helper_0*helper_9;
result = (helper_1*(-9.5355979057079837e-14*helper_0*helper_10 - 3.4737644859374424e-14*helper_0*helper_4 - 2.1005095287349963e-13*helper_0*y - 9.1293509945063498*helper_10*helper_16 + 2.496142581611986e-13*helper_10*helper_7 + 7.333333333333206*helper_10 + 0.63530949930768088*helper_11 - 40.727187642883862*helper_12 - 14.510366655235416*helper_13 - 2.0267630547487718e-13*helper_14*y + 3.5527136788005009e-15*helper_15 + 12.606917295385802*helper_16*y + 2.9235872981793912e-14*helper_16 + 5.0936428326509748*helper_17*y + 3.2665228546747284e-14*helper_17 - 6.0236411179684168*helper_18*y + 2.881645539472296e-14*helper_18 - 1.5000346643824953e-14*helper_2 + 10.666666666666666*helper_3 - 0.45061728395117484*helper_4*y - 6.2369862361156025e-14*helper_4 + 31.99922839506614*helper_5*x - 15.999999999999561*helper_5 + 4.4064555768889643e-13*helper_6 + 5.5264435003572239e-14*helper_7*helper_9 - 5.5422884265125837e-13*helper_7*y + 14.846607796068492*helper_8 + 2.9235872981793912e-14*helper_9 - 1.0*y) + helper_11*helper_7*(248.88888888889079*helper_10 + 330.21527777777135*helper_8 + 78.222222222225511*helper_9 - 107.56867283952414*x - 181.78549382717242*y + 22.179941129401517) + helper_15*y*(-101.33410493826891*helper_10 + 191.54783950616877*helper_12 + 87.986882716040498*helper_13 + 10.666666666667936*helper_4 + 88.888888888889255*helper_5 - 148.29586048240884*helper_8 - 27.117283950620894*helper_9 + 14.846607796068785*x + 29.513274462734927*y - 1.0) + 113.77777777777709*helper_2*helper_3 + helper_4*helper_6*(170.66666666666822*x + 284.44444444444491*y - 96.451388888902457))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_4(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 1.9767372530011187e-27*helper_2;
double helper_13 = 3.1178763274888675e-14*helper_3;
double helper_14 = 4.9418431325027968e-28*x;
double helper_15 = helper_0*y;
result = (helper_0*helper_7*(1.5527832046920039e-12*helper_4 + 2.225608289006302e-12*x + 2.1560283287910506e-12*y + 20.590277777775473) + helper_1*(10.666666666666666*pow(helper_0, 4) + helper_0*helper_12 - helper_0*helper_13 - helper_0*helper_14 + 55.694444444438787*helper_0*helper_4 - 4.6768144912332542e-14*helper_0*helper_5 + 1.4825529397508378e-27*helper_0*helper_6 + 35.104166666663886*helper_0*helper_8 - 1.5589381637443429e-14*helper_1*y + 26.666666666666668*helper_1 - helper_10*helper_14 - 4.6768144912332107e-14*helper_10*helper_5 + 1.9767372530011187e-27*helper_10*helper_6 - 1.9486727046804272e-14*helper_10*y + 23.333333333333332*helper_10 + 35.104166666663318*helper_11 + helper_12 - helper_13 + 35.104166666664014*helper_15*helper_6 + 3.8973454093627417e-15*helper_15 + 6.7949259036774496e-13*helper_2*y + 5.7206636082416129e-13*helper_3*x + 20.590277777775469*helper_4 + 6.5523779064093092e-28*helper_5 - 4.9418431325027968e-28*helper_6 + 35.104166666666167*helper_7 + 55.694444444439384*helper_8 + 55.69444444443949*helper_9 + 8.3333333333333321*z - 7.3333333333333321) + helper_11*(55.69444444444165*helper_4 + 5.7206636082416129e-13*helper_5 + 6.7949259036774496e-13*helper_6 + 2.1560283287910506e-12*helper_8 + 2.225608289006302e-12*helper_9 + 20.590277777775476*x + 20.59027777777548*y) + 1.5527832046920039e-12*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_5(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = helper_0*x;
double helper_15 = -helper_0;
result = -(helper_0*helper_4*helper_5*(1137.7777777778233*x + 1024.0000000000314*y + 161.77777777780852) + helper_1*(897.13305898491092*helper_0*helper_13 - 2.3206331689390451e-12*helper_0*helper_9 + 7.2438869412868359e-13*helper_0*y + 2135.1083676269554*helper_10 + 42.666666666666693*helper_11*x - 1.0477450335469493e-12*helper_11*y + 212.86053955191807*helper_12 + 335.87288523090382*helper_13 + 575.00960219480862*helper_14*helper_9 + 149.99634202105338*helper_14*y + 5.3333333333329165*helper_14 - 128.00000000000225*helper_15*helper_4 - 1.4529362319279368e-13*helper_15*helper_5 - 64.000000000002188*helper_15*helper_8 + 42.666666666666494*helper_2 - 5.6843418860808116e-14*helper_3 + 985.87654320988327*helper_4*y + 32.000000000002103*helper_4 + 521.53086419754936*helper_5*x + 2.6608588827200688e-13*helper_5 + 266.81207133060059*helper_6*helper_7 + 128.00000000000279*helper_6*helper_8 - 8.2902866099228005e-13*helper_6*helper_9 + 31.999999999999613*helper_6*x - 2.7429456359049438e-12*helper_6*y + 16.517604023790135*helper_7 + 5.3333333333329165*helper_8 + 6.0015778355252574e-14*helper_9) + helper_10*helper_6*(2721.7777777779193*helper_7 + 995.55555555560318*helper_8 + 739.55555555558215*helper_9 + 315.6543209876541*x + 256.64197530867182*y + 11.184270690453683) + helper_11*helper_7*(2105.5308641976417*helper_12 + 2555.6543209877591*helper_13 + 355.55555555556964*helper_4 + 170.66666666666987*helper_5 + 528.51486053956387*helper_7 + 185.87654320985271*helper_8 + 94.864197530870626*helper_9 + 16.517604023791243*x + 11.184270690455788*y) + 455.11111111112183*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_6(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_10*x;
double helper_13 = helper_9*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = helper_0*y;
double helper_17 = helper_7*y;
double helper_18 = helper_0*helper_10;
result = (helper_1*(128.00000000000253*helper_0*helper_4 - 15.999999999997325*helper_0*helper_9 - 4.0000000000004237*helper_0*x + 3.0649282837353259e-13*helper_10*helper_7 + 2.1506664761836465e-13*helper_10 + 1699.8662551439338*helper_11 - 82.604252400551928*helper_12 - 67.900548696886219*helper_13 - 8.3282991842643894e-13*helper_14*y + 1.4210854715202004e-14*helper_15 + 578.56995884770049*helper_16*helper_9 - 67.900548696816301*helper_16*x + 7.4573326480586356e-13*helper_16 + 45.236625514415714*helper_17*x - 2.7637447657369339e-12*helper_17 + 206.53292181069202*helper_18*x - 1.1615262935284924e-12*helper_18 + 63.999999999999716*helper_2 - 7.5791225147744167e-14*helper_3 + 1066.6666666666408*helper_4*y + 2.3234747459356504e-12*helper_4 + 289.29629629628738*helper_5*x + 1.2103770203759776e-12*helper_5 + 1.0493206669370273e-12*helper_6 + 64.000000000003268*helper_7*helper_9 - 16.000000000000416*helper_7*x - 13.137174211226265*helper_8 - 4.0000000000004237*helper_9) + helper_11*helper_7*(810.66666666670278*helper_10 + 3413.3333333334008*helper_8 + 1493.3333333333992*helper_9 - 1.6090477098866693e-10*x - 30.703703703822818*y - 9.1371742112297678) - helper_15*y*(30.703703703728884*helper_10 - 1995.9629629629549*helper_12 - 2986.6666666666506*helper_13 - 533.3333333333527*helper_4 - 128.00000000000418*helper_5 + 82.60425240072189*helper_8 + 6.815525921411155e-11*helper_9 + 13.137174211224409*x + 9.1371742112275136*y) + 682.66666666668073*helper_2*helper_3 + helper_4*helper_6*(1706.666666666729*x + 1365.3333333333751*y - 8.4005099173134069e-11))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_7(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = helper_0*x;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
result = -(helper_0*helper_4*helper_5*(1137.7777777778158*x + 796.44444444446913*y - 161.77777777788964) + helper_1*(-35.880201188827797*helper_0*helper_7 - 31.999999999998646*helper_0*helper_8 + 441.47873799710624*helper_10 - 3.1007173661471753e-13*helper_11*y - 102.79378143577537*helper_12 - 221.75674439875948*helper_13 + 10.466392318221381*helper_14*helper_9 + 5.3333333333331847*helper_14 + 72.836762688578077*helper_15*helper_8 + 1.5007237945851064e-13*helper_15 - 42.666666666667808*helper_16*helper_4 - 1.0727958682961118e-12*helper_16*helper_5 - 1.6864653249586023e-13*helper_16*helper_9 + 42.666666666666494*helper_2 - 4.4211548002850774e-14*helper_3 + 436.34567901231696*helper_4*y - 31.999999999999009*helper_4 + 75.308641975286989*helper_5*x + 1.1667454078021655e-12*helper_5 - 7.9533607681700573*helper_6*helper_7 + 1.683196791822911e-12*helper_6*helper_8 + 8.132516151132222e-13*helper_6*helper_9 - 1.4052956329478979e-13*helper_6*x - 1.354099807438712e-12*helper_6*y + 16.517604023790447*helper_7 + 5.3333333333331847*helper_8 + 1.0497912552966317e-13*helper_9) + helper_10*helper_6*(1829.3333333333139*helper_7 + 995.55555555559613*helper_8 + 398.222222222244*helper_9 - 315.65432098782526*x - 228.69135802484072*y + 11.184270690455484) + helper_11*helper_7*(766.86419753078292*helper_12 + 1426.5679012344858*helper_13 + 355.55555555556748*helper_4 + 56.888888888891316*helper_5 - 418.44810242360529*helper_7 - 185.87654320993119*helper_8 - 66.913580246944932*helper_9 + 16.517604023791748*x + 11.184270690456586*y) + 455.11111111111927*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_8(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
result = -(-helper_1*(-14.713305898461797*helper_0*helper_10*x - 2.2005397047314326e-13*helper_0*helper_10 - 9.5719012658129484e-13*helper_0*helper_4 - 34.417009602158231*helper_0*helper_9*y - 7.2146404077273267e-13*helper_0*helper_9 + 8.86785550982448*helper_0*x*y + 5.3838963465785072e-15*helper_0*x + 9.4794161851107983e-14*helper_0*y - 7.7003241460397688e-13*helper_10*helper_7 + 48.769090077759117*helper_10*x - 6.2467178206488395e-14*helper_10 - 255.84910836747054*helper_11 + 2.7237471537470675e-14*helper_12*x + 1.9900009090665223e-13*helper_12*y + 1.4526651486651421e-13*helper_2 + 8.1690929296220772e-29*helper_3 - 56.098765432074572*helper_4*x - 9.5719012658129161e-13*helper_4 - 161.13580246910732*helper_5*y - 3.1858192353047521e-13*helper_5 - 4.2496596091233571e-13*helper_6 - 8.5247584026133814e-13*helper_7*helper_9 - 1.2812071330594783*helper_7*x*y + 7.6536856413671495e-15*helper_7*x + 9.9113431986883358e-13*helper_7*y - 5.8509373571186627*helper_8 + 71.732053040735224*helper_9*y + 5.3838963465785072e-15*helper_9) + helper_11*helper_7*(312.88888888890267*helper_10 + 1374.2222222221708*helper_8 + 739.55555555558067*helper_9 - 256.64197530879864*x - 169.67901234582294*y + 11.184270690450631) + helper_12*x*y*(534.32098765422313*helper_10*x - 39.901234567932534*helper_10 + 42.666666666668384*helper_4 + 170.66666666667442*helper_5 - 241.41106538656146*helper_8 + 852.69135802459186*helper_9*y - 94.864197530909905*helper_9 + 11.184270690453141*x + 5.8509373571181476*y) + 455.1111111111137*helper_2*helper_3 + helper_4*helper_6*(1024.0000000000209*x + 682.66666666667902*y - 161.77777777788549))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_9(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_10*x;
double helper_13 = helper_9*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = helper_0*x;
double helper_17 = helper_0*y;
double helper_18 = helper_7*x;
result = (helper_1*(2.3576648178350676e-13*helper_0*helper_10 + 1.2894783549119472e-12*helper_0*helper_4 + 8.3446829693107209e-13*helper_0*helper_9 + 75.94032921806712*helper_10*helper_16 + 1.1027612293832231e-12*helper_10*helper_7 + 1.4486457301217052e-13*helper_10 + 575.97736625495054*helper_11 - 5.1968449931801928*helper_12 + 9.5068587105036642*helper_13 - 1.9382255288632491e-13*helper_14*y - 3.6119255734471978e-14*helper_15 - 5.1968449931387983*helper_16*y + 5.6169061171711329e-15*helper_16 + 106.64403292175646*helper_17*helper_9 + 3.4624109706313257e-14*helper_17 + 11.940329218103233*helper_18*y - 7.483725573459685e-16*helper_18 - 1.6579330501069435e-13*helper_2 - 1.2393881333138956e-28*helper_3 + 127.99999999997132*helper_4*x + 1.2894783549119438e-12*helper_4 + 222.70370370366265*helper_5*y + 3.3783675445633906e-13*helper_5 + 4.8557043159979316e-13*helper_6 + 1.0557809770472788e-12*helper_7*helper_9 - 1.2217755385363207e-12*helper_7*y - 5.137174211239266*helper_8 + 5.6169061171711329e-15*helper_9) + helper_11*helper_7*(469.33333333334576*helper_10 + 2047.9999999999293*helper_8 + 810.66666666668925*helper_9 + 30.703703703520048*x - 1.6292567295295157e-10*y - 9.1371742112418541) + helper_15*y*(-3.6839864492321794e-11*helper_10 + 938.66666666655146*helper_12 + 1246.7037037035764*helper_13 + 64.000000000001705*helper_4 + 128.00000000000773*helper_5 + 9.5068587103340008*helper_8 + 30.703703703644489*helper_9 - 9.1371742112377401*x - 5.1371742112404641*y) + 682.66666666666436*helper_2*helper_3 + helper_4*helper_6*(1365.3333333333471*x + 1024.0000000000055*y - 1.1952376864132644e-10))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_10(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = helper_0*x;
double helper_15 = helper_6*x;
double helper_16 = -helper_0;
result = -(helper_0*helper_4*helper_5*(796.44444444444912*x + 682.66666666666742*y + 161.77777777772576) + helper_1*(125.87379972561695*helper_0*helper_13 + 9.9250892522065064e-14*helper_0*y + 530.21947873790123*helper_10 - 2.092153610849196e-14*helper_11*x - 1.0136625573367452e-13*helper_11*y + 70.934613625948558*helper_12 + 97.946959304946006*helper_13 + 102.4170096021729*helper_14*helper_9 + 31.033379058068025*helper_14*y + 8.4322124043058815e-15*helper_14 + 30.515775034288577*helper_15*y + 3.0044553950275375e-15*helper_15 - 2.623306236161534e-13*helper_16*helper_4 - 8.0747175445056228e-13*helper_16*helper_5 - 4.4001016821887629e-13*helper_16*helper_8 - 9.5539031647129252e-14*helper_16*helper_9 - 8.2107160576724668e-14*helper_2 - 8.3560821812299196e-29*helper_3 + 152.2469135802225*helper_4*y + 1.6542048937776249e-13*helper_4 + 114.56790123455549*helper_5*x + 8.0747175445056026e-13*helper_5 + 5.9510147153046887e-13*helper_6*helper_8 + 7.1084245028862174e-13*helper_6*helper_9 - 7.4520210152010532e-13*helper_6*y + 5.8509373571140504*helper_7 + 8.4322124043058815e-15*helper_8 + 1.0020379587808616e-13*helper_9) + helper_10*helper_6*(1356.4444444444148*helper_7 + 398.22222222223274*helper_8 + 312.88888888889488*helper_9 + 228.69135802460218*x + 169.679012345606*y + 11.184270690445402) + helper_11*helper_7*(717.23456790118667*helper_12 + 826.02469135796377*helper_13 + 56.888888888892794*helper_4 + 42.666666666667624*helper_5 + 267.62597165054666*helper_7 + 66.91358024687969*helper_8 + 39.901234567884167*helper_9 + 11.18427069044829*x + 5.8509373571130645*y) + 455.11111111110841*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_11(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = helper_0*x;
double helper_15 = helper_9*x;
double helper_16 = -helper_0;
result = -(helper_0*helper_4*helper_5*(682.66666666667231*x + 796.44444444444628*y + 161.77777777773821) + helper_1*(102.41700960216426*helper_0*helper_13 - 6.8265999468202367e-14*helper_0*y + 8.8119178163739177e-13*helper_10 - 1.4210854715202004e-14*helper_11*x - 1.7806210704963339e-13*helper_11*y + 97.946959304969212*helper_12 + 70.934613625936805*helper_13 + 125.87379972563949*helper_14*helper_8 + 31.033379058068263*helper_14*y + 3.4342898895065417e-14*helper_14 + 30.51577503428976*helper_15*y + 2.7632217501775896e-14*helper_15 - 1.4526651486653642e-13*helper_16*helper_4 - 1.0529379532580463e-12*helper_16*helper_5 - 3.0000693287651213e-13*helper_16*helper_6 - 5.1387792055474153e-14*helper_16*helper_8 - 6.7106813932900422e-14*helper_2 - 7.7133066288276313e-29*helper_3 + 114.56790123454681*helper_4*y + 4.1843072217010918e-14*helper_4 + 152.24691358024316*helper_5*x + 1.0529379532580445e-12*helper_5 + 530.21947873792635*helper_6*helper_8 + 4.2000970602711055e-13*helper_6*helper_9 + 3.4342898895065417e-14*helper_6 + 5.8509373571128434*helper_7 - 3.0613371919780804e-14*helper_8 - 9.5736914814025232e-13*helper_9*y) + helper_10*helper_6*(312.88888888889971*helper_6 + 1356.4444444444287*helper_7 + 398.22222222222865*helper_8 + 169.67901234560549*x + 228.6913580246389*y + 11.184270690444221) + helper_11*helper_7*(826.02469135799834*helper_12 + 717.23456790119019*helper_13 + 42.666666666670665*helper_4 + 56.888888888890087*helper_5 + 39.901234567870461*helper_6 + 267.62597165057014*helper_7 + 66.91358024690463*helper_8 + 5.8509373571138106*x + 11.184270690445203*y) + 455.1111111111091*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_12(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_7*x;
double helper_14 = helper_9*y;
double helper_15 = helper_0*x;
double helper_16 = helper_10*x;
double helper_17 = helper_0*helper_9;
double helper_18 = helper_0*helper_7;
result = (helper_1*(1.9584864137396319e-12*helper_0*helper_4 - 3.9778822668378445e-13*helper_0*y + 1.538017998281771e-12*helper_10*helper_7 - 1.8466100847796473e-12*helper_10*y + 5.8738199489506233e-13*helper_11 - 4.3467927017306327e-13*helper_12*y + 9.5068587105532885*helper_13 - 5.1968449931839302*helper_14 - 5.1968449931366933*helper_15*y + 8.3488771451802003e-14*helper_15 + 11.940329218108225*helper_16*y + 9.4739031434670777e-14*helper_16 + 75.940329218070644*helper_17*y + 4.0737783516916456e-13*helper_17 + 106.64403292180516*helper_18*x + 3.8336960492340746e-14*helper_18 - 9.0002079862949263e-14*helper_2 - 1.3463226115771865e-28*helper_3 + 222.70370370370401*helper_4*x + 1.9584864137396295e-12*helper_4 + 127.99999999997112*helper_5*y - 3.3158661002094724e-14*helper_5 + 1.3263464400859518e-13*helper_6 + 575.97736625504035*helper_7*helper_9 - 2.2468858049853334e-13*helper_7 - 5.1371742112432361*helper_8 + 8.3488771451802003e-14*helper_9) + helper_11*helper_7*(810.66666666667686*helper_7 + 2047.9999999999641*helper_8 + 469.33333333335042*helper_9 - 1.2232703738845942e-10*x + 30.703703703616192*y - 9.1371742112458758) + helper_12*helper_8*(1246.703703703658*helper_13 + 938.66666666659103*helper_14 + 128.00000000000193*helper_4 + 64.000000000006423*helper_5 + 30.703703703695481*helper_7 + 9.5068587104243676*helper_8 - 4.4489449161726093e-11*helper_9 - 5.1371742112416259*x - 9.1371742112446555*y) + 682.6666666666631*helper_2*helper_3 + helper_4*helper_6*(1024.0000000000082*x + 1365.3333333333358*y - 7.2968002010990702e-11))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_13(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_7*x;
double helper_13 = pow(helper_0, 3);
result = -(-helper_1*(-34.417009602203947*helper_0*helper_12 - 1.5180994702446193e-12*helper_0*helper_4 + 1.5329964092843262e-13*helper_0*helper_7 - 14.713305898478637*helper_0*helper_9*y - 2.1474180458530425e-13*helper_0*helper_9 + 8.8678555098217515*helper_0*x*y - 7.5199106201270218e-14*helper_0*x + 4.9078193117494846e-13*helper_0*y - 1.0476375140057314e-12*helper_10*helper_7 - 1.2812071330655179*helper_10*x*y - 8.0922922683782476e-14*helper_10*x + 1.6361656733017865e-12*helper_10*y - 3.0632286830549936e-13*helper_11 + 4.8940976252582394e-13*helper_13*y + 6.710681393290075e-14*helper_2 + 7.7133066288276268e-29*helper_3 - 161.1358024691433*helper_4*x - 1.5180994702446177e-12*helper_4 - 56.098765432086054*helper_5*y + 1.0026547493500468e-13*helper_5 - 3.1579677145207259e-15*helper_6 - 255.84910836757615*helper_7*helper_9 + 71.732053040688214*helper_7*x + 3.0176615664454863e-13*helper_7 - 5.8509373571138994*helper_8 + 48.769090077748984*helper_9*y - 7.5199106201270218e-14*helper_9) + helper_11*helper_7*(739.55555555556305*helper_7 + 1374.2222222221999*helper_8 + 312.88888888890165*helper_9 - 169.67901234575672*x - 256.64197530869535*y + 11.184270690445679) + helper_13*x*y*(852.69135802466826*helper_12 + 170.66666666666811*helper_4 + 42.666666666671432*helper_5 - 94.864197530862967*helper_7 - 241.41106538644905*helper_8 + 534.32098765427781*helper_9*y - 39.901234567925428*helper_9 + 5.8509373571150274*x + 11.184270690446327*y) + 455.11111111110887*helper_2*helper_3 + helper_4*helper_6*(682.66666666667334*x + 1024.0000000000025*y - 161.77777777782796))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_14(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_6*y;
double helper_13 = pow(helper_0, 3);
double helper_14 = helper_13*x;
double helper_15 = helper_0*x;
double helper_16 = helper_0*y;
double helper_17 = helper_9*x;
double helper_18 = -helper_0;
result = -(helper_0*helper_4*helper_5*(796.44444444445537*x + 1137.777777777784*y - 161.77777777780938) + helper_1*(-32.000000000001037*helper_0*helper_8 + 7.0139047747643979e-13*helper_10 - 221.75674439870073*helper_11 - 102.79378143576366*helper_12 - 9.3786584960608345e-13*helper_13*y + 1.4605600679513295e-14*helper_14 + 72.836762688635503*helper_15*helper_8 + 2.1568480883316888e-14*helper_15 + 10.466392318241702*helper_16*helper_6 - 35.880201188827044*helper_16*x + 5.3333333333326181*helper_16 - 7.9533607681638685*helper_17*y + 4.2786076465042679e-14*helper_17 - 2.8083982327612313e-13*helper_18*helper_4 - 42.666666666668263*helper_18*helper_5 - 4.8053742055486116e-13*helper_18*helper_6 - 8.2107160576725956e-14*helper_2 + 42.666666666666664*helper_3 + 75.308641975302862*helper_4*y + 1.661661206437288e-13*helper_4 + 436.34567901236073*helper_5*x - 31.999999999998394*helper_5 + 441.47873799724351*helper_6*helper_8 + 6.4913780797844624e-13*helper_6*helper_9 + 2.1568480883316888e-14*helper_6 + 16.517604023782841*helper_7 + 5.3333333333328383*helper_8 - 2.5641915050660013e-12*helper_9*y) + helper_10*helper_6*(398.22222222223951*helper_6 + 1829.3333333333414*helper_7 + 995.55555555556634*helper_8 - 228.69135802474733*x - 315.65432098768412*y + 11.184270690447443) + helper_14*y*(1426.5679012345777*helper_11 + 766.86419753085329*helper_12 + 56.888888888894996*helper_4 + 355.55555555555748*helper_5 - 66.913580246933833*helper_6 - 418.44810242345415*helper_7 - 185.87654320986931*helper_8 + 11.184270690450553*x + 16.517604023781697*y) + 455.11111111111006*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_15(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_10*x;
double helper_13 = helper_9*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = helper_0*y;
double helper_17 = helper_0*helper_9;
double helper_18 = helper_7*y;
double helper_19 = helper_0*helper_10;
result = (helper_1*(1.2765755530854894e-12*helper_0*helper_4 - 1.6075207009151348e-13*helper_0*x + 64.000000000000327*helper_10*helper_7 - 4.0000000000006537*helper_10 + 1699.866255144083*helper_11 - 67.900548696811384*helper_12 - 82.604252400554159*helper_13 - 1.6447853595803433e-12*helper_14*y + 2.6645352591004009e-14*helper_15 - 67.900548696817808*helper_16*x - 4.0000000000006661*helper_16 + 206.53292181070069*helper_17*y + 1.4747955942894571e-12*helper_17 + 45.236625514420879*helper_18*x - 16.000000000004391*helper_18 + 578.56995884777257*helper_19*x - 16.000000000002647*helper_19 - 1.657933050106966e-13*helper_2 + 64.0*helper_3 + 289.2962962962963*helper_4*y + 1.0974595717792958e-12*helper_4 + 1066.6666666666947*helper_5*x + 1.9306033680729531e-12*helper_5 + 128.0000000000019*helper_6 + 1.9095507068584562e-12*helper_7*helper_9 - 1.2004389252192329e-13*helper_7*x - 13.137174211235669*helper_8 - 1.6075207009151348e-13*helper_9) + helper_11*helper_7*(1493.3333333333546*helper_10 + 3413.3333333334231*helper_8 + 810.66666666670142*helper_9 - 30.703703703729026*x + 1.4287238059296214e-11*y - 9.1371742112398202) + helper_15*y*(1.4782841617488884e-11*helper_10 + 2986.6666666667543*helper_12 + 1995.962962963024*helper_13 + 128.00000000001162*helper_4 + 533.33333333333667*helper_5 - 82.604252400552866*helper_8 - 30.703703703730596*helper_9 - 9.1371742112343011*x - 13.137174211237511*y) + 682.66666666666765*helper_2*helper_3 + helper_4*helper_6*(1365.3333333333587*x + 1706.6666666666827*y + 9.4106500639270478e-12))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_16(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = helper_0*y;
double helper_15 = -helper_0;
result = -(helper_0*helper_4*helper_5*(1024.0000000000327*x + 1137.7777777778006*y + 161.7777777778405) + helper_1*(897.1330589849432*helper_0*helper_12 + 149.99634202105079*helper_0*helper_7 - 2.8357014952924765e-13*helper_0*x + 127.99999999999932*helper_10 + 2.0921536108491992e-14*helper_11*x + 42.666666666665293*helper_11*y + 335.87288523093861*helper_12 + 212.86053955190053*helper_13 + 575.00960219479771*helper_14*helper_6 + 5.3333333333333792*helper_14 - 1.6340289889349434e-12*helper_15*helper_4 - 128.00000000000063*helper_15*helper_5 - 1.6643805675241655e-12*helper_15*helper_6 - 63.999999999997122*helper_15*helper_8 - 1.4526651486651633e-13*helper_2 + 42.666666666666664*helper_3 + 521.5308641975455*helper_4*y + 1.5035654477300997e-12*helper_4 + 985.87654320990578*helper_5*x + 32.000000000000647*helper_5 + 2135.1083676270014*helper_6*helper_8 + 2.1096978759299135e-12*helper_6*helper_9 - 2.8357014952924765e-13*helper_6 + 266.8120713306007*helper_7*helper_9 + 16.517604023786376*helper_7 + 5.3333333333330009*helper_8 - 2.4972068308956402e-13*helper_9*x + 31.999999999996575*helper_9*y) + helper_10*helper_6*(739.55555555559181*helper_6 + 2721.7777777779265*helper_7 + 995.55555555557748*helper_8 + 256.64197530868802*x + 315.65432098772709*y + 11.184270690449567) + helper_11*helper_7*(2555.654320987795*helper_12 + 2105.5308641976603*helper_13 + 170.66666666667774*helper_4 + 355.55555555555867*helper_5 + 94.86419753085417*helper_6 + 528.51486053961617*helper_7 + 185.87654320989427*helper_8 + 11.184270690453841*x + 16.517604023785154*y) + 455.11111111111757*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_17(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_5*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 42.666666666667147*helper_2;
double helper_14 = 42.666666666665812*helper_3;
double helper_15 = helper_0*x;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(249.00000000007782*helper_4 + 474.00000000009538*x + 474.00000000009271*y + 96.611111111100485) + helper_1*(42.666666666666664*pow(helper_0, 4) + 698.91666666666561*helper_0*helper_8 + 698.91666666666833*helper_0*helper_9 + 69.333333333333783*helper_0*y + 128.00000000000003*helper_1*x + 127.99999999999976*helper_1*y + 74.666666666666671*helper_1 + 128.00000000000045*helper_10*helper_5 + 127.99999999999892*helper_10*helper_6 + 191.99999999999997*helper_10*x + 37.333333333333329*helper_10 + 191.99999999999969*helper_11 + 559.24999999998397*helper_12 - helper_13*helper_16 + helper_13 - helper_14*helper_16 + helper_14 + 751.86111111108528*helper_15*y + 69.333333333333229*helper_15 - 160.00000000000037*helper_16*helper_5 - 159.99999999999875*helper_16*helper_6 + 267.6666666666851*helper_2*y + 267.66666666668164*helper_3*x + 197.94444444443454*helper_4 + 31.999999999999886*helper_5 + 32.000000000000114*helper_6 + 1044.9166666667597*helper_7 + 827.527777777766*helper_8 + 827.52777777776964*helper_9 + 5.3333333333333339*x + 5.3333333333333339*y + 5.3333333333333339*z - 5.3333333333333339) + helper_12*(1141.5277777778599*helper_4 + 267.6666666666851*helper_5 + 267.66666666668164*helper_6 + 474.00000000009271*helper_8 + 474.00000000009538*helper_9 + 128.61111111110108*x + 128.61111111110114*y + 5.3333333333333339) + 249.00000000007782*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_18(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = x*y;
double helper_6 = pow(y, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_4*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_5;
double helper_12 = 1.4788170688010137e-13*helper_2;
double helper_13 = 3.8327674367622573e-13*helper_3;
double helper_14 = 128.0*helper_1;
result = (helper_1*(64.0*pow(helper_0, 4) + helper_0*helper_12 - helper_0*helper_13 + 112.0000000000001*helper_0*helper_4 + 763.43749999996771*helper_0*helper_5 + 387.81249999998693*helper_0*helper_9 + 123.99999999999997*helper_0*x + 124.00000000000016*helper_0*y + 127.99999999999986*helper_1*y + 64.000000000000142*helper_10*helper_4 + 63.999999999999481*helper_10*helper_6 + 240.00000000000003*helper_10*x + 239.99999999999983*helper_10*y + 76.0*helper_10 + 450.56249999998005*helper_11 + helper_12 - helper_13 + helper_14*x + helper_14 + 65.250000000007105*helper_2*y + 65.250000000006509*helper_3*x + 490.31250000002046*helper_4*helper_6 + 47.999999999999964*helper_4 + 324.87499999998761*helper_5 + 48.000000000000043*helper_6 + 387.81249999998658*helper_7*x + 111.99999999999943*helper_7 + 588.68749999997397*helper_8 + 588.68749999997476*helper_9 + 12.0*x + 12.0*y + 12.0*z - 12.0) + helper_11*(65.250000000007105*helper_4 + 643.18750000000773*helper_5 + 65.250000000006509*helper_6 + 166.50000000003396*helper_8 + 166.50000000003422*helper_9 + 200.87499999998766*x + 200.87499999998772*y + 12.0) + 101.25000000002743*helper_2*helper_3 + helper_4*helper_7*(101.25000000002743*helper_5 + 166.50000000003422*x + 166.50000000003396*y + 152.87499999998747))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_19(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_11*y;
double helper_13 = 4.7961634663821921e-14*helper_2;
double helper_14 = 1.627216879758953e-13*helper_3;
double helper_15 = helper_0*x;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(27.000000000012804*helper_4 + 54.000000000017017*x + 54.000000000016527*y + 75.611111111102346) + helper_1*(42.666666666666664*pow(helper_0, 4) - helper_0*helper_14 - 2.4238944185128574e-13*helper_0*helper_5 + 167.41666666665822*helper_0*helper_9 + 69.333333333333385*helper_0*y + 42.666666666666664*helper_1*x + 42.6666666666666*helper_1*y + 96.0*helper_1 - 2.2696659366753134e-13*helper_10*helper_5 + 4.7961634663821921e-14*helper_10*helper_6 + 95.999999999999943*helper_10*y + 69.333333333333329*helper_10 + 96.0*helper_11 + 183.08333333332061*helper_12 - helper_13*helper_16 + helper_13 - helper_14 + 167.4166666666577*helper_15*helper_5 + 312.0277777777564*helper_15*y + 69.333333333333314*helper_15 - 3.5971225997866406e-14*helper_16*helper_6 + 27.00000000000432*helper_2*y + 27.000000000003698*helper_3*x + 144.94444444443567*helper_4 + 1.005677023140154e-14*helper_5 - 1.199040866595548e-14*helper_6 + 221.41666666667462*helper_7 + 243.02777777776006*helper_8 + 243.02777777776066*helper_9 + 16.0*x + 16.0*y + 16.0*z - 16.0) + helper_12*(297.02777777777692*helper_4 + 27.000000000003698*helper_5 + 27.00000000000432*helper_6 + 54.000000000016527*helper_8 + 54.000000000017017*helper_9 + 75.611111111102389*x + 75.611111111102446*y + 16.0) + 27.000000000012804*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_20(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 42.666666666667119*helper_2;
double helper_14 = 4.3235785322321246e-13*helper_3;
double helper_15 = helper_0*x;
double helper_16 = helper_0*y;
result = (helper_0*helper_7*(249.00000000003811*helper_4 + 474.0000000000415*x + 273.00000000004252*y - 96.611111111110915) + helper_1*(helper_0*helper_13 - helper_0*helper_14 - 2.8755701523647829e-13*helper_0*helper_5 - 31.999999999999684*helper_0*helper_6 + 7.1054273576010019e-15*helper_1*x + 8.2822637637043191e-14*helper_1*y - 3.258504577274993e-13*helper_10*helper_5 + 4.4941828036830053e-13*helper_10*helper_6 - 1.0524914273447413e-13*helper_10*x + 2.4338864257345752e-13*helper_11 - 35.583333333334132*helper_12 + helper_13 - helper_14 + 31.083333333336981*helper_15*helper_5 + 5.3333333333332194*helper_15 + 104.08333333333616*helper_16*helper_6 - 100.19444444444423*helper_16*x + 5.4394451905659117e-13*helper_16 + 267.66666666667084*helper_2*y + 66.666666666671176*helper_3*x - 59.277777777776819*helper_4 + 2.2068458171988003e-13*helper_5 - 32.000000000000099*helper_6 + 377.08333333337737*helper_7 - 33.527777777774418*helper_8 - 24.527777777774006*helper_9 + 5.3333333333333339*x) + helper_12*(280.47222222226594*helper_4 + 66.666666666671176*helper_5 + 267.66666666667084*helper_6 + 273.00000000004252*helper_8 + 474.0000000000415*helper_9 - 128.61111111111035*x - 64.611111111110503*y + 5.3333333333333339) + 249.00000000003811*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_21(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 1.4788170688007842e-13*helper_2;
double helper_14 = 1.8068879725769576e-14*helper_3;
double helper_15 = helper_0*x;
double helper_16 = helper_0*y;
double helper_17 = -helper_0;
result = -(helper_0*helper_7*(101.25000000000925*helper_4 + 166.50000000000838*x + 137.25000000000878*y - 152.8749999999981) + helper_1*(-111.9999999999999*helper_0*helper_6 + 9.9309449552721641e-14*helper_1*y + 1.3514189767249372e-13*helper_10*helper_5 - 63.999999999999851*helper_10*helper_6 + 15.999999999999963*helper_10*x + 1.6728979312930096e-13*helper_11 - 129.31249999999753*helper_12 - helper_13*helper_17 + helper_13 - helper_14*helper_17 + helper_14 - 93.312499999997712*helper_15*helper_5 - 218.18749999999528*helper_15*y + 27.999999999999964*helper_15 - 192.06249999999849*helper_16*helper_6 + 1.656938475314079e-13*helper_16 - 1.2635031909624177e-13*helper_17*helper_5 + 65.249999999999432*helper_2*y + 35.999999999999844*helper_3*x - 76.874999999997726*helper_4 + 7.0131400686793803e-14*helper_5 - 48.000000000000036*helper_6 + 9.1875000000098339*helper_7 - 198.18749999999602*helper_8 - 392.93749999999648*helper_9 + 12.0*x) + helper_12*(-143.68749999998849*helper_4 + 35.999999999999844*helper_5 + 65.249999999999432*helper_6 + 137.25000000000878*helper_8 + 166.50000000000838*helper_9 - 200.8749999999979*x - 104.8749999999979*y + 12.0) + 101.25000000000925*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_22(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = x*y;
double helper_6 = pow(y, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_4*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_11*y;
double helper_13 = 4.796163466380473e-14*helper_2;
double helper_14 = 6.2727600891322405e-14*helper_3;
double helper_15 = helper_0*x;
double helper_16 = helper_0*helper_4;
result = (helper_1*(helper_0*helper_13 + helper_0*helper_14 + 4.4362661692311778e-14*helper_0*y + 42.666666666666664*helper_1*x + 6.2838623193783557e-14*helper_1*y + 4.796163466380473e-14*helper_10*helper_4 + 1.2319774829923894e-13*helper_10*helper_6 + 1.016316660458941e-13*helper_10*y + 96.0*helper_11 - 70.749999999998096*helper_12 + helper_13 + helper_14 - 113.41666666666583*helper_15*helper_6 - 93.027777777774119*helper_15*y + 69.333333333333314*helper_15 - 86.416666666666188*helper_16*y + 3.5971225997853519e-14*helper_16 + 26.999999999998689*helper_2*y - 1.0871303857129533e-12*helper_3*x - 59.416666666668505*helper_4*helper_6 - 1.1990408665951183e-14*helper_4 - 6.2777777777760306*helper_5 + 2.2047178897347431e-14*helper_6 + 1.2875811528090379e-13*helper_7 - 189.02777777777541*helper_8 - 162.02777777777561*helper_9 + 16.0*x) + helper_12*(26.999999999998689*helper_4 - 135.02777777777806*helper_5 - 1.0871303857129533e-12*helper_6 + 26.999999999997826*helper_8 + 53.999999999997669*helper_9 - 75.611111111109409*x - 75.611111111109423*y + 16.0) + 26.999999999999059*helper_2*helper_3 + helper_4*helper_7*(26.999999999999059*helper_5 + 53.999999999997669*x + 26.999999999997826*y - 75.611111111109452))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_23(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 2.4247270857816165e-13*helper_2;
double helper_13 = 2.6267876762632274e-13*helper_3;
double helper_14 = 6.0618177144540413e-14*x;
double helper_15 = -helper_0;
result = -(helper_0*helper_7*(249.00000000002979*helper_4 + 273.0000000000307*x + 273.00000000003411*y + 96.611111111108983) + helper_1*(-helper_0*helper_13 - helper_0*helper_14 + 166.86111111110483*helper_0*helper_4 - 3.3874754853022419e-13*helper_0*helper_5 + 168.91666666666612*helper_0*helper_8 + 168.91666666666288*helper_0*helper_9 + 2.5465740627341896e-13*helper_0*y - 1.7652546091533173e-14*helper_1*y - helper_10*helper_14 - 2.8033131371785596e-13*helper_10*helper_5 + 2.4247270857816165e-13*helper_10*helper_6 - 1.7402745910992399e-14*helper_10*y + 102.24999999999511*helper_11 - helper_12*helper_15 + helper_12 - helper_13 - 1.8185453143362111e-13*helper_15*helper_6 + 66.666666666668192*helper_2*y + 66.666666666671119*helper_3*x + 69.944444444442908*helper_4 + 9.5645713571467434e-14*helper_5 - 6.0618177144540413e-14*helper_6 + 441.91666666669602*helper_7 + 233.5277777777749*helper_8 + 233.52777777777243*helper_9) + helper_11*(538.52777777780466*helper_4 + 66.666666666671119*helper_5 + 66.666666666668192*helper_6 + 273.00000000003411*helper_8 + 273.0000000000307*helper_9 + 64.611111111109452*x + 64.611111111109366*y + 5.3333333333333339) + 249.00000000002979*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_24(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 1.21902488103869e-13*helper_2;
double helper_13 = 2.5174307083377091e-13*helper_3;
double helper_14 = 3.0475622025967249e-14*x;
double helper_15 = helper_0*y;
result = (helper_0*helper_7*(101.25000000002478*helper_4 + 137.25000000003007*x + 137.25000000003061*y + 152.87499999999719) + helper_1*(helper_0*helper_12 - helper_0*helper_13 - helper_0*helper_14 + 286.18749999999363*helper_0*helper_4 - 3.525096881063141e-13*helper_0*helper_5 + 9.1426866077901672e-14*helper_0*helper_6 + 201.31250000000193*helper_0*helper_8 - 7.0915495697927835e-14*helper_1*y - helper_10*helper_14 - 3.2265856653169874e-13*helper_10*helper_5 + 1.21902488103869e-13*helper_10*helper_6 - 8.9643570344572826e-14*helper_10*y + 165.31249999999622*helper_11 + helper_12 - helper_13 + 201.31250000000159*helper_15*helper_6 + 1.3263695697321113e-13*helper_15 + 36.000000000005571*helper_2*y + 36.000000000005777*helper_3*x + 132.87499999999739*helper_4 + 4.0342729157326225e-14*helper_5 - 3.0475622025967249e-14*helper_6 + 402.56250000003172*helper_7 + 306.18749999999903*helper_8 + 306.18749999999903*helper_9) + helper_11*(555.43750000002888*helper_4 + 36.000000000005777*helper_5 + 36.000000000005571*helper_6 + 137.25000000003061*helper_8 + 137.25000000003007*helper_9 + 104.87499999999736*x + 104.87499999999737*y + 12.0) + 101.25000000002478*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_25(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 2.3980817331910128e-14*helper_2;
double helper_13 = 1.3167245072054654e-13*helper_3;
double helper_14 = 5.9952043329775319e-15*x;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(27.000000000005947*helper_4 + 27.000000000007574*x + 27.000000000007766*y + 75.611111111109196) + helper_1*(-helper_0*helper_13 - helper_0*helper_14 - 1.7580381594939667e-13*helper_0*helper_5 + 113.41666666666556*helper_0*helper_8 - 4.4371913550850533e-14*helper_1*y - helper_10*helper_14 - 1.7604436427139705e-13*helper_10*helper_5 + 2.3980817331910128e-14*helper_10*helper_6 - 5.0727940366829295e-14*helper_10*y + 156.0833333333305*helper_11 - helper_12*helper_16 + helper_12 - helper_13 + 113.41666666666558*helper_15*helper_6 + 285.02777777777311*helper_15*x + 3.9810747291356341e-14*helper_15 - 1.7985612998932585e-14*helper_16*helper_6 + 1.7035262089848402e-12*helper_2*y + 1.723066134218243e-12*helper_3*x + 144.94444444444255*helper_4 + 1.5228559154444024e-14*helper_5 - 5.9952043329775319e-15*helper_6 + 140.41666666667322*helper_7 + 189.02777777777479*helper_8 + 189.02777777777482*helper_9) + helper_11*(216.02777777778243*helper_4 + 1.723066134218243e-12*helper_5 + 1.7035262089848402e-12*helper_6 + 27.000000000007766*helper_8 + 27.000000000007574*helper_9 + 75.611111111109238*x + 75.611111111109238*y + 16.0) + 27.000000000005947*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_26(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_5*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 2.3536728122056459e-13*helper_2;
double helper_14 = 42.666666666666529*helper_3;
double helper_15 = 5.8841820305141147e-14*x;
double helper_16 = helper_0*y;
double helper_17 = helper_0*helper_5;
result = (helper_0*helper_7*(249.00000000003479*helper_4 + 273.00000000003519*x + 474.00000000003564*y - 96.611111111111455) + helper_1*(helper_0*helper_13 + helper_0*helper_14 - helper_0*helper_15 - 32.000000000000405*helper_0*helper_6 + 104.08333333333157*helper_0*helper_8 - 4.5704181180393186e-14*helper_1*y - helper_10*helper_15 + 2.3536728122056459e-13*helper_10*helper_5 - 1.7885692926711648e-13*helper_10*helper_6 - 1.4528193463072914e-13*helper_11 - 35.583333333336626*helper_12 + helper_13 + helper_14 - 100.1944444444473*helper_16*x + 5.3333333333334547*helper_16 + 31.083333333329918*helper_17*y + 1.7652546091542332e-13*helper_17 + 66.666666666667524*helper_2*y + 267.66666666666822*helper_3*x - 59.277777777777423*helper_4 - 5.8841820305141147e-14*helper_5 - 31.999999999999993*helper_6 + 377.08333333336486*helper_7 - 24.52777777777996*helper_8 - 33.527777777781068*helper_9 + 5.3333333333333339*y) + helper_12*(280.47222222225309*helper_4 + 66.666666666667524*helper_5 + 267.66666666666822*helper_6 + 474.00000000003564*helper_8 + 273.00000000003519*helper_9 - 64.611111111111015*x - 128.61111111111097*y + 5.3333333333333339) + 249.00000000003479*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_27(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = x*y;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1.2190248810384673e-13*helper_2;
double helper_9 = 9.936496070394318e-15*helper_3;
result = -(helper_0*helper_4*helper_6*(101.25000000000739*helper_5 + 137.250000000005*x + 166.5000000000062*y - 152.87499999999807) - helper_1*(93.312500000000725*helper_0*helper_4*y - 9.1426866077884973e-14*helper_0*helper_4 + 192.06249999999912*helper_0*helper_6*x + 112.0*helper_0*helper_6 - helper_0*helper_8 - helper_0*helper_9 + 218.18749999999585*helper_0*x*y + 3.0475622025961683e-14*helper_0*x - 28.000000000000057*helper_0*y - 4.6934678366030247e-14*helper_1*y - 35.99999999999784*helper_2*y - 65.249999999999076*helper_3*x - 9.1875000000051159*helper_4*helper_6 - 1.2190248810384673e-13*helper_4*helper_7 + 198.18749999999852*helper_4*y + 3.0475622025961683e-14*helper_4 + 63.999999999999943*helper_6*helper_7 + 392.93749999999727*helper_6*x + 47.999999999999972*helper_6 + 129.31249999999818*helper_7*x*y + 3.0475622025961683e-14*helper_7*x - 16.000000000000043*helper_7*y - helper_8 - helper_9 + 76.87499999999774*x*y - 12.0*y) + 101.25000000000739*helper_2*helper_3 + helper_5*helper_7*(137.250000000005*helper_4*y + 35.99999999999784*helper_4 - 143.68749999999312*helper_5 + 166.5000000000062*helper_6*x + 65.249999999999076*helper_6 - 104.87499999999785*x - 200.87499999999793*y + 12.0))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_28(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 2.3980817331901248e-14*helper_2;
double helper_14 = 6.9796020814769862e-14*helper_3;
double helper_15 = 5.995204332975312e-15*x;
double helper_16 = helper_0*y;
double helper_17 = helper_0*helper_4;
result = (helper_1*(helper_0*helper_13 + helper_0*helper_14 - helper_0*helper_15 + 42.666666666666707*helper_1*y - helper_10*helper_15 + 1.1416793436562177e-13*helper_10*helper_4 + 2.3980817331901248e-14*helper_10*helper_6 + 96.000000000000043*helper_11 - 70.749999999998252*helper_12 + helper_13 + helper_14 - 93.027777777774347*helper_16*x + 69.333333333333329*helper_16 - 86.416666666665904*helper_17*x + 9.342526752220874e-14*helper_17 - 1.6431300764452317e-12*helper_2*y + 26.999999999999019*helper_3*x - 59.416666666668668*helper_4*helper_6 + 3.2381504884894693e-15*helper_4 - 6.2777777777761239*helper_5 - 5.995204332975312e-15*helper_6 - 113.41666666666673*helper_7*y + 1.798561299892592e-14*helper_7 - 162.0277777777755*helper_8 - 189.02777777777621*helper_9 + 16.0*y) + helper_12*(26.999999999999019*helper_4 - 135.02777777777823*helper_5 - 1.6431300764452317e-12*helper_6 + 53.99999999999811*helper_8 + 26.999999999997577*helper_9 - 75.611111111109494*x - 75.611111111109494*y + 16.0) + 26.999999999999247*helper_2*helper_3 + helper_4*helper_7*(26.999999999999247*helper_5 + 26.999999999997577*x + 53.99999999999811*y - 75.611111111109537))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_29(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_5*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_11*y;
double helper_13 = 128.00000000000114*helper_2;
double helper_14 = 1.8083312625095685e-12*helper_3;
double helper_15 = helper_0*x;
double helper_16 = helper_0*y;
result = (helper_0*helper_7*(576.00000000015746*helper_4 + 1152.0000000001889*x + 936.00000000018736*y + 66.666666666657719) + helper_1*(helper_0*helper_13 - helper_0*helper_14 + 288.0000000000008*helper_0*helper_5 - 2.3225865675159535e-12*helper_0*helper_6 + 128.0*helper_1*x - 3.7125857943462877e-13*helper_1*y + 256.00000000000114*helper_10*helper_5 - 2.1866952693017984e-12*helper_10*helper_6 - 3.6637359812627243e-13*helper_10*y + 159.99999999999977*helper_11 + 587.99999999998806*helper_12 + helper_13 - helper_14 + 820.00000000001774*helper_15*helper_6 + 686.66666666664742*helper_15*y + 31.999999999999716*helper_15 + 1292.0000000000205*helper_16*helper_5 + 1.1346479311670675e-12*helper_16 + 704.00000000003365*helper_2*y + 360.00000000002984*helper_3*x + 98.66666666665931*helper_4 + 31.999999999999716*helper_5 + 3.4283687000431145e-13*helper_6 + 1972.0000000002046*helper_7 + 886.6666666666747*helper_8 + 1390.6666666666797*helper_9) + helper_12*(2038.6666666668616*helper_4 + 704.00000000003365*helper_5 + 360.00000000002984*helper_6 + 936.00000000018736*helper_8 + 1152.0000000001889*helper_9 + 98.666666666659026*x + 66.666666666658912*y) + 576.00000000015746*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_30(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_5;
double helper_12 = 2.9811550257494136e-27*helper_2;
double helper_13 = 1.8829382497641923e-13*helper_3;
double helper_14 = helper_0*x;
result = (-helper_1*(-helper_0*helper_12 - helper_0*helper_13 - 2.4780177909632575e-13*helper_0*helper_4 + 1.2878587085647919e-14*helper_0*y + 128.0*helper_1*x - 9.0594198809410931e-14*helper_1*y - 2.8599345114343114e-13*helper_10*helper_4 + 128.0*helper_10*helper_6 + 224.00000000000006*helper_10*x - 8.8373752760160529e-14*helper_10*y + 339.99999999999335*helper_11 - helper_12 - helper_13 + 139.99999999999358*helper_14*helper_4 + 625.33333333332325*helper_14*y + 96.0*helper_14 - 1.5498483934322839e-13*helper_2*y - 71.999999999999687*helper_3*x + 139.99999999999088*helper_4*helper_6 + 1.4210854715200398e-14*helper_4 + 285.33333333332985*helper_5 + 96.0*helper_6 + 339.99999999999346*helper_7*y + 224.0*helper_7 + 329.33333333332354*helper_8 + 625.33333333332359*helper_9) + helper_11*(71.999999999999687*helper_4 - 329.33333333332092*helper_5 + 1.5498483934322839e-13*helper_6 + 72.000000000002501*helper_8 + 3.3564821657746815e-12*helper_9 - 285.33333333333002*x - 189.33333333332993*y) + 3.1559041675535225e-12*helper_2*helper_3 + helper_4*helper_7*(3.1559041675535225e-12*helper_5 + 3.3564821657746815e-12*x + 72.000000000002501*y - 189.33333333333002))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_31(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 128.00000000000114*helper_2;
double helper_14 = 1.5027978861327012e-12*helper_3;
double helper_15 = 32.000000000000284*x;
double helper_16 = helper_0*y;
double helper_17 = -helper_0;
result = -(helper_0*helper_7*(576.00000000012835*helper_4 + 1152.0000000001492*x + 792.00000000015007*y - 66.666666666669911) + helper_1*(-helper_0*helper_14 - helper_0*helper_15 - 1.6351364706680301e-12*helper_0*helper_5 + 332.00000000001654*helper_0*helper_8 + 820.00000000001705*helper_0*helper_9 - 1.2612133559739895e-13*helper_1*y - helper_10*helper_15 - 1.6289192217301004e-12*helper_10*helper_5 + 128.00000000000114*helper_10*helper_6 + 1.1102230246274133e-14*helper_11 + 115.99999999999504*helper_12 - helper_13*helper_17 + helper_13 - helper_14 + 17.333333333326934*helper_16*x + 1.2030376694839438e-12*helper_16 - 96.000000000000767*helper_17*helper_6 + 704.00000000002308*helper_2*y + 216.00000000002171*helper_3*x - 98.666666666668206*helper_4 + 4.2454928461671025e-13*helper_5 - 32.000000000000284*helper_6 + 1484.0000000001637*helper_7 + 265.33333333334559*helper_8 + 721.33333333334872*helper_9) + helper_12*(1417.3333333334929*helper_4 + 216.00000000002171*helper_5 + 704.00000000002308*helper_6 + 792.00000000015007*helper_8 + 1152.0000000001492*helper_9 - 98.666666666668618*x - 66.666666666668846*y) + 576.00000000012835*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_32(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 8.3133500083937901e-13*helper_2;
double helper_14 = 7.6649797620123413e-13*helper_3;
double helper_15 = 2.0783375020984475e-13*x;
double helper_16 = helper_0*y;
double helper_17 = -helper_0;
result = -(helper_0*helper_7*(576.00000000006844*helper_4 + 936.00000000006878*x + 576.00000000007469*y - 66.666666666667794) + helper_1*(-helper_0*helper_14 - helper_0*helper_15 - 5.8020255266913347e-13*helper_0*helper_5 + 155.99999999999937*helper_0*helper_8 + 1.3233858453533416e-13*helper_1*y - helper_10*helper_15 - 6.3415939166590002e-13*helper_10*helper_5 + 8.3133500083937901e-13*helper_10*helper_6 + 3.1530333899356117e-13*helper_11 + 27.999999999993193*helper_12 - helper_13*helper_17 + helper_13 - helper_14 + 259.9999999999942*helper_16*helper_6 - 6.6666666666729952*helper_16*x + 9.565681580170894e-13*helper_16 - 6.2350125062953376e-13*helper_17*helper_6 + 360.0000000000021*helper_2*y + 128.00000000000642*helper_3*x - 34.666666666666238*helper_4 + 3.9368508453210479e-13*helper_5 - 2.0783375020984475e-13*helper_6 + 836.00000000006617*helper_7 + 121.33333333333047*helper_8 + 193.33333333332777*helper_9) + helper_12*(769.33333333339715*helper_4 + 128.00000000000642*helper_5 + 360.0000000000021*helper_6 + 576.00000000007469*helper_8 + 936.00000000006878*helper_9 - 66.666666666666671*x - 34.66666666666697*y) + 576.00000000006844*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_33(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 6.3948846218432228e-14*helper_2;
double helper_13 = 3.1530333899356112e-13*helper_3;
double helper_14 = 1.5987211554608057e-14*x;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(1.8950232008599917e-11*helper_4 + 72.000000000026063*x + 2.4847774879325669e-11*y + 189.33333333332934) + helper_1*(-helper_0*helper_13 - helper_0*helper_14 - 4.3365311341860488e-13*helper_0*helper_5 + 156.00000000000125*helper_0*helper_8 + 356.00000000000296*helper_0*helper_9 - 1.3411494137471626e-13*helper_1*y - helper_10*helper_14 - 4.4941828036827735e-13*helper_10*helper_5 + 6.3948846218432228e-14*helper_10*helper_6 - 1.5187850976871768e-13*helper_10*y + 155.99999999999557*helper_11 - helper_12*helper_16 + helper_12 - helper_13 + 249.33333333332484*helper_15*x + 8.0824236192733194e-14*helper_15 - 4.7961634663824124e-14*helper_16*helper_6 + 72.000000000007262*helper_2*y + 5.6275366311369416e-12*helper_3*x + 93.33333333332935*helper_4 + 1.9761969838335909e-14*helper_5 - 1.5987211554608057e-14*helper_6 + 356.00000000002763*helper_7 + 249.33333333333059*helper_8 + 545.33333333333246*helper_9) + helper_11*(545.33333333335702*helper_4 + 5.6275366311369416e-12*helper_5 + 72.000000000007262*helper_6 + 2.4847774879325669e-11*helper_8 + 72.000000000026063*helper_9 + 189.33333333332939*x + 93.333333333329378*y) + 1.8950232008599917e-11*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_34(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_5;
double helper_12 = 7.0343730840255967e-13*helper_2;
double helper_13 = 6.4748206796141543e-13*helper_3;
double helper_14 = 1.7585932710063992e-13*x;
result = (helper_1*(helper_0*helper_12 - helper_0*helper_13 - helper_0*helper_14 - 6.081801728896868e-13*helper_0*helper_4 + 134.66666666665685*helper_0*helper_5 + 227.9999999999992*helper_0*helper_8 + 7.5228712148605868e-13*helper_0*y + 5.9507954119923575e-14*helper_1*y - helper_10*helper_14 - 5.8797411384149199e-13*helper_10*helper_4 + 7.0343730840255967e-13*helper_10*helper_6 + 1.5010215292933692e-13*helper_10*y + 99.999999999991331*helper_11 + helper_12 - helper_13 + 216.00000000000236*helper_2*y + 128.00000000000813*helper_3*x + 892.0000000000648*helper_4*helper_6 + 3.1152858070984195e-13*helper_4 + 34.666666666665378*helper_5 - 1.7585932710063992e-13*helper_6 + 315.99999999999267*helper_7*y + 5.2757798130191942e-13*helper_7 + 262.66666666666219*helper_8 + 382.66666666665793*helper_9) + helper_11*(128.00000000000813*helper_4 + 958.66666666672779*helper_5 + 216.00000000000236*helper_6 + 576.00000000007469*helper_8 + 792.00000000006753*helper_9 + 66.666666666665037*x + 34.666666666664753*y) + 576.00000000006673*helper_2*helper_3 + helper_4*helper_7*(576.00000000006673*helper_5 + 792.00000000006753*x + 576.00000000007469*y + 66.666666666663929))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_35(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_5;
double helper_12 = 5.1159076974732908e-13*helper_2;
double helper_13 = 4.7251091928048409e-13*helper_3;
double helper_14 = 1.2789769243683227e-13*x;
double helper_15 = helper_0*y;
result = (helper_1*(helper_0*helper_12 - helper_0*helper_13 - helper_0*helper_14 - 8.1623596770443124e-13*helper_0*helper_4 + 315.99999999999557*helper_0*helper_8 - 5.50670620213907e-14*helper_1*y - helper_10*helper_14 - 5.2757798130187489e-13*helper_10*helper_4 + 5.1159076974732908e-13*helper_10*helper_6 - 1.540989558179535e-13*helper_10*y + 99.999999999989342*helper_11 + helper_12 - helper_13 + 134.66666666665424*helper_15*x + 4.5163872641756275e-13*helper_15 + 127.99999999999994*helper_2*y + 216.00000000000639*helper_3*x + 892.00000000005673*helper_4*helper_6 + 1.3855583347324183e-13*helper_4 + 34.666666666664653*helper_5 - 1.2789769243683227e-13*helper_6 + 227.99999999998789*helper_7*y + 3.8369307731049651e-13*helper_7 + 382.66666666665844*helper_8 + 262.6666666666523*helper_9) + helper_11*(216.00000000000639*helper_4 + 958.66666666671927*helper_5 + 127.99999999999994*helper_6 + 792.00000000007071*helper_8 + 576.00000000006366*helper_9 + 34.666666666664284*x + 66.66666666666417*y) + 576.00000000006492*helper_2*helper_3 + helper_4*helper_7*(576.00000000006492*helper_5 + 576.00000000006366*x + 792.00000000007071*y + 66.666666666663247))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_36(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_5*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 2.7369463312065102e-26*helper_2;
double helper_13 = 1.3855583347323963e-13*helper_3;
double helper_14 = 6.8423658280162755e-27*x;
double helper_15 = -helper_0;
result = -(helper_0*helper_7*(2.2410008012700788e-11*helper_4 + 3.0815276193386688e-11*x + 72.000000000029445*y + 189.33333333332897) + helper_1*(-helper_0*helper_13 - helper_0*helper_14 + 249.33333333332479*helper_0*helper_4 - 2.9576341376016467e-13*helper_0*helper_6 + 356.00000000000284*helper_0*helper_8 + 156.00000000000404*helper_0*helper_9 + 1.3766765505377729e-14*helper_0*y - 1.0125233984581142e-13*helper_1*y - helper_10*helper_14 + 2.7369463312065102e-26*helper_10*helper_5 - 2.3980817331905113e-13*helper_10*helper_6 - 1.5853984791646844e-13*helper_10*y + 155.99999999999574*helper_11 - helper_12*helper_15 + helper_12 - helper_13 - 2.0527097484048812e-26*helper_15*helper_5 + 8.4153342027758968e-12*helper_2*y + 72.000000000007049*helper_3*x + 93.333333333329023*helper_4 - 6.8423658280162755e-27*helper_5 - 3.1974423109194954e-14*helper_6 + 356.00000000003354*helper_7 + 545.33333333333212*helper_8 + 249.33333333333309*helper_9) + helper_11*(545.3333333333627*helper_4 + 8.4153342027758968e-12*helper_5 + 72.000000000007049*helper_6 + 72.000000000029445*helper_8 + 3.0815276193386688e-11*helper_9 + 93.333333333329008*x + 189.33333333332911*y) + 2.2410008012700788e-11*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_37(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = x*y;
double helper_6 = pow(y, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_4*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_5;
double helper_12 = 5.0448534238973e-13*helper_2;
double helper_13 = helper_0*y;
result = -(-helper_1*(-helper_0*helper_12 + 4.0856207306207568e-13*helper_0*helper_3 - 3.7836400679229717e-13*helper_0*helper_4 + 8.961720254774407e-13*helper_0*helper_6 + 6.6666666666770311*helper_0*x*y + 1.261213355974325e-13*helper_0*x + 8.7041485130593719e-14*helper_1*y - 5.0448534238973e-13*helper_10*helper_4 + 4.9560355819266948e-13*helper_10*helper_6 + 1.261213355974325e-13*helper_10*x + 2.5801583092286548e-13*helper_10*y - 27.999999999990237*helper_11 - helper_12 - 155.99999999998781*helper_13*helper_4 - 3.5127456499145123e-13*helper_13 - 127.99999999999935*helper_2*y - 360.00000000000443*helper_3*x + 4.0856207306207568e-13*helper_3 - 836.00000000005707*helper_4*helper_6 + 1.261213355974325e-13*helper_4 - 7.4606987254834652e-14*helper_6 - 259.99999999999454*helper_7*x - 193.3333333333253*helper_8 - 121.33333333331996*helper_9 + 34.666666666667552*x*y) + helper_11*(127.99999999999935*helper_4 + 769.33333333338737*helper_5 + 360.00000000000443*helper_6 + 936.00000000007094*helper_8 + 576.00000000006582*helper_9 - 34.666666666667986*x - 66.666666666668007*y) + 576.00000000006753*helper_2*helper_3 + helper_4*helper_7*(576.00000000006753*helper_5 + 576.00000000006582*x + 936.00000000007094*y - 66.666666666669002))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_38(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_5*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 7.0343730840262591e-13*helper_2;
double helper_14 = 127.99999999999935*helper_3;
double helper_15 = 1.7585932710065648e-13*x;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(576.00000000012483*helper_4 + 792.00000000014199*x + 1152.0000000001389*y - 66.666666666671048) + helper_1*(-helper_0*helper_15 + 17.333333333321548*helper_0*helper_4 + 820.00000000000637*helper_0*helper_8 + 332.00000000000688*helper_0*helper_9 - 31.999999999999496*helper_0*y - 2.0339285811130429e-13*helper_1*y - helper_10*helper_15 + 7.0343730840262591e-13*helper_10*helper_5 + 127.99999999999912*helper_10*helper_6 - 32.000000000000441*helper_11 + 115.99999999999061*helper_12 - helper_13*helper_16 + helper_13 - helper_14*helper_16 + helper_14 - 5.27577981301969e-13*helper_16*helper_5 - 95.999999999998664*helper_16*helper_6 + 216.00000000001859*helper_2*y + 704.00000000001569*helper_3*x - 98.666666666669329*helper_4 - 1.7585932710065648e-13*helper_5 - 31.99999999999994*helper_6 + 1484.0000000001439*helper_7 + 721.33333333333508*helper_8 + 265.33333333333712*helper_9) + helper_12*(1417.3333333334722*helper_4 + 216.00000000001859*helper_5 + 704.00000000001569*helper_6 + 1152.0000000001389*helper_8 + 792.00000000014199*helper_9 - 66.666666666669954*x - 98.66666666666967*y) + 576.00000000012483*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_39(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_6*x;
double helper_9 = helper_5*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 6.3948846218408007e-14*helper_2;
double helper_14 = 1.2523315717771382e-13*helper_3;
double helper_15 = 1.5987211554602002e-14*x;
double helper_16 = helper_0*y;
result = (helper_0*helper_7*(5.1216843266486259e-14*helper_4 + 71.999999999998806*x - 1.3215132113608593e-12*y - 189.33333333332956) - helper_1*(-helper_0*helper_13 - helper_0*helper_14 + helper_0*helper_15 + 625.3333333333228*helper_0*helper_4 - 4.7961634663805961e-14*helper_0*helper_5 + 223.9999999999998*helper_0*helper_6 + 339.99999999999449*helper_0*helper_8 + 127.99999999999991*helper_1*y + helper_10*helper_15 - 6.3948846218408007e-14*helper_10*helper_5 + 127.99999999999979*helper_10*helper_6 + 223.99999999999989*helper_11 + 339.99999999999335*helper_12 - helper_13 - helper_14 + 139.99999999999451*helper_16*helper_5 + 95.999999999999972*helper_16 - 71.999999999998835*helper_2*y + 1.2021997980449216e-12*helper_3*x + 285.33333333332951*helper_4 + 1.5987211554602002e-14*helper_5 + 95.999999999999972*helper_6 + 139.99999999999602*helper_7 + 625.33333333332439*helper_8 + 329.33333333332411*helper_9) - helper_12*(329.33333333332564*helper_4 - 71.999999999998835*helper_5 + 1.2021997980449216e-12*helper_6 + 1.3215132113608593e-12*helper_8 - 71.999999999998806*helper_9 + 189.33333333332953*x + 285.33333333332962*y) + 5.1216843266486259e-14*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_40(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 8.3133500083947787e-13*helper_2;
double helper_14 = 127.99999999999872*helper_3;
double helper_15 = 2.0783375020986947e-13*x;
double helper_16 = helper_0*y;
result = (helper_1*(helper_0*helper_13 + helper_0*helper_14 - helper_0*helper_15 + 287.99999999999801*helper_0*helper_4 + 1292.0000000000082*helper_0*helper_8 + 127.99999999999963*helper_1*y - helper_10*helper_15 + 255.99999999999838*helper_10*helper_4 + 8.3133500083947787e-13*helper_10*helper_6 + 159.99999999999943*helper_11 + 587.99999999998431*helper_12 + helper_13 + helper_14 + 686.66666666664332*helper_16*x + 32.00000000000076*helper_16 + 360.0000000000295*helper_2*y + 704.00000000002365*helper_3*x + 1972.0000000001842*helper_4*helper_6 + 32.000000000000135*helper_4 + 98.666666666658799*helper_5 - 2.0783375020986947e-13*helper_6 + 820.0000000000116*helper_7*y + 6.2350125062960787e-13*helper_7 + 1390.6666666666647*helper_8 + 886.66666666666993*helper_9) + helper_12*(704.00000000002365*helper_4 + 2038.6666666668402*helper_5 + 360.0000000000295*helper_6 + 1152.0000000001751*helper_8 + 936.00000000018019*helper_9 + 66.666666666658159*x + 98.666666666658486*y) + 576.00000000015234*helper_2*helper_3 + helper_4*helper_7*(576.00000000015234*helper_5 + 936.00000000018019*x + 1152.0000000001751*y + 66.66666666665698))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_41(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(x, 2);
double helper_8 = x*y;
double helper_9 = pow(y, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_9*x;
double helper_13 = helper_7*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = helper_0*y;
result = (helper_1*(1336.1865569273252*helper_0*helper_12 + 1336.1865569272309*helper_0*helper_13 + 5.2204715173529575e-12*helper_0*helper_4 + 5.6435514697690756e-12*helper_0*helper_7 - 5.8503837348468797e-12*helper_0*helper_9 - 8.2352780287860203e-13*helper_0*x + 7.086830436616207e-12*helper_10*helper_7 + 343.79149519893139*helper_10*helper_8 - 7.1536740865733855e-13*helper_10*x - 9.2796601069371826e-12*helper_10*y + 3.9572551911162819e-13*helper_11 + 228.72610882489857*helper_12 + 228.72610882478762*helper_13 - 3.1687525641069935e-12*helper_14*y + 4.7369515717340324e-14*helper_15 + 89.664380429872352*helper_16*x + 6.0347763975448381e-13*helper_16 - 5.8106605946606329e-13*helper_2 - 3.7304355695785182e-28*helper_3 + 1675.0617283950185*helper_4*y + 4.7167756668918997e-12*helper_4 + 1675.0617283950901*helper_5*x + 3.4886868580708686e-12*helper_5 + 3.4886868580708816e-12*helper_6 + 5400.5816186556285*helper_7*helper_9 - 8.2352780287860203e-13*helper_7 + 1.8728852309527233*helper_8 - 3.3088483941822605e-13*helper_9) + helper_11*helper_7*(2958.2222222223568*helper_7 + 9443.555555555793*helper_8 + 2958.2222222223018*helper_9 + 366.61728395038023*x + 366.61728395049875*y + 1.8728852309417623) + helper_15*y*(7022.6172839507499*helper_12 + 7022.6172839506899*helper_13 + 682.6666666667079*helper_4 + 682.66666666667777*helper_5 + 139.06172839492652*helper_7 + 595.34339277524862*helper_8 + 139.06172839504833*helper_9 + 1.8728852309570856*x + 1.8728852309482167*y) + 1820.4444444444666*helper_2*helper_3 + helper_4*helper_6*(4096.0000000001201*x + 4096.0000000000809*y + 227.55555555547679))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_42(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_7*x;
double helper_13 = helper_9*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*y;
double helper_16 = helper_0*y;
double helper_17 = helper_10*y;
result = -(-helper_1*(-1181.7613168724481*helper_0*helper_12 - 7.211085866521058e-12*helper_0*helper_4 + 5.6345367329378288e-12*helper_0*helper_7 - 5.3081489820926787e-12*helper_0*helper_9 + 4.7974792862627778e-13*helper_0*x - 3.3799684587518816e-12*helper_10*helper_7 + 3.5066599829652242e-13*helper_10*x - 6.8501583006802949e-12*helper_11 + 37.750342935513615*helper_12 + 85.157750343079755*helper_13 - 6.1580370432542454e-14*helper_14*x + 3.9448042454907904e-12*helper_15 - 750.35390946489917*helper_16*helper_9 + 37.750342935457439*helper_16*x + 1.071782351941171e-12*helper_16 - 157.76131687246246*helper_17*x + 1.2336390254095628e-11*helper_17 + 6.6317322004278518e-13*helper_2 + 5.5676049666265282e-28*helper_3 - 2048.0000000000236*helper_4*x - 7.2110858665210427e-12*helper_4 - 1104.5925925924946*helper_5*y - 3.6692953202608656e-12*helper_5 - 4.3632587255199022e-12*helper_6 - 5187.6872427980788*helper_7*helper_9 + 1.1040592308700703e-12*helper_7 + 3.5116598079075501*helper_8 + 4.7974792862627778e-13*helper_9) + helper_11*helper_7*(4437.3333333334131*helper_7 + 12288.000000000078*helper_8 + 3242.6666666667984*helper_9 - 47.407407407859139*x - 2.8492763703980017e-10*y - 3.5116598079220265) - helper_15*x*(-8874.666666666657*helper_12 - 7248.5925925924657*helper_13 - 1024.0000000000123*helper_4 - 512.00000000004343*helper_5 + 2.5565327632648405e-11*helper_7 + 85.157750343411905*helper_8 + 47.407407407605866*helper_9 + 3.5116598079009647*x + 3.5116598079144898*y) + 2730.6666666666706*helper_2*helper_3 + helper_4*helper_6*(5461.3333333334303*x + 6144.0000000000591*y - 2.2435444914549252e-10))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_43(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(x, 2);
double helper_8 = x*y;
double helper_9 = pow(y, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_9*x;
double helper_13 = helper_7*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = helper_0*x;
double helper_17 = helper_10*x;
double helper_18 = helper_0*helper_7;
result = (helper_1*(5.5604671910953163e-12*helper_0*helper_4 - 2.2212264035408417e-12*helper_0*helper_9 - 1.5012475335415458e-12*helper_0*y + 2.7035712489047069e-12*helper_10*helper_7 - 7.4072100381412386e-12*helper_10*y + 3.3544342429492255e-12*helper_11 - 188.4590763603037*helper_12 - 137.89117512583769*helper_13 - 2.2818241732938338e-12*helper_14*y + 3.4737644859382956e-14*helper_15 + 407.00137174212671*helper_16*helper_9 - 49.397347965212845*helper_16*y + 4.1755350891532838e-14*helper_16 + 34.063100137194084*helper_17*y + 1.088621648244332e-13*helper_17 + 230.01371742104607*helper_18*y + 2.0024146949626341e-12*helper_18 - 3.2842864230690256e-13*helper_2 - 3.6930377192568733e-28*helper_3 + 1055.6049382716214*helper_4*x + 5.5604671910953074e-12*helper_4 + 423.50617283943814*helper_5*y + 8.0721163635384623e-13*helper_5 + 1.2540640679540876e-12*helper_6 + 2082.0631001369479*helper_7*helper_9 + 4.1755350891532838e-14*helper_7 + 1.8728852309324617*helper_8 - 1.0257309405141674e-12*helper_9) + helper_11*helper_7*(1592.8888888889512*helper_7 + 6940.4444444444007*helper_8 + 2958.2222222222599*helper_9 - 316.04938271637133*x - 366.61728395083333*y + 1.8728852309241883) + helper_15*y*(4810.2716049381979*helper_12 + 3267.9506172837928*helper_13 + 682.66666666667311*helper_4 + 227.55555555557794*helper_5 - 88.493827160615282*helper_7 - 504.50845907669674*helper_8 - 139.06172839507292*helper_9 + 1.8728852309368427*x + 1.8728852309279933*y) + 1820.444444444438*helper_2*helper_3 + helper_4*helper_6*(3185.7777777778147*x + 4096.0000000000182*y - 227.55555555573991))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_44(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_9*y;
double helper_13 = pow(helper_0, 3);
result = -(-helper_1*(-1181.7613168722344*helper_0*helper_12 - 5.7751480780939e-12*helper_0*helper_4 - 750.35390946494999*helper_0*helper_7*x + 3.3988724043756669e-12*helper_0*helper_7 - 7.2001663890360602e-12*helper_0*helper_9 + 37.750342935460864*helper_0*x*y + 8.6686213762739776e-13*helper_0*x - 1.0775888640473118e-12*helper_0*y - 3.1652266541668386e-12*helper_10*helper_7 - 157.7613168724487*helper_10*x*y + 7.5791225147751091e-13*helper_10*x + 9.6480756625827057e-12*helper_10*y - 8.9433645674342424e-12*helper_11 + 2.6667648427878658e-12*helper_13*y + 8.7159908919909201e-13*helper_2 + 5.3852904463088283e-28*helper_3 - 1104.5925925925426*helper_4*x - 5.7751480780938815e-12*helper_4 - 2047.9999999998545*helper_5*y - 5.4948638232118068e-12*helper_5 - 6.2148804621153866e-12*helper_6 - 5187.6872427977933*helper_7*helper_9 + 85.157750343002007*helper_7*x - 3.5500820390364004e-13*helper_7 + 37.750342935744627*helper_9*y + 8.6686213762739776e-13*helper_9 + 3.5116598078923267*x*y) + helper_11*helper_7*(3242.6666666667738*helper_7 + 12288.000000000035*helper_8 + 4437.3333333335195*helper_9 - 6.9849193096162502e-10*x - 47.407407407944319*y - 3.5116598079059691) - helper_13*x*y*(-8874.666666666395*helper_12 - 512.00000000001455*helper_4 - 1024.0000000000571*helper_5 - 7248.5925925924003*helper_7*x + 47.407407407513368*helper_7 + 85.15775034373506*helper_8 + 2.7254524563129465e-10*helper_9 + 3.5116598078852022*x + 3.5116598078982761*y) + 2730.6666666666943*helper_2*helper_3 + helper_4*helper_6*(6144.0000000001628*x + 5461.3333333334394*y - 3.9578177772152741e-10))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_45(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_7*x;
double helper_14 = helper_9*y;
result = (helper_1*(923.75308641964921*helper_0*helper_13 + 923.75308641946413*helper_0*helper_14 + 1.0171638565720142e-11*helper_0*helper_4 - 2.9482013538288553e-12*helper_0*helper_7 + 13.00411522640205*helper_0*helper_8 + 6.9917405198797711e-12*helper_0*helper_9 - 4.6540549192295004e-13*helper_0*x - 3.9826441184798304e-13*helper_0*y + 6.9028075436703683e-12*helper_10*helper_7 + 155.75308641976756*helper_10*helper_8 - 3.4106051316492766e-13*helper_10*x - 1.3052421969040149e-11*helper_10*y + 8.9812601800081155e-12*helper_11 - 3.3540961503409836e-12*helper_12*y + 13.004115226231306*helper_13 + 13.004115226018243*helper_14 - 9.9475983006417419e-13*helper_2 - 8.0779356694631591e-28*helper_3 + 1535.9999999999354*helper_4*x + 1.0171638565720121e-11*helper_4 + 1535.9999999997724*helper_5*y + 4.3485215428522041e-12*helper_5 + 5.3432813729163492e-12*helper_6 + 5787.7530864188902*helper_7*helper_9 - 2.3381296898636453e-13*helper_7 + 1.2510288066526334*helper_8 - 4.6540549192295004e-13*helper_9) + helper_11*helper_7*(4864.0000000001037*helper_7 + 16383.999999999827*helper_8 + 4864.0000000001764*helper_9 - 9.7907104645856842e-10*x - 7.5138473221159074e-10*y + 1.2510288066333635) + helper_12*helper_8*(9727.9999999996198*helper_13 + 9727.9999999994579*helper_14 + 768.00000000001558*helper_4 + 768.00000000005912*helper_5 - 1.3008616406295914e-10*helper_7 + 13.004115225204941*helper_8 - 3.6538949643727935e-10*helper_9 + 1.2510288066630073*x + 1.2510288066431121*y) + 4095.9999999999991*helper_2*helper_3 + helper_4*helper_6*(8192.0000000001255*x + 8192.0000000000691*y - 5.7445959100733086e-10))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_46(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = helper_0*x;
double helper_15 = helper_9*x;
double helper_16 = -helper_0;
result = -(helper_0*helper_4*helper_5*(4778.6666666667152*x + 5461.3333333333539*y - 3.4818488832873741e-10) + helper_1*(508.57613168719013*helper_0*helper_12 + 418.50205761299151*helper_0*helper_13 - 8.3039199659101745e-13*helper_0*helper_8 - 1.1278897340561557e-12*helper_0*y + 5.5527818283087616e-12*helper_10 - 1.9637222803496256e-12*helper_11*y + 57.064471879225493*helper_12 + 9.6570644716797531*helper_13 + 9.6570644719053078*helper_14*y + 9.0002079862900619e-14*helper_14 + 77.16872427983931*helper_15*y + 1.5158245029544509e-13*helper_15 - 1.6674069532505725e-12*helper_16*helper_4 - 7.4596606897975807e-12*helper_16*helper_5 - 2.8042753304667231e-12*helper_16*helper_6 - 4.9264296346035215e-13*helper_2 - 5.3852904463087521e-28*helper_3 + 682.666666666526*helper_4*y + 1.0231815394947486e-12*helper_4 + 943.40740740738261*helper_5*x + 7.4596606897975678e-12*helper_5 + 2897.9094650200536*helper_6*helper_8 + 3.7895612573874258e-12*helper_6*helper_9 + 9.0002079862900619e-14*helper_6 - 3.5116598079251702*helper_7 - 5.8830306881193594e-13*helper_8 - 8.0072811664261341e-12*helper_9*y) + helper_10*helper_6*(2389.3333333334185*helper_6 + 9557.3333333331739*helper_7 + 3242.666666666717*helper_8 - 5.8450192833940567e-10*x + 47.407407406969142*y - 3.5116598079369936) + helper_11*helper_7*(5722.0740740738311*helper_12 + 4778.6666666663059*helper_13 + 341.33333333336384*helper_4 + 512.00000000000853*helper_5 - 2.137312549166402e-10*helper_6 + 57.064471878612494*helper_7 + 47.407407407345211*helper_8 - 3.5116598079185004*x - 3.5116598079314989*y) + 2730.6666666666556*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_47(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_7*x;
double helper_14 = helper_9*y;
double helper_15 = helper_0*helper_9;
double helper_16 = helper_0*helper_7;
result = (helper_1*(3.0534038946097039e-12*helper_0*helper_4 - 49.39734796521801*helper_0*helper_8 - 3.3228838062462068e-13*helper_0*x + 4.5175538359635091e-13*helper_0*y + 2.9013316669142961e-12*helper_10*helper_7 + 34.06310013717794*helper_10*helper_8 - 2.9991921155110236e-13*helper_10*x - 4.7149904770021803e-12*helper_10*y + 4.818707847009283e-12*helper_11 - 5.3685451146319007e-14*helper_12*x - 1.0387362361809544e-12*helper_12*y - 137.89117512583158*helper_13 - 188.45907636047895*helper_14 + 407.00137174195936*helper_15*y + 3.9424595275638797e-12*helper_15 + 230.01371742103106*helper_16*x - 1.3747485089011398e-13*helper_16 - 5.8106605946605925e-13*helper_2 - 3.431252767005792e-28*helper_3 + 1055.6049382714882*helper_4*y + 2.6002355275804776e-12*helper_4 + 423.50617283943399*helper_5*x + 3.914804161379326e-12*helper_5 + 3.9148041613793381e-12*helper_6 + 2082.0631001366323*helper_7*helper_9 + 4.3250177102251236e-13*helper_7 + 1.8728852309484418*helper_8 - 3.3228838062462068e-13*helper_9) + helper_11*helper_7*(1592.8888888889528*helper_7 + 6940.4444444443361*helper_8 + 2958.2222222223354*helper_9 - 366.61728395119536*x - 316.04938271654316*y + 1.8728852309409447) + helper_12*helper_8*(3267.9506172836655*helper_13 + 4810.2716049379524*helper_14 + 682.6666666667013*helper_4 + 227.55555555556398*helper_5 - 88.493827160599182*helper_7 - 504.50845907704513*helper_8 - 139.06172839525581*helper_9 + 1.8728852309531629*x + 1.8728852309452115*y) + 1820.4444444444594*helper_2*helper_3 + helper_4*helper_6*(4096.0000000000973*x + 3185.7777777778388*y - 227.55555555592261))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_48(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(x, 3);
double helper_5 = pow(y, 3);
double helper_6 = pow(y, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_6*x;
double helper_12 = helper_8*y;
double helper_13 = pow(helper_0, 3);
double helper_14 = helper_13*x;
double helper_15 = helper_0*y;
double helper_16 = helper_9*y;
double helper_17 = -helper_0;
result = -(helper_0*helper_4*helper_5*(5461.333333333404*x + 4778.6666666667024*y - 4.5831795971564993e-10) + helper_1*(418.50205761304238*helper_0*helper_11 - 1.4079272727101529e-13*helper_0*x + 5.0964335632335601e-12*helper_10 + 9.6570644717534861*helper_11 + 57.064471879047119*helper_12 - 1.3057794443339879e-12*helper_13*y - 7.1054273576010473e-14*helper_14 + 508.57613168702301*helper_15*helper_8 + 9.6570644719010446*helper_15*x + 4.5484420984428143e-14*helper_15 + 77.168724279828524*helper_16*x - 6.4164960496234201e-12*helper_16 - 2.7421686320817338e-12*helper_17*helper_4 - 6.1242350925330335e-12*helper_17*helper_5 - 1.1234360492676968e-13*helper_17*helper_6 - 3.992855429007624e-12*helper_17*helper_8 - 6.6317322004278023e-13*helper_2 - 5.1749275382498276e-28*helper_3 + 943.407407407238*helper_4*y + 2.1145225488269763e-12*helper_4 + 682.66666666658023*helper_5*x + 6.1242350925330182e-12*helper_5 + 2897.9094650198513*helper_6*helper_8 + 4.8563512607729155e-12*helper_6*helper_9 + 3.3663606881463865e-13*helper_6 - 3.5116598079158297*helper_7 - 1.4079272727101529e-13*helper_8 - 1.1118677994767512e-13*helper_9*x) + helper_10*helper_6*(2389.3333333333935*helper_6 + 9557.3333333331084*helper_7 + 3242.6666666667716*helper_8 + 47.407407406674288*x - 6.1127281014705659e-10*y - 3.5116598079273431) + helper_14*y*(4778.6666666662768*helper_11 + 5722.074074073601*helper_12 + 512.00000000003524*helper_4 + 341.33333333334213*helper_5 - 1.2461498499760637e-10*helper_6 + 57.06447187840142*helper_7 + 47.407407407155652*helper_8 - 3.5116598079091266*x - 3.5116598079214634*y) + 2730.6666666666624*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_49(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 4);
double helper_3 = pow(y, 4);
double helper_4 = pow(y, 3);
double helper_5 = pow(x, 3);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_7*x;
double helper_14 = helper_9*y;
double helper_15 = helper_0*x;
double helper_16 = helper_10*x;
double helper_17 = helper_0*helper_9;
double helper_18 = helper_0*helper_7;
result = (helper_1*(4.2804924366507697e-12*helper_0*helper_4 - 2.6129289252044484e-13*helper_0*y + 3.5452500560310959e-12*helper_10*helper_7 - 3.9831588720833393e-12*helper_10*y + 2.3302292879916367e-12*helper_11 - 4.1053580288361602e-14*helper_12*x - 7.6050612233558798e-13*helper_12*y + 127.59030635566631*helper_13 + 127.59030635558973*helper_14 + 39.096479195247909*helper_15*y + 7.8247422258989439e-14*helper_15 + 65.66803840876986*helper_16*y + 8.8510817331080975e-14*helper_16 + 324.82853223580497*helper_17*y + 1.7224082242778284e-12*helper_17 + 324.82853223587227*helper_18*x + 3.1930653822904682e-13*helper_18 - 3.2842864230690024e-13*helper_2 - 3.4686506173273346e-28*helper_3 + 486.71604938267774*helper_4*x + 4.2804924366507608e-12*helper_4 + 486.71604938261419*helper_5*y + 5.4755651316239774e-13*helper_5 + 9.5651333218876562e-13*helper_6 + 1835.5445816182557*helper_7*helper_9 + 1.141199617645187e-14*helper_7 + 1.8728852309280564*helper_8 + 7.8247422258989439e-14*helper_9) + helper_11*helper_7*(1592.8888888889178*helper_7 + 5802.6666666665324*helper_8 + 1592.8888888889385*helper_9 + 316.04938271564748*x + 316.04938271572934*y + 1.8728852309204287) + helper_12*helper_8*(3103.6049382714004*helper_13 + 3103.604938271334*helper_14 + 227.55555555556037*helper_4 + 227.55555555557351*helper_5 + 88.493827160433767*helper_7 + 443.63968907129669*helper_8 + 88.493827160348687*helper_9 + 1.872885230932273*x + 1.8728852309240795*y) + 1820.4444444444357*helper_2*helper_3 + helper_4*helper_6*(3185.7777777778037*x + 3185.7777777777869*y + 227.55555555531257))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_50(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 3.4015282073038801e-26*helper_2;
double helper_13 = 5.7553961596569943e-13*helper_3;
double helper_14 = 8.5038205182597004e-27*x;
double helper_15 = helper_0*y;
result = (helper_0*helper_7*(2.6651694767140311e-11*helper_4 + 3.8297802917259932e-11*x + 3.6936433404046677e-11*y + 431.99999999998829) + helper_1*(helper_0*helper_12 - helper_0*helper_13 - helper_0*helper_14 - 8.6330942394854081e-13*helper_0*helper_5 + 2.5511461554779082e-26*helper_0*helper_6 + 647.99999999999466*helper_0*helper_8 + 647.99999999999693*helper_0*helper_9 - 2.8776980798283396e-13*helper_1*y - helper_10*helper_14 - 8.6330942394853344e-13*helper_10*helper_5 + 3.4015282073038801e-26*helper_10*helper_6 - 3.5971225997854219e-13*helper_10*y + 647.99999999998488*helper_11 + helper_12 - helper_13 + 1079.9999999999734*helper_15*x + 7.1942451995741004e-14*helper_15 + 1.1742031419447303e-11*helper_2*y + 9.7091990209407227e-12*helper_3*x + 431.99999999998823*helper_4 + 1.1324013841668443e-26*helper_5 - 8.5038205182597004e-27*helper_6 + 648.00000000003376*helper_7 + 1079.9999999999832*helper_8 + 1079.9999999999854*helper_9) + helper_11*(1080.0000000000221*helper_4 + 9.7091990209407227e-12*helper_5 + 1.1742031419447303e-11*helper_6 + 3.6936433404046677e-11*helper_8 + 3.8297802917259932e-11*helper_9 + 431.99999999998835*x + 431.99999999998846*y) + 2.6651694767140311e-11*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_51(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 2.7315649742125347e-12*helper_2;
double helper_13 = 3.6420866322830789e-12*helper_3;
double helper_14 = 6.8289124355313368e-13*x;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(1845.2812500004031*helper_4 + 3075.4687500004652*x + 3075.468750000467*y + 34.171874999978037) + helper_1*(-helper_0*helper_13 - helper_0*helper_14 - 5.0533952022926942e-12*helper_0*helper_5 + 2101.5703125000305*helper_0*helper_8 - 6.6012820210119113e-13*helper_1*y - helper_10*helper_14 - 4.3022148343842708e-12*helper_10*helper_5 + 2.7315649742125347e-12*helper_10*helper_6 - 8.5930481480414303e-13*helper_10*y + 871.38281249996476*helper_11 - helper_12*helper_16 + helper_12 - helper_13 + 2101.5703125000268*helper_15*helper_6 + 905.55468749994839*helper_15*x + 2.6575850894942591e-12*helper_15 - 2.0486737306593994e-12*helper_16*helper_6 + 1230.1875000000673*helper_2*y + 1230.1875000000659*helper_3*x + 34.171874999983295*helper_4 + 7.511803679084914e-13*helper_5 - 6.8289124355313368e-13*helper_6 + 5177.0390625004857*helper_7 + 2135.7421875000064*helper_8 + 2135.7421875000095*helper_9) + helper_11*(5211.2109375004611*helper_4 + 1230.1875000000659*helper_5 + 1230.1875000000673*helper_6 + 3075.468750000467*helper_8 + 3075.4687500004652*helper_9 + 34.171874999981746*x + 34.171874999981839*y) + 1845.2812500004031*helper_2*helper_3)/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_52(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 2.1852519793700167e-12*helper_2;
double helper_14 = 2.1624889379181041e-12*helper_3;
double helper_15 = 5.4631299484250418e-13*x;
double helper_16 = helper_0*y;
result = (helper_1*(helper_0*helper_13 - helper_0*helper_14 - helper_0*helper_15 - 3.5339621853871158e-12*helper_0*helper_4 + 1588.9921875000173*helper_0*helper_8 - 3.6420866322821914e-13*helper_1*y - helper_10*helper_15 - 2.5266976011463237e-12*helper_10*helper_4 + 2.1852519793700167e-12*helper_10*helper_6 - 7.4548960754531664e-13*helper_11 + 358.80468749997112*helper_12 + helper_13 - helper_14 + 324.63281249996351*helper_16*x + 1.8153525557787645e-12*helper_16 + 615.09375000003888*helper_2*y + 1230.1875000000464*helper_3*x + 4049.3671875003661*helper_4*helper_6 + 4.6095158939841927e-13*helper_4 - 34.171875000008392*helper_5 - 5.4631299484250418e-13*helper_6 + 973.89843750000432*helper_7*y + 1.6389389845275112e-12*helper_7 + 1554.8203125000025*helper_8 + 939.72656249999511*helper_9) + helper_12*(1230.1875000000464*helper_4 + 4015.1953125003497*helper_5 + 615.09375000003888*helper_6 + 3075.4687500003683*helper_8 + 2460.3750000003593*helper_9 - 34.171875000009962*x - 34.17187500000972*y) + 1845.2812500003249*helper_2*helper_3 + helper_4*helper_7*(1845.2812500003249*helper_5 + 2460.3750000003593*x + 3075.4687500003683*y - 34.171875000013337))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_53(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = x*y;
double helper_6 = pow(x, 2);
double helper_7 = helper_0*helper_6;
double helper_8 = helper_4*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_11*x;
double helper_13 = 2.7315649742124285e-12*helper_2;
double helper_14 = 3.1868258032476448e-12*helper_3;
double helper_15 = 6.8289124355310713e-13*x;
double helper_16 = helper_0*y;
result = (helper_1*(helper_0*helper_13 - helper_0*helper_14 - helper_0*helper_15 - 3.6193235908312434e-12*helper_0*helper_4 + 973.89843750001785*helper_0*helper_8 - 1.5934129016231449e-13*helper_1*y - helper_10*helper_15 - 3.3461670934099599e-12*helper_10*helper_4 + 2.7315649742124285e-12*helper_10*helper_6 + 3.9835322540661365e-14*helper_11 + 358.80468749997385*helper_12 + helper_13 - helper_14 + 324.63281249996822*helper_16*x + 2.8738339832859489e-12*helper_16 + 1230.1875000000373*helper_2*y + 615.09375000004411*helper_3*x + 4049.3671875003552*helper_4*helper_6 + 1.0243368653296636e-12*helper_4 - 34.171875000005791*helper_5 - 6.8289124355310713e-13*helper_6 + 1588.992187500007*helper_7*y + 2.0486737306593199e-12*helper_7 + 939.72656250000387*helper_8 + 1554.8203125000007*helper_9) + helper_12*(615.09375000004411*helper_4 + 4015.1953125003411*helper_5 + 1230.1875000000373*helper_6 + 2460.3750000003565*helper_8 + 3075.4687500003456*helper_9 - 34.171875000007191*x - 34.171875000007738*y) + 1845.2812500003135*helper_2*helper_3 + helper_4*helper_7*(1845.2812500003135*helper_5 + 3075.4687500003456*x + 2460.3750000003565*y - 34.171875000010878))/helper_1;
return result;
}

inline POLYFEM_BOTH double pyramid_4_basis_value_3d_single_54(double x, double y, double z) {
double result;
double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = x*y;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = helper_5*helper_6;
double helper_8 = helper_5*x;
double helper_9 = helper_6*y;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_4;
double helper_12 = 2.1852519793699371e-12*helper_2;
double helper_13 = 2.0714367721109724e-12*helper_3;
double helper_14 = 5.4631299484248429e-13*x;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
result = -(helper_0*helper_7*(1845.2812500002565*helper_4 + 2460.3750000002701*x + 2460.3750000002874*y + 34.171874999992276) + helper_1*(-helper_0*helper_13 - helper_0*helper_14 + 290.46093749997232*helper_0*helper_4 - 2.6462035687681136e-12*helper_0*helper_5 + 871.38281250000682*helper_0*helper_8 - 4.5526082903481456e-14*helper_1*y - helper_10*helper_14 - 2.116962855014454e-12*helper_10*helper_5 + 2.1852519793699371e-12*helper_10*helper_6 - 7.3979884718185834e-14*helper_10*y + 256.2890624999747*helper_11 - helper_12*helper_16 + helper_12 - helper_13 + 871.38281249998829*helper_15*helper_6 + 2.0771275324740145e-12*helper_15 - 1.6389389845274514e-12*helper_16*helper_6 + 615.09375000001796*helper_2*y + 615.09375000003263*helper_3*x + 34.171874999996973*helper_4 + 7.3410808681960022e-13*helper_5 - 5.4631299484248429e-13*helper_6 + 3331.7578125002683*helper_7 + 905.55468749999682*helper_8 + 905.55468749998465*helper_9) + helper_11*(3365.9296875002578*helper_4 + 615.09375000003263*helper_5 + 615.09375000001796*helper_6 + 2460.3750000002874*helper_8 + 2460.3750000002701*helper_9 + 34.171874999995673*x + 34.171874999995268*y) + 1845.2812500002565*helper_2*helper_3)/helper_1;
return result;
}



POLYFEM_BOTH void pyramid_4_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_14(x[i], y[i], z[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_15(x[i], y[i], z[i]);
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_16(x[i], y[i], z[i]);
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_17(x[i], y[i], z[i]);
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_18(x[i], y[i], z[i]);
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_19(x[i], y[i], z[i]);
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_20(x[i], y[i], z[i]);
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_21(x[i], y[i], z[i]);
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_22(x[i], y[i], z[i]);
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_23(x[i], y[i], z[i]);
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_24(x[i], y[i], z[i]);
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_25(x[i], y[i], z[i]);
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_26(x[i], y[i], z[i]);
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_27(x[i], y[i], z[i]);
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_28(x[i], y[i], z[i]);
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_29(x[i], y[i], z[i]);
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_30(x[i], y[i], z[i]);
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_31(x[i], y[i], z[i]);
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_32(x[i], y[i], z[i]);
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_33(x[i], y[i], z[i]);
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_34(x[i], y[i], z[i]);
		break;
	case 35:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_35(x[i], y[i], z[i]);
		break;
	case 36:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_36(x[i], y[i], z[i]);
		break;
	case 37:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_37(x[i], y[i], z[i]);
		break;
	case 38:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_38(x[i], y[i], z[i]);
		break;
	case 39:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_39(x[i], y[i], z[i]);
		break;
	case 40:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_40(x[i], y[i], z[i]);
		break;
	case 41:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_41(x[i], y[i], z[i]);
		break;
	case 42:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_42(x[i], y[i], z[i]);
		break;
	case 43:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_43(x[i], y[i], z[i]);
		break;
	case 44:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_44(x[i], y[i], z[i]);
		break;
	case 45:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_45(x[i], y[i], z[i]);
		break;
	case 46:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_46(x[i], y[i], z[i]);
		break;
	case 47:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_47(x[i], y[i], z[i]);
		break;
	case 48:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_48(x[i], y[i], z[i]);
		break;
	case 49:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_49(x[i], y[i], z[i]);
		break;
	case 50:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_50(x[i], y[i], z[i]);
		break;
	case 51:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_51(x[i], y[i], z[i]);
		break;
	case 52:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_52(x[i], y[i], z[i]);
		break;
	case 53:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_53(x[i], y[i], z[i]);
		break;
	case 54:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = pyramid_4_basis_value_3d_single_54(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 807.56250000007196*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 397.06062099913544*y;
double helper_11 = x*y;
double helper_12 = 804.0077160494543*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (284.44444444445674*helper_0*helper_2*helper_3 + helper_1*(428.20629715364714*helper_0*helper_6 + 209.94102223365456*helper_0*y + 526.55025434385755*helper_11 + 1838.6502486283935*helper_13 + 970.66898148151677*helper_14 + 856.41259430728587*helper_15*y + 96.000000000001336*helper_15 + 42.666666666666487*helper_2 + 323.55632716050701*helper_3 + 48.000000000002061*helper_4 + 128.00000000000216*helper_5 + 263.27512717193429*helper_6 + 128.00000000000171*helper_7*x + 236.20552554869772*helper_7*y + 47.999999999999886*helper_7 + 42.666666666666671*helper_9 + 14.666666666666384*x + 44.179941129402209*y + 14.666666666666524*z - 13.666666666666524) + helper_11*helper_9*(helper_10 + 1608.0154320989088*helper_11 + helper_12 + 266.66666666667788*helper_4 + 202.66820987655061*x + 29.51327446273574) + 2*helper_13*helper_7*(248.88888888890159*helper_4 + 248.8888888888961*helper_6 + helper_8*x + 181.78549382720408*x + 181.78549382721039*y + 22.179941129401165) + 455.11111111112319*helper_2*pow(y, 4) + 3*helper_3*helper_5*(284.44444444445674*x + 284.4444444444531*y + 96.451388888930992) + helper_4*helper_6*helper_7*(helper_8 + 497.77777777780318*x + 181.78549382720408) + helper_9*y*(helper_10*x + helper_12*x + 804.00771604945442*helper_14 + 88.888888888892623*helper_2 + 88.888888888889767*helper_3 + 101.3341049382753*helper_4 + 101.3341049382815*helper_6 + 29.51327446273574*x + 29.513274462735268*y + 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 807.56250000007196*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 397.06062099913544*x;
double helper_11 = x*y;
double helper_12 = 804.00771604945442*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
val[1] = (284.4444444444531*helper_0*helper_2*helper_3 + helper_1*(856.41259430729428*helper_0*helper_11 + 428.20629715364294*helper_0*helper_6 + 209.94102223365456*helper_0*x + 95.999999999997627*helper_0*y + 526.55025434386857*helper_11 + 1838.6502486283935*helper_13 + 970.66898148152109*helper_14 + 42.6666666666666*helper_2 + 323.55632716050559*helper_3 + 47.999999999999559*helper_4 + 127.99999999999946*helper_5 + 263.27512717192877*helper_6 + 236.20552554869772*helper_7*x + 127.99999999999879*helper_7*y + 47.999999999998963*helper_7 + 42.666666666666195*helper_9 + 44.179941129402209*x + 14.666666666666597*y + 14.666666666666892*z - 13.666666666666892) + helper_11*helper_9*(helper_10 + 1608.0154320989086*helper_11 + helper_12 + 266.6666666666693*helper_4 + 202.668209876563*y + 29.513274462735268) + 2*helper_13*helper_7*(248.8888888888961*helper_4 + 248.88888888890159*helper_6 + helper_8*y + 181.78549382720408*x + 181.78549382721039*y + 22.179941129401165) + 455.11111111112319*helper_2*pow(x, 4) + 3*helper_3*helper_5*(284.44444444445674*x + 284.4444444444531*y + 96.451388888930992) + helper_4*helper_6*helper_7*(helper_8 + 497.77777777779221*y + 181.78549382721039) + helper_9*x*(helper_10*y + helper_12*y + 804.0077160494543*helper_14 + 88.888888888889767*helper_2 + 88.888888888892623*helper_3 + 101.3341049382815*helper_4 + 101.3341049382753*helper_6 + 29.51327446273574*x + 29.513274462735268*y + 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(284.44444444445674*x + 284.4444444444531*y + 96.451388888930992);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = 804.0077160494543*helper_12 + 804.00771604945442*helper_13 + 88.888888888892623*helper_3 + 88.888888888889767*helper_4 + 397.06062099913544*helper_7 + 101.3341049382753*helper_8 + 101.3341049382815*helper_9 + 29.51327446273574*x + 29.513274462735268*y + 1.0;
double helper_15 = pow(helper_0, 4);
double helper_16 = -helper_0;
double helper_17 = 170.66666666666595*helper_4;
double helper_18 = 192.00000000000267*helper_8;
double helper_19 = 170.66666666666956*helper_3;
double helper_20 = 191.99999999999525*helper_9;
double helper_21 = helper_6*x;
double helper_22 = helper_0*y;
double helper_23 = helper_6*y;
double helper_24 = helper_0*x;
double helper_25 = helper_0*helper_7;
double helper_26 = helper_6*helper_7;
double helper_27 = 1712.8251886145886*helper_0*helper_12 + 42.666666666666487*helper_1 + 3677.3004972567869*helper_10 + 170.66666666666669*helper_11*x + 170.66666666666478*helper_11*y + 64.0*helper_11 + 1053.1005086877371*helper_12 + 1053.1005086877151*helper_13 + 42.666666666666664*helper_15 + 42.6666666666666*helper_2 + 191.99999999999955*helper_21 + 1712.8251886145717*helper_22*helper_8 + 58.666666666667567*helper_22 + 191.99999999999585*helper_23 + 58.666666666666096*helper_24 + 839.76408893461826*helper_25 + 944.82210219479089*helper_26 + 1294.2253086420224*helper_3*y + 64.000000000002743*helper_3 + 1294.225308642028*helper_4*x + 63.999999999999417*helper_4 + 256.00000000000341*helper_6*helper_8 + 255.99999999999758*helper_6*helper_9 + 29.333333333333329*helper_6 + 176.71976451760884*helper_7 + 29.333333333332767*helper_8 + 29.333333333333194*helper_9 + 4.0*x + 4.0*y + 4.0*z - 4.0;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_10*(807.56250000007196*helper_7 + 248.88888888890159*helper_8 + 248.8888888888961*helper_9 + 181.78549382720408*x + 181.78549382721039*y + 22.179941129401165) + helper_11*(helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_0*helper_20 + helper_27) + 3*helper_14*helper_26 + helper_15*(128.00000000000171*helper_0*helper_8 + 127.99999999999879*helper_0*helper_9 + 42.666666666666664*helper_11 + 428.20629715364714*helper_12 + 428.20629715364294*helper_13 + 128.0*helper_21 + 95.999999999997925*helper_22 + 127.99999999999858*helper_23 + 95.999999999999773*helper_24 + 472.41105109739544*helper_25 + 42.666666666667389*helper_3 + 42.666666666666487*helper_4 + 48.0*helper_6 + 209.94102223365456*helper_7 + 48.000000000000668*helper_8 + 47.999999999998813*helper_9 + 14.666666666666524*x + 14.666666666666892*y + 14.666666666666664*z - 13.666666666666664) + helper_5) + 455.11111111112319*helper_1*helper_2 + helper_10*helper_6*(3230.2500000002879*helper_7 + 995.55555555560636*helper_8 + 995.55555555558442*helper_9 + 727.14197530881631*x + 727.14197530884155*y + 88.719764517604659) + 4*helper_11*helper_14*helper_7 + helper_15*(-helper_16*helper_17 - helper_16*helper_18 - helper_16*helper_19 - helper_16*helper_20 + helper_27))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 330.21527777776021*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 148.29586048245488*y;
double helper_11 = x*y;
double helper_12 = 87.986882716015629*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (284.44444444445236*helper_0*helper_2*helper_3 + helper_1*(-6.0236411179779417*helper_0*helper_6 + 12.606917295384916*helper_0*y - 81.454375285800239*helper_11 + 1.270618998529244*helper_13 + 95.997685185161671*helper_14 - 18.258701989045036*helper_15*y + 4.5632633474374437e-13*helper_15 + 42.666666666666494*helper_2 - 0.45061728395837974*helper_3 - 47.99999999999973*helper_4 + 3.5053441630836907e-13*helper_5 - 14.510366655242242*helper_6 + 4.7211617331619765e-13*helper_7*x + 5.0936428326485643*helper_7*y + 3.0592812234082594e-15*helper_7 + 14.666666666666671*x + 14.84660779607055*y + 3.4540271877191976e-15*z - 1.0000000000000036) + helper_11*helper_9*(-helper_10 + 383.09567901228149*helper_11 + helper_12 + 266.66666666667453*helper_4 - 202.66820987657033*x + 29.51327446273752) + 2*helper_13*helper_7*(248.88888888889772*helper_4 + 78.222222222226705*helper_6 + helper_8*x - 181.78549382721192*x - 107.56867283955444*y + 22.179941129403684) + 455.11111111111688*helper_2*pow(y, 4) + 3*helper_3*helper_5*(284.44444444445236*x + 170.66666666667143*y - 96.451388888925962) + helper_4*helper_6*helper_7*(helper_8 + 497.77777777779545*x - 181.78549382721192) + helper_9*y*(-helper_10*x + helper_12*x + 191.54783950614075*helper_14 + 88.888888888891515*helper_2 + 10.6666666666671*helper_3 - 101.33410493828517*helper_4 - 27.117283950627051*helper_6 + 29.51327446273752*x + 14.846607796070508*y - 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 330.21527777776021*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 148.29586048245488*x;
double helper_11 = x*y;
double helper_12 = 191.54783950614075*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (170.66666666667143*helper_0*helper_2*helper_3 + helper_1*(-9.1293509945225182*helper_0*helper_6 + 12.606917295384916*helper_0*x - 29.020733310484484*helper_11 + 1.270618998529244*helper_13 - 1.3518518518751392*helper_14 - 12.047282235955883*helper_15*x + 2.6066621712060094e-13*helper_15 - 3.7895612573872102e-14*helper_2 + 31.99922839505389*helper_3 + 1.0562703261207333e-12*helper_4 + 9.8699240938412655e-13*helper_5 - 40.72718764290012*helper_6 + 5.0936428326485643*helper_7*x + 5.3464257483943942e-13*helper_7*y - 3.1605443618517131e-13*helper_7 - 7.647915603665759e-14*helper_9 + 14.84660779607055*x - 4.3555308536819258e-14*y - 1.0068469025672859e-13*z + 1.0068469025672859e-13) + helper_11*helper_9*(-helper_10 + 175.97376543203126*helper_11 + helper_12 + 32.0000000000013*helper_4 - 54.234567901254103*y + 14.846607796070508) + 2*helper_13*helper_7*(78.222222222226705*helper_4 + 248.88888888889772*helper_6 + helper_8*y - 181.78549382721192*x - 107.56867283955444*y + 22.179941129403684) + 455.11111111111688*helper_2*pow(x, 4) + 3*helper_3*helper_5*(284.44444444445236*x + 170.66666666667143*y - 96.451388888925962) + helper_4*helper_6*helper_7*(helper_8 + 156.44444444445341*y - 107.56867283955444) + helper_9*x*(-helper_10*y + helper_12*y + 87.986882716015629*helper_14 + 10.6666666666671*helper_2 + 88.888888888891515*helper_3 - 27.117283950627051*helper_4 - 101.33410493828517*helper_6 + 29.51327446273752*x + 14.846607796070508*y - 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 3);
double helper_6 = helper_5*(284.44444444445236*x + 170.66666666667143*y - 96.451388888925962);
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_10*x;
double helper_13 = helper_9*y;
double helper_14 = -27.117283950627051*helper_10 + 87.986882716015629*helper_12 + 191.54783950614075*helper_13 + 88.888888888891515*helper_3 + 10.6666666666671*helper_5 - 148.29586048245488*helper_8 - 101.33410493828517*helper_9 + 29.51327446273752*x + 14.846607796070508*y - 1.0;
double helper_15 = pow(helper_0, 3);
double helper_16 = pow(helper_0, 4);
double helper_17 = helper_0*x;
double helper_18 = helper_0*y;
double helper_19 = helper_18*x;
double helper_20 = helper_0*helper_9;
double helper_21 = helper_0*helper_10;
double helper_22 = helper_7*y;
double helper_23 = helper_22*x;
double helper_24 = -24.094564471911767*helper_0*helper_10*x + 1.3159898791788353e-12*helper_0*helper_5 - 36.517403978090073*helper_0*helper_9*y - 4.0273876102691437e-13*helper_0*y + 42.666666666666494*helper_1 + 1.0692851496788788e-12*helper_10*helper_7 - 58.041466620968968*helper_10*x - 8.7110617073638515e-14*helper_10 + 2.541237997058488*helper_11 - 3.0591662414663036e-13*helper_15*y + 1.381610875087679e-14*helper_17 + 50.427669181539663*helper_19 - 3.7895612573872102e-14*helper_2 + 9.1265266948748875e-13*helper_20 + 5.2133243424120187e-13*helper_21 + 20.374571330594257*helper_23 + 127.99691358021556*helper_3*y - 63.999999999999638*helper_3 + 4.6737922174449209e-13*helper_4 - 1.802469135833519*helper_5*x + 1.4083604348276445e-12*helper_5 + 9.4423234663239531e-13*helper_7*helper_9 + 1.2237124893633038e-14*helper_7*x - 1.2642177447406852e-12*helper_7*y + 59.386431184282202*helper_8 - 162.90875057160048*helper_9*y + 29.333333333333343*helper_9 - 4.0*x;
val[2] = -(-helper_0*(2*helper_0*helper_11*(78.222222222226705*helper_10 + 330.21527777776021*helper_8 + 248.88888888889772*helper_9 - 181.78549382721192*x - 107.56867283955444*y + 22.179941129403684) + 3*helper_14*helper_23 + helper_15*helper_24 + helper_16*(1.3033310856030047e-13*helper_10 - 6.0236411179779417*helper_12 - 9.1293509945225182*helper_13 + 6.1185624468165188e-15*helper_17 - 6.3210887237034262e-13*helper_18 + 10.187285665297129*helper_19 + 4.7211617331619765e-13*helper_20 + 5.3464257483943942e-13*helper_21 - 2.2943746810997277e-13*helper_22 + 1.1684480543612302e-13*helper_3 + 3.2899746979470883e-13*helper_5 + 12.606917295384916*helper_8 + 2.2816316737187219e-13*helper_9 + 3.4540271877191976e-15*x - 1.0068469025672859e-13*y) + helper_3*helper_6) + 455.11111111111688*helper_1*helper_2 + helper_11*helper_7*(312.88888888890682*helper_10 + 1320.8611111110408*helper_8 + 995.5555555555909*helper_9 - 727.14197530884769*x - 430.27469135821775*y + 88.719764517614735) + 4*helper_14*helper_15*x*y + helper_16*helper_24 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 352.45138888888886*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 174.41015803610387*y;
double helper_11 = x*y;
double helper_12 = 224.90200617283739*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
val[0] = (170.66666666666742*helper_0*helper_2*helper_3 + helper_1*(118.65642146774724*helper_0*helper_11 + 59.328210733877711*helper_0*helper_6 + 2.0684688529906186e-13*helper_0*x + 39.724201245997087*helper_0*y + 133.68297039321351*helper_11 + 412.0159893689713*helper_13 + 129.35185185183687*helper_14 - 6.0001386575298941e-14*helper_2 + 43.117283950616155*helper_3 + 1.4329278504497109e-13*helper_4 + 2.1553129651391416e-13*helper_5 + 66.841485196611174*helper_6 + 3.0948083601996672e-13*helper_7*x + 26.877593449929265*helper_7*y - 2.8619082412572395e-15*helper_7 - 5.3290705182007514e-15*helper_9 - 2.3684757858696933e-15*x + 14.846607796068016*y - 1.1842378929348466e-15*z + 1.1842378929348466e-15) + helper_11*helper_9*(helper_10 + 449.80401234566625*helper_11 + helper_12 + 32.000000000002508*helper_4 + 54.23456790122048*x + 14.846607796068218) + 2*helper_13*helper_7*(78.222222222224332*helper_4 + 78.222222222223422*helper_6 + helper_8*x + 107.5686728394939*x + 107.5686728394987*y + 22.17994112940077) + 455.11111111110819*helper_2*pow(y, 4) + 3*helper_3*helper_5*(170.66666666666742*x + 170.66666666666657*y + 96.451388888884509) + helper_4*helper_6*helper_7*(helper_8 + 156.44444444444866*x + 107.5686728394939) + helper_9*y*(helper_10*x + helper_12*x + 224.90200617283313*helper_14 + 10.666666666667503*helper_2 + 10.666666666666888*helper_3 + 27.11728395061024*helper_4 + 27.117283950615121*helper_6 + 14.846607796068218*x + 14.846607796067751*y + 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 352.45138888888886*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 174.41015803610387*x;
double helper_11 = x*y;
double helper_12 = 224.90200617283313*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_4*x;
val[1] = (170.66666666666657*helper_0*helper_2*helper_3 + helper_1*(118.65642146775542*helper_0*helper_11 + 59.328210733873618*helper_0*helper_6 + 39.724201245997087*helper_0*x - 5.4454354834163984e-14*helper_0*y + 133.68297039322235*helper_11 + 2.9230898680077835e-13*helper_13 + 129.35185185184847*helper_14 - 7.7133066288276133e-29*helper_2 + 43.11728395061229*helper_3 + 5.1448585903678218e-13*helper_4 + 5.1448585903678339e-13*helper_5 + 412.0159893689713*helper_6*y + 66.841485196606754*helper_6 + 26.877593449929265*helper_7*x - 1.9076957996458532e-13*helper_7 - 2.6525030838138875e-14*helper_9 + 14.846607796068016*x + 5.8011808662719093e-14*y + 5.2924131222120068e-14*z - 5.2924131222120068e-14) + helper_11*helper_9*(helper_10 + 449.80401234567478*helper_11 + helper_12 + 32.000000000000668*helper_4 + 54.234567901230243*y + 14.846607796067751) + 2*helper_13*helper_6*(78.222222222223422*helper_4 + 78.222222222224332*helper_6 + helper_8*y + 107.5686728394939*x + 107.5686728394987*y + 22.17994112940077) + 455.11111111110819*helper_2*pow(x, 4) + 3*helper_3*helper_5*(170.66666666666742*x + 170.66666666666657*y + 96.451388888884509) + helper_4*helper_6*helper_7*(helper_8 + 156.44444444444684*y + 107.5686728394987) + helper_9*x*(helper_10*y + helper_12*y + 224.90200617283739*helper_14 + 10.666666666666888*helper_2 + 10.666666666667503*helper_3 + 27.117283950615121*helper_4 + 27.11728395061024*helper_6 + 14.846607796068218*x + 14.846607796067751*y + 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(170.66666666666742*x + 170.66666666666657*y + 96.451388888884509);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = 224.90200617283739*helper_12 + 224.90200617283313*helper_13 + 10.666666666667503*helper_3 + 10.666666666666888*helper_4 + 27.11728395061024*helper_6 + 174.41015803610387*helper_7 + 27.117283950615121*helper_8 + 14.846607796068218*x + 14.846607796067751*y + 1.0;
double helper_15 = pow(helper_0, 4);
double helper_16 = -helper_0;
double helper_17 = 6.8598114538237786e-13*helper_4;
double helper_18 = 2.8737506201855222e-13*helper_3;
double helper_19 = 4.1369377059812372e-13*helper_6;
double helper_20 = helper_0*y;
double helper_21 = helper_6*helper_8;
double helper_22 = helper_0*x;
double helper_23 = helper_0*helper_8;
double helper_24 = helper_9*x;
double helper_25 = helper_9*y;
double helper_26 = helper_7*helper_9;
double helper_27 = helper_20*x;
double helper_28 = 237.31284293551084*helper_0*helper_12 + 237.31284293549447*helper_0*helper_13 - 6.0001386575298941e-14*helper_1 + 5.8461797360155669e-13*helper_10 - 2.1316282072803006e-14*helper_11*x - 1.061001233525555e-13*helper_11*y + 267.3659407864447*helper_12 + 267.36594078642702*helper_13 - 7.7133066288276133e-29*helper_2 + 2.1169652488848027e-13*helper_20 + 824.03197873794261*helper_21 - 4.7369515717393865e-15*helper_22 - 1.0890870966832797e-13*helper_23 - 1.1447632965028958e-14*helper_24 - 7.630783198583413e-13*helper_25 + 107.51037379971706*helper_26 + 158.89680498398835*helper_27 + 172.46913580244916*helper_3*y + 1.9105704672662812e-13*helper_3 + 172.46913580246462*helper_4*x + 6.8598114538237624e-13*helper_4 + 6.1896167203993345e-13*helper_6*helper_9 - 4.7369515717393865e-15*helper_6 + 59.386431184272062*helper_7 + 1.1602361732543819e-13*helper_8;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_21*(78.222222222224332*helper_6 + 352.45138888888886*helper_7 + 78.222222222223422*helper_8 + 107.5686728394939*x + 107.5686728394987*y + 22.17994112940077) + helper_11*(helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_28) + 3*helper_14*helper_26 + helper_15*(3.0948083601996672e-13*helper_0*helper_6 + 59.328210733877711*helper_12 + 59.328210733873618*helper_13 - 3.8153915992917065e-13*helper_20 - 5.723816482514479e-15*helper_22 + 2.9230898680077835e-13*helper_23 - 1.5987211554602254e-14*helper_24 - 7.9575092514416626e-14*helper_25 + 53.755186899858529*helper_27 + 7.1843765504638054e-14*helper_3 + 1.7149528634559446e-13*helper_4 + 1.0342344264953093e-13*helper_6 + 39.724201245997087*helper_7 - 2.7227177417081992e-14*helper_8 - 1.1842378929348466e-15*x + 5.2924131222120068e-14*y) + helper_5) + 455.11111111110819*helper_1*helper_2 + helper_10*helper_6*(312.88888888889733*helper_6 + 1409.8055555555554*helper_7 + 312.88888888889369*helper_8 + 430.2746913579756*x + 430.27469135799481*y + 88.719764517603082) + 4*helper_11*helper_14*helper_7 + helper_15*(-helper_16*helper_17 - helper_16*helper_18 - helper_16*helper_19 + helper_28))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 330.21527777777135*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 148.29586048240884*y;
double helper_11 = x*y;
double helper_12 = 191.54783950616877*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (170.66666666666822*helper_0*helper_2*helper_3 + helper_1*(-9.1293509945063498*helper_0*helper_6 + 12.606917295385802*helper_0*y - 29.020733310470831*helper_11 + 1.2706189986153618*helper_13 - 1.3518518518535245*helper_14 - 12.047282235936834*helper_15*y + 5.7632910789445921e-14*helper_15 - 6.0001386575299812e-14*helper_2 + 31.99922839506614*helper_3 - 1.8710958708346807e-13*helper_4 - 1.0421293457812327e-13*helper_5 - 40.727187642883862*helper_6 + 1.1052887000714448e-13*helper_7*x + 5.0936428326509748*helper_7*y + 3.2665228546747284e-14*helper_7 + 3.5527136788005009e-15*helper_9 + 5.8471745963587825e-14*x + 14.846607796068492*y + 2.9235872981793912e-14*z - 2.9235872981793912e-14) + helper_11*helper_9*(-helper_10 + 175.973765432081*helper_11 + helper_12 + 32.000000000003809*helper_4 - 54.234567901241789*x + 14.846607796068785) + 2*helper_13*helper_7*(78.222222222225511*helper_4 + 248.88888888889079*helper_6 + helper_8*x - 107.56867283952414*x - 181.78549382717242*y + 22.179941129401517) + 455.11111111110836*helper_2*pow(y, 4) + 3*helper_3*helper_5*(170.66666666666822*x + 284.44444444444491*y - 96.451388888902457) + helper_4*helper_6*helper_7*(helper_8 + 156.44444444445102*x - 107.56867283952414) + helper_9*y*(-helper_10*x + helper_12*x + 87.986882716040498*helper_14 + 10.666666666667936*helper_2 + 88.888888888889255*helper_3 - 27.117283950620894*helper_4 - 101.33410493826891*helper_6 + 14.846607796068785*x + 29.513274462734927*y - 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 330.21527777777135*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 148.29586048240884*x;
double helper_11 = x*y;
double helper_12 = 87.986882716040498*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (284.44444444444491*helper_0*helper_2*helper_3 - helper_1*(6.0236411179684168*helper_0*helper_6 - 12.606917295385802*helper_0*x + 81.454375285767725*helper_11 - 1.2706189986153618*helper_13 - 95.997685185198421*helper_14 + 18.2587019890127*helper_15*x + 1.9071195811415967e-13*helper_15 - 42.666666666666664*helper_2 + 0.45061728395117484*helper_3 + 47.999999999998685*helper_4 - 1.3219366730666893e-12*helper_5 + 14.510366655235416*helper_6 - 5.0936428326509748*helper_7*x - 4.992285163223972e-13*helper_7*y + 5.5422884265125837e-13*helper_7 + 2.0267630547487718e-13*helper_9 - 14.846607796068492*x - 14.666666666666412*y + 2.1005095287349963e-13*z + 0.99999999999978995) + helper_11*helper_9*(-helper_10 + 383.09567901233754*helper_11 + helper_12 + 266.66666666666777*helper_4 - 202.66820987653782*y + 29.513274462734927) + 2*helper_13*helper_7*(248.88888888889079*helper_4 + 78.222222222225511*helper_6 + helper_8*y - 107.56867283952414*x - 181.78549382717242*y + 22.179941129401517) + 455.11111111110836*helper_2*pow(x, 4) + 3*helper_3*helper_5*(170.66666666666822*x + 284.44444444444491*y - 96.451388888902457) + helper_4*helper_6*helper_7*(helper_8 + 497.77777777778158*y - 181.78549382717242) + helper_9*x*(-helper_10*y + helper_12*y + 191.54783950616877*helper_14 + 88.888888888889255*helper_2 + 10.666666666667936*helper_3 - 101.33410493826891*helper_4 - 27.117283950620894*helper_6 + 14.846607796068785*x + 29.513274462734927*y - 1.0))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_4*(170.66666666666822*x + 284.44444444444491*y - 96.451388888902457);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_9*x;
double helper_12 = helper_8*y;
double helper_13 = 191.54783950616877*helper_11 + 87.986882716040498*helper_12 + 10.666666666667936*helper_3 + 88.888888888889255*helper_4 - 148.29586048240884*helper_7 - 27.117283950620894*helper_8 - 101.33410493826891*helper_9 + 14.846607796068785*x + 29.513274462734927*y - 1.0;
double helper_14 = pow(helper_0, 3);
double helper_15 = pow(helper_0, 4);
double helper_16 = helper_0*x;
double helper_17 = helper_0*y;
double helper_18 = helper_17*x;
double helper_19 = helper_6*x;
double helper_20 = helper_0*helper_8;
double helper_21 = helper_6*y;
double helper_22 = helper_21*x;
double helper_23 = -1.389505794374977e-13*helper_0*helper_3 + 1.7625822307555857e-12*helper_0*helper_4 - 24.094564471873667*helper_0*helper_8*y - 36.517403978025399*helper_0*helper_9*x - 3.8142391622831935e-13*helper_0*helper_9 - 8.402038114939985e-13*helper_0*y - 6.0001386575299812e-14*helper_1 + 2.5412379972307235*helper_10 + 1.4210854715202004e-14*helper_14*x - 8.1070522189950871e-13*helper_14*y + 1.1694349192717565e-13*helper_16 + 50.427669181543209*helper_18 + 1.3066091418698914e-13*helper_19 + 42.666666666666664*helper_2 + 1.1526582157889184e-13*helper_20 + 20.374571330603899*helper_22 - 1.8024691358046994*helper_3*y - 2.494794494446241e-13*helper_3 + 127.99691358026456*helper_4*x - 63.999999999998245*helper_4 + 2.2105774001428896e-13*helper_6*helper_8 + 9.984570326447944e-13*helper_6*helper_9 - 2.2169153706050335e-12*helper_6*y + 59.386431184273967*helper_7 - 58.041466620941662*helper_8*y + 1.1694349192717565e-13*helper_8 - 162.90875057153545*helper_9*x + 29.333333333332824*helper_9 - 4.0*y;
double helper_24 = helper_0*helper_9;
val[2] = -(4*helper_0*helper_3*helper_5 - helper_0*(3*helper_13*helper_22 + helper_14*helper_23 + helper_15*(-9.1293509945063498*helper_11 - 6.0236411179684168*helper_12 + 6.5330457093494568e-14*helper_16 - 1.1084576853025167e-12*helper_17 + 10.18728566530195*helper_18 + 1.0658141036401503e-14*helper_19 + 1.1052887000714448e-13*helper_20 - 6.0802891642463148e-13*helper_21 + 4.992285163223972e-13*helper_24 - 3.4737644859374424e-14*helper_3 + 4.4064555768889643e-13*helper_4 + 12.606917295385802*helper_7 + 2.881645539472296e-14*helper_8 - 9.5355979057079837e-14*helper_9 + 2.9235872981793912e-14*x - 2.1005095287349963e-13*y) + 2*helper_24*helper_8*(330.21527777777135*helper_7 + 78.222222222225511*helper_8 + 248.88888888889079*helper_9 - 107.56867283952414*x - 181.78549382717242*y + 22.179941129401517) + helper_3*helper_5) + 455.11111111110836*helper_1*helper_2 + helper_10*helper_6*(1320.8611111110854*helper_7 + 312.88888888890204*helper_8 + 995.55555555556316*helper_9 - 430.27469135809656*x - 727.14197530868967*y + 88.719764517606066) + 4*helper_13*helper_14*x*y + helper_15*helper_23)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 1.5527832046920039e-12*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 55.69444444444165*y;
double helper_8 = x*y;
double helper_9 = 2.1560283287910506e-12*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 5.9302117590033562e-27*helper_2;
double helper_15 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_14 + 2.9651058795016756e-27*helper_0*x - 4.9418431325027968e-28*helper_10 + 3.9534745060022374e-27*helper_11 + 2.0384777711032349e-12*helper_12 + 35.104166666663318*helper_13 + helper_14 + 70.208333333328028*helper_15*x + 55.694444444438787*helper_15 + 5.7206636082416129e-13*helper_3 + 70.208333333332334*helper_5*x + 55.694444444439384*helper_5 + 35.104166666663886*helper_6 + 111.38888888887898*helper_8 - 9.8836862650055936e-28*x + 20.590277777775469*y - 4.9418431325027968e-28*z + 4.9418431325027968e-28) + helper_11*y*(helper_7 + 4.451216578012604e-12*helper_8 + helper_9 + 1.3589851807354899e-12*x + 20.590277777775476) + helper_13*(2.225608289006302e-12*helper_12 + 6.7949259036774496e-13*helper_2 + 5.7206636082416129e-13*helper_5 + helper_7*x + helper_9*x + 20.590277777775476*x + 20.59027777777548*y) + 4.6583496140760117e-12*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 2.225608289006302e-12) + 2*helper_6*x*(helper_4*x + 2.225608289006302e-12*x + 2.1560283287910506e-12*y + 20.590277777775473))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 1.5527832046920039e-12*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 55.69444444444165*x;
double helper_8 = x*y;
double helper_9 = 2.225608289006302e-12*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 9.3536289824666031e-14*helper_3;
double helper_14 = helper_0*x;
val[1] = (helper_1*(-helper_0*helper_13 - 9.3536289824665084e-14*helper_0*y - 1.5589381637443429e-14*helper_1 - 9.3536289824664213e-14*helper_10*y - 1.9486727046804272e-14*helper_10 + 35.104166666663318*helper_11 + 1.7161990824724839e-12*helper_12 - helper_13 + 70.208333333327772*helper_14*y + 55.694444444438787*helper_14 + 6.7949259036774496e-13*helper_2 + 70.208333333332334*helper_5*y + 55.69444444443949*helper_5 + 35.104166666664014*helper_6 + 111.38888888887877*helper_8 + 20.590277777775469*x + 1.3104755812818618e-27*y + 3.8973454093627417e-15*z - 3.8973454093627417e-15) + helper_11*y*(helper_7 + 4.3120566575821013e-12*helper_8 + helper_9 + 1.1441327216483226e-12*y + 20.59027777777548) + helper_11*(2.1560283287910506e-12*helper_12 + 5.7206636082416129e-13*helper_3 + 6.7949259036774496e-13*helper_5 + helper_7*y + helper_9*y + 20.590277777775476*x + 20.59027777777548*y) + 4.6583496140760117e-12*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 2.1560283287910506e-12) + 2*helper_6*y*(helper_4*y + 2.225608289006302e-12*x + 2.1560283287910506e-12*y + 20.590277777775473))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 1.9767372530011187e-27*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 3.1178763274888542e-14*helper_10;
double helper_12 = pow(z, 5);
double helper_13 = pow(z, 6);
double helper_14 = helper_9*x;
double helper_15 = 5.7206636082416098e-13*x;
double helper_16 = helper_1*x;
double helper_17 = helper_12*x;
double helper_18 = helper_6*y;
double helper_19 = 6.7949259036774435e-13*helper_7;
double helper_20 = 7.9069490120044749e-27*helper_7;
double helper_21 = helper_0*y;
double helper_22 = 1.2471505309955417e-13*helper_10;
double helper_23 = helper_6*helper_9;
double helper_24 = 2.156028328791051e-12*helper_6;
double helper_25 = helper_0*helper_6;
double helper_26 = helper_1*helper_6;
double helper_27 = helper_2*helper_6;
double helper_28 = 2.2256082890063016e-12*helper_7*helper_9;
double helper_29 = helper_10*helper_7;
double helper_30 = helper_0*helper_10;
val[2] = -(-190.03472222220785*helper_0*helper_14 + 55.694444444441672*helper_0*helper_23 + helper_0*helper_28 + 367.9166666666336*helper_0*helper_3 - 1.1860423518006709e-26*helper_0*helper_7 - 6.5475402877264629e-13*helper_0*helper_9 - 6.9185803855039155e-27*helper_0*x + 112.66666666666657*helper_0 + helper_1*helper_20 - helper_1*helper_22 - 479.30555555551121*helper_1*helper_3 + 7.4829031859731865e-13*helper_1*helper_9 - 5.3003897567307e-13*helper_1*y - 326.66666666666657*helper_1 + helper_10*helper_15 - helper_10*helper_24 - 1.144132721648323e-12*helper_10*helper_4 + helper_11*helper_2 + helper_11 - 3.9534745060022374e-27*helper_12*helper_6 + 9.3536289824664895e-14*helper_12*helper_9 - 2.416354153803732e-13*helper_12*y - 462.66666666666663*helper_12 + 4.6768144912330264e-14*helper_13*y + 218.66666666666663*helper_13 - 35.104166666663886*helper_14*helper_2 - 14.513888888888403*helper_14 + helper_15*helper_30 + 140.41666666665554*helper_16*helper_9 + 7.9069490120044763e-27*helper_16 - 70.208333333326649*helper_17*y + 9.8836862650055936e-28*helper_17 + 99.236111111105075*helper_18*z - 14.513888888888538*helper_18 + helper_19*helper_21 + helper_19*y + 295.34722222219432*helper_2*helper_3 - helper_2*helper_8 - 4.2091330421098824e-13*helper_2*helper_9 - 4.4476588192525178e-27*helper_2*x + 5.0275755780754966e-13*helper_2*y + 518.33333333333326*helper_2 + helper_20*z + 2.8840356029269156e-13*helper_21 - helper_22*z - 70.208333333332391*helper_23*z + 14.513888888890705*helper_23 + helper_24*helper_30 - 190.03472222220859*helper_25*y + 3.063942742151735e-26*helper_25 + 140.41666666665606*helper_26*y - 3.3604533301019021e-26*helper_26 - 35.104166666664014*helper_27*y + 1.8284819590260352e-26*helper_27 - helper_28 + 3.1055664093840078e-12*helper_29*z + 1.5527832046920045e-12*helper_29 - 128.263888888878*helper_3*z + 14.513888888887848*helper_3 + 1.8707257964933241e-13*helper_30 + 99.236111111104606*helper_4*helper_9 + 2.9651058795016781e-27*helper_4 - 1.3589851807354905e-12*helper_5*helper_7 - 7.0152217368488006e-14*helper_5 - 1.3837160771007837e-26*helper_6*z + 2.4709215662513998e-27*helper_6 - helper_8 + 2.8060886947399049e-13*helper_9*z - 4.6768144912331356e-14*helper_9 - 4.9418431325027968e-28*x + 3.897345409359001e-15*y - 42.666666666666657*pow(z, 7) - 18.666666666666629*z + 1.0)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2721.7777777779193*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 528.51486053956387*y;
double helper_11 = x*y;
double helper_12 = 2105.5308641976417*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_5*y;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_1*(671.74577046180764*helper_11 + 4270.2167352539109*helper_13 + 2957.6296296296496*helper_14 + 1794.2661179698218*helper_15*x + 149.99634202105338*helper_15 - 384.00000000000671*helper_16*helper_5 - 575.00960219480862*helper_16*helper_6 - 128.00000000000438*helper_16*x + 170.66666666666598*helper_2 + 521.53086419754936*helper_3 + 96.00000000000631*helper_5 + 212.86053955191807*helper_6 + 256.00000000000557*helper_7*x + 266.81207133060059*helper_7*y + 31.999999999999613*helper_7 + 42.666666666666693*helper_9 + 10.666666666665833*x + 16.517604023790135*y + 5.3333333333329165*z - 5.3333333333329165) + helper_11*helper_9*(helper_10 + 5111.3086419755182*helper_11 + helper_12 + 1066.6666666667088*helper_5 + 371.75308641970543*x + 16.517604023791243) + 2*helper_13*helper_7*(995.55555555560318*helper_5 + 739.55555555558215*helper_6 + helper_8*x + 315.6543209876541*x + 256.64197530867182*y + 11.184270690453683) + 1137.7777777778233*helper_2*helper_4 + 1820.4444444444873*helper_2*pow(y, 4) + 3*helper_4*helper_5*(1137.7777777778233*x + 1024.0000000000314*y + 161.77777777780852) + helper_5*helper_6*helper_7*(helper_8 + 1991.1111111112064*x + 315.6543209876541) + helper_9*y*(helper_10*x + helper_12*x + 2555.6543209877591*helper_14 + 355.55555555556964*helper_2 + 170.66666666666987*helper_3 + 185.87654320985271*helper_5 + 94.864197530870626*helper_6 + 16.517604023791243*x + 11.184270690455788*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2721.7777777779193*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 528.51486053956387*x;
double helper_11 = x*y;
double helper_12 = 2555.6543209877591*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_5*x;
double helper_15 = -helper_0;
val[1] = -(helper_1*(1150.0192043896172*helper_0*helper_11 + 149.99634202105338*helper_0*x - 4.6412663378780902e-12*helper_0*y + 425.72107910383613*helper_11 + 4270.2167352539109*helper_13 + 1564.5925925926481*helper_14 - 4.3588086957838105e-13*helper_15*helper_5 - 897.13305898491092*helper_15*helper_6 - 2.2737367544323246e-13*helper_2 + 985.87654320988327*helper_3 + 7.9825766481602063e-13*helper_5 + 335.87288523090382*helper_6 + 266.81207133060059*helper_7*x - 1.6580573219845601e-12*helper_7*y - 2.7429456359049438e-12*helper_7 - 1.0477450335469493e-12*helper_9 + 16.517604023790135*x + 1.2003155671050515e-13*y + 7.2438869412868359e-13*z - 7.2438869412868359e-13) + helper_11*helper_9*(helper_10 + 4211.0617283952834*helper_11 + helper_12 + 512.00000000000955*helper_5 + 189.72839506174125*y + 11.184270690455788) + 2*helper_13*helper_7*(739.55555555558215*helper_5 + 995.55555555560318*helper_6 + helper_8*y + 315.6543209876541*x + 256.64197530867182*y + 11.184270690453683) + 1024.0000000000314*helper_2*helper_4 + 1820.4444444444873*helper_2*pow(x, 4) + 3*helper_4*helper_5*(1137.7777777778233*x + 1024.0000000000314*y + 161.77777777780852) + helper_5*helper_6*helper_7*(helper_8 + 1479.1111111111643*y + 256.64197530867182) + helper_9*x*(helper_10*y + helper_12*y + 2105.5308641976417*helper_14 + 170.66666666666987*helper_2 + 355.55555555556964*helper_3 + 94.864197530870626*helper_5 + 185.87654320985271*helper_6 + 16.517604023791243*x + 11.184270690455788*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(1137.7777777778233*x + 1024.0000000000314*y + 161.77777777780852);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = 2105.5308641976417*helper_12 + 2555.6543209877591*helper_13 + 355.55555555556964*helper_3 + 170.66666666666987*helper_4 + 528.51486053956387*helper_7 + 185.87654320985271*helper_8 + 94.864197530870626*helper_9 + 16.517604023791243*x + 11.184270690455788*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = 5.8117449277117473e-13*helper_4;
double helper_17 = 256.00000000000875*helper_8;
double helper_18 = 512.00000000000898*helper_3;
double helper_19 = helper_0*x;
double helper_20 = helper_0*y;
double helper_21 = helper_6*x;
double helper_22 = helper_0*helper_9;
double helper_23 = helper_6*y;
double helper_24 = helper_6*helper_7;
double helper_25 = helper_19*y;
double helper_26 = 3588.5322359396437*helper_0*helper_13 + 170.66666666666598*helper_1 + 8540.4334705078218*helper_10 + 170.66666666666677*helper_11*x - 4.1909801341877972e-12*helper_11*y + 851.44215820767226*helper_12 + 1343.4915409236153*helper_13 + 2300.0384087792345*helper_19*helper_9 + 21.333333333331666*helper_19 - 2.2737367544323246e-13*helper_2 + 2.8975547765147343e-12*helper_20 + 127.99999999999845*helper_21 - 9.2825326757561804e-12*helper_22 - 1.0971782543619775e-11*helper_23 + 1067.2482853224024*helper_24 + 599.98536808421352*helper_25 + 3943.5061728395331*helper_3*y + 128.00000000000841*helper_3 + 2086.1234567901975*helper_4*x + 1.0643435530880275e-12*helper_4 + 512.00000000001114*helper_6*helper_8 - 3.3161146439691202e-12*helper_6*helper_9 + 66.070416095160539*helper_7 + 21.333333333331666*helper_8 + 2.400631134210103e-13*helper_9;
double helper_27 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_10*(2721.7777777779193*helper_7 + 995.55555555560318*helper_8 + 739.55555555558215*helper_9 + 315.6543209876541*x + 256.64197530867182*y + 11.184270690453683) + helper_11*(-helper_16*helper_27 - helper_17*helper_27 - helper_18*helper_27 + helper_26) + 3*helper_14*helper_24 + helper_15*(575.00960219480862*helper_12 + 897.13305898491092*helper_13 + 63.999999999999226*helper_19 - 5.4858912718098876e-12*helper_20 + 128.00000000000009*helper_21 - 1.6580573219845601e-12*helper_22 - 3.1432351006408479e-12*helper_23 + 533.62414266120118*helper_25 - 256.00000000000557*helper_27*helper_8 + 128.00000000000225*helper_3 + 1.4529362319279368e-13*helper_4 + 149.99634202105338*helper_7 + 64.000000000002188*helper_8 - 2.3206331689390451e-12*helper_9 + 5.3333333333329165*x + 7.2438869412868359e-13*y) + helper_5) + 1820.4444444444873*helper_1*helper_2 + helper_10*helper_6*(10887.111111111677*helper_7 + 3982.2222222224127*helper_8 + 2958.2222222223286*helper_9 + 1262.6172839506164*x + 1026.5679012346873*y + 44.737082761814733) + 4*helper_11*helper_14*helper_7 + helper_15*(helper_0*helper_16 + helper_0*helper_17 + helper_0*helper_18 + helper_26))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 3413.3333333334008*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 82.60425240072189*y;
double helper_11 = x*y;
double helper_12 = 1995.9629629629549*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (1706.666666666729*helper_0*helper_2*helper_3 + helper_1*(206.53292181069202*helper_0*helper_6 - 67.900548696816301*helper_0*y - 135.80109739377244*helper_11 + 3399.7325102878676*helper_13 + 3199.9999999999227*helper_14 + 1157.139917695401*helper_15*y - 31.99999999999465*helper_15 + 255.99999999999886*helper_2 + 289.29629629628738*helper_3 + 6.9704242378069512e-12*helper_4 + 384.00000000000762*helper_5 - 82.604252400551928*helper_6 + 128.00000000000654*helper_7*x + 45.236625514415714*helper_7*y - 16.000000000000416*helper_7 + 1.4210854715202004e-14*helper_9 - 8.0000000000008473*x - 13.137174211226265*y - 4.0000000000004237*z + 4.0000000000004237) + helper_11*helper_9*(-helper_10 + 5973.3333333333012*helper_11 + helper_12 + 1600.0000000000582*helper_4 - 1.363105184282231e-10*x - 13.137174211224409) + 2*helper_13*helper_7*(1493.3333333333992*helper_4 + 810.66666666670278*helper_6 + helper_8*x - 1.6090477098866693e-10*x - 30.703703703822818*y - 9.1371742112297678) + 2730.6666666667229*helper_2*pow(y, 4) + 3*helper_3*helper_5*(1706.666666666729*x + 1365.3333333333751*y - 8.4005099173134069e-11) + helper_4*helper_6*helper_7*(helper_8 + 2986.6666666667984*x - 1.6090477098866693e-10) - helper_9*y*(helper_10*x - helper_12*x - 2986.6666666666506*helper_14 - 533.3333333333527*helper_2 - 128.00000000000418*helper_3 + 6.815525921411155e-11*helper_4 + 30.703703703728884*helper_6 + 13.137174211224409*x + 9.1371742112275136*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 3413.3333333334008*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 82.60425240072189*x;
double helper_11 = x*y;
double helper_12 = 2986.6666666666506*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (1365.3333333333751*helper_0*helper_2*helper_3 + helper_1*(578.56995884770049*helper_0*helper_6 - 67.900548696816301*helper_0*x - 165.20850480110386*helper_11 + 3399.7325102878676*helper_13 + 867.88888888886208*helper_14 + 413.06584362138403*helper_15*x - 2.3230525870569848e-12*helper_15 - 3.0316490059097667e-13*helper_2 + 1066.6666666666408*helper_3 + 3.631131061127933e-12*helper_4 + 3.1479620008110819e-12*helper_5 - 67.900548696886219*helper_6 + 45.236625514415714*helper_7*x + 6.1298565674706518e-13*helper_7*y - 2.7637447657369339e-12*helper_7 - 8.3282991842643894e-13*helper_9 - 13.137174211226265*x + 4.3013329523672931e-13*y + 7.4573326480586356e-13*z - 7.4573326480586356e-13) + helper_11*helper_9*(-helper_10 + 3991.9259259259097*helper_11 + helper_12 + 384.00000000001251*helper_4 - 61.407407407457768*y - 9.1371742112275136) + 2*helper_13*helper_7*(810.66666666670278*helper_4 + 1493.3333333333992*helper_6 + helper_8*y - 1.6090477098866693e-10*x - 30.703703703822818*y - 9.1371742112297678) + 2730.6666666667229*helper_2*pow(x, 4) + 3*helper_3*helper_5*(1706.666666666729*x + 1365.3333333333751*y - 8.4005099173134069e-11) + helper_4*helper_6*helper_7*(helper_8 + 1621.3333333334056*y - 30.703703703822818) - helper_9*x*(helper_10*y - helper_12*y - 1995.9629629629549*helper_14 - 128.00000000000418*helper_2 - 533.3333333333527*helper_3 + 30.703703703728884*helper_4 + 6.815525921411155e-11*helper_6 + 13.137174211224409*x + 9.1371742112275136*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(1706.666666666729*x + 1365.3333333333751*y - 8.4005099173134069e-11);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_9*x;
double helper_12 = helper_8*y;
double helper_13 = -1995.9629629629549*helper_11 - 2986.6666666666506*helper_12 - 533.3333333333527*helper_3 - 128.00000000000418*helper_4 + 82.60425240072189*helper_7 + 6.815525921411155e-11*helper_8 + 30.703703703728884*helper_9 + 13.137174211224409*x + 9.1371742112275136*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = pow(helper_0, 4);
double helper_17 = -helper_0;
double helper_18 = 4.1972826677481092e-12*helper_4;
double helper_19 = 512.00000000001012*helper_3;
double helper_20 = helper_0*y;
double helper_21 = helper_0*helper_9;
double helper_22 = helper_6*y;
double helper_23 = helper_0*helper_8;
double helper_24 = helper_6*x;
double helper_25 = helper_0*x;
double helper_26 = helper_22*x;
double helper_27 = helper_20*x;
double helper_28 = 255.99999999999886*helper_1 + 6799.4650205757353*helper_10 - 330.41700960220771*helper_11 - 271.60219478754487*helper_12 - 3.3313196737057558e-12*helper_14*y + 5.6843418860808015e-14*helper_15 - 3.0316490059097667e-13*helper_2 + 2314.279835390802*helper_20*helper_8 + 2.9829330592234542e-12*helper_20 + 826.13168724276807*helper_21*x - 4.6461051741139695e-12*helper_21 - 1.1054979062947735e-11*helper_22 - 63.999999999989299*helper_23 - 64.000000000001663*helper_24 - 16.000000000001695*helper_25 + 180.94650205766285*helper_26 - 271.6021947872652*helper_27 + 4266.6666666665633*helper_3*y + 9.2938989837426016e-12*helper_3 + 1157.1851851851495*helper_4*x + 4.8415080815039105e-12*helper_4 + 256.00000000001307*helper_6*helper_8 + 1.2259713134941304e-12*helper_6*helper_9 - 52.548696844905059*helper_7 - 16.000000000001695*helper_8 + 8.6026659047345861e-13*helper_9;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_10*(3413.3333333334008*helper_7 + 1493.3333333333992*helper_8 + 810.66666666670278*helper_9 - 1.6090477098866693e-10*x - 30.703703703822818*y - 9.1371742112297678) - 3*helper_13*helper_26 + helper_14*(helper_0*helper_18 + helper_0*helper_19 + helper_28) + helper_16*(206.53292181069202*helper_11 + 578.56995884770049*helper_12 - 5.5274895314738677e-12*helper_20 + 6.1298565674706518e-13*helper_21 - 2.4984897552793167e-12*helper_22 + 128.00000000000654*helper_23 + 4.2632564145606011e-14*helper_24 - 32.000000000000831*helper_25 + 90.473251028831427*helper_27 + 128.00000000000253*helper_3 + 1.0493206669370273e-12*helper_4 - 67.900548696816301*helper_7 - 15.999999999997325*helper_8 - 1.1615262935284924e-12*helper_9 - 4.0000000000004237*x + 7.4573326480586356e-13*y) + helper_5) + 2730.6666666667229*helper_1*helper_2 + helper_10*helper_6*(13653.333333333603*helper_7 + 5973.3333333335968*helper_8 + 3242.6666666668111*helper_9 - 6.4361908395466772e-10*x - 122.81481481529127*y - 36.548696844919071) - 4*helper_13*helper_15*y + helper_16*(-helper_17*helper_18 - helper_17*helper_19 + helper_28))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1829.3333333333139*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 418.44810242360529*y;
double helper_11 = x*y;
double helper_12 = 766.86419753078292*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
val[0] = -(1137.7777777778158*helper_0*helper_2*helper_3 - helper_1*(-10.466392318221381*helper_0*helper_6 - 145.67352537715615*helper_0*x*y + 63.999999999997293*helper_0*x + 35.880201188827797*helper_0*y - 882.95747599421247*helper_13 - 1309.037037036951*helper_14 - 170.66666666666598*helper_2 - 75.308641975286989*helper_3 + 95.99999999999703*helper_4 - 128.00000000000341*helper_5 + 102.79378143577537*helper_6 - 3.366393583645822e-12*helper_7*x + 7.9533607681700573*helper_7*y + 1.4052956329478979e-13*helper_7 + 443.51348879751896*x*y - 10.666666666666369*x - 16.517604023790447*y - 5.3333333333331847*z + 5.3333333333331847) + helper_11*helper_9*(-helper_10 + 2853.1358024689716*helper_11 + helper_12 + 1066.6666666667024*helper_4 - 371.75308641986237*x + 16.517604023791748) + 2*helper_13*helper_7*(995.55555555559613*helper_4 + 398.222222222244*helper_6 + helper_8*x - 315.65432098782526*x - 228.69135802484072*y + 11.184270690455484) + 1820.4444444444771*helper_2*pow(y, 4) + 3*helper_3*helper_5*(1137.7777777778158*x + 796.44444444446913*y - 161.77777777788964) + helper_4*helper_6*helper_7*(helper_8 + 1991.1111111111923*x - 315.65432098782526) + helper_9*y*(-helper_10*x + helper_12*x + 1426.5679012344858*helper_14 + 355.55555555556748*helper_2 + 56.888888888891316*helper_3 - 185.87654320993119*helper_4 - 66.913580246944932*helper_6 + 16.517604023791748*x + 11.184270690456586*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1829.3333333333139*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 418.44810242360529*x;
double helper_11 = x*y;
double helper_12 = 1426.5679012344858*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
val[1] = -(796.44444444446913*helper_0*helper_2*helper_3 - helper_1*(-20.932784636442761*helper_0*helper_11 - 72.836762688578077*helper_0*helper_6 + 35.880201188827797*helper_0*x - 3.3729306499172046e-13*helper_0*y - 882.95747599421247*helper_13 - 225.92592592586095*helper_14 + 1.7684619201140309e-13*helper_2 - 436.34567901231696*helper_3 - 3.5002362234064962e-12*helper_4 - 3.2183876048883355e-12*helper_5 + 221.75674439875948*helper_6 + 7.9533607681700573*helper_7*x - 1.6265032302264444e-12*helper_7*y + 1.354099807438712e-12*helper_7 + 3.1007173661471753e-13*helper_9 + 205.58756287155074*x*y - 16.517604023790447*x - 2.0995825105932634e-13*y - 1.5007237945851064e-13*z + 1.5007237945851064e-13) + helper_11*helper_9*(-helper_10 + 1533.7283950615658*helper_11 + helper_12 + 170.66666666667396*helper_4 - 133.82716049388986*y + 11.184270690456586) + 2*helper_13*helper_7*(398.222222222244*helper_4 + 995.55555555559613*helper_6 + helper_8*y - 315.65432098782526*x - 228.69135802484072*y + 11.184270690455484) + 1820.4444444444771*helper_2*pow(x, 4) + 3*helper_3*helper_5*(1137.7777777778158*x + 796.44444444446913*y - 161.77777777788964) + helper_4*helper_6*helper_7*(helper_8 + 796.444444444488*y - 228.69135802484072) + helper_9*x*(-helper_10*y + helper_12*y + 766.86419753078292*helper_14 + 56.888888888891316*helper_2 + 355.55555555556748*helper_3 - 66.913580246944932*helper_4 - 185.87654320993119*helper_6 + 16.517604023791748*x + 11.184270690456586*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(1137.7777777778158*x + 796.44444444446913*y - 161.77777777788964);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = 766.86419753078292*helper_12 + 1426.5679012344858*helper_13 + 355.55555555556748*helper_3 + 56.888888888891316*helper_4 - 418.44810242360529*helper_7 - 185.87654320993119*helper_8 - 66.913580246944932*helper_9 + 16.517604023791748*x + 11.184270690456586*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = 6.7458612998344092e-13*helper_9;
double helper_17 = 170.66666666667123*helper_3;
double helper_18 = 4.2911834731844471e-12*helper_4;
double helper_19 = helper_0*x;
double helper_20 = helper_0*y;
double helper_21 = helper_0*helper_8;
double helper_22 = helper_6*helper_7;
double helper_23 = -143.52080475531119*helper_0*helper_7 + 170.66666666666598*helper_1 + 1765.9149519884249*helper_10 - 1.2402869464588701e-12*helper_11*y - 411.17512574310149*helper_12 - 887.02697759503792*helper_13 + 41.865569272885523*helper_19*helper_9 + 21.333333333332739*helper_19 - 1.7684619201140309e-13*helper_2 + 291.34705075431231*helper_20*helper_8 + 6.0028951783404256e-13*helper_20 - 127.99999999999459*helper_21 - 31.813443072680229*helper_22 + 1745.3827160492679*helper_3*y - 127.99999999999604*helper_3 + 301.23456790114795*helper_4*x + 4.6669816312086619e-12*helper_4 + 6.732787167291644e-12*helper_6*helper_8 + 3.2530064604528888e-12*helper_6*helper_9 - 5.6211825317915916e-13*helper_6*x - 5.416399229754848e-12*helper_6*y + 66.070416095161789*helper_7 + 21.333333333332739*helper_8 + 4.1991650211865269e-13*helper_9;
double helper_24 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_10*(1829.3333333333139*helper_7 + 995.55555555559613*helper_8 + 398.222222222244*helper_9 - 315.65432098782526*x - 228.69135802484072*y + 11.184270690455484) + helper_11*(-helper_16*helper_24 - helper_17*helper_24 - helper_18*helper_24 + helper_23) + 3*helper_14*helper_22 - helper_15*(-1.6265032302264444e-12*helper_0*helper_9 + 15.906721536340115*helper_0*x*y + 2.8105912658957958e-13*helper_0*x + 2.708199614877424e-12*helper_0*y - 10.466392318221381*helper_12 - 72.836762688578077*helper_13 - 3.366393583645822e-12*helper_21 - 42.666666666667808*helper_3 - 1.0727958682961118e-12*helper_4 + 9.3021520984415258e-13*helper_6*y + 31.999999999998646*helper_8 - 1.6864653249586023e-13*helper_9 + 35.880201188827797*x*y - 5.3333333333331847*x - 1.5007237945851064e-13*y) + helper_5) + 1820.4444444444771*helper_1*helper_2 + helper_10*helper_6*(7317.3333333332557*helper_7 + 3982.2222222223845*helper_8 + 1592.888888888976*helper_9 - 1262.617283951301*x - 914.7654320993629*y + 44.737082761821938) + 4*helper_11*helper_14*helper_7 + helper_15*(helper_0*helper_16 + helper_0*helper_17 + helper_0*helper_18 + helper_23))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1374.2222222221708*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 241.41106538656146*y;
double helper_11 = x*y;
double helper_12 = 534.32098765422313*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
val[0] = -(1024.0000000000209*helper_0*helper_2*helper_3 - helper_1*(-14.713305898461797*helper_0*helper_6 - 68.834019204316462*helper_0*x*y - 1.4429280815454653e-12*helper_0*x + 8.86785550982448*helper_0*y - 511.69821673494107*helper_13 - 483.40740740732195*helper_14 + 5.8106605946605683e-13*helper_2 - 56.098765432074572*helper_3 - 9.5574577059142564e-13*helper_4 - 1.2748978827370071e-12*helper_5 + 48.769090077759117*helper_6 - 1.7049516805226763e-12*helper_7*x - 1.2812071330594783*helper_7*y + 7.6536856413671495e-15*helper_7 + 2.7237471537470675e-14*helper_9 + 143.46410608147045*x*y + 1.0767792693157014e-14*x - 5.8509373571186627*y + 5.3838963465785072e-15*z - 5.3838963465785072e-15) + helper_11*helper_9*(-helper_10 + 1705.3827160491837*helper_11 + helper_12 + 512.00000000002319*helper_4 - 189.72839506181981*x + 11.184270690453141) + 2*helper_13*helper_7*(739.55555555558067*helper_4 + 312.88888888890267*helper_6 + helper_8*x - 256.64197530879864*x - 169.67901234582294*y + 11.184270690450631) + 1820.4444444444548*helper_2*pow(y, 4) + 3*helper_3*helper_5*(1024.0000000000209*x + 682.66666666667902*y - 161.77777777788549) + helper_4*helper_6*helper_7*(helper_8 + 1479.1111111111613*x - 256.64197530879864) + helper_9*y*(-helper_10*x + helper_12*x + 852.69135802459186*helper_14 + 170.66666666667442*helper_2 + 42.666666666668384*helper_3 - 94.864197530909905*helper_4 - 39.901234567932534*helper_6 + 11.184270690453141*x + 5.8509373571181476*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1374.2222222221708*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 241.41106538656146*x;
double helper_11 = x*y;
double helper_12 = 852.69135802459186*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_5*x;
double helper_15 = helper_0*x;
double helper_16 = -helper_0;
val[1] = -(helper_1*(-97.538180155518234*helper_11 + 511.69821673494107*helper_13 + 168.29629629622372*helper_14 + 29.426611796923595*helper_15*y - 8.86785550982448*helper_15 - 2.8715703797438847e-12*helper_16*helper_5 - 34.417009602158231*helper_16*helper_6 - 4.4010794094628651e-13*helper_16*y - 3.2676371718488309e-28*helper_2 + 161.13580246910732*helper_3 + 2.871570379743875e-12*helper_5 - 71.732053040735224*helper_6 + 1.2812071330594783*helper_7*x + 1.5400648292079538e-12*helper_7*y - 9.9113431986883358e-13*helper_7 - 1.9900009090665223e-13*helper_9 + 5.8509373571186627*x + 1.2493435641297679e-13*y - 9.4794161851107983e-14*z + 9.4794161851107983e-14) + helper_11*helper_9*(-helper_10 + 1068.6419753084463*helper_11 + helper_12 + 128.00000000000514*helper_5 - 79.802469135865067*y + 5.8509373571181476) + 2*helper_13*helper_7*(312.88888888890267*helper_5 + 739.55555555558067*helper_6 + helper_8*y - 256.64197530879864*x - 169.67901234582294*y + 11.184270690450631) + 682.66666666667902*helper_2*helper_4 + 1820.4444444444548*helper_2*pow(x, 4) + 3*helper_4*helper_5*(1024.0000000000209*x + 682.66666666667902*y - 161.77777777788549) + helper_5*helper_6*helper_7*(helper_8 + 625.77777777780534*y - 169.67901234582294) + helper_9*x*(-helper_10*y + helper_12*y + 534.32098765422313*helper_14 + 42.666666666668384*helper_2 + 170.66666666667442*helper_3 - 39.901234567932534*helper_5 - 94.864197530909905*helper_6 + 11.184270690453141*x + 5.8509373571181476*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 3);
double helper_6 = helper_5*(1024.0000000000209*x + 682.66666666667902*y - 161.77777777788549);
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_10*x;
double helper_14 = helper_9*y;
double helper_15 = -39.901234567932534*helper_10 + 534.32098765422313*helper_13 + 852.69135802459186*helper_14 + 170.66666666667442*helper_3 + 42.666666666668384*helper_5 - 241.41106538656146*helper_8 - 94.864197530909905*helper_9 + 11.184270690453141*x + 5.8509373571181476*y;
double helper_16 = pow(helper_0, 4);
double helper_17 = helper_0*helper_8;
double helper_18 = helper_0*helper_9;
double helper_19 = helper_0*helper_10;
double helper_20 = helper_7*helper_8;
double helper_21 = 3.8287605063251794e-12*helper_0*helper_5 - 2.1535585386314029e-14*helper_0*x - 3.7917664740443193e-13*helper_0*y - 5.8106605946605683e-13*helper_1 + 3.0801296584159075e-12*helper_10*helper_7 + 2.4986871282595358e-13*helper_10 + 1023.3964334698821*helper_11 - 1.089498861498827e-13*helper_12*x - 7.9600036362660894e-13*helper_12*y - 195.07636031103647*helper_13 - 286.92821216294089*helper_14 - 35.47142203929792*helper_17 + 137.66803840863292*helper_18*y + 2.8858561630909307e-12*helper_18 + 58.85322359384719*helper_19*x + 8.8021588189257303e-13*helper_19 - 3.2676371718488309e-28*helper_2 + 5.1248285322379132*helper_20 + 644.54320987642927*helper_3*y + 1.2743276941219009e-12*helper_3 + 1.6998638436493428e-12*helper_4 + 224.39506172829829*helper_5*x + 3.8287605063251664e-12*helper_5 + 3.4099033610453526e-12*helper_7*helper_9 - 3.0614742565468598e-14*helper_7*x - 3.9645372794753343e-12*helper_7*y + 23.403749428474651*helper_8 - 2.1535585386314029e-14*helper_9;
val[2] = (-helper_0*(2*helper_0*helper_11*(312.88888888890267*helper_10 + 1374.2222222221708*helper_8 + 739.55555555558067*helper_9 - 256.64197530879864*x - 169.67901234582294*y + 11.184270690450631) + helper_12*helper_21 + 3*helper_15*helper_20 - helper_16*(1.5307371282734299e-14*helper_0*x + 1.9822686397376672e-12*helper_0*y - 2.2005397047314326e-13*helper_10 - 14.713305898461797*helper_13 - 34.417009602158231*helper_14 - 2.5624142661189566*helper_17 - 1.7049516805226763e-12*helper_18 - 1.5400648292079538e-12*helper_19 - 4.2496596091233571e-13*helper_3 - 9.5719012658129484e-13*helper_5 + 8.1712414612412026e-14*helper_7*x + 5.9700027271995675e-13*helper_7*y - 7.2146404077273267e-13*helper_9 + 8.86785550982448*x*y + 5.3838963465785072e-15*x + 9.4794161851107983e-14*y) + helper_3*helper_6) + 1820.4444444444548*helper_1*helper_2 + helper_11*helper_7*(1251.5555555556107*helper_10 + 5496.8888888886831*helper_8 + 2958.2222222223227*helper_9 - 1026.5679012351945*x - 678.71604938329176*y + 44.737082761802526) + 4*helper_12*helper_15*helper_8 + helper_16*helper_21 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2047.9999999999293*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 9.5068587103340008*y;
double helper_11 = x*y;
double helper_12 = 938.66666666655146*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
val[0] = (1365.3333333333471*helper_0*helper_2*helper_3 + helper_1*(213.28806584351292*helper_0*helper_11 + 75.94032921806712*helper_0*helper_6 + 1.6689365938621442e-12*helper_0*x - 5.1968449931387983*helper_0*y + 19.013717421007328*helper_11 + 1151.9547325099011*helper_13 + 668.11111111098796*helper_14 - 6.6317322004277741e-13*helper_2 + 127.99999999997132*helper_3 + 1.0135102633690172e-12*helper_4 + 1.4567112947993795e-12*helper_5 - 5.1968449931801928*helper_6 + 2.1115619540945576e-12*helper_7*x + 11.940329218103233*helper_7*y - 7.483725573459685e-16*helper_7 - 3.6119255734471978e-14*helper_9 + 1.1233812234342266e-14*x - 5.137174211239266*y + 5.6169061171711329e-15*z - 5.6169061171711329e-15) + helper_11*helper_9*(helper_10 + 2493.4074074071527*helper_11 + helper_12 + 384.00000000002319*helper_4 + 61.407407407288979*x - 9.1371742112377401) + 2*helper_13*helper_7*(810.66666666668925*helper_4 + 469.33333333334576*helper_6 + helper_8*x + 30.703703703520048*x - 1.6292567295295157e-10*y - 9.1371742112418541) + 2730.6666666666574*helper_2*pow(y, 4) + 3*helper_3*helper_5*(1365.3333333333471*x + 1024.0000000000055*y - 1.1952376864132644e-10) + helper_4*helper_6*helper_7*(helper_8 + 1621.3333333333785*x + 30.703703703520048) + helper_9*y*(helper_10*x + helper_12*x + 1246.7037037035764*helper_14 + 128.00000000000773*helper_2 + 64.000000000001705*helper_3 + 30.703703703644489*helper_4 - 3.6839864492321794e-11*helper_6 - 9.1371742112377401*x - 5.1371742112404641*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2047.9999999999293*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 9.5068587103340008*x;
double helper_11 = x*y;
double helper_12 = 1246.7037037035764*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (1024.0000000000055*helper_0*helper_2*helper_3 + helper_1*(106.64403292175646*helper_0*helper_6 - 5.1968449931387983*helper_0*x - 10.393689986360386*helper_11 + 1151.9547325099011*helper_13 + 383.99999999991394*helper_14 + 151.88065843613424*helper_15*x + 4.7153296356701352e-13*helper_15 - 4.9575525332555826e-28*helper_2 + 222.70370370366265*helper_3 + 3.8684350647358316e-12*helper_4 + 3.8684350647358413e-12*helper_5 + 9.5068587105036642*helper_6 + 11.940329218103233*helper_7*x + 2.2055224587664463e-12*helper_7*y - 1.2217755385363207e-12*helper_7 - 1.9382255288632491e-13*helper_9 - 5.137174211239266*x + 2.8972914602434105e-13*y + 3.4624109706313257e-14*z - 3.4624109706313257e-14) + helper_11*helper_9*(helper_10 + 1877.3333333331029*helper_11 + helper_12 + 192.00000000000512*helper_4 - 7.3679728984643589e-11*y - 5.1371742112404641) + 2*helper_13*helper_7*(469.33333333334576*helper_4 + 810.66666666668925*helper_6 + helper_8*y + 30.703703703520048*x - 1.6292567295295157e-10*y - 9.1371742112418541) + 2730.6666666666574*helper_2*pow(x, 4) + 3*helper_3*helper_5*(1365.3333333333471*x + 1024.0000000000055*y - 1.1952376864132644e-10) + helper_4*helper_6*helper_7*(helper_8 + 938.66666666669153*y - 1.6292567295295157e-10) + helper_9*x*(helper_10*y + helper_12*y + 938.66666666655146*helper_14 + 64.000000000001705*helper_2 + 128.00000000000773*helper_3 - 3.6839864492321794e-11*helper_4 + 30.703703703644489*helper_6 - 9.1371742112377401*x - 5.1371742112404641*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(1365.3333333333471*x + 1024.0000000000055*y - 1.1952376864132644e-10);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_9*x;
double helper_12 = helper_8*y;
double helper_13 = 938.66666666655146*helper_11 + 1246.7037037035764*helper_12 + 128.00000000000773*helper_3 + 64.000000000001705*helper_4 + 9.5068587103340008*helper_7 + 30.703703703644489*helper_8 - 3.6839864492321794e-11*helper_9 - 9.1371742112377401*x - 5.1371742112404641*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = pow(helper_0, 4);
double helper_17 = -helper_0;
double helper_18 = 1.9422817263991726e-12*helper_3;
double helper_19 = 9.4306592713402704e-13*helper_9;
double helper_20 = 5.157913419647789e-12*helper_4;
double helper_21 = 3.3378731877242883e-12*helper_8;
double helper_22 = helper_0*x;
double helper_23 = helper_0*y;
double helper_24 = helper_6*x;
double helper_25 = helper_6*y;
double helper_26 = helper_24*y;
double helper_27 = helper_22*y;
double helper_28 = -6.6317322004277741e-13*helper_1 + 2303.9094650198022*helper_10 - 20.787379972720771*helper_11 + 38.027434842014657*helper_12 - 7.7529021154529962e-13*helper_14*y - 1.4447702293788791e-13*helper_15 - 4.9575525332555826e-28*helper_2 + 303.76131687226848*helper_22*helper_9 + 2.2467624468684531e-14*helper_22 + 426.57613168702585*helper_23*helper_8 + 1.3849643882525303e-13*helper_23 - 2.993490229383874e-15*helper_24 - 4.8871021541452828e-12*helper_25 + 47.761316872412934*helper_26 - 20.787379972555193*helper_27 + 890.81481481465062*helper_3*y + 1.3513470178253563e-12*helper_3 + 511.99999999988529*helper_4*x + 5.1579134196477752e-12*helper_4 + 4.2231239081891152e-12*helper_6*helper_8 + 4.4110449175328926e-12*helper_6*helper_9 - 20.548696844957064*helper_7 + 2.2467624468684531e-14*helper_8 + 5.794582920486821e-13*helper_9;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_10*(2047.9999999999293*helper_7 + 810.66666666668925*helper_8 + 469.33333333334576*helper_9 + 30.703703703520048*x - 1.6292567295295157e-10*y - 9.1371742112418541) + 3*helper_13*helper_26 + helper_14*(helper_0*helper_18 + helper_0*helper_19 + helper_0*helper_20 + helper_0*helper_21 + helper_28) + helper_16*(2.1115619540945576e-12*helper_0*helper_8 + 2.2055224587664463e-12*helper_0*helper_9 + 75.94032921806712*helper_11 + 106.64403292175646*helper_12 - 1.496745114691937e-15*helper_22 - 2.4435510770726414e-12*helper_23 - 1.0835776720341593e-13*helper_24 - 5.8146765865897472e-13*helper_25 + 23.880658436206467*helper_27 + 4.8557043159979316e-13*helper_3 + 1.2894783549119472e-12*helper_4 - 5.1968449931387983*helper_7 + 8.3446829693107209e-13*helper_8 + 2.3576648178350676e-13*helper_9 + 5.6169061171711329e-15*x + 3.4624109706313257e-14*y) + helper_5) + 2730.6666666666574*helper_1*helper_2 + helper_10*helper_6*(8191.9999999997171*helper_7 + 3242.666666666757*helper_8 + 1877.3333333333831*helper_9 + 122.81481481408019*x - 6.5170269181180629e-10*y - 36.548696844967417) + 4*helper_13*helper_15*y + helper_16*(-helper_17*helper_18 - helper_17*helper_19 - helper_17*helper_20 - helper_17*helper_21 + helper_28))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1356.4444444444148*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 267.62597165054666*y;
double helper_11 = x*y;
double helper_12 = 717.23456790118667*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_5*y;
double helper_15 = -helper_0;
val[0] = -(helper_1*(251.7475994512339*helper_0*helper_11 + 31.033379058068025*helper_0*y + 195.89391860989201*helper_11 + 1060.4389574758025*helper_13 + 456.74074074066749*helper_14 - 7.8699187084846021e-13*helper_15*helper_5 - 102.4170096021729*helper_15*helper_6 - 8.8002033643775258e-13*helper_15*x - 3.2842864230689867e-13*helper_2 + 114.56790123455549*helper_3 + 4.9626146813328746e-13*helper_5 + 70.934613625948558*helper_6 + 1.1902029430609377e-12*helper_7*x + 30.515775034288577*helper_7*y + 3.0044553950275375e-15*helper_7 - 2.092153610849196e-14*helper_9 + 1.6864424808611763e-14*x + 5.8509373571140504*y + 8.4322124043058815e-15*z - 8.4322124043058815e-15) + helper_11*helper_9*(helper_10 + 1652.0493827159275*helper_11 + helper_12 + 170.6666666666784*helper_5 + 133.82716049375938*x + 11.18427069044829) + 2*helper_13*helper_7*(398.22222222223274*helper_5 + 312.88888888889488*helper_6 + helper_8*x + 228.69135802460218*x + 169.679012345606*y + 11.184270690445402) + 796.44444444444912*helper_2*helper_4 + 1820.4444444444337*helper_2*pow(y, 4) + 3*helper_4*helper_5*(796.44444444444912*x + 682.66666666666742*y + 161.77777777772576) + helper_5*helper_6*helper_7*(helper_8 + 796.44444444446549*x + 228.69135802460218) + helper_9*y*(helper_10*x + helper_12*x + 826.02469135796377*helper_14 + 56.888888888892794*helper_2 + 42.666666666667624*helper_3 + 66.91358024687969*helper_5 + 39.901234567884167*helper_6 + 11.18427069044829*x + 5.8509373571130645*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1356.4444444444148*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 267.62597165054666*x;
double helper_11 = x*y;
double helper_12 = 826.02469135796377*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_5*x;
double helper_15 = -helper_0;
val[1] = -(helper_1*(204.83401920434579*helper_0*helper_11 + 31.033379058068025*helper_0*x + 141.86922725189712*helper_11 + 1060.4389574758025*helper_13 + 343.70370370366646*helper_14 - 2.4224152633516868e-12*helper_15*helper_5 - 125.87379972561695*helper_15*helper_6 - 1.910780632942585e-13*helper_15*y - 3.3424328724919678e-28*helper_2 + 152.2469135802225*helper_3 + 2.4224152633516808e-12*helper_5 + 97.946959304946006*helper_6 + 30.515775034288577*helper_7*x + 1.4216849005772435e-12*helper_7*y - 7.4520210152010532e-13*helper_7 - 1.0136625573367452e-13*helper_9 + 5.8509373571140504*x + 2.0040759175617231e-13*y + 9.9250892522065064e-14*z - 9.9250892522065064e-14) + helper_11*helper_9*(helper_10 + 1434.4691358023733*helper_11 + helper_12 + 128.00000000000287*helper_5 + 79.802469135768334*y + 5.8509373571130645) + 2*helper_13*helper_7*(312.88888888889488*helper_5 + 398.22222222223274*helper_6 + helper_8*y + 228.69135802460218*x + 169.679012345606*y + 11.184270690445402) + 682.66666666666742*helper_2*helper_4 + 1820.4444444444337*helper_2*pow(x, 4) + 3*helper_4*helper_5*(796.44444444444912*x + 682.66666666666742*y + 161.77777777772576) + helper_5*helper_6*helper_7*(helper_8 + 625.77777777778977*y + 169.679012345606) + helper_9*x*(helper_10*y + helper_12*y + 717.23456790118667*helper_14 + 42.666666666667624*helper_2 + 56.888888888892794*helper_3 + 39.901234567884167*helper_5 + 66.91358024687969*helper_6 + 11.18427069044829*x + 5.8509373571130645*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(796.44444444444912*x + 682.66666666666742*y + 161.77777777772576);
double helper_6 = pow(helper_0, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(y, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_9*x;
double helper_13 = helper_8*y;
double helper_14 = 717.23456790118667*helper_12 + 826.02469135796377*helper_13 + 56.888888888892794*helper_3 + 42.666666666667624*helper_4 + 267.62597165054666*helper_7 + 66.91358024687969*helper_8 + 39.901234567884167*helper_9 + 11.18427069044829*x + 5.8509373571130645*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = 3.2298870178022491e-12*helper_4;
double helper_17 = 3.8215612658851701e-13*helper_9;
double helper_18 = 1.7600406728755052e-12*helper_8;
double helper_19 = 1.0493224944646136e-12*helper_3;
double helper_20 = helper_6*x;
double helper_21 = helper_0*x;
double helper_22 = helper_0*y;
double helper_23 = helper_6*y;
double helper_24 = helper_20*y;
double helper_25 = helper_21*y;
double helper_26 = 503.49519890246779*helper_0*helper_13 - 3.2842864230689867e-13*helper_1 + 2120.8779149516049*helper_10 - 8.368614443396784e-14*helper_11*x - 4.0546502293469808e-13*helper_11*y + 283.73845450379423*helper_12 + 391.78783721978402*helper_13 - 3.3424328724919678e-28*helper_2 + 1.201782158011015e-14*helper_20 + 409.66803840869159*helper_21*helper_9 + 3.3728849617223526e-14*helper_21 + 3.9700357008826026e-13*helper_22 - 2.9808084060804213e-12*helper_23 + 122.06310013715431*helper_24 + 124.1335162322721*helper_25 + 608.98765432088999*helper_3*y + 6.6168195751104995e-13*helper_3 + 458.27160493822197*helper_4*x + 3.229887017802241e-12*helper_4 + 2.3804058861218755e-12*helper_6*helper_8 + 2.843369801154487e-12*helper_6*helper_9 + 23.403749428456202*helper_7 + 3.3728849617223526e-14*helper_8 + 4.0081518351234463e-13*helper_9;
double helper_27 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_10*(1356.4444444444148*helper_7 + 398.22222222223274*helper_8 + 312.88888888889488*helper_9 + 228.69135802460218*x + 169.679012345606*y + 11.184270690445402) + helper_11*(-helper_16*helper_27 - helper_17*helper_27 - helper_18*helper_27 - helper_19*helper_27 + helper_26) + 3*helper_14*helper_24 + helper_15*(102.4170096021729*helper_12 + 125.87379972561695*helper_13 - 6.2764608325475887e-14*helper_20 + 6.008910790055075e-15*helper_21 - 1.4904042030402106e-12*helper_22 - 3.0409876720102356e-13*helper_23 + 61.031550068577154*helper_25 - 1.1902029430609377e-12*helper_27*helper_8 - 1.4216849005772435e-12*helper_27*helper_9 + 2.623306236161534e-13*helper_3 + 8.0747175445056228e-13*helper_4 + 31.033379058068025*helper_7 + 4.4001016821887629e-13*helper_8 + 9.5539031647129252e-14*helper_9 + 8.4322124043058815e-15*x + 9.9250892522065064e-14*y) + helper_5) + 1820.4444444444337*helper_1*helper_2 + helper_10*helper_6*(5425.7777777776591*helper_7 + 1592.888888888931*helper_8 + 1251.5555555555795*helper_9 + 914.76543209840872*x + 678.71604938242399*y + 44.737082761781608) + 4*helper_11*helper_14*helper_7 + helper_15*(helper_0*helper_16 + helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_26))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1356.4444444444287*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 267.62597165057014*y;
double helper_11 = x*y;
double helper_12 = 826.02469135799834*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_5*y;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_1*(141.86922725187361*helper_11 + 8.4001941205422109e-13*helper_13 + 343.70370370364043*helper_14 + 204.83401920432851*helper_15*x + 31.033379058068263*helper_15 - 4.3579954459960926e-13*helper_16*helper_5 - 125.87379972563949*helper_16*helper_6 - 6.0001386575302427e-13*helper_16*x - 2.6842725573160169e-13*helper_2 + 152.24691358024316*helper_3 + 1.2552921665103275e-13*helper_5 + 1060.4389574758527*helper_6*x + 97.946959304969212*helper_6 + 30.51577503428976*helper_7*y + 2.7632217501775896e-14*helper_7 - 1.4210854715202004e-14*helper_9 + 6.8685797790130834e-14*x + 5.8509373571128434*y + 3.4342898895065417e-14*z - 3.4342898895065417e-14) + helper_11*helper_9*(helper_10 + 1434.4691358023804*helper_11 + helper_12 + 128.00000000001199*helper_5 + 79.802469135740921*x + 5.8509373571138106) + 2*helper_13*helper_6*(312.88888888889971*helper_5 + 398.22222222222865*helper_6 + helper_8*x + 169.67901234560549*x + 228.6913580246389*y + 11.184270690444221) + 682.66666666667231*helper_2*helper_4 + 1820.4444444444364*helper_2*pow(y, 4) + 3*helper_4*helper_5*(682.66666666667231*x + 796.44444444444628*y + 161.77777777773821) + helper_5*helper_6*helper_7*(helper_8 + 625.77777777779943*x + 169.67901234560549) + helper_9*y*(helper_10*x + helper_12*x + 717.23456790119019*helper_14 + 42.666666666670665*helper_2 + 56.888888888890087*helper_3 + 39.901234567870461*helper_5 + 66.91358024690463*helper_6 + 5.8509373571138106*x + 11.184270690445203*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1356.4444444444287*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 267.62597165057014*x;
double helper_11 = x*y;
double helper_12 = 717.23456790119019*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_5*x;
double helper_15 = helper_0*x;
double helper_16 = -helper_0;
val[1] = -(helper_1*(195.89391860993842*helper_11 + 1.7623835632747835e-12*helper_13 + 456.74074074072951*helper_14 + 251.74759945127897*helper_15*y + 31.033379058068263*helper_15 - 3.1588138597741387e-12*helper_16*helper_5 - 102.41700960216426*helper_16*helper_6 - 1.0277558411094831e-13*helper_16*y - 3.0853226515310525e-28*helper_2 + 114.56790123454681*helper_3 + 3.1588138597741335e-12*helper_5 + 1060.4389574758527*helper_6*y + 70.934613625936805*helper_6 + 30.51577503428976*helper_7*x - 9.5736914814025232e-13*helper_7 - 1.7806210704963339e-13*helper_9 + 5.8509373571128434*x - 6.1226743839561609e-14*y - 6.8265999468202367e-14*z + 6.8265999468202367e-14) + helper_11*helper_9*(helper_10 + 1652.0493827159967*helper_11 + helper_12 + 170.66666666667027*helper_5 + 133.82716049380926*y + 11.184270690445203) + 2*helper_13*helper_6*(398.22222222222865*helper_5 + 312.88888888889971*helper_6 + helper_8*y + 169.67901234560549*x + 228.6913580246389*y + 11.184270690444221) + 796.44444444444628*helper_2*helper_4 + 1820.4444444444364*helper_2*pow(x, 4) + 3*helper_4*helper_5*(682.66666666667231*x + 796.44444444444628*y + 161.77777777773821) + helper_5*helper_6*helper_7*(helper_8 + 796.4444444444573*y + 228.6913580246389) + helper_9*x*(helper_10*y + helper_12*y + 826.02469135799834*helper_14 + 56.888888888890087*helper_2 + 42.666666666670665*helper_3 + 66.91358024690463*helper_5 + 39.901234567870461*helper_6 + 5.8509373571138106*x + 11.184270690445203*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(682.66666666667231*x + 796.44444444444628*y + 161.77777777773821);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = 826.02469135799834*helper_12 + 717.23456790119019*helper_13 + 42.666666666670665*helper_3 + 56.888888888890087*helper_4 + 39.901234567870461*helper_6 + 267.62597165057014*helper_7 + 66.91358024690463*helper_8 + 5.8509373571138106*x + 11.184270690445203*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = 2.0555116822189661e-13*helper_8;
double helper_17 = 5.8106605946614568e-13*helper_3;
double helper_18 = 1.2000277315060485e-12*helper_6;
double helper_19 = 4.2117518130321852e-12*helper_4;
double helper_20 = helper_9*x;
double helper_21 = helper_0*x;
double helper_22 = helper_6*helper_8;
double helper_23 = helper_9*y;
double helper_24 = helper_0*y;
double helper_25 = helper_20*y;
double helper_26 = helper_21*y;
double helper_27 = 409.66803840865703*helper_0*helper_13 - 2.6842725573160169e-13*helper_1 + 3.5247671265495671e-12*helper_10 - 5.6843418860808015e-14*helper_11*x - 7.1224842819853357e-13*helper_11*y + 391.78783721987685*helper_12 + 283.73845450374722*helper_13 - 3.0853226515310525e-28*helper_2 + 1.1052887000710358e-13*helper_20 + 503.49519890255795*helper_21*helper_8 + 1.3737159558026167e-13*helper_21 + 2120.8779149517054*helper_22 - 3.8294765925610093e-12*helper_23 - 2.7306399787280947e-13*helper_24 + 122.06310013715904*helper_25 + 124.13351623227305*helper_26 + 458.27160493818724*helper_3*y + 1.6737228886804367e-13*helper_3 + 608.98765432097264*helper_4*x + 4.211751813032178e-12*helper_4 + 1.6800388241084422e-12*helper_6*helper_9 + 1.3737159558026167e-13*helper_6 + 23.403749428451373*helper_7 - 1.2245348767912322e-13*helper_8;
double helper_28 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_22*(312.88888888889971*helper_6 + 1356.4444444444287*helper_7 + 398.22222222222865*helper_8 + 169.67901234560549*x + 228.6913580246389*y + 11.184270690444221) + helper_11*(-helper_16*helper_28 - helper_17*helper_28 - helper_18*helper_28 - helper_19*helper_28 + helper_27) + 3*helper_14*helper_25 + helper_15*(125.87379972563949*helper_12 + 102.41700960216426*helper_13 - 4.2632564145606011e-14*helper_20 + 5.5264435003551792e-14*helper_21 - 5.3418632114890018e-13*helper_23 - 1.9147382962805046e-12*helper_24 + 61.03155006857952*helper_26 - 8.4001941205422109e-13*helper_28*helper_6 - 1.7623835632747835e-12*helper_28*helper_8 + 1.4526651486653642e-13*helper_3 + 1.0529379532580463e-12*helper_4 + 3.0000693287651213e-13*helper_6 + 31.033379058068263*helper_7 + 5.1387792055474153e-14*helper_8 + 3.4342898895065417e-14*x - 6.8265999468202367e-14*y) + helper_5) + 1820.4444444444364*helper_1*helper_2 + helper_10*helper_6*(1251.5555555555989*helper_6 + 5425.7777777777146*helper_7 + 1592.8888888889146*helper_8 + 678.71604938242194*x + 914.76543209855561*y + 44.737082761776882) + 4*helper_11*helper_14*helper_7 + helper_15*(helper_0*helper_16 + helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_27))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2047.9999999999641*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 9.5068587104243676*y;
double helper_11 = x*y;
double helper_12 = 1246.703703703658*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (1024.0000000000082*helper_0*helper_2*helper_3 + helper_1*(106.64403292180516*helper_0*helper_6 - 5.1968449931366933*helper_0*y - 10.39368998636786*helper_11 + 1.1747639897901247e-12*helper_13 + 383.99999999991337*helper_14 + 151.88065843614129*helper_15*y + 8.1475567033832913e-13*helper_15 - 3.6000831945179705e-13*helper_2 + 222.70370370370401*helper_3 - 9.9475983006284173e-14*helper_4 + 3.9790393202578555e-13*helper_5 + 1151.9547325100807*helper_6*x + 9.5068587105532885*helper_6 + 11.940329218108225*helper_7*y + 9.4739031434670777e-14*helper_7 + 1.6697754290360401e-13*x - 5.1371742112432361*y + 8.3488771451802003e-14*z - 8.3488771451802003e-14) + helper_11*helper_9*(helper_10 + 1877.3333333331821*helper_11 + helper_12 + 192.00000000001927*helper_4 - 8.8978898323452186e-11*x - 5.1371742112416259) + 2*helper_13*helper_6*(469.33333333335042*helper_4 + 810.66666666667686*helper_6 + helper_8*x - 1.2232703738845942e-10*x + 30.703703703616192*y - 9.1371742112458758) + 2730.6666666666524*helper_2*pow(y, 4) + 3*helper_3*helper_5*(1024.0000000000082*x + 1365.3333333333358*y - 7.2968002010990702e-11) + helper_4*helper_6*helper_7*(helper_8 + 938.66666666670085*x - 1.2232703738845942e-10) + helper_9*y*(helper_10*x + helper_12*x + 938.66666666659103*helper_14 + 64.000000000006423*helper_2 + 128.00000000000193*helper_3 - 4.4489449161726093e-11*helper_4 + 30.703703703695481*helper_6 - 5.1371742112416259*x - 9.1371742112446555*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2047.9999999999641*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 9.5068587104243676*x;
double helper_11 = x*y;
double helper_12 = 938.66666666659103*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (1365.3333333333358*helper_0*helper_2*helper_3 + helper_1*(75.940329218070644*helper_0*helper_6 - 5.1968449931366933*helper_0*x + 19.013717421106577*helper_11 + 1151.9547325100807*helper_13 + 668.111111111112*helper_14 + 213.28806584361033*helper_15*x + 7.6673920984681493e-14*helper_15 - 5.3852904463087458e-28*helper_2 + 127.99999999997112*helper_3 + 5.875459241218889e-12*helper_4 + 5.8754592412188954e-12*helper_5 - 5.1968449931839302*helper_6 + 11.940329218108225*helper_7*x + 3.076035996563542e-12*helper_7*y - 1.8466100847796473e-12*helper_7 - 4.3467927017306327e-13*helper_9 - 5.1371742112432361*x - 4.4937716099706667e-13*y - 3.9778822668378445e-13*z + 3.9778822668378445e-13) + helper_11*helper_9*(helper_10 + 2493.407407407316*helper_11 + helper_12 + 384.0000000000058*helper_4 + 61.407407407390963*y - 9.1371742112446555) + 2*helper_13*helper_7*(810.66666666667686*helper_4 + 469.33333333335042*helper_6 + helper_8*y - 1.2232703738845942e-10*x + 30.703703703616192*y - 9.1371742112458758) + 2730.6666666666524*helper_2*pow(x, 4) + 3*helper_3*helper_5*(1024.0000000000082*x + 1365.3333333333358*y - 7.2968002010990702e-11) + helper_4*helper_6*helper_7*(helper_8 + 1621.3333333333537*y + 30.703703703616192) + helper_9*x*(helper_10*y + helper_12*y + 1246.703703703658*helper_14 + 128.00000000000193*helper_2 + 64.000000000006423*helper_3 + 30.703703703695481*helper_4 - 4.4489449161726093e-11*helper_6 - 5.1371742112416259*x - 9.1371742112446555*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 3);
double helper_6 = helper_5*(1024.0000000000082*x + 1365.3333333333358*y - 7.2968002010990702e-11);
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_7*x;
double helper_14 = helper_9*y;
double helper_15 = 1246.703703703658*helper_13 + 938.66666666659103*helper_14 + 64.000000000006423*helper_3 + 128.00000000000193*helper_5 + 30.703703703695481*helper_7 + 9.5068587104243676*helper_8 - 4.4489449161726093e-11*helper_9 - 5.1371742112416259*x - 9.1371742112446555*y;
double helper_16 = pow(helper_0, 4);
double helper_17 = helper_0*x;
double helper_18 = helper_0*helper_9;
double helper_19 = helper_0*helper_7;
double helper_20 = helper_10*y;
double helper_21 = helper_20*x;
double helper_22 = 303.76131687228258*helper_0*helper_14 + 7.8339456549585278e-12*helper_0*helper_5 - 20.787379972546773*helper_0*x*y - 1.5911529067351378e-12*helper_0*y - 3.6000831945179705e-13*helper_1 + 6.1520719931270839e-12*helper_10*helper_7 + 3.7895612573868311e-13*helper_10*x - 7.3864403391185893e-12*helper_10*y + 2.3495279795802493e-12*helper_11 - 1.7387170806922531e-12*helper_12*y + 38.027434842213154*helper_13 + 3.3395508580720801e-13*helper_17 + 1.6295113406766583e-12*helper_18 + 426.57613168722065*helper_19*x + 1.5334784196936299e-13*helper_19 - 5.3852904463087458e-28*helper_2 + 47.7613168724329*helper_21 + 511.99999999988449*helper_3*y - 1.326346440083789e-13*helper_3 + 5.3053857603438074e-13*helper_4 + 890.81481481481603*helper_5*x + 7.8339456549585181e-12*helper_5 + 2303.9094650201614*helper_7*helper_9 - 8.9875432199413335e-13*helper_7 - 20.787379972735721*helper_9*y + 3.3395508580720801e-13*helper_9 - 20.548696844972945*x*y;
val[2] = -(-helper_0*(helper_12*helper_22 + 3*helper_15*helper_21 + helper_16*(23.88065843621645*helper_0*helper_8 - 3.6932201695592946e-12*helper_0*y + 106.64403292180516*helper_13 + 75.940329218070644*helper_14 + 1.8947806286934155e-13*helper_17 + 1.1747639897901247e-12*helper_18 + 3.076035996563542e-12*helper_19 - 1.3040378105191897e-12*helper_20 + 1.3263464400859518e-13*helper_3 + 1.9584864137396319e-12*helper_5 + 3.8336960492340746e-14*helper_7 - 5.1968449931366933*helper_8 + 4.0737783516916456e-13*helper_9 + 8.3488771451802003e-14*x - 3.9778822668378445e-13*y) + 2*helper_19*helper_9*(810.66666666667686*helper_7 + 2047.9999999999641*helper_8 + 469.33333333335042*helper_9 - 1.2232703738845942e-10*x + 30.703703703616192*y - 9.1371742112458758) + helper_3*helper_6) + 2730.6666666666524*helper_1*helper_2 + helper_11*helper_7*(3242.6666666667074*helper_7 + 8191.9999999998563*helper_8 + 1877.3333333334017*helper_9 - 4.8930814955383767e-10*x + 122.81481481446477*y - 36.548696844983503) + 4*helper_12*helper_15*helper_8 + helper_16*helper_22 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1374.2222222221999*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 241.41106538644905*y;
double helper_11 = x*y;
double helper_12 = 852.69135802466826*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_5*y;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_1*(-97.538180155497969*helper_11 + 6.1264573661099873e-13*helper_13 + 168.29629629625816*helper_14 + 29.426611796957275*helper_15*x - 8.8678555098217515*helper_15 - 9.4739031435621777e-15*helper_16*helper_5 - 34.417009602203947*helper_16*helper_6 - 4.294836091706085e-13*helper_16*x - 2.68427255731603e-13*helper_2 + 161.1358024691433*helper_3 - 3.0079642480501403e-13*helper_5 + 511.6982167351523*helper_6*x - 71.732053040688214*helper_6 + 1.2812071330655179*helper_7*y + 8.0922922683782476e-14*helper_7 + 1.5039821240254044e-13*x + 5.8509373571138994*y + 7.5199106201270218e-14*z - 7.5199106201270218e-14) + helper_11*helper_9*(-helper_10 + 1068.6419753085556*helper_11 + helper_12 + 128.0000000000143*helper_5 - 79.802469135850856*x + 5.8509373571150274) + 2*helper_13*helper_6*(312.88888888890165*helper_5 + 739.55555555556305*helper_6 + helper_8*x - 169.67901234575672*x - 256.64197530869535*y + 11.184270690445679) + 682.66666666667334*helper_2*helper_4 + 1820.4444444444355*helper_2*pow(y, 4) + 3*helper_4*helper_5*(682.66666666667334*x + 1024.0000000000025*y - 161.77777777782796) + helper_5*helper_6*helper_7*(helper_8 + 625.77777777780329*x - 169.67901234575672) + helper_9*y*(-helper_10*x + helper_12*x + 534.32098765427781*helper_14 + 42.666666666671432*helper_2 + 170.66666666666811*helper_3 - 39.901234567925428*helper_5 - 94.864197530862967*helper_6 + 5.8509373571150274*x + 11.184270690446327*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1374.2222222221999*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 241.41106538644905*x;
double helper_11 = x*y;
double helper_12 = 534.32098765427781*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
val[1] = -(1024.0000000000025*helper_0*helper_2*helper_3 - helper_1*(-14.713305898478637*helper_0*helper_6 - 68.834019204407895*helper_0*x*y + 8.8678555098217515*helper_0*x + 3.0659928185686523e-13*helper_0*y - 511.6982167351523*helper_13 - 483.4074074074299*helper_14 + 3.0853226515310507e-28*helper_2 - 56.098765432086054*helper_3 - 4.5542984107338531e-12*helper_4 - 4.5542984107338579e-12*helper_5 + 48.769090077748984*helper_6 - 1.2812071330655179*helper_7*x - 2.0952750280114628e-12*helper_7*y + 1.6361656733017865e-12*helper_7 + 4.8940976252582394e-13*helper_9 + 143.46410608137643*x*y - 5.8509373571138994*x + 6.0353231328909726e-13*y + 4.9078193117494846e-13*z - 4.9078193117494846e-13) + helper_11*helper_9*(-helper_10 + 1705.3827160493365*helper_11 + helper_12 + 512.00000000000432*helper_4 - 189.72839506172593*y + 11.184270690446327) + 2*helper_13*helper_7*(739.55555555556305*helper_4 + 312.88888888890165*helper_6 + helper_8*y - 169.67901234575672*x - 256.64197530869535*y + 11.184270690445679) + 1820.4444444444355*helper_2*pow(x, 4) + 3*helper_3*helper_5*(682.66666666667334*x + 1024.0000000000025*y - 161.77777777782796) + helper_4*helper_6*helper_7*(helper_8 + 1479.1111111111261*y - 256.64197530869535) + helper_9*x*(-helper_10*y + helper_12*y + 852.69135802466826*helper_14 + 170.66666666666811*helper_2 + 42.666666666671432*helper_3 - 94.864197530862967*helper_4 - 39.901234567925428*helper_6 + 5.8509373571150274*x + 11.184270690446327*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 3);
double helper_6 = helper_5*(682.66666666667334*x + 1024.0000000000025*y - 161.77777777782796);
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = pow(helper_0, 3);
double helper_13 = helper_7*x;
double helper_14 = helper_9*y;
double helper_15 = 852.69135802466826*helper_13 + 534.32098765427781*helper_14 + 42.666666666671432*helper_3 + 170.66666666666811*helper_5 - 94.864197530862967*helper_7 - 241.41106538644905*helper_8 - 39.901234567925428*helper_9 + 5.8509373571150274*x + 11.184270690446327*y;
double helper_16 = pow(helper_0, 4);
double helper_17 = helper_0*x;
double helper_18 = helper_0*y;
double helper_19 = helper_0*helper_8;
double helper_20 = helper_0*helper_9;
double helper_21 = helper_10*y;
double helper_22 = helper_10*helper_8;
double helper_23 = 6.0723978809784772e-12*helper_0*helper_5 - 6.1319856371373046e-13*helper_0*helper_7 - 2.68427255731603e-13*helper_1 + 4.1905500560229256e-12*helper_10*helper_7 + 3.236916907351299e-13*helper_10*x + 1.2252914732219975e-12*helper_11 - 1.9576390501032958e-12*helper_12*y - 286.92821216275286*helper_13 - 195.07636031099594*helper_14 + 137.66803840881579*helper_17*helper_7 + 3.0079642480508087e-13*helper_17 - 1.9631277246997939e-12*helper_18 - 35.471422039287006*helper_19 - 3.0853226515310507e-28*helper_2 + 58.853223593914549*helper_20*y + 8.5896721834121701e-13*helper_20 - 6.544662693207146e-12*helper_21 + 5.1248285322620717*helper_22 + 224.39506172834422*helper_3*y - 4.010618997400187e-13*helper_3 + 1.2631870858082904e-14*helper_4 + 644.5432098765732*helper_5*x + 6.0723978809784707e-12*helper_5 + 1023.3964334703046*helper_7*helper_9 - 1.2070646265781945e-12*helper_7 + 23.403749428455598*helper_8 + 3.0079642480508087e-13*helper_9;
double helper_24 = -helper_0;
val[2] = (-helper_0*(helper_12*helper_23 + 3*helper_15*helper_22 + helper_16*(34.417009602203947*helper_13 + 14.713305898478637*helper_14 + 1.6184584536756495e-13*helper_17 - 3.272331346603573e-12*helper_18 + 2.5624142661310358*helper_19 - 1.4682292875774719e-12*helper_21 - 2.0952750280114628e-12*helper_24*helper_7 - 6.1264573661099873e-13*helper_24*helper_9 + 3.1579677145207259e-15*helper_3 + 1.5180994702446193e-12*helper_5 - 1.5329964092843262e-13*helper_7 - 8.8678555098217515*helper_8 + 2.1474180458530425e-13*helper_9 + 7.5199106201270218e-14*x - 4.9078193117494846e-13*y) + 2*helper_20*helper_7*(739.55555555556305*helper_7 + 1374.2222222221999*helper_8 + 312.88888888890165*helper_9 - 169.67901234575672*x - 256.64197530869535*y + 11.184270690445679) + helper_3*helper_6) + 1820.4444444444355*helper_1*helper_2 + helper_11*helper_7*(2958.2222222222522*helper_7 + 5496.8888888887996*helper_8 + 1251.5555555556066*helper_9 - 678.71604938302687*x - 1026.5679012347814*y + 44.737082761782716) + 4*helper_12*helper_15*helper_8 + helper_16*helper_23 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1829.3333333333414*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 418.44810242345415*y;
double helper_11 = x*y;
double helper_12 = 1426.5679012345777*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_5*y;
double helper_15 = -helper_0;
val[0] = -(helper_1*(20.932784636483404*helper_0*helper_11 - 35.880201188827044*helper_0*y - 205.58756287152733*helper_11 + 1.2982756159568925e-12*helper_13 + 225.92592592590859*helper_14 - 8.425194698283694e-13*helper_15*helper_5 - 72.836762688635503*helper_15*helper_6 - 9.6107484110972232e-13*helper_15*x - 3.2842864230690382e-13*helper_2 + 436.34567901236073*helper_3 + 4.9849836193118641e-13*helper_5 + 882.95747599448703*helper_6*x - 221.75674439870073*helper_6 - 7.9533607681638685*helper_7*y + 4.2786076465042679e-14*helper_7 + 1.4605600679513295e-14*helper_9 + 4.3136961766633775e-14*x + 16.517604023782841*y + 2.1568480883316888e-14*z - 2.1568480883316888e-14) + helper_11*helper_9*(-helper_10 + 1533.7283950617066*helper_11 + helper_12 + 170.66666666668499*helper_5 - 133.82716049386767*x + 11.184270690450553) + 2*helper_13*helper_6*(398.22222222223951*helper_5 + 995.55555555556634*helper_6 + helper_8*x - 228.69135802474733*x - 315.65432098768412*y + 11.184270690447443) + 796.44444444445537*helper_2*helper_4 + 1820.4444444444403*helper_2*pow(y, 4) + 3*helper_4*helper_5*(796.44444444445537*x + 1137.777777777784*y - 161.77777777780938) + helper_5*helper_6*helper_7*(helper_8 + 796.44444444447902*x - 228.69135802474733) + helper_9*y*(-helper_10*x + helper_12*x + 766.86419753085329*helper_14 + 56.888888888894996*helper_2 + 355.55555555555748*helper_3 - 66.913580246933833*helper_5 - 185.87654320986931*helper_6 + 11.184270690450553*x + 16.517604023781697*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 1829.3333333333414*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 418.44810242345415*x;
double helper_11 = x*y;
double helper_12 = 766.86419753085329*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_4*x;
val[1] = -(1137.777777777784*helper_0*helper_2*helper_3 - helper_1*(-10.466392318241702*helper_0*helper_6 - 145.67352537727101*helper_0*x*y + 35.880201188827044*helper_0*x + 64.000000000002075*helper_0*y - 1.4027809549528796e-12*helper_13 - 1309.0370370370822*helper_14 - 170.66666666666666*helper_2 - 75.308641975302862*helper_3 + 95.999999999995183*helper_4 - 128.00000000000477*helper_5 - 882.95747599448703*helper_6*y + 102.79378143576366*helper_6 + 7.9533607681638685*helper_7*x + 2.5641915050660013e-12*helper_7 + 9.3786584960608345e-13*helper_9 + 443.51348879740146*x*y - 16.517604023782841*x - 10.666666666665677*y - 5.3333333333326181*z + 5.3333333333326181) + helper_11*helper_9*(-helper_10 + 2853.1358024691554*helper_11 + helper_12 + 1066.6666666666724*helper_4 - 371.75308641973862*y + 16.517604023781697) + 2*helper_13*helper_6*(995.55555555556634*helper_4 + 398.22222222223951*helper_6 + helper_8*y - 228.69135802474733*x - 315.65432098768412*y + 11.184270690447443) + 1820.4444444444403*helper_2*pow(x, 4) + 3*helper_3*helper_5*(796.44444444445537*x + 1137.777777777784*y - 161.77777777780938) + helper_4*helper_6*helper_7*(helper_8 + 1991.1111111111327*y - 315.65432098768412) + helper_9*x*(-helper_10*y + helper_12*y + 1426.5679012345777*helper_14 + 355.55555555555748*helper_2 + 56.888888888894996*helper_3 - 185.87654320986931*helper_4 - 66.913580246933833*helper_6 + 11.184270690450553*x + 16.517604023781697*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(796.44444444445537*x + 1137.777777777784*y - 161.77777777780938);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_6*y;
double helper_13 = 1426.5679012345777*helper_11 + 766.86419753085329*helper_12 + 56.888888888894996*helper_3 + 355.55555555555748*helper_4 - 66.913580246933833*helper_6 - 418.44810242345415*helper_7 - 185.87654320986931*helper_8 + 11.184270690450553*x + 16.517604023781697*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = pow(helper_0, 4);
double helper_17 = 1.1233592931044925e-12*helper_3;
double helper_18 = 170.66666666667305*helper_4;
double helper_19 = 1.9221496822194446e-12*helper_6;
double helper_20 = helper_0*y;
double helper_21 = helper_0*x;
double helper_22 = helper_9*x;
double helper_23 = helper_6*helper_8;
double helper_24 = helper_9*y;
double helper_25 = helper_22*y;
double helper_26 = helper_20*x;
double helper_27 = -128.00000000000415*helper_0*helper_8 - 3.2842864230690382e-13*helper_1 + 2.8055619099057592e-12*helper_10 - 887.02697759480293*helper_11 - 411.17512574305465*helper_12 - 3.7514633984243338e-12*helper_14*y + 5.8422402718053181e-14*helper_15 + 170.66666666666666*helper_2 + 41.865569272966809*helper_20*helper_6 + 21.333333333330472*helper_20 + 291.34705075454201*helper_21*helper_8 + 8.627392353326755e-14*helper_21 + 1.7114430586017072e-13*helper_22 + 1765.9149519889741*helper_23 - 1.0256766020264005e-11*helper_24 - 31.813443072655474*helper_25 - 143.52080475530818*helper_26 + 301.23456790121145*helper_3*y + 6.6466448257491522e-13*helper_3 + 1745.3827160494429*helper_4*x - 127.99999999999358*helper_4 + 2.596551231913785e-12*helper_6*helper_9 + 8.627392353326755e-14*helper_6 + 66.070416095131364*helper_7 + 21.333333333331353*helper_8;
double helper_28 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_23*(398.22222222223951*helper_6 + 1829.3333333333414*helper_7 + 995.55555555556634*helper_8 - 228.69135802474733*x - 315.65432098768412*y + 11.184270690447443) + 3*helper_13*helper_25 + helper_14*(-helper_17*helper_28 - helper_18*helper_28 - helper_19*helper_28 + helper_27) + helper_16*(72.836762688635503*helper_11 + 10.466392318241702*helper_12 - 5.1283830101320026e-12*helper_20 + 8.5572152930085358e-14*helper_21 + 4.3816802038539886e-14*helper_22 - 2.8135975488182504e-12*helper_24 - 15.906721536327737*helper_26 - 1.2982756159568925e-12*helper_28*helper_6 - 1.4027809549528796e-12*helper_28*helper_8 + 2.8083982327612313e-13*helper_3 + 42.666666666668263*helper_4 + 4.8053742055486116e-13*helper_6 - 35.880201188827044*helper_7 - 32.000000000001037*helper_8 + 2.1568480883316888e-14*x + 5.3333333333326181*y) + helper_5) + 1820.4444444444403*helper_1*helper_2 + helper_10*helper_6*(1592.888888888958*helper_6 + 7317.3333333333658*helper_7 + 3982.2222222222654*helper_8 - 914.76543209898932*x - 1262.6172839507365*y + 44.737082761789772) + 4*helper_13*helper_15*y + helper_16*(helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_27))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 3413.3333333334231*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 82.604252400552866*y;
double helper_11 = x*y;
double helper_12 = 2986.6666666667543*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (1365.3333333333587*helper_0*helper_2*helper_3 + helper_1*(578.56995884777257*helper_0*helper_6 - 67.900548696817808*helper_0*y - 165.20850480110832*helper_11 + 3399.7325102881659*helper_13 + 867.88888888888891*helper_14 + 413.06584362140137*helper_15*y + 2.9495911885789142e-12*helper_15 - 6.6317322004278639e-13*helper_2 + 1066.6666666666947*helper_3 + 3.2923787153378873e-12*helper_4 + 3.8297266592564679e-12*helper_5 - 67.900548696811384*helper_6 + 3.8191014137169125e-12*helper_7*x + 45.236625514420879*helper_7*y - 1.2004389252192329e-13*helper_7 + 2.6645352591004009e-14*helper_9 - 3.2150414018302697e-13*x - 13.137174211235669*y - 1.6075207009151348e-13*z + 1.6075207009151348e-13) + helper_11*helper_9*(-helper_10 + 3991.925925926048*helper_11 + helper_12 + 384.0000000000349*helper_4 - 61.407407407461193*x - 9.1371742112343011) + 2*helper_13*helper_7*(810.66666666670142*helper_4 + 1493.3333333333546*helper_6 + helper_8*x - 30.703703703729026*x + 1.4287238059296214e-11*y - 9.1371742112398202) + 2730.6666666666706*helper_2*pow(y, 4) + 3*helper_3*helper_5*(1365.3333333333587*x + 1706.6666666666827*y + 9.4106500639270478e-12) + helper_4*helper_6*helper_7*(helper_8 + 1621.3333333334028*x - 30.703703703729026) + helper_9*y*(-helper_10*x + helper_12*x + 1995.962962963024*helper_14 + 128.00000000001162*helper_2 + 533.33333333333667*helper_3 - 30.703703703730596*helper_4 + 1.4782841617488884e-11*helper_6 - 9.1371742112343011*x - 13.137174211237511*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 3413.3333333334231*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 82.604252400552866*x;
double helper_11 = x*y;
double helper_12 = 1995.962962963024*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (1706.6666666666827*helper_0*helper_2*helper_3 + helper_1*(206.53292181070069*helper_0*helper_6 - 67.900548696817808*helper_0*x - 135.80109739362277*helper_11 + 3399.7325102881659*helper_13 + 3200.0000000000841*helper_14 + 1157.1399176955451*helper_15*x - 32.000000000005294*helper_15 + 256.0*helper_2 + 289.2962962962963*helper_3 + 5.7918101042188592e-12*helper_4 + 384.00000000000568*helper_5 - 82.604252400554159*helper_6 + 45.236625514420879*helper_7*x + 128.00000000000065*helper_7*y - 16.000000000004391*helper_7 - 1.6447853595803433e-12*helper_9 - 13.137174211235669*x - 8.0000000000013074*y - 4.0000000000006661*z + 4.0000000000006661) + helper_11*helper_9*(-helper_10 + 5973.3333333335086*helper_11 + helper_12 + 1600.00000000001*helper_4 + 2.9565683234977769e-11*y - 13.137174211237511) + 2*helper_13*helper_7*(1493.3333333333546*helper_4 + 810.66666666670142*helper_6 + helper_8*y - 30.703703703729026*x + 1.4287238059296214e-11*y - 9.1371742112398202) + 2730.6666666666706*helper_2*pow(x, 4) + 3*helper_3*helper_5*(1365.3333333333587*x + 1706.6666666666827*y + 9.4106500639270478e-12) + helper_4*helper_6*helper_7*(helper_8 + 2986.6666666667093*y + 1.4287238059296214e-11) + helper_9*x*(-helper_10*y + helper_12*y + 2986.6666666667543*helper_14 + 533.33333333333667*helper_2 + 128.00000000001162*helper_3 + 1.4782841617488884e-11*helper_4 - 30.703703703730596*helper_6 - 9.1371742112343011*x - 13.137174211237511*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 3);
double helper_6 = helper_5*(1365.3333333333587*x + 1706.6666666666827*y + 9.4106500639270478e-12);
double helper_7 = pow(helper_0, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(y, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_10*x;
double helper_13 = helper_9*y;
double helper_14 = 1.4782841617488884e-11*helper_10 + 2986.6666666667543*helper_12 + 1995.962962963024*helper_13 + 533.33333333333667*helper_3 + 128.00000000001162*helper_5 - 82.604252400552866*helper_8 - 30.703703703730596*helper_9 - 9.1371742112343011*x - 13.137174211237511*y;
double helper_15 = pow(helper_0, 3);
double helper_16 = pow(helper_0, 4);
double helper_17 = helper_0*helper_9;
double helper_18 = helper_7*y;
double helper_19 = helper_18*x;
double helper_20 = helper_0*helper_10;
double helper_21 = helper_0*y;
double helper_22 = -64.000000000010587*helper_0*helper_10 + 5.1063022123419574e-12*helper_0*helper_5 - 271.60219478727123*helper_0*x*y - 6.4300828036605394e-13*helper_0*x - 16.000000000002665*helper_0*y - 6.6317322004278639e-13*helper_1 + 256.00000000000131*helper_10*helper_7 - 271.60219478724554*helper_10*x - 16.000000000002615*helper_10 + 6799.4650205763319*helper_11 + 1.0658141036401604e-13*helper_15*x - 6.5791414383213732e-12*helper_15*y + 5.8991823771578283e-12*helper_17 + 180.94650205768352*helper_19 + 256.0*helper_2 + 2314.2798353910903*helper_20*x + 826.13168724280274*helper_21*helper_9 + 4266.6666666667788*helper_3*x + 7.7224134722918123e-12*helper_3 + 512.00000000000762*helper_4 + 1157.1851851851852*helper_5*y + 4.389838287117183e-12*helper_5 + 7.638202827433825e-12*helper_7*helper_9 - 4.8017557008769317e-13*helper_7*x - 64.000000000017565*helper_7*y - 330.41700960221664*helper_9*y - 6.4300828036605394e-13*helper_9 - 52.548696844942675*x*y;
val[2] = -(-helper_0*(3*helper_14*helper_19 + helper_15*helper_22 + helper_16*(-2.4008778504384658e-13*helper_0*x - 16.000000000002647*helper_10 + 578.56995884777257*helper_12 + 206.53292181070069*helper_13 + 3.8191014137169125e-12*helper_17 - 4.9343560787410303e-12*helper_18 + 128.00000000000065*helper_20 + 90.473251028841759*helper_21*x - 32.000000000008782*helper_21 + 128.0000000000019*helper_3 + 1.2765755530854894e-12*helper_5 + 7.9936057773012028e-14*helper_7*x - 67.900548696817808*helper_8 + 1.4747955942894571e-12*helper_9 - 1.6075207009151348e-13*x - 4.0000000000006661*y) + 2*helper_20*helper_9*(1493.3333333333546*helper_10 + 3413.3333333334231*helper_8 + 810.66666666670142*helper_9 - 30.703703703729026*x + 1.4287238059296214e-11*y - 9.1371742112398202) + helper_3*helper_6) + 2730.6666666666706*helper_1*helper_2 + helper_11*helper_7*(5973.3333333334185*helper_10 + 13653.333333333692*helper_8 + 3242.6666666668057*helper_9 - 122.8148148149161*x + 5.7148952237184858e-11*y - 36.548696844959281) + 4*helper_14*helper_15*x*y + helper_16*helper_22 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2721.7777777779265*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 528.51486053961617*y;
double helper_11 = x*y;
double helper_12 = 2555.654320987795*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_5*y;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_1*(425.72107910380106*helper_11 + 4270.2167352540027*helper_13 + 1564.5925925926365*helper_14 + 1150.0192043895954*helper_15*x + 149.99634202105079*helper_15 - 4.9020869668048301e-12*helper_16*helper_5 - 897.1330589849432*helper_16*helper_6 - 3.3287611350483311e-12*helper_16*x - 5.8106605946606531e-13*helper_2 + 985.87654320990578*helper_3 + 4.5106963431902991e-12*helper_5 + 335.87288523093861*helper_6 + 4.2193957518598271e-12*helper_7*x + 266.8120713306007*helper_7*y - 2.4972068308956402e-13*helper_7 + 2.0921536108491992e-14*helper_9 - 5.6714029905849531e-13*x + 16.517604023786376*y - 2.8357014952924765e-13*z + 2.8357014952924765e-13) + helper_11*helper_9*(helper_10 + 4211.0617283953206*helper_11 + helper_12 + 512.0000000000332*helper_5 + 189.72839506170834*x + 11.184270690453841) + 2*helper_13*helper_7*(739.55555555559181*helper_5 + 995.55555555557748*helper_6 + helper_8*x + 256.64197530868802*x + 315.65432098772709*y + 11.184270690449567) + 1024.0000000000327*helper_2*helper_4 + 1820.4444444444703*helper_2*pow(y, 4) + 3*helper_4*helper_5*(1024.0000000000327*x + 1137.7777777778006*y + 161.7777777778405) + helper_5*helper_6*helper_7*(helper_8 + 1479.1111111111836*x + 256.64197530868802) + helper_9*y*(helper_10*x + helper_12*x + 2105.5308641976603*helper_14 + 170.66666666667774*helper_2 + 355.55555555555867*helper_3 + 94.86419753085417*helper_5 + 185.87654320989427*helper_6 + 11.184270690453841*x + 16.517604023785154*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 2721.7777777779265*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 528.51486053961617*x;
double helper_11 = x*y;
double helper_12 = 2105.5308641976603*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_5*x;
double helper_15 = helper_0*x;
double helper_16 = -helper_0;
val[1] = -(helper_1*(671.74577046187721*helper_11 + 255.99999999999864*helper_13 + 2957.6296296297173*helper_14 + 1794.2661179698864*helper_15*y + 149.99634202105079*helper_15 - 384.00000000000188*helper_16*helper_5 - 575.00960219479771*helper_16*helper_6 - 127.99999999999424*helper_16*y + 170.66666666666666*helper_2 + 521.5308641975455*helper_3 + 96.000000000001933*helper_5 + 4270.2167352540027*helper_6*y + 212.86053955190053*helper_6 + 266.8120713306007*helper_7*x + 31.999999999996575*helper_7 + 42.666666666665293*helper_9 + 16.517604023786376*x + 10.666666666666002*y + 5.3333333333333792*z - 5.3333333333333792) + helper_11*helper_9*(helper_10 + 5111.30864197559*helper_11 + helper_12 + 1066.6666666666761*helper_5 + 371.75308641978853*y + 16.517604023785154) + 2*helper_13*helper_6*(995.55555555557748*helper_5 + 739.55555555559181*helper_6 + helper_8*y + 256.64197530868802*x + 315.65432098772709*y + 11.184270690449567) + 1137.7777777778006*helper_2*helper_4 + 1820.4444444444703*helper_2*pow(x, 4) + 3*helper_4*helper_5*(1024.0000000000327*x + 1137.7777777778006*y + 161.7777777778405) + helper_5*helper_6*helper_7*(helper_8 + 1991.111111111155*y + 315.65432098772709) + helper_9*x*(helper_10*y + helper_12*y + 2555.654320987795*helper_14 + 355.55555555555867*helper_2 + 170.66666666667774*helper_3 + 185.87654320989427*helper_5 + 94.86419753085417*helper_6 + 11.184270690453841*x + 16.517604023785154*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(1024.0000000000327*x + 1137.7777777778006*y + 161.7777777778405);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = 2555.654320987795*helper_12 + 2105.5308641976603*helper_13 + 170.66666666667774*helper_3 + 355.55555555555867*helper_4 + 94.86419753085417*helper_6 + 528.51486053961617*helper_7 + 185.87654320989427*helper_8 + 11.184270690453841*x + 16.517604023785154*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = 6.5361159557397735e-12*helper_3;
double helper_17 = 6.6575222700966621e-12*helper_6;
double helper_18 = 512.0000000000025*helper_4;
double helper_19 = 255.99999999998849*helper_8;
double helper_20 = helper_9*y;
double helper_21 = helper_6*helper_8;
double helper_22 = helper_0*y;
double helper_23 = helper_0*x;
double helper_24 = helper_9*x;
double helper_25 = helper_7*helper_9;
double helper_26 = helper_0*helper_7;
double helper_27 = 3588.5322359397728*helper_0*helper_12 - 5.8106605946606531e-13*helper_1 + 511.99999999999727*helper_10 + 8.3686144433967967e-14*helper_11*x + 170.66666666666117*helper_11*y + 1343.4915409237544*helper_12 + 851.44215820760212*helper_13 + 170.66666666666666*helper_2 + 127.9999999999863*helper_20 + 8540.4334705080055*helper_21 + 2300.0384087791908*helper_22*helper_6 + 21.333333333333517*helper_22 - 1.1342805981169906e-12*helper_23 - 9.9888273235825606e-13*helper_24 + 1067.2482853224028*helper_25 + 599.98536808420317*helper_26 + 2086.123456790182*helper_3*y + 6.0142617909203988e-12*helper_3 + 3943.5061728396231*helper_4*x + 128.00000000000259*helper_4 + 8.4387915037196542e-12*helper_6*helper_9 - 1.1342805981169906e-12*helper_6 + 66.070416095145504*helper_7 + 21.333333333332003*helper_8;
double helper_28 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_21*(739.55555555559181*helper_6 + 2721.7777777779265*helper_7 + 995.55555555557748*helper_8 + 256.64197530868802*x + 315.65432098772709*y + 11.184270690449567) + helper_11*(-helper_16*helper_28 - helper_17*helper_28 - helper_18*helper_28 - helper_19*helper_28 + helper_27) + 3*helper_14*helper_25 + helper_15*(897.1330589849432*helper_12 + 575.00960219479771*helper_13 + 127.99999999999588*helper_20 + 63.99999999999315*helper_22 - 4.9944136617912803e-13*helper_23 + 6.2764608325475975e-14*helper_24 + 533.6241426612014*helper_26 - 4.2193957518598271e-12*helper_28*helper_6 - 255.99999999999864*helper_28*helper_8 + 1.6340289889349434e-12*helper_3 + 128.00000000000063*helper_4 + 1.6643805675241655e-12*helper_6 + 149.99634202105079*helper_7 + 63.999999999997122*helper_8 - 2.8357014952924765e-13*x + 5.3333333333333792*y) + helper_5) + 1820.4444444444703*helper_1*helper_2 + helper_10*helper_6*(2958.2222222223672*helper_6 + 10887.111111111706*helper_7 + 3982.2222222223099*helper_8 + 1026.5679012347521*x + 1262.6172839509084*y + 44.73708276179827) + 4*helper_11*helper_14*helper_7 + helper_15*(helper_0*helper_16 + helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_27))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 249.00000000007782*y;
double helper_6 = helper_4*x;
double helper_7 = 1141.5277777778599*y;
double helper_8 = x*y;
double helper_9 = 474.00000000009271*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 128.00000000000145*helper_2;
double helper_15 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 474.00000000009538) + 2*helper_0*helper_6*(helper_5*x + 474.00000000009538*x + 474.00000000009271*y + 96.611111111100485) + helper_1*(1397.8333333333367*helper_0*helper_8 + 751.86111111108528*helper_0*y + 128.00000000000003*helper_1 + 191.99999999999997*helper_10 + 256.00000000000091*helper_11 + 803.00000000005525*helper_12 + 559.24999999998397*helper_13 - helper_14*helper_15 + helper_14 - 698.91666666666561*helper_15*helper_4 - 320.00000000000074*helper_15*x + 267.66666666668164*helper_3 + 827.527777777766*helper_4 + 2089.8333333335195*helper_6 + 1655.0555555555393*helper_8 + 63.999999999999773*x + 197.94444444443454*y + 69.333333333333229*z - 63.999999999999893) + helper_11*y*(helper_7 + 948.00000000019077*helper_8 + helper_9 + 535.33333333337021*x + 128.61111111110108) + helper_13*(474.00000000009538*helper_12 + 267.6666666666851*helper_2 + 267.66666666668164*helper_4 + helper_7*x + helper_9*x + 128.61111111110108*x + 128.61111111110114*y + 5.3333333333333339) + 747.00000000023351*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 249.00000000007782*x;
double helper_6 = helper_4*y;
double helper_7 = pow(helper_0, 2);
double helper_8 = 1141.5277777778599*x;
double helper_9 = x*y;
double helper_10 = 474.00000000009538*helper_4;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 127.99999999999744*helper_3;
double helper_14 = -helper_0;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 474.00000000009271) + 2*helper_0*helper_6*(helper_5*y + 474.00000000009538*x + 474.00000000009271*y + 96.611111111100485) + helper_1*(1397.8333333333312*helper_0*helper_9 + 751.86111111108528*helper_0*x + 127.99999999999976*helper_1 + 803.00000000004491*helper_11 + 559.24999999998397*helper_12 - helper_13*helper_14 + helper_13 - 698.91666666666833*helper_14*helper_4 - 319.9999999999975*helper_14*y + 267.6666666666851*helper_2 + 827.52777777776964*helper_4 + 2089.8333333335195*helper_6 + 255.99999999999784*helper_7*y + 191.99999999999969*helper_7 + 1655.055555555532*helper_9 + 197.94444444443454*x + 64.000000000000227*y + 69.333333333333783*z - 64.000000000000455) + helper_12*(helper_10*y + 474.00000000009271*helper_11 + 267.66666666668164*helper_3 + 267.6666666666851*helper_4 + helper_8*y + 128.61111111110108*x + 128.61111111110114*y + 5.3333333333333339) + 747.00000000023351*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 948.00000000018542*helper_9 + 535.33333333336327*y + 128.61111111110114))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = y*z;
double helper_4 = pow(x, 2);
double helper_5 = pow(x, 3);
double helper_6 = pow(y, 2);
double helper_7 = pow(y, 3);
double helper_8 = pow(z, 5);
double helper_9 = pow(z, 6);
double helper_10 = helper_6*x;
double helper_11 = helper_7*x;
double helper_12 = helper_2*x;
double helper_13 = helper_4*y;
double helper_14 = helper_5*y;
double helper_15 = helper_0*y;
double helper_16 = helper_1*y;
double helper_17 = helper_2*y;
double helper_18 = helper_8*y;
double helper_19 = helper_6*z;
double helper_20 = 170.66666666666328*helper_7;
double helper_21 = helper_4*helper_7;
double helper_22 = helper_0*helper_4;
double helper_23 = helper_5*helper_7;
double helper_24 = helper_0*helper_5;
double helper_25 = helper_0*helper_7;
val[2] = -(4064.8888888888919*helper_0*helper_10 - 474.00000000009265*helper_0*helper_21 - 1599.9999999999861*helper_0*helper_6 + 2336.0000000000023*helper_0*x - 938.6666666666664*helper_0 - 2795.6666666666624*helper_1*helper_10 - 2795.6666666666742*helper_1*helper_13 - helper_1*helper_20 + 1920.0000000000077*helper_1*helper_4 - 170.66666666666856*helper_1*helper_5 + 1919.9999999999839*helper_1*helper_6 - 4117.3333333333367*helper_1*x + 2218.6666666666661*helper_1 - 2538.4444444444603*helper_10*z + 570.30555555556452*helper_10 + 535.33333333336327*helper_11*z - 267.66666666668164*helper_11 + 698.91666666666561*helper_12*helper_6 - 4840.638888888755*helper_12*y + 3909.3333333333358*helper_12 - 2538.4444444444716*helper_13*z + 570.30555555556725*helper_13 + 535.33333333337032*helper_14*z - 267.66666666668505*helper_14 + 4064.8888888889096*helper_15*helper_4 - 6679.1666666665005*helper_15*x + 2335.9999999999982*helper_15 + 8177.5555555553383*helper_16*x - 4117.3333333333267*helper_16 + 698.91666666666833*helper_17*helper_4 + 3909.3333333333267*helper_17 + 1118.4999999999682*helper_18*x - 1919.9999999999964*helper_18 + 2089.8333333335195*helper_19*helper_4 + 639.99999999999432*helper_19 - 1120.0000000000043*helper_2*helper_4 + 42.66666666666714*helper_2*helper_5 - 1119.9999999999907*helper_2*helper_6 + 42.666666666665805*helper_2*helper_7 - 2981.333333333333*helper_2 - helper_20*z + 474.00000000009277*helper_21 - 1141.527777777861*helper_22*helper_6 - 1600.0000000000068*helper_22 - 498.00000000015569*helper_23*z - 249.00000000007776*helper_23 - 474.00000000009527*helper_24*helper_6 - 267.66666666668516*helper_24*y + 256.00000000000296*helper_24 - 267.66666666668186*helper_25*x + 255.99999999999494*helper_25 + 2595.7222222221653*helper_3*x - 661.33333333333371*helper_3 - 948.30555555565866*helper_4*helper_6 + 256.00000000000091*helper_4*helper_8 + 640.00000000000307*helper_4*z - 96.00000000000054*helper_4 + 474.00000000009538*helper_5*helper_6 - 170.66666666666859*helper_5*z + 42.666666666667147*helper_5 + 255.99999999999778*helper_6*helper_8 - 95.999999999999091*helper_6 + 42.666666666665812*helper_7 - 1920.0000000000009*helper_8*x + 2314.6666666666665*helper_8 + 384.00000000000011*helper_9*x + 383.99999999999926*helper_9*y - 970.66666666666652*helper_9 - 371.97222222221598*x*y - 661.33333333333405*x*z + 69.333333333333428*x + 69.333333333333655*y + 170.66666666666663*pow(z, 7) + 202.66666666666652*z - 15.999999999999972)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 101.25000000002743*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*x;
double helper_8 = 643.18750000000773*y;
double helper_9 = x*y;
double helper_10 = 166.50000000003396*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*x;
double helper_13 = helper_2*y;
double helper_14 = helper_11*y;
double helper_15 = 4.4364512064030414e-13*helper_2;
double helper_16 = helper_0*y;
val[0] = (2*helper_0*helper_7*(helper_4*x + 166.50000000003422*x + 166.50000000003396*y + 152.87499999998747) + helper_1*(helper_0*helper_15 + 224.0000000000002*helper_0*x + 128.0*helper_1 + 240.00000000000003*helper_11 + 128.00000000000028*helper_12 + 195.75000000002132*helper_13 + 450.56249999998005*helper_14 + helper_15 + 775.62499999997385*helper_16*x + 763.43749999996771*helper_16 + 65.250000000006509*helper_3 + 588.68749999997397*helper_5 + 387.81249999998658*helper_6 + 980.62500000004093*helper_7 + 1177.3749999999495*helper_9 + 95.999999999999929*x + 324.87499999998761*y + 123.99999999999997*z - 111.99999999999997) + helper_12*y*(helper_10 + helper_8 + 333.00000000006844*helper_9 + 130.50000000001421*x + 200.87499999998766) + helper_14*(helper_10*x + 166.50000000003422*helper_13 + 65.250000000007105*helper_2 + 65.250000000006509*helper_5 + helper_8*x + 200.87499999998766*x + 200.87499999998772*y + 12.0) + 303.75000000008231*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 166.50000000003422))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 101.25000000002743*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 643.18750000000773*x;
double helper_10 = x*y;
double helper_11 = 166.50000000003422*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 1.1498302310286772e-12*helper_3;
val[1] = (helper_1*(-helper_0*helper_14 + 763.43749999996771*helper_0*x + 127.99999999999986*helper_1 + 1177.3749999999479*helper_10 + 195.75000000001953*helper_12 + 450.56249999998005*helper_13 - helper_14 + 65.250000000007105*helper_2 + 980.62500000004093*helper_5*y + 588.68749999997476*helper_5 + 387.81249999998693*helper_6 + 775.62499999997317*helper_7*x + 223.99999999999886*helper_7 + 127.99999999999896*helper_8*y + 239.99999999999983*helper_8 + 324.87499999998761*x + 96.000000000000085*y + 124.00000000000016*z - 112.00000000000016) + helper_10*helper_8*(333.00000000006793*helper_10 + helper_11 + helper_9 + 130.50000000001302*y + 200.87499999998772) + helper_13*(helper_11*y + 166.50000000003396*helper_12 + 65.250000000006509*helper_3 + 65.250000000007105*helper_5 + helper_9*y + 200.87499999998766*x + 200.87499999998772*y + 12.0) + 303.75000000008231*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 166.50000000003396) + 2*helper_5*helper_7*(helper_4*y + 166.50000000003422*x + 166.50000000003396*y + 152.87499999998747))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 1.4788170688010152e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 3.8327674367622588e-13*helper_10;
double helper_12 = pow(z, 5);
double helper_13 = pow(z, 6);
double helper_14 = helper_10*x;
double helper_15 = helper_1*x;
double helper_16 = helper_12*x;
double helper_17 = helper_7*y;
double helper_18 = 5.9152682752040609e-13*helper_7;
double helper_19 = helper_1*y;
double helper_20 = 1.5331069747049035e-12*helper_10;
double helper_21 = helper_6*helper_9;
double helper_22 = helper_0*helper_6;
double helper_23 = helper_2*helper_6;
double helper_24 = 166.50000000003422*helper_7;
double helper_25 = helper_10*helper_7;
double helper_26 = helper_0*helper_9;
double helper_27 = helper_2*helper_9;
double helper_28 = helper_0*helper_10;
val[2] = -(65.250000000007162*helper_0*helper_17 + 643.18750000000819*helper_0*helper_21 + 4442.6249999997963*helper_0*helper_3 - 8.8729024128060908e-13*helper_0*helper_7 - 1703.9999999999991*helper_0*x - 1703.9999999999977*helper_0*y + 1064.0*helper_0 + helper_1*helper_18 - helper_1*helper_20 - 5957.4999999997308*helper_1*helper_3 - 832.0000000000025*helper_1*helper_6 - 831.99999999999181*helper_1*helper_9 - 2752.0*helper_1 - 166.50000000003399*helper_10*helper_6 + helper_11*helper_2 + helper_11 - 128.00000000000026*helper_12*helper_6 - 127.99999999999895*helper_12*helper_9 + 1823.9999999999977*helper_12*y - 3224.0*helper_12 - 384.0*helper_13*x - 383.99999999999955*helper_13*y + 1408.0*helper_13 - 130.50000000001302*helper_14*z + 65.250000000006509*helper_14 + 1551.2499999999463*helper_15*helper_9 + 3375.9999999999995*helper_15 - 901.12499999996021*helper_16*y + 1824.0*helper_16 - 130.50000000001427*helper_17*z + 65.250000000007105*helper_17 + helper_18*z + 1551.2499999999482*helper_19*helper_6 + 3375.9999999999945*helper_19 + 3742.1874999998327*helper_2*helper_3 - helper_2*helper_8 - 3484.0*helper_2*x - 3483.999999999995*helper_2*y + 3948.0*helper_2 - helper_20*z - 980.62500000004127*helper_21*z + 337.43750000003286*helper_21 - 2125.9999999999336*helper_22*y + 608.00000000000216*helper_22 - 387.81249999998681*helper_23*y + 528.00000000000136*helper_23 + helper_24*helper_26 - helper_24*helper_9 + 202.50000000005488*helper_25*z + 101.2500000000274*helper_25 - 2125.9999999999322*helper_26*x + 607.99999999999307*helper_26 - 387.81249999998658*helper_27*x + 527.99999999999523*helper_27 + 166.50000000003394*helper_28*helper_6 + 65.250000000006565*helper_28*x + 2.2996604620573535e-12*helper_28 - 1475.8749999999297*helper_3*z + 149.68749999999238*helper_3 + 1149.4999999999709*helper_4*helper_9 + 399.99999999999972*helper_4 + 1149.4999999999727*helper_5*helper_6 + 399.99999999999966*helper_5 - 186.93749999999932*helper_6*y - 192.00000000000097*helper_6*z + 16.000000000000156*helper_6 - helper_8 - 186.93749999999886*helper_9*x - 191.99999999999716*helper_9*z + 15.999999999999559*helper_9 - 27.999999999999943*x - 28.000000000000057*y - 256.0*pow(z, 7) - 200.0*z + 12.0)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 27.000000000012804*y;
double helper_6 = helper_4*x;
double helper_7 = pow(helper_0, 2);
double helper_8 = 297.02777777777692*y;
double helper_9 = x*y;
double helper_10 = 54.000000000016527*helper_4;
double helper_11 = helper_2*y;
double helper_12 = helper_7*y;
double helper_13 = 1.4388490399146578e-13*helper_2;
double helper_14 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 54.000000000017017) + 2*helper_0*helper_6*(helper_5*x + 54.000000000017017*x + 54.000000000016527*y + 75.611111111102346) + helper_1*(334.83333333331643*helper_0*helper_9 + 312.0277777777564*helper_0*y + 42.666666666666664*helper_1 + 81.00000000001296*helper_11 + 183.08333333332061*helper_12 - helper_13*helper_14 + helper_13 - 167.4166666666577*helper_14*helper_4 - 7.1942451995732813e-14*helper_14*x + 27.000000000003698*helper_3 + 243.02777777776006*helper_4 + 442.83333333334923*helper_6 + 9.5923269327643843e-14*helper_7*x + 96.0*helper_7 + 486.05555555552132*helper_9 - 2.3980817331910961e-14*x + 144.94444444443567*y + 69.333333333333314*z - 53.333333333333314) + helper_12*(helper_10*x + 54.000000000017017*helper_11 + 27.00000000000432*helper_2 + 27.000000000003698*helper_4 + helper_8*x + 75.611111111102389*x + 75.611111111102446*y + 16.0) + 81.000000000038412*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 108.00000000003403*helper_9 + 54.00000000000864*x + 75.611111111102389))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 27.000000000012804*x;
double helper_6 = helper_4*y;
double helper_7 = 297.02777777777692*x;
double helper_8 = x*y;
double helper_9 = 54.000000000017017*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 4.8816506392768587e-13*helper_3;
double helper_14 = helper_0*x;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 54.000000000016527) + 2*helper_0*helper_6*(helper_5*y + 54.000000000017017*x + 54.000000000016527*y + 75.611111111102346) + helper_1*(-helper_0*helper_13 + 167.41666666665822*helper_0*helper_4 - 4.8477888370257147e-13*helper_0*y + 42.6666666666666*helper_1 - 4.5393318733506268e-13*helper_10*y + 95.999999999999943*helper_10 + 183.08333333332061*helper_11 + 81.000000000011099*helper_12 - helper_13 + 334.83333333331541*helper_14*y + 312.0277777777564*helper_14 + 27.00000000000432*helper_2 + 243.02777777776066*helper_4 + 442.83333333334923*helper_6 + 486.05555555552013*helper_8 + 144.94444444443567*x + 2.0113540462803081e-14*y + 69.333333333333385*z - 53.333333333333385) + helper_11*y*(helper_7 + 108.00000000003305*helper_8 + helper_9 + 54.000000000007397*y + 75.611111111102446) + helper_11*(54.000000000016527*helper_12 + 27.000000000003698*helper_3 + 27.00000000000432*helper_4 + helper_7*y + helper_9*y + 75.611111111102389*x + 75.611111111102446*y + 16.0) + 81.000000000038412*helper_2*helper_3)/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 4.7961634663821505e-14*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 1.6272168797589545e-13*helper_10;
double helper_12 = pow(z, 5);
double helper_13 = pow(z, 6);
double helper_14 = helper_9*x;
double helper_15 = helper_1*x;
double helper_16 = helper_12*x;
double helper_17 = helper_7*y;
double helper_18 = 27.00000000000432*helper_17;
double helper_19 = 1.9184653865528784e-13*helper_7;
double helper_20 = helper_1*y;
double helper_21 = helper_9*z;
double helper_22 = 6.5088675190358181e-13*helper_10;
double helper_23 = helper_6*helper_9;
double helper_24 = helper_0*helper_6;
double helper_25 = helper_2*helper_6;
double helper_26 = helper_7*helper_9;
double helper_27 = helper_10*helper_7;
double helper_28 = helper_0*helper_10;
val[2] = (-928.88888888884389*helper_0*helper_14 + helper_0*helper_18 + 297.02777777777692*helper_0*helper_23 + 54.00000000001701*helper_0*helper_26 + 1805.499999999874*helper_0*helper_3 - 2.87769807982931e-13*helper_0*helper_7 - 3.0849952222429133e-12*helper_0*helper_9 - 415.99999999999989*helper_0*x - 415.99999999999852*helper_0*y + 554.66666666666629*helper_0 + helper_1*helper_19 - helper_1*helper_22 - 2413.555555555386*helper_1*helper_3 - 8.1534778928497406e-13*helper_1*helper_6 + 3.569774105945483e-12*helper_1*helper_9 - 1536.0*helper_1 - 54.00000000000739*helper_10*helper_4 - 54.000000000016527*helper_10*helper_6 + 27.000000000003702*helper_10*x + helper_11*helper_2 + helper_11 - 9.5923269327644537e-14*helper_12*helper_6 + 4.5393318733506227e-13*helper_12*helper_9 + 575.99999999999886*helper_12*y - 1994.6666666666665*helper_12 - 128.0*helper_13*x - 127.9999999999998*helper_13*y + 906.66666666666652*helper_13 - 167.4166666666577*helper_14*helper_2 - 91.805555555555259*helper_14 + 669.66666666663104*helper_15*helper_9 + 917.33333333333326*helper_15 - 366.16666666664128*helper_16*y + 576.0*helper_16 - 54.00000000000864*helper_17*z + helper_18 + helper_19*z + 1518.8055555554499*helper_2*helper_3 - helper_2*helper_8 - 2.0272764948240262e-12*helper_2*helper_9 - 1029.3333333333333*helper_2*x - 1029.333333333331*helper_2*y + 2330.6666666666665*helper_2 + 669.66666666663264*helper_20*helper_6 + 917.33333333333076*helper_20 - 442.833333333349*helper_21*helper_6 + 1.3001081692701707e-12*helper_21 - helper_22*z + 145.8055555555722*helper_23 - 928.88888888884662*helper_24*y + 7.4340533728923433e-13*helper_24 - 167.41666666665822*helper_25*y + 4.4364512064035225e-13*helper_25 - 54.000000000017025*helper_26 + 54.000000000025608*helper_27*z + 27.000000000012804*helper_27 + 54.000000000016513*helper_28*helper_6 + 27.000000000003709*helper_28*x + 9.7633012785537276e-13*helper_28 - 614.72222222218056*helper_3*z + 70.138888888884821*helper_3 + 518.44444444442593*helper_4*helper_9 + 85.333333333333258*helper_4 + 518.44444444442809*helper_5*helper_6 + 85.333333333332916*helper_5 - 91.805555555555827*helper_6*y - 3.3573144264675371e-13*helper_6*z + 5.9952043329777234e-14*helper_6 - helper_8 - 2.1154374548377671e-13*helper_9 - 5.3333333333333144*x - 5.3333333333333002*y - 170.66666666666663*pow(z, 7) - 95.999999999999886*z + 5.3333333333333144)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 249.00000000003811*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 280.47222222226594*y;
double helper_8 = x*y;
double helper_9 = 273.00000000004252*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 128.00000000000136*helper_2;
double helper_15 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_14 - 63.999999999999368*helper_0*x + 7.1054273576010019e-15*helper_1 - 1.0524914273447413e-13*helper_10 + 8.9883656073660105e-13*helper_11 + 803.00000000001251*helper_12 - 35.583333333334132*helper_13 + helper_14 + 208.16666666667231*helper_15*x - 100.19444444444423*helper_15 + 66.666666666671176*helper_3 + 754.16666666675474*helper_5*x - 33.527777777774418*helper_5 + 31.083333333336981*helper_6 - 49.055555555548011*helper_8 - 64.000000000000199*x - 59.277777777776819*y + 5.3333333333332194*z + 1.1457501614131615e-13) + helper_11*y*(helper_7 + 948.00000000008299*helper_8 + helper_9 + 535.33333333334167*x - 128.61111111111035) + helper_13*(474.0000000000415*helper_12 + 267.66666666667084*helper_2 + 66.666666666671176*helper_5 + helper_7*x + helper_9*x - 128.61111111111035*x - 64.611111111110503*y + 5.3333333333333339) + 747.00000000011437*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 474.0000000000415) + 2*helper_6*x*(helper_4*x + 474.0000000000415*x + 273.00000000004252*y - 96.611111111110915))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 249.00000000003811*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 280.47222222226594*x;
double helper_10 = x*y;
double helper_11 = 474.0000000000415*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 1.2970735596696374e-12*helper_3;
val[1] = (-helper_1*(-62.166666666673962*helper_0*helper_10 + helper_0*helper_14 + 100.19444444444423*helper_0*x - 8.2822637637043191e-14*helper_1 + 67.055555555548835*helper_10 - 200.00000000001353*helper_12 + 35.583333333334132*helper_13 + helper_14 - 267.66666666667084*helper_2 - 754.16666666675474*helper_5*y + 24.527777777774006*helper_5 - 104.08333333333616*helper_6 + 5.7511403047295657e-13*helper_7 + 6.517009154549986e-13*helper_8*y - 2.4338864257345752e-13*helper_8 + 59.277777777776819*x - 4.4136916343976006e-13*y - 5.4394451905659117e-13*z + 5.4394451905659117e-13) + helper_10*helper_8*(546.00000000008504*helper_10 + helper_11 + helper_9 + 133.33333333334235*y - 64.611111111110503) + helper_13*(helper_11*y + 273.00000000004252*helper_12 + 66.666666666671176*helper_3 + 267.66666666667084*helper_5 + helper_9*y - 128.61111111111035*x - 64.611111111110503*y + 5.3333333333333339) + 747.00000000011437*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 273.00000000004252) + 2*helper_5*helper_7*(helper_4*y + 474.0000000000415*x + 273.00000000004252*y - 96.611111111110915))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = pow(y, 2);
double helper_8 = pow(y, 3);
double helper_9 = 4.3235785322321211e-13*helper_8;
double helper_10 = helper_7*x;
double helper_11 = helper_8*x;
double helper_12 = 66.666666666671176*helper_11;
double helper_13 = helper_1*x;
double helper_14 = helper_2*x;
double helper_15 = pow(z, 5);
double helper_16 = pow(z, 6);
double helper_17 = helper_6*y;
double helper_18 = helper_1*y;
double helper_19 = helper_15*y;
double helper_20 = helper_7*z;
double helper_21 = 1.7294314128928494e-12*helper_8;
double helper_22 = 273.00000000004252*helper_5;
double helper_23 = helper_0*helper_5;
double helper_24 = helper_2*helper_5;
double helper_25 = helper_6*helper_8;
double helper_26 = helper_0*helper_6;
double helper_27 = helper_0*helper_8;
val[2] = -(-251.11111111113229*helper_0*helper_10 + helper_0*helper_12 + 267.66666666667106*helper_0*helper_17 - 105.16666666668391*helper_0*helper_3 - 4.791667063131118e-12*helper_0*helper_7 - 32.000000000001748*helper_0*x - 2.12291295653734e-12*helper_0*y - helper_1*helper_21 - 128.00000000000773*helper_1*helper_5 + 170.66666666666845*helper_1*helper_6 + 5.3667810936040736e-12*helper_1*helper_7 - 95.69444444444747*helper_10 - 133.33333333334235*helper_11*z + helper_12 + 124.33333333334792*helper_13*helper_7 + 310.88888888890568*helper_13*y + 21.333333333335418*helper_13 - 31.083333333336981*helper_14*helper_7 - 5.3333333333345898*helper_14 - 8.9883656073660105e-13*helper_15*helper_5 + 6.5170091545499819e-13*helper_15*helper_7 + 3.383959779057674e-13*helper_15*x - 2.1316282072803006e-14*helper_16*x - 2.4846791291113043e-13*helper_16*y - 535.3333333333419*helper_17*z + 267.66666666667084*helper_17 + 416.3333333333444*helper_18*helper_5 + 2.2773634829798064e-12*helper_18 + 71.166666666668277*helper_19*x + 1.0040301923198621e-12*helper_19 - 255.6388888888971*helper_2*helper_3 - 42.666666666667112*helper_2*helper_6 - 2.9709475620385138e-12*helper_2*helper_7 + helper_2*helper_9 - 1.8370767869889597e-12*helper_2*y - 754.16666666675383*helper_20*helper_5 + 253.55555555556884*helper_20*x + 2.1082765163290796e-12*helper_20 - helper_21*z + helper_22*helper_27 - helper_22*helper_8 + 280.47222222226571*helper_23*helper_7 - 753.111111111127*helper_23*y + 192.00000000000711*helper_23 - 104.08333333333616*helper_24*y + 32.000000000004178*helper_24 + 498.00000000007628*helper_25*z + 249.00000000003809*helper_25 + 474.00000000004161*helper_26*helper_7 - 256.00000000000273*helper_26 + 2.5941471193392749e-12*helper_27 - 55.611111111102275*helper_3*z + 34.361111111109295*helper_3 + 673.55555555556532*helper_4*helper_5 + 1.232699127958567e-12*helper_4 + 473.69444444448783*helper_5*helper_7 - 232.6944444444465*helper_5*y - 128.00000000000324*helper_5*z + 32.000000000000583*helper_5 - 474.0000000000415*helper_6*helper_7 + 170.6666666666685*helper_6*z - 42.666666666667119*helper_6 - 3.6414390021851975e-13*helper_7 + helper_9 + 21.333333333334057*x*z - 5.3333333333334512*x - 3.0563514682080475e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 101.25000000000925*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = 143.68749999998849*y;
double helper_9 = x*y;
double helper_10 = 137.25000000000878*helper_5;
double helper_11 = helper_2*y;
double helper_12 = helper_7*y;
double helper_13 = 4.4364512064023527e-13*helper_2;
val[0] = -(-helper_1*(-helper_0*helper_13 + 384.12499999999699*helper_0*helper_9 + 223.9999999999998*helper_0*x + 218.18749999999528*helper_0*y - 195.74999999999829*helper_11 + 129.31249999999753*helper_12 - helper_13 - 35.999999999999844*helper_3 - 18.375000000019668*helper_5*x + 198.18749999999602*helper_5 + 93.312499999997712*helper_6 + 127.9999999999997*helper_7*x - 15.999999999999963*helper_7 + 785.87499999999295*helper_9 + 96.000000000000071*x + 76.874999999997726*y - 27.999999999999964*z + 15.999999999999964) + helper_12*(helper_10*x + 166.50000000000838*helper_11 + 65.249999999999432*helper_2 + 35.999999999999844*helper_5 - helper_8*x - 200.8749999999979*x - 104.8749999999979*y + 12.0) + 303.75000000002774*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 166.50000000000838) + 2*helper_6*x*(helper_4*x + 166.50000000000838*x + 137.25000000000878*y - 152.8749999999981) + helper_7*helper_9*(helper_10 - helper_8 + 333.00000000001677*helper_9 + 130.49999999999886*x - 200.8749999999979))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 101.25000000000925*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*y;
double helper_8 = 143.68749999998849*x;
double helper_9 = x*y;
double helper_10 = 166.50000000000838*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*y;
double helper_13 = helper_3*x;
double helper_14 = helper_11*x;
double helper_15 = 5.4206639177308728e-14*helper_3;
double helper_16 = helper_0*y;
val[1] = -(2*helper_0*helper_7*(helper_4*y + 166.50000000000838*x + 137.25000000000878*y - 152.8749999999981) + helper_1*(helper_0*helper_15 - 218.18749999999528*helper_0*x + 9.9309449552721641e-14*helper_1 + 1.6728979312930096e-13*helper_11 + 2.7028379534498744e-13*helper_12 + 107.99999999999953*helper_13 - 129.31249999999753*helper_14 + helper_15 - 186.62499999999542*helper_16*x + 2.5270063819248355e-13*helper_16 + 65.249999999999432*helper_2 - 392.93749999999648*helper_5 - 192.06249999999849*helper_6 + 18.375000000019668*helper_7 - 396.37499999999204*helper_9 - 76.874999999997726*x + 1.4026280137358761e-13*y + 1.656938475314079e-13*z - 1.656938475314079e-13) + helper_12*x*(helper_10 - helper_8 + 274.50000000001756*helper_9 + 71.999999999999687*y - 104.8749999999979) + helper_14*(helper_10*y + 137.25000000000878*helper_13 + 35.999999999999844*helper_3 + 65.249999999999432*helper_5 - helper_8*y - 200.8749999999979*x - 104.8749999999979*y + 12.0) + 303.75000000002774*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 137.25000000000878))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 1.4788170688007966e-13*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 1.8068879725769576e-14*helper_9;
double helper_11 = helper_8*x;
double helper_12 = pow(z, 5);
double helper_13 = helper_5*y;
double helper_14 = helper_6*y;
double helper_15 = 5.9152682752031158e-13*helper_6;
double helper_16 = helper_0*y;
double helper_17 = helper_8*z;
double helper_18 = 7.2275518903077976e-14*helper_9;
double helper_19 = helper_5*helper_9;
double helper_20 = helper_0*helper_5;
double helper_21 = helper_6*helper_8;
double helper_22 = helper_6*helper_9;
double helper_23 = helper_0*helper_9;
val[2] = -(-454.99999999998835*helper_0*helper_11 - 951.49999999999318*helper_0*helper_13 - 137.25000000000881*helper_0*helper_19 - 166.50000000000836*helper_0*helper_21 + 1265.1249999999789*helper_0*helper_3 + 8.8729024128047105e-13*helper_0*helper_6 - 1.9447360388724241e-12*helper_0*helper_8 - 151.99999999999946*helper_0*x + 373.24999999999091*helper_1*helper_11 + 768.24999999999409*helper_1*helper_13 - helper_1*helper_15 - helper_1*helper_18 - 1713.49999999997*helper_1*helper_3 - 831.99999999999739*helper_1*helper_5 + 2.197436677064907e-12*helper_1*helper_8 + 207.99999999999937*helper_1*x - 3.2755465007029122e-12*helper_1*y + helper_10*helper_2 + helper_10 - 93.312499999997726*helper_11*helper_2 + 163.49999999999488*helper_11*z + 11.562500000000171*helper_11 - 258.62499999999511*helper_12*helper_3 - 127.99999999999972*helper_12*helper_5 + 2.7028379534498689e-13*helper_12*helper_8 + 31.999999999999929*helper_12*x - 1.4529905056903868e-12*helper_12*y - 192.06249999999852*helper_13*helper_2 + 366.49999999999818*helper_13*z + 8.8124999999993179*helper_13 + 130.49999999999886*helper_14*z - 65.249999999999432*helper_14 - helper_15*z - 65.249999999999432*helper_16*helper_6 + 2.1172924524749017e-12*helper_16 + 18.37500000001927*helper_17*helper_5 + 8.4601770033996956e-13*helper_17 - helper_18*z + 137.25000000000875*helper_19 + 1074.9374999999802*helper_2*helper_3 + 527.99999999999864*helper_2*helper_5 + helper_2*helper_7 - 1.2250686576286952e-12*helper_2*helper_8 - 131.99999999999966*helper_2*x + 2.9617211461108717e-12*helper_2*y + 143.68749999998846*helper_20*helper_8 + 607.99999999999773*helper_20 + 166.50000000000841*helper_21 - 202.50000000001847*helper_22*z - 101.25000000000927*helper_22 - 35.999999999999872*helper_23*x + 1.0841327835461696e-13*helper_23 - 396.37499999999415*helper_3*z + 28.437499999999773*helper_3 + 71.999999999999687*helper_4*helper_9 + 47.999999999999773*helper_4 - 162.06250000000773*helper_5*helper_8 - 191.99999999999889*helper_5*z + 15.999999999999801*helper_5 + helper_7 - 1.4393347624874509e-13*helper_8 - 35.999999999999844*helper_9*x - 3.9999999999999609*x + 2.9792834865816616e-13*y*pow(z, 6) - 7.7744755078161165e-13*y*z + 1.2904260993097247e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 26.999999999999059*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 135.02777777777806*y;
double helper_8 = x*y;
double helper_9 = 26.999999999997826*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 1.4388490399141418e-13*helper_2;
val[0] = (-helper_1*(-helper_0*helper_13 + 172.83333333333238*helper_0*helper_8 - 7.1942451995707039e-14*helper_0*x + 93.027777777774119*helper_0*y - 42.666666666666664*helper_1 - 9.5923269327609461e-14*helper_10*x - 96.0*helper_10 + 70.749999999998096*helper_11 - 80.999999999996064*helper_12 - helper_13 + 1.0871303857129533e-12*helper_3 + 118.83333333333701*helper_5*x + 189.02777777777541*helper_5 + 113.41666666666583*helper_6 + 324.05555555555122*helper_8 + 2.3980817331902365e-14*x + 6.2777777777760306*y - 69.333333333333314*z + 53.333333333333314) + helper_11*x*(-helper_7 + 107.99999999999534*helper_8 + helper_9 + 53.999999999997378*x - 75.611111111109409) + helper_11*(53.999999999997669*helper_12 + 26.999999999998689*helper_2 - 1.0871303857129533e-12*helper_5 - helper_7*x + helper_9*x - 75.611111111109409*x - 75.611111111109423*y + 16.0) + 80.999999999997172*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 53.999999999997669) + 2*helper_6*x*(helper_4*x + 53.999999999997669*x + 26.999999999997826*y - 75.611111111109452))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 26.999999999999059*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*y;
double helper_8 = 135.02777777777806*x;
double helper_9 = x*y;
double helper_10 = 53.999999999997669*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*x;
double helper_13 = helper_3*x;
double helper_14 = 1.8818280267396721e-13*helper_3;
double helper_15 = helper_0*x;
val[1] = (2*helper_0*helper_7*(helper_4*y + 53.999999999997669*x + 26.999999999997826*y - 75.611111111109452) - helper_1*(-helper_0*helper_14 - 2.5751623056180758e-13*helper_0*y - 6.2838623193783557e-14*helper_1 - 2.4639549659847789e-13*helper_11*y - 1.016316660458941e-13*helper_11 + 70.749999999998096*helper_12 + 3.2613911571388599e-12*helper_13 - helper_14 + 226.83333333333167*helper_15*y + 93.027777777774119*helper_15 - 26.999999999998689*helper_2 + 162.02777777777561*helper_5 + 86.416666666666188*helper_6 + 118.83333333333701*helper_7 + 378.05555555555082*helper_9 + 6.2777777777760306*x - 4.4094357794694862e-14*y - 4.4362661692311778e-14*z + 4.4362661692311778e-14) - helper_12*y*(-helper_10 + helper_8 - 53.999999999995651*helper_9 + 2.1742607714259066e-12*y + 75.611111111109423) + helper_12*(helper_10*y + 26.999999999997826*helper_13 - 1.0871303857129533e-12*helper_3 + 26.999999999998689*helper_5 - helper_8*y - 75.611111111109409*x - 75.611111111109423*y + 16.0) + 80.999999999997172*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 26.999999999997826))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 4.7961634663804251e-14*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 6.2727600891322821e-14*helper_9;
double helper_11 = helper_8*x;
double helper_12 = 1.0871303857129571e-12*helper_9;
double helper_13 = helper_0*x;
double helper_14 = pow(z, 5);
double helper_15 = helper_14*x;
double helper_16 = pow(z, 6);
double helper_17 = helper_5*y;
double helper_18 = helper_6*y;
double helper_19 = 1.91846538655217e-13*helper_6;
double helper_20 = helper_1*y;
double helper_21 = 2.5091040356529129e-13*helper_9;
double helper_22 = helper_5*helper_8;
double helper_23 = 26.999999999997826*helper_9;
double helper_24 = helper_0*helper_5;
double helper_25 = 53.999999999997669*helper_6;
double helper_26 = helper_6*helper_9;
double helper_27 = helper_0*helper_8;
val[2] = -(604.88888888888573*helper_0*helper_11 + 442.88888888888778*helper_0*helper_17 + 26.999999999998678*helper_0*helper_18 - 135.027777777778*helper_0*helper_22 - 840.83333333331723*helper_0*helper_3 - 2.8776980798282896e-13*helper_0*helper_6 - 3.763656053479369e-13*helper_0*helper_9 - 1.0612806929562488e-12*helper_0*y - 453.66666666666333*helper_1*helper_11 - 345.6666666666647*helper_1*helper_17 + helper_1*helper_19 + helper_1*helper_21 - 8.1534778928468093e-13*helper_1*helper_5 - 1.9489225048611639e-12*helper_1*helper_8 + 917.33333333333326*helper_1*x - helper_10*helper_2 - helper_10 + 113.41666666666583*helper_11*helper_2 - 302.44444444444457*helper_11*z + 37.805555555556452*helper_11 - helper_12*helper_13 - helper_12*x - 415.99999999999989*helper_13 - 9.5923269327608502e-14*helper_14*helper_5 - 2.4639549659848016e-13*helper_14*helper_8 + 9.2783188539631549e-13*helper_14*y + 141.49999999999616*helper_15*y + 576.0*helper_15 - 128.0*helper_16*x - 1.8851586958135024e-13*helper_16*y + 86.416666666666174*helper_17*helper_2 - 194.44444444444599*helper_17*z + 10.805555555556765*helper_17 - 53.999999999997385*helper_18*z + 26.999999999998685*helper_18 + helper_19*z - 614.47222222220694*helper_2*helper_3 + 4.4364512064019408e-13*helper_2*helper_5 - helper_2*helper_7 + 1.1032193677114859e-12*helper_2*helper_8 - 1029.3333333333333*helper_2*x - 1.8557840449536323e-12*helper_2*y + 1042.8888888888653*helper_20*x + 1.9151347174783781e-12*helper_20 + helper_21*z + 118.83333333333712*helper_22*z + 16.194444444440876*helper_22 + helper_23*helper_24 - helper_23*helper_5 + 7.4340533728897473e-13*helper_24 + helper_25*helper_27 - helper_25*helper_8 + 53.99999999999811*helper_26*z + 26.999999999999062*helper_26 + 1.6914062742993524e-12*helper_27 - 32.472222222222072*helper_3 + 2.1742607714259021e-12*helper_4*helper_9 + 303.38888888888448*helper_4*y + 85.333333333333258*helper_4 - 3.357314426466335e-13*helper_5*z + 5.9952043329755676e-14*helper_5 - helper_7 - 7.1694502186877337e-13*helper_8*z + 1.176373813175742e-13*helper_8 - 5.3333333333333144*x + 2.9222920379841039e-13*y*z - 2.9615199181873842e-14*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 249.00000000002979*y;
double helper_6 = helper_4*x;
double helper_7 = 538.52777777780466*y;
double helper_8 = x*y;
double helper_9 = 273.00000000003411*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 7.2741812573448496e-13*helper_2;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 273.0000000000307) + 2*helper_0*helper_6*(helper_5*x + 273.0000000000307*x + 273.00000000003411*y + 96.611111111108983) + helper_1*(-6.0618177144540413e-14*helper_10 + 4.8494541715632331e-13*helper_11 + 200.00000000000458*helper_12 + 102.24999999999511*helper_13 - helper_14*helper_16 + helper_14 + 337.83333333332575*helper_15*x + 166.86111111110483*helper_15 - 168.91666666666612*helper_16*helper_4 - 3.6370906286724223e-13*helper_16*x + 66.666666666671119*helper_3 + 233.5277777777749*helper_4 + 883.83333333339203*helper_6 + 467.05555555554486*helper_8 - 1.2123635428908083e-13*x + 69.944444444442908*y - 6.0618177144540413e-14*z + 6.0618177144540413e-14) + helper_11*y*(helper_7 + 546.00000000006139*helper_8 + helper_9 + 133.33333333333638*x + 64.611111111109452) + helper_13*(273.0000000000307*helper_12 + 66.666666666668192*helper_2 + 66.666666666671119*helper_4 + helper_7*x + helper_9*x + 64.611111111109452*x + 64.611111111109366*y + 5.3333333333333339) + 747.00000000008936*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 249.00000000002979*x;
double helper_6 = helper_4*y;
double helper_7 = 538.52777777780466*x;
double helper_8 = x*y;
double helper_9 = 273.0000000000307*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 7.8803630287896822e-13*helper_3;
double helper_14 = helper_0*x;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 273.00000000003411) + 2*helper_0*helper_6*(helper_5*y + 273.0000000000307*x + 273.00000000003411*y + 96.611111111108983) + helper_1*(-helper_0*helper_13 + 168.91666666666288*helper_0*helper_4 - 6.7749509706044839e-13*helper_0*y - 1.7652546091533173e-14*helper_1 - 5.6066262743571193e-13*helper_10*y - 1.7402745910992399e-14*helper_10 + 102.24999999999511*helper_11 + 200.00000000001336*helper_12 - helper_13 + 337.83333333333223*helper_14*y + 166.86111111110483*helper_14 + 66.666666666668192*helper_2 + 233.52777777777243*helper_4 + 883.83333333339203*helper_6 + 467.0555555555498*helper_8 + 69.944444444442908*x + 1.9129142714293487e-13*y + 2.5465740627341896e-13*z - 2.5465740627341896e-13) + helper_11*y*(helper_7 + 546.00000000006821*helper_8 + helper_9 + 133.33333333334224*y + 64.611111111109366) + helper_11*(273.00000000003411*helper_12 + 66.666666666671119*helper_3 + 66.666666666668192*helper_4 + helper_7*y + helper_9*y + 64.611111111109452*x + 64.611111111109366*y + 5.3333333333333339) + 747.00000000008936*helper_2*helper_3)/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 2.424727085781612e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 2.6267876762632254e-13*helper_10;
double helper_12 = helper_9*x;
double helper_13 = helper_10*x;
double helper_14 = 9.6989083431264762e-13*helper_1;
double helper_15 = pow(z, 5);
double helper_16 = helper_7*y;
double helper_17 = helper_1*y;
double helper_18 = helper_9*z;
double helper_19 = 1.0507150705052914e-12*helper_10;
double helper_20 = 273.00000000003411*helper_6;
double helper_21 = helper_0*helper_6;
double helper_22 = helper_2*helper_6;
double helper_23 = 273.0000000000307*helper_7*helper_9;
double helper_24 = helper_10*helper_7;
double helper_25 = helper_1*helper_9;
double helper_26 = helper_2*helper_9;
double helper_27 = helper_0*helper_10;
val[2] = (-948.88888888888778*helper_0*helper_12 + 66.666666666668135*helper_0*helper_16 + helper_0*helper_23 + 1049.1666666666065*helper_0*helper_3 - 1.4548362514689701e-12*helper_0*helper_7 - 3.5741409831757745e-12*helper_0*helper_9 - 8.4865448002356649e-13*helper_0*x - 1.0816347817413681e-12*helper_0*y - helper_1*helper_19 - 4.1220360458287489e-12*helper_1*helper_6 - helper_10*helper_20 + helper_11*helper_2 + helper_11 - 104.30555555555674*helper_12 - 133.3333333333423*helper_13*z + 66.666666666671119*helper_13 + helper_14*helper_7 + helper_14*x - 204.49999999999022*helper_15*helper_3 - 4.849454171563224e-13*helper_15*helper_6 + 5.6066262743571617e-13*helper_15*helper_9 + 1.2123635428908222e-13*helper_15*x - 2.829403378256129e-13*helper_15*y - 133.33333333333638*helper_16*z + 66.666666666668192*helper_16 + 675.66666666665151*helper_17*helper_6 - 1377.5555555554829*helper_17*x + 3.0753177782153369e-13*helper_17 - 883.83333333339192*helper_18*helper_6 + 546.44444444444582*helper_18*x + 1.4483229430576631e-12*helper_18 - helper_19*z + 855.63888888884617*helper_2*helper_3 - helper_2*helper_8 - 5.4556359430086342e-13*helper_2*x + 3.6567970873565012e-13*helper_2*y + helper_20*helper_27 + 538.52777777780466*helper_21*helper_9 - 948.88888888886765*helper_21*y + 3.7583269829614652e-12*helper_21 - 168.91666666666288*helper_22*y + 2.2428725543479957e-12*helper_22 - helper_23 + 498.00000000005957*helper_24*z + 249.00000000002979*helper_24 + 675.6666666666647*helper_25*x + 4.2516360802362237e-12*helper_25 - 168.91666666666612*helper_26*x - 2.4645655886483348e-12*helper_26 + 66.666666666671176*helper_27*x + 1.5760726057579352e-12*helper_27 + 42.972222222218718*helper_3 - 365.72222222219841*helper_4*y + 3.6370906286724228e-13*helper_4 + 546.44444444443252*helper_5*helper_6 + 8.7491125455600274e-13*helper_5 + 345.30555555558675*helper_6*helper_9 - 104.30555555555338*helper_6*y - 1.697308960047133e-12*helper_6*z + 3.0309088572270222e-13*helper_6 + 9.6989083431264762e-13*helper_7*z - helper_8 - 2.2191507890548736e-13*helper_9 - 6.0618177144541108e-14*x + 5.2957638274599412e-14*y*pow(z, 6) - 2.3650525982080386e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 101.25000000002478*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 555.43750000002888*y;
double helper_8 = x*y;
double helper_9 = 137.25000000003061*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 3.6570746431160699e-13*helper_2;
double helper_15 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_14 + 1.8285373215580334e-13*helper_0*x - 3.0475622025967249e-14*helper_10 + 2.4380497620773799e-13*helper_11 + 108.00000000001671*helper_12 + 165.31249999999622*helper_13 + helper_14 + 402.62500000000318*helper_15*x + 286.18749999999363*helper_15 + 36.000000000005777*helper_3 + 805.12500000006344*helper_5*x + 306.18749999999903*helper_5 + 201.31250000000193*helper_6 + 612.37499999999807*helper_8 - 6.0951244051934498e-14*x + 132.87499999999739*y - 3.0475622025967249e-14*z + 3.0475622025967249e-14) + helper_11*y*(helper_7 + 274.50000000006014*helper_8 + helper_9 + 72.000000000011141*x + 104.87499999999736) + helper_13*(137.25000000003007*helper_12 + 36.000000000005571*helper_2 + 36.000000000005777*helper_5 + helper_7*x + helper_9*x + 104.87499999999736*x + 104.87499999999737*y + 12.0) + 303.75000000007435*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 137.25000000003007) + 2*helper_6*x*(helper_4*x + 137.25000000003007*x + 137.25000000003061*y + 152.87499999999719))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 101.25000000002478*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = 555.43750000002888*x;
double helper_9 = x*y;
double helper_10 = 137.25000000003007*helper_5;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 7.5522921250131272e-13*helper_3;
double helper_14 = helper_0*x;
val[1] = (helper_1*(-helper_0*helper_13 - 7.050193762126282e-13*helper_0*y - 7.0915495697927835e-14*helper_1 + 108.00000000001734*helper_11 + 165.31249999999622*helper_12 - helper_13 + 402.62500000000387*helper_14*y + 286.18749999999363*helper_14 + 36.000000000005571*helper_2 + 805.12500000006344*helper_5*y + 306.18749999999903*helper_5 + 201.31250000000159*helper_6 - 6.4531713306339748e-13*helper_7*y - 8.9643570344572826e-14*helper_7 + 612.37499999999807*helper_9 + 132.87499999999739*x + 8.0685458314652449e-14*y + 1.3263695697321113e-13*z - 1.3263695697321113e-13) + helper_12*(helper_10*y + 137.25000000003061*helper_11 + 36.000000000005777*helper_3 + 36.000000000005571*helper_5 + helper_8*y + 104.87499999999736*x + 104.87499999999737*y + 12.0) + 303.75000000007435*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 137.25000000003061) + 2*helper_6*y*(helper_4*y + 137.25000000003007*x + 137.25000000003061*y + 152.87499999999719) + helper_7*helper_9*(helper_10 + helper_8 + 274.50000000006122*helper_9 + 72.000000000011553*y + 104.87499999999737))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 1.219024881038693e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 2.5174307083377141e-13*helper_10;
double helper_12 = helper_10*x;
double helper_13 = helper_0*x;
double helper_14 = 4.8760995241547518e-13*helper_1;
double helper_15 = helper_2*x;
double helper_16 = pow(z, 5);
double helper_17 = helper_6*y;
double helper_18 = helper_7*y;
double helper_19 = 36.000000000005571*helper_18;
double helper_20 = helper_0*y;
double helper_21 = helper_2*y;
double helper_22 = 1.0069722833350836e-12*helper_10;
double helper_23 = helper_6*helper_9;
double helper_24 = helper_0*helper_6;
double helper_25 = 137.25000000003007*helper_9;
double helper_26 = helper_10*helper_7;
double helper_27 = helper_0*helper_7;
double helper_28 = helper_1*helper_9;
double helper_29 = helper_0*helper_10;
val[2] = -(helper_0*helper_19 + 555.43750000002888*helper_0*helper_23 - 4.3381132019960904e-12*helper_0*helper_9 + 805.25000000000614*helper_1*helper_17 - helper_1*helper_22 - 2161.49999999995*helper_1*helper_3 - 2.0723422977657736e-12*helper_1*helper_6 - 1.9315105070913685e-12*helper_1*y - 137.25000000003058*helper_10*helper_6 + helper_11*helper_2 + helper_11 - 72.000000000011596*helper_12*z + 36.000000000005784*helper_12 - 1103.0000000000143*helper_13*helper_9 - 4.2665870836354055e-13*helper_13 + helper_14*helper_7 + helper_14*x - 201.31250000000193*helper_15*helper_9 - 2.7428059823370591e-13*helper_15 - 330.6249999999925*helper_16*helper_3 - 2.438049762077386e-13*helper_16*helper_6 + 6.4531713306339698e-13*helper_16*helper_9 + 6.095124405193465e-14*helper_16*x - 1.0971917818735549e-12*helper_16*y - 96.437500000004263*helper_17 - 72.000000000011141*helper_18*z + helper_19 + 1366.9374999999686*helper_2*helper_3 + 1.1275980149607894e-12*helper_2*helper_6 - helper_2*helper_8 - 2.8740759772106724e-12*helper_2*helper_9 + 1601.1249999999627*helper_20*x + 6.0250415767602796e-13*helper_20 - 201.31250000000159*helper_21*helper_6 + 2.1621246459878121e-12*helper_21 - helper_22*z - 805.12500000006344*helper_23*z + 249.68750000003456*helper_23 - 1103.0000000000123*helper_24*y + 1.8894885656099707e-12*helper_24 + helper_25*helper_27 - helper_25*helper_7 + 202.50000000004957*helper_26*z + 101.25000000002478*helper_26 - 7.3141492862321388e-13*helper_27 + 805.25000000000773*helper_28*x + 5.0431325782087184e-12*helper_28 + 137.25000000003064*helper_29*helper_6 + 36.000000000005798*helper_29*x + 1.5104584250026256e-12*helper_29 - 532.37499999998761*helper_3*z + 56.437499999998806*helper_3 + 595.50000000001307*helper_4*helper_9 + 1.8285373215580395e-13*helper_4 + 595.50000000001182*helper_5*helper_6 + 1.5050460877587289e-13*helper_5 - 8.5331741672708313e-13*helper_6*z + 1.5237811012983661e-13*helper_6 + 4.8760995241547518e-13*helper_7*z - helper_8 - 96.437500000004604*helper_9*x + 1.8165469128917306e-12*helper_9*z - 2.9280744495708298e-13*helper_9 - 3.0475622025967325e-14*x + 2.1274648709378356e-13*y*pow(z, 6) - 9.9177610568573783e-14*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 27.000000000005947*y;
double helper_6 = helper_4*x;
double helper_7 = pow(helper_0, 2);
double helper_8 = 216.02777777778243*y;
double helper_9 = x*y;
double helper_10 = 27.000000000007766*helper_4;
double helper_11 = helper_2*y;
double helper_12 = helper_7*y;
double helper_13 = 7.1942451995730389e-14*helper_2;
double helper_14 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 27.000000000007574) + 2*helper_0*helper_6*(helper_5*x + 27.000000000007574*x + 27.000000000007766*y + 75.611111111109196) + helper_1*(226.83333333333115*helper_0*helper_9 + 285.02777777777311*helper_0*y + 5.1105786269545206e-12*helper_11 + 156.0833333333305*helper_12 - helper_13*helper_14 + helper_13 - 113.41666666666556*helper_14*helper_4 - 3.5971225997865169e-14*helper_14*x + 1.723066134218243e-12*helper_3 + 189.02777777777479*helper_4 + 280.83333333334645*helper_6 + 4.7961634663820255e-14*helper_7*x - 5.9952043329775319e-15*helper_7 + 378.05555555554963*helper_9 - 1.1990408665955064e-14*x + 144.94444444444255*y - 5.9952043329775319e-15*z + 5.9952043329775319e-15) + helper_12*(helper_10*x + 27.000000000007574*helper_11 + 1.7035262089848402e-12*helper_2 + 1.723066134218243e-12*helper_4 + helper_8*x + 75.611111111109238*x + 75.611111111109238*y + 16.0) + 81.000000000017849*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 54.000000000015149*helper_9 + 3.4070524179696804e-12*x + 75.611111111109238))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 27.000000000005947*x;
double helper_6 = helper_4*y;
double helper_7 = 216.02777777778243*x;
double helper_8 = x*y;
double helper_9 = 27.000000000007574*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 3.9501735216163963e-13*helper_3;
double helper_14 = helper_0*x;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 27.000000000007766) + 2*helper_0*helper_6*(helper_5*y + 27.000000000007574*x + 27.000000000007766*y + 75.611111111109196) + helper_1*(-helper_0*helper_13 + 113.41666666666558*helper_0*helper_4 - 3.5160763189879334e-13*helper_0*y - 4.4371913550850533e-14*helper_1 - 3.520887285427941e-13*helper_10*y - 5.0727940366829295e-14*helper_10 + 156.0833333333305*helper_11 + 5.1691984026547289e-12*helper_12 - helper_13 + 226.83333333333113*helper_14*y + 285.02777777777311*helper_14 + 1.7035262089848402e-12*helper_2 + 189.02777777777482*helper_4 + 280.83333333334645*helper_6 + 378.05555555554957*helper_8 + 144.94444444444255*x + 3.0457118308888047e-14*y + 3.9810747291356341e-14*z - 3.9810747291356341e-14) + helper_11*y*(helper_7 + 54.000000000015532*helper_8 + helper_9 + 3.4461322684364859e-12*y + 75.611111111109238) + helper_11*(27.000000000007766*helper_12 + 1.723066134218243e-12*helper_3 + 1.7035262089848402e-12*helper_4 + helper_7*y + helper_9*y + 75.611111111109238*x + 75.611111111109238*y + 16.0) + 81.000000000017849*helper_2*helper_3)/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = pow(x, 2);
double helper_5 = pow(x, 3);
double helper_6 = 2.3980817331910175e-14*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = pow(y, 3);
double helper_9 = 1.3167245072054586e-13*helper_8;
double helper_10 = helper_7*x;
double helper_11 = helper_8*x;
double helper_12 = 1.723066134218249e-12*helper_11;
double helper_13 = 9.59232693276407e-14*helper_1;
double helper_14 = pow(z, 5);
double helper_15 = helper_4*y;
double helper_16 = 1.7035262089848394e-12*helper_5;
double helper_17 = helper_5*z;
double helper_18 = helper_0*y;
double helper_19 = helper_7*z;
double helper_20 = 5.2668980288218678e-13*helper_8;
double helper_21 = helper_4*helper_7;
double helper_22 = helper_0*helper_4;
double helper_23 = helper_1*helper_4;
double helper_24 = helper_2*helper_4;
double helper_25 = 27.000000000007574*helper_5*helper_7;
double helper_26 = helper_5*helper_8;
double helper_27 = helper_1*helper_7;
val[2] = (-604.88888888888425*helper_0*helper_10 + helper_0*helper_12 + 216.02777777778238*helper_0*helper_21 + helper_0*helper_25 + 1427.4999999999714*helper_0*helper_3 - 1.4388490399146106e-13*helper_0*helper_5 - 2.4660643897315606e-12*helper_0*helper_7 + 7.9003470432327523e-13*helper_0*helper_8 - 8.3932860661685614e-14*helper_0*x - helper_1*helper_20 - 1981.5555555555175*helper_1*helper_3 - 1.48851301654902e-12*helper_1*y - 113.41666666666555*helper_10*helper_2 - 37.80555555555631*helper_10 - 3.446132268436498e-12*helper_11*z + helper_12 + helper_13*helper_5 + helper_13*x - 312.16666666666094*helper_14*helper_3 - 4.796163466382035e-14*helper_14*helper_4 + 3.5208872854279229e-13*helper_14*helper_7 + 1.1990408665955061e-14*helper_14*x - 6.9723856318165008e-13*helper_14*y + 302.44444444444383*helper_15*z - 37.80555555555631*helper_15 + helper_16*helper_18 + helper_16*y - 3.4070524179696808e-12*helper_17*y + 9.59232693276407e-14*helper_17 - 604.88888888888425*helper_18*helper_4 + 7.4331281870354965e-13*helper_18 + 302.44444444444383*helper_19*x + 1.0572283789163844e-12*helper_19 + 1275.8055555555318*helper_2*helper_3 - helper_2*helper_6 - 1.5846398267645737e-12*helper_2*helper_7 + helper_2*helper_9 - 5.3956838996797413e-14*helper_2*x + 1.4496459588286234e-12*helper_2*y - helper_20*z - 280.83333333334645*helper_21*z + 64.805555555564126*helper_21 + 27.000000000007773*helper_22*helper_8 + 3.7170266864460712e-13*helper_22 + 453.66666666666242*helper_23*y - 4.07673894642473e-13*helper_23 - 113.41666666666561*helper_24*y + 2.2182256032016868e-13*helper_24 - helper_25 + 54.000000000011902*helper_26*z + 27.00000000000594*helper_26 + 453.66666666666219*helper_27*x + 2.8176720216303538e-12*helper_27 - 452.72222222221251*helper_3*z + 43.138888888887891*helper_3 - 27.000000000007766*helper_4*helper_8 - 1.6786572132337123e-13*helper_4*z + 2.9976021664887721e-14*helper_4 - helper_6 - 1.7628491259339769e-13*helper_7 + helper_9 + 3.5971225997865264e-14*x*z - 5.9952043329775319e-15*x + 1.3311574065255142e-13*y*pow(z, 6) - 1.3217205108159103e-13*y*z - 8.1508873724633332e-15*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 249.00000000003479*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*x;
double helper_8 = 280.47222222225309*y;
double helper_9 = x*y;
double helper_10 = 474.00000000003564*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*x;
double helper_13 = helper_2*y;
double helper_14 = helper_11*y;
double helper_15 = 7.0610184366169377e-13*helper_2;
double helper_16 = helper_0*x;
val[0] = (2*helper_0*helper_7*(helper_4*x + 273.00000000003519*x + 474.00000000003564*y - 96.611111111111455) + helper_1*(helper_0*helper_15 - 100.1944444444473*helper_0*y - 5.8841820305141147e-14*helper_11 + 4.7073456244112918e-13*helper_12 + 200.00000000000256*helper_13 - 35.583333333336626*helper_14 + helper_15 + 62.166666666659836*helper_16*y + 3.5305092183084663e-13*helper_16 + 267.66666666666822*helper_3 - 24.52777777777996*helper_5 + 104.08333333333157*helper_6 + 754.16666666672972*helper_7 - 67.055555555562137*helper_9 - 1.1768364061028229e-13*x - 59.277777777777423*y - 5.8841820305141147e-14*z + 5.8841820305141147e-14) + helper_12*y*(helper_10 + helper_8 + 546.00000000007037*helper_9 + 133.33333333333505*x - 64.611111111111015) + helper_14*(helper_10*x + 273.00000000003519*helper_13 + 66.666666666667524*helper_2 + 267.66666666666822*helper_5 + helper_8*x - 64.611111111111015*x - 128.61111111111097*y + 5.3333333333333339) + 747.00000000010436*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 273.00000000003519))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 249.00000000003479*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 280.47222222225309*x;
double helper_10 = x*y;
double helper_11 = 273.00000000003519*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 127.99999999999959*helper_3;
double helper_15 = helper_0*x;
val[1] = (-helper_1*(-helper_0*helper_14 + 4.5704181180393186e-14*helper_1 + 49.05555555555992*helper_10 - 803.00000000000466*helper_12 + 35.583333333336626*helper_13 - helper_14 - 208.16666666666313*helper_15*y + 100.1944444444473*helper_15 - 66.666666666667524*helper_2 - 754.16666666672972*helper_5*y + 33.527777777781068*helper_5 - 31.083333333329918*helper_6 + 64.00000000000081*helper_7 + 3.5771385853423296e-13*helper_8*y + 1.4528193463072914e-13*helper_8 + 59.277777777777423*x + 63.999999999999986*y - 5.3333333333334547*z + 1.2079226507921703e-13) + helper_10*helper_8*(948.00000000007128*helper_10 + helper_11 + helper_9 + 535.33333333333644*y - 128.61111111111097) + helper_13*(helper_11*y + 474.00000000003564*helper_12 + 267.66666666666822*helper_3 + 66.666666666667524*helper_5 + helper_9*y - 64.611111111111015*x - 128.61111111111097*y + 5.3333333333333339) + 747.00000000010436*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 474.00000000003564) + 2*helper_5*helper_7*(helper_4*y + 273.00000000003519*x + 474.00000000003564*y - 96.611111111111455))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 2.3536728122056373e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 42.666666666666529*helper_10;
double helper_12 = helper_9*x;
double helper_13 = helper_10*x;
double helper_14 = 9.4146912488225816e-13*helper_1;
double helper_15 = pow(z, 5);
double helper_16 = helper_7*y;
double helper_17 = helper_1*y;
double helper_18 = helper_6*helper_9;
double helper_19 = helper_0*helper_6;
double helper_20 = helper_2*helper_6;
double helper_21 = helper_10*helper_7;
double helper_22 = helper_0*helper_9;
double helper_23 = helper_2*helper_9;
val[2] = -(-255.9999999999992*helper_0*helper_10 + 267.66666666666799*helper_0*helper_13 + 66.666666666667425*helper_0*helper_16 - 105.16666666671541*helper_0*helper_3 - 1.4122036873233877e-12*helper_0*helper_7 - 8.2378548427197621e-13*helper_0*x - 32.000000000001563*helper_0*y + 170.66666666666617*helper_1*helper_10 + 416.33333333332621*helper_1*helper_12 + 310.88888888894337*helper_1*helper_3 - 4.0012437807496005e-12*helper_1*helper_6 - 127.99999999999807*helper_1*helper_9 + 474.00000000003558*helper_10*helper_19 - 474.0000000000357*helper_10*helper_6 + 170.66666666666612*helper_10*z - helper_11*helper_2 - helper_11 - 232.69444444444252*helper_12 - 535.33333333333644*helper_13*z + 267.66666666666822*helper_13 + helper_14*helper_7 + helper_14*x + 71.166666666673237*helper_15*helper_3 - 4.7073456244112999e-13*helper_15*helper_6 + 3.5771385853423321e-13*helper_15*helper_9 + 1.1768364061028277e-13*helper_15*x - 5.3211139198561884e-13*helper_15*y + 66.66666666666751*helper_16 + 124.33333333331973*helper_17*helper_6 + 21.333333333333975*helper_17 - 754.16666666672927*helper_18*z + 473.69444444447595*helper_18 - 251.11111111109051*helper_19*y + 3.6481928589187515e-12*helper_19 - 255.63888888891896*helper_2*helper_3 - helper_2*helper_8 - 5.2957638274626995e-13*helper_2*x - 5.333333333332849*helper_2*y - 31.083333333329918*helper_20*y + 2.1771473512902224e-12*helper_20 + 498.00000000006958*helper_21*z + 249.00000000003479*helper_21 + 280.47222222225309*helper_22*helper_6 + 273.00000000003513*helper_22*helper_7 - 753.11111111110029*helper_22*x + 191.99999999999886*helper_22 - 104.08333333333155*helper_23*x + 31.999999999998607*helper_23 + 34.361111111107377*helper_3 + 673.55555555554815*helper_4*helper_9 + 3.5305092183084658e-13*helper_4 + 253.55555555554167*helper_5*helper_6 - 133.33333333333496*helper_5*helper_7 - 55.611111111089599*helper_5*x + 21.333333333334451*helper_5 - 95.694444444440904*helper_6*y - 1.6475709685439524e-12*helper_6*z + 2.9420910152570612e-13*helper_6 - 273.00000000003524*helper_7*helper_9 + 9.4146912488225816e-13*helper_7*z - helper_8 - 127.99999999999986*helper_9*z + 32.00000000000005*helper_9 - 5.8841820305141968e-14*x + 1.3711254354117922e-13*y*pow(z, 6) - 5.3333333333336101*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 101.25000000000739*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 143.68749999999312*y;
double helper_8 = x*y;
double helper_9 = 166.5000000000062*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 3.657074643115402e-13*helper_2;
val[0] = -(-helper_1*(-helper_0*helper_13 + 186.62500000000145*helper_0*helper_8 - 1.8285373215576995e-13*helper_0*x + 218.18749999999585*helper_0*y - 2.4380497620769346e-13*helper_10*x + 3.0475622025961683e-14*helper_10 + 129.31249999999818*helper_11 - 107.99999999999352*helper_12 - helper_13 - 65.249999999999076*helper_3 - 18.375000000010232*helper_5*x + 392.93749999999727*helper_5 + 192.06249999999912*helper_6 + 396.37499999999704*helper_8 + 6.0951244051923366e-14*x + 76.87499999999774*y + 3.0475622025961683e-14*z - 3.0475622025961683e-14) + helper_11*x*(-helper_7 + 274.50000000001*helper_8 + helper_9 + 71.99999999999568*x - 104.87499999999785) + helper_11*(137.250000000005*helper_12 + 35.99999999999784*helper_2 + 65.249999999999076*helper_5 - helper_7*x + helper_9*x - 104.87499999999785*x - 200.87499999999793*y + 12.0) + 303.75000000002217*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 137.250000000005) + 2*helper_6*x*(helper_4*x + 137.250000000005*x + 166.5000000000062*y - 152.87499999999807))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 101.25000000000739*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*y;
double helper_8 = 143.68749999999312*x;
double helper_9 = x*y;
double helper_10 = 137.250000000005*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*x;
double helper_13 = helper_3*x;
double helper_14 = 2.9809488211182954e-14*helper_3;
val[1] = -(-helper_1*(-helper_0*helper_14 + 218.18749999999585*helper_0*x - 4.6934678366030247e-14*helper_1 + 127.99999999999989*helper_11*y - 16.000000000000043*helper_11 + 129.31249999999818*helper_12 - 195.74999999999721*helper_13 - helper_14 - 35.99999999999784*helper_2 - 18.375000000010232*helper_5*y + 198.18749999999852*helper_5 + 93.312500000000725*helper_6 + 384.12499999999824*helper_7*x + 224.0*helper_7 + 785.87499999999454*helper_9 + 76.87499999999774*x + 95.999999999999943*y - 28.000000000000057*z + 16.000000000000057) + helper_12*y*(helper_10 - helper_8 + 333.00000000001239*helper_9 + 130.49999999999815*y - 200.87499999999793) + helper_12*(helper_10*y + 166.5000000000062*helper_13 + 65.249999999999076*helper_3 + 35.99999999999784*helper_5 - helper_8*y - 104.87499999999785*x - 200.87499999999793*y + 12.0) + 303.75000000002217*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 166.5000000000062) + 2*helper_5*helper_7*(helper_4*y + 137.250000000005*x + 166.5000000000062*y - 152.87499999999807))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 1.2190248810384701e-13*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 9.936496070394318e-15*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_9*x;
double helper_13 = 4.8760995241538804e-13*helper_1;
double helper_14 = pow(z, 5);
double helper_15 = helper_5*y;
double helper_16 = helper_5*z;
double helper_17 = 35.99999999999784*y;
double helper_18 = 3.9745984281577089e-14*helper_9;
double helper_19 = 166.5000000000062*helper_5*helper_9;
double helper_20 = 137.250000000005*helper_6*helper_8;
double helper_21 = helper_6*helper_9;
double helper_22 = helper_0*helper_6;
double helper_23 = helper_0*helper_8;
double helper_24 = helper_0*helper_9;
val[2] = -(-951.4999999999967*helper_0*helper_11 - 455.00000000000642*helper_0*helper_15 - helper_0*helper_19 - helper_0*helper_20 + 1265.1249999999886*helper_0*helper_3 - 1.8894885656096246e-12*helper_0*helper_5 + 4.2665870836346316e-13*helper_0*x - 151.99999999999841*helper_0*y + 768.24999999999636*helper_1*helper_11 + 373.25000000000284*helper_1*helper_15 - helper_1*helper_18 - 1713.4999999999804*helper_1*helper_3 + 2.0723422977653952e-12*helper_1*helper_5 - 831.99999999999875*helper_1*helper_8 + 207.99999999999778*helper_1*y + helper_10*helper_2 + helper_10 - 192.06249999999909*helper_11*helper_2 + 366.50000000000068*helper_11*z + 8.8124999999987494*helper_11 + 130.49999999999818*helper_12*z - 65.249999999999062*helper_12 - helper_13*helper_6 - helper_13*x - 258.62499999999636*helper_14*helper_3 + 2.4380497620769402e-13*helper_14*helper_5 - 127.99999999999989*helper_14*helper_8 - 6.0951244051923505e-14*helper_14*x + 31.999999999999233*helper_14*y - 93.312500000000739*helper_15*helper_2 + 163.50000000000716*helper_15*z + 11.562499999997101*helper_15 + 18.375000000009891*helper_16*helper_8 + 8.5331741672692632e-13*helper_16 - helper_17*helper_22 - helper_17*helper_6 - helper_18*z + helper_19 + 1074.9374999999857*helper_2*helper_3 - 1.1275980149605826e-12*helper_2*helper_5 + helper_2*helper_7 + 527.99999999999943*helper_2*helper_8 + 2.7428059823365492e-13*helper_2*x - 131.99999999999824*helper_2*y + helper_20 - 202.50000000001478*helper_21*z - 101.25000000000739*helper_21 + 7.3141492862308201e-13*helper_22 + 143.68749999999312*helper_23*helper_5 + 607.99999999999875*helper_23 - 65.249999999999091*helper_24*x + 5.9618976422365643e-14*helper_24 - 396.37499999999841*helper_3*z + 28.437500000000512*helper_3 + 71.99999999999568*helper_4*helper_6 + 47.999999999999346*helper_4 - 162.06250000000301*helper_5*helper_8 - 1.5237811012980877e-13*helper_5 - 4.8760995241538804e-13*helper_6*z + helper_7 - 191.99999999999943*helper_8*z + 15.999999999999886*helper_8 - 1.828537321557705e-13*x*z + 3.0475622025961752e-14*x + 1.4080403509808982e-13*y*pow(z, 6) - 3.9999999999998863*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 26.999999999999247*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 135.02777777777823*y;
double helper_8 = x*y;
double helper_9 = 53.99999999999811*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 7.1942451995703745e-14*helper_2;
double helper_14 = helper_0*y;
val[0] = (-helper_1*(-helper_0*helper_13 - 3.5971225997851841e-14*helper_0*x - 4.7961634663802496e-14*helper_10*x + 5.995204332975312e-15*helper_10 + 70.749999999998252*helper_11 + 4.929390229335695e-12*helper_12 - helper_13 + 226.83333333333346*helper_14*x + 93.027777777774347*helper_14 - 26.999999999999019*helper_3 + 118.83333333333734*helper_5*x + 162.0277777777755*helper_5 + 86.416666666665904*helper_6 + 378.05555555555242*helper_8 + 1.1990408665950624e-14*x + 6.2777777777761239*y + 5.995204332975312e-15*z - 5.995204332975312e-15) - helper_11*x*(helper_7 - 53.999999999995154*helper_8 - helper_9 + 3.2862601528904634e-12*x + 75.611111111109494) + helper_11*(26.999999999997577*helper_12 - 1.6431300764452317e-12*helper_2 + 26.999999999999019*helper_5 - helper_7*x + helper_9*x - 75.611111111109494*x - 75.611111111109494*y + 16.0) + 80.99999999999774*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 26.999999999997577) + 2*helper_6*x*(helper_4*x + 26.999999999997577*x + 53.99999999999811*y - 75.611111111109537))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 26.999999999999247*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = 135.02777777777823*x;
double helper_9 = x*y;
double helper_10 = 26.999999999997577*helper_5;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 2.0938806244430957e-13*helper_3;
val[1] = (-helper_1*(-helper_0*helper_13 + 172.83333333333181*helper_0*helper_9 + 93.027777777774347*helper_0*x - 1.8685053504441748e-13*helper_0*y - 42.666666666666707*helper_1 - 80.999999999997058*helper_11 + 70.749999999998252*helper_12 - helper_13 + 1.6431300764452317e-12*helper_2 + 118.83333333333734*helper_5*y + 189.02777777777621*helper_5 + 113.41666666666673*helper_6 - 2.2833586873124354e-13*helper_7*y - 96.000000000000043*helper_7 + 324.055555555551*helper_9 + 6.2777777777761239*x - 6.4763009769789386e-15*y - 69.333333333333329*z + 53.333333333333329) + helper_12*(helper_10*y + 53.99999999999811*helper_11 + 26.999999999999019*helper_3 - 1.6431300764452317e-12*helper_5 - helper_8*y - 75.611111111109494*x - 75.611111111109494*y + 16.0) + 80.99999999999774*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 53.99999999999811) + 2*helper_6*y*(helper_4*y + 26.999999999997577*x + 53.99999999999811*y - 75.611111111109537) + helper_7*helper_9*(helper_10 - helper_8 + 107.99999999999622*helper_9 + 53.999999999998039*y - 75.611111111109494))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 2.3980817331900974e-14*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 6.9796020814769167e-14*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_9*x;
double helper_13 = helper_0*x;
double helper_14 = 9.5923269327604665e-14*helper_1;
double helper_15 = pow(z, 5);
double helper_16 = helper_15*x;
double helper_17 = helper_5*y;
double helper_18 = 1.6431300764452311e-12*y;
double helper_19 = helper_1*y;
double helper_20 = 2.7918408325907884e-13*helper_9;
double helper_21 = helper_5*helper_8;
double helper_22 = helper_0*helper_5;
double helper_23 = helper_6*helper_9;
double helper_24 = helper_0*helper_6;
double helper_25 = helper_0*helper_8;
val[2] = -(442.88888888888607*helper_0*helper_11 + 26.999999999998991*helper_0*helper_12 + 604.88888888889073*helper_0*helper_17 - 135.02777777777823*helper_0*helper_21 - 4.1877612488861869e-13*helper_0*helper_9 - 416.0000000000008*helper_0*y - 345.66666666666356*helper_1*helper_11 - 453.66666666666697*helper_1*helper_17 + helper_1*helper_20 - 4.0767389464232083e-13*helper_1*helper_5 - 1.9096576172236008e-12*helper_1*helper_8 - helper_10*helper_2 - helper_10 + 86.41666666666589*helper_11*helper_2 - 194.44444444444468*helper_11*z + 10.805555555556396*helper_11 - 53.999999999998039*helper_12*z + 26.999999999999019*helper_12 - 840.83333333331927*helper_13*y - 8.3932860661653908e-14*helper_13 + helper_14*helper_6 + helper_14*x - 4.7961634663801947e-14*helper_15*helper_5 - 2.2833586873124314e-13*helper_15*helper_8 + 576.00000000000057*helper_15*y + 141.49999999999648*helper_16*y + 1.1990408665950627e-14*helper_16 + 113.41666666666674*helper_17*helper_2 - 302.44444444444787*helper_17*z + 37.805555555557248*helper_17 - helper_18*helper_24 - helper_18*helper_6 + 1042.8888888888678*helper_19*x + 917.33333333333462*helper_19 - 614.4722222222083*helper_2*helper_3 + 2.2182256032008666e-13*helper_2*helper_5 - helper_2*helper_7 + 1.0482540761340087e-12*helper_2*helper_8 - 5.3956838996778487e-14*helper_2*x - 1029.3333333333346*helper_2*y + helper_20*z + 118.83333333333741*helper_21*z + 16.194444444440848*helper_21 + 53.999999999998096*helper_22*helper_9 + 3.7170266864446899e-13*helper_22 + 53.999999999998494*helper_23*z + 26.999999999999247*helper_23 - 1.4388490399140585e-13*helper_24 + 26.99999999999757*helper_25*helper_6 + 1.7228070821791827e-12*helper_25 - 32.472222222222157*helper_3 + 3.2862601528904634e-12*helper_4*helper_6 + 303.38888888888516*helper_4*x + 85.333333333333513*helper_4 - 53.99999999999811*helper_5*helper_9 - 1.6786572132330923e-13*helper_5*z + 2.9976021664876936e-14*helper_5 - 26.999999999997577*helper_6*helper_8 + 9.5923269327604665e-14*helper_6*z - helper_7 - 7.6797827356738249e-13*helper_8*z + 1.349106012090346e-13*helper_8 + 3.5971225997852327e-14*x*z - 5.995204332975312e-15*x - 128.00000000000011*y*pow(z, 6) - 5.3333333333333428*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 576.00000000015746*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 2038.6666666668616*y;
double helper_8 = x*y;
double helper_9 = 936.00000000018736*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 384.00000000000341*helper_2;
double helper_15 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_14 + 576.00000000000159*helper_0*x + 128.0*helper_1 + 159.99999999999977*helper_10 + 512.00000000000227*helper_11 + 2112.000000000101*helper_12 + 587.99999999998806*helper_13 + helper_14 + 2584.0000000000409*helper_15*x + 686.66666666664742*helper_15 + 360.00000000002984*helper_3 + 3944.0000000004093*helper_5*x + 886.6666666666747*helper_5 + 820.00000000001774*helper_6 + 2781.3333333333594*helper_8 + 63.999999999999432*x + 98.66666666665931*y + 31.999999999999716*z - 31.999999999999716) + helper_11*y*(helper_7 + 2304.0000000003779*helper_8 + helper_9 + 1408.0000000000673*x + 98.666666666659026) + helper_13*(1152.0000000001889*helper_12 + 704.00000000003365*helper_2 + 360.00000000002984*helper_5 + helper_7*x + helper_9*x + 98.666666666659026*x + 66.666666666658912*y) + 1728.0000000004725*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 1152.0000000001889) + 2*helper_6*x*(helper_4*x + 1152.0000000001889*x + 936.00000000018736*y + 66.666666666657719))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 576.00000000015746*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 2038.6666666668616*x;
double helper_10 = x*y;
double helper_11 = 1152.0000000001889*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 5.4249937875287058e-12*helper_3;
double helper_15 = helper_0*x;
val[1] = (2*helper_0*helper_7*(helper_4*y + 1152.0000000001889*x + 936.00000000018736*y + 66.666666666657719) + helper_1*(-helper_0*helper_14 - 4.645173135031907e-12*helper_0*y - 3.7125857943462877e-13*helper_1 + 1773.3333333333494*helper_10 + 1080.0000000000896*helper_12 + 587.99999999998806*helper_13 - helper_14 + 1640.0000000000355*helper_15*y + 686.66666666664742*helper_15 + 704.00000000003365*helper_2 + 1390.6666666666797*helper_5 + 1292.0000000000205*helper_6 + 3944.0000000004093*helper_7 - 4.3733905386035968e-12*helper_8*y - 3.6637359812627243e-13*helper_8 + 98.66666666665931*x + 6.856737400086229e-13*y + 1.1346479311670675e-12*z - 1.1346479311670675e-12) + helper_10*helper_8*(1872.0000000003747*helper_10 + helper_11 + helper_9 + 720.00000000005969*y + 66.666666666658912) + helper_13*(helper_11*y + 936.00000000018736*helper_12 + 360.00000000002984*helper_3 + 704.00000000003365*helper_5 + helper_9*y + 98.666666666659026*x + 66.666666666658912*y) + 1728.0000000004725*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 936.00000000018736))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 128.00000000000114*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 1.8083312625095683e-12*helper_10;
double helper_12 = helper_10*x;
double helper_13 = helper_0*x;
double helper_14 = helper_1*x;
double helper_15 = pow(z, 5);
double helper_16 = pow(z, 6);
double helper_17 = helper_7*y;
double helper_18 = 704.00000000003365*helper_17;
double helper_19 = 512.00000000000455*helper_7;
double helper_20 = helper_0*y;
double helper_21 = helper_9*z;
double helper_22 = helper_6*helper_9;
double helper_23 = 936.00000000018736*helper_6;
double helper_24 = helper_1*helper_6;
double helper_25 = helper_2*helper_6;
double helper_26 = helper_7*helper_9;
double helper_27 = helper_10*helper_7;
double helper_28 = helper_1*helper_9;
double helper_29 = helper_2*helper_9;
double helper_30 = helper_0*helper_10;
val[2] = -(360.00000000003001*helper_0*helper_12 + helper_0*helper_18 + 2038.6666666668634*helper_0*helper_22 + 1152.0000000001887*helper_0*helper_26 + 7639.9999999998772*helper_0*helper_3 + 3392.0000000000177*helper_0*helper_6 - 768.00000000000682*helper_0*helper_7 - 2.9798385980940242e-11*helper_0*helper_9 - 7.2333250500382731e-12*helper_1*helper_10 + helper_1*helper_19 - 1.040945107888401e-11*helper_1*y - helper_10*helper_23 - 720.00000000006003*helper_10*helper_4 - 7.2333250500382747e-12*helper_10*z + helper_11*helper_2 + helper_11 + 360.0000000000299*helper_12 - 4853.3333333334467*helper_13*helper_9 - 2752.0000000000018*helper_13 - 9013.333333333172*helper_14*y + 4608.0000000000036*helper_14 - 1175.9999999999759*helper_15*helper_3 - 512.00000000000227*helper_15*helper_6 + 4.3733905386035968e-12*helper_15*helper_9 + 1984.0000000000002*helper_15*x - 5.9499072335707746e-12*helper_15*y - 384.0*helper_16*x + 1.1137757383038866e-12*helper_16*y - 1408.0000000000673*helper_17*z + helper_18 + helper_19*z + 5193.3333333332339*helper_2*helper_3 - helper_2*helper_8 - 4192.0000000000018*helper_2*x + 1.1908252162128503e-11*helper_2*y - 7653.333333333464*helper_20*helper_6 + 2.5712765250304412e-12*helper_20 - 3944.0000000004088*helper_21*helper_6 + 1.2576606422954171e-11*helper_21 + 1905.3333333335463*helper_22 + helper_23*helper_30 + 5168.0000000000855*helper_24*y - 3968.00000000002*helper_24 - 1292.0000000000205*helper_25*y + 2272.0000000000109*helper_25 - 1152.0000000001892*helper_26 + 1152.0000000003151*helper_27*z + 576.00000000015734*helper_27 + 3280.0000000000709*helper_28*x + 3.4443559115972149e-11*helper_28 - 820.00000000001774*helper_29*x - 1.9544366125502031e-11*helper_29 + 489.33333333332871*helper_3 + 1.0849987575057412e-11*helper_30 + 3146.6666666667529*helper_4*helper_9 - 3133.3333333332912*helper_4*y + 832.00000000000091*helper_4 + 4970.6666666667634*helper_5*helper_6 + 1.519673276107678e-12*helper_5 - 1193.3333333333615*helper_6*y - 1408.000000000008*helper_6*z + 224.00000000000148*helper_6 - helper_8 - 753.33333333335872*helper_9*x - 2.0508039710876433e-12*helper_9 - 96.000000000000171*x - 7.536193891157259e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_30(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 3.1559041675535225e-12*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*x;
double helper_8 = 329.33333333332092*y;
double helper_9 = x*y;
double helper_10 = 72.000000000002501*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*x;
double helper_13 = helper_2*y;
double helper_14 = helper_11*y;
double helper_15 = 8.9434650772482409e-27*helper_2;
val[0] = (-helper_1*(-helper_0*helper_15 + 625.33333333332325*helper_0*y + 128.0*helper_1 + 224.00000000000006*helper_11 + 256.0*helper_12 - 4.6495451802968516e-13*helper_13 + 339.99999999999335*helper_14 - helper_15 - 71.999999999999687*helper_3 + 279.99999999998175*helper_5*x + 329.33333333332354*helper_5 + 139.99999999999358*helper_6 + 679.99999999998693*helper_7*y + 448.0*helper_7 + 1250.6666666666472*helper_9 + 192.0*x + 285.33333333332985*y + 96.0*z - 96.0) + helper_12*y*(helper_10 - helper_8 + 6.712964331549363e-12*helper_9 + 3.0996967868645677e-13*x - 285.33333333333002) + helper_14*(helper_10*x + 3.3564821657746815e-12*helper_13 + 1.5498483934322839e-13*helper_2 + 71.999999999999687*helper_5 - helper_8*x - 285.33333333333002*x - 189.33333333332993*y) + 9.4677125026605667e-12*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 3.3564821657746815e-12) + 2*helper_5*helper_7*(helper_4*x + 3.3564821657746815e-12*x + 72.000000000002501*y - 189.33333333333002))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 3.1559041675535225e-12*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 329.33333333332092*x;
double helper_10 = x*y;
double helper_11 = 3.3564821657746815e-12*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 5.6488147492925764e-13*helper_3;
double helper_15 = helper_0*x;
val[1] = (2*helper_0*helper_7*(helper_4*y + 3.3564821657746815e-12*x + 72.000000000002501*y - 189.33333333333002) - helper_1*(-helper_0*helper_14 - 4.956035581926515e-13*helper_0*y - 9.0594198809410931e-14*helper_1 + 658.66666666664707*helper_10 - 215.99999999999906*helper_12 + 339.99999999999335*helper_13 - helper_14 + 279.99999999998715*helper_15*y + 625.33333333332325*helper_15 - 1.5498483934322839e-13*helper_2 + 625.33333333332359*helper_5 + 339.99999999999346*helper_6 + 279.99999999998175*helper_7 - 5.7198690228686227e-13*helper_8*y - 8.8373752760160529e-14*helper_8 + 285.33333333332985*x + 2.8421709430400795e-14*y + 1.2878587085647919e-14*z - 1.2878587085647919e-14) + helper_10*helper_8*(144.000000000005*helper_10 + helper_11 - helper_9 + 143.99999999999937*y - 189.33333333332993) + helper_13*(helper_11*y + 72.000000000002501*helper_12 + 71.999999999999687*helper_3 + 1.5498483934322839e-13*helper_5 - helper_9*y - 285.33333333333002*x - 189.33333333332993*y) + 9.4677125026605667e-12*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 72.000000000002501))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = 54.666666666663446*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 2.9811550257494136e-27*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 1.8829382497641905e-13*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_9*x;
double helper_13 = helper_0*x;
double helper_14 = helper_2*x;
double helper_15 = pow(z, 5);
double helper_16 = pow(z, 6);
double helper_17 = helper_5*z;
double helper_18 = helper_6*y;
double helper_19 = 1.5498483934322793e-13*helper_18;
double helper_20 = 1.1924620102997654e-26*helper_6;
double helper_21 = helper_1*y;
double helper_22 = helper_15*y;
double helper_23 = 7.5317529990567621e-13*helper_9;
double helper_24 = helper_5*helper_8;
double helper_25 = 72.000000000002501*helper_9;
double helper_26 = helper_0*helper_5;
double helper_27 = helper_1*helper_5;
double helper_28 = helper_2*helper_5;
double helper_29 = 3.3564821657746811e-12*helper_6;
double helper_30 = helper_6*helper_9;
double helper_31 = helper_0*helper_8;
double helper_32 = helper_2*helper_8;
val[2] = -(helper_0*helper_19 - 329.33333333332098*helper_0*helper_24 - 1.7886930154496482e-26*helper_0*helper_6 - 1.1297629498585163e-12*helper_0*helper_9 - 2.2319923687063937e-12*helper_0*y - 559.99999999997431*helper_1*helper_11 + helper_1*helper_20 + helper_1*helper_23 - 4.7286619064833189e-12*helper_1*helper_8 - 3583.9999999999986*helper_1*x - helper_10*helper_2 - helper_10 - 181.33333333331439*helper_11*z - 49.333333333336327*helper_11 - 143.99999999999937*helper_12*z + 71.999999999999687*helper_12 + 650.6666666666315*helper_13*helper_8 + 71.999999999999744*helper_13*helper_9 - 3047.9999999999281*helper_13*y + 1855.9999999999991*helper_13 - 2774.6666666666106*helper_14*y + 3615.9999999999991*helper_14 + 256.0*helper_15*helper_5 - 5.7198690228686258e-13*helper_15*helper_8 - 1856.0000000000002*helper_15*x + 384.0*helper_16*x - 2.7178259642823257e-13*helper_16*y + 279.99999999998181*helper_17*helper_8 - 789.33333333331393*helper_17*y + 384.0*helper_17 - 3.0996967868645718e-13*helper_18*z + helper_19 - helper_2*helper_7 - 3.180122831736239e-12*helper_2*y + helper_20*z + 4298.6666666665733*helper_21*x + 3.616662525018853e-12*helper_21 + 679.99999999998681*helper_22*x + 1.4539480730490745e-12*helper_22 + helper_23*z + 49.333333333339169*helper_24 + helper_25*helper_26 - helper_25*helper_5 + 1754.666666666631*helper_26*y - 1216.0*helper_26 - 1359.9999999999736*helper_27*y + 1664.0*helper_27 + 339.99999999999341*helper_28*y - 1056.0*helper_28 + helper_29*helper_31 - helper_29*helper_8 + helper_3*helper_5 - helper_3*x + 6.3118083351070442e-12*helper_30*z + 3.1559041675535237e-12*helper_30 + 4.2330583482906692e-12*helper_31 + 139.99999999999358*helper_32*x + 2.6121327323379843e-12*helper_32 + 898.66666666664059*helper_4*x + 6.9544370262519947e-13*helper_4 - 32.0*helper_5 - helper_7 - 1.8687273950490082e-12*helper_8*z + 3.241851231905339e-13*helper_8 - 447.99999999999943*x*z + 31.999999999999886*x - 8.2156503822264676e-14*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_31(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 576.00000000012835*y;
double helper_6 = helper_4*x;
double helper_7 = 1417.3333333334929*y;
double helper_8 = x*y;
double helper_9 = 792.00000000015007*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 384.00000000000341*helper_2;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 1152.0000000001492) + 2*helper_0*helper_6*(helper_5*x + 1152.0000000001492*x + 792.00000000015007*y - 66.666666666669911) + helper_1*(-32.000000000000284*helper_10 + 256.00000000000227*helper_11 + 2112.0000000000691*helper_12 + 115.99999999999504*helper_13 - helper_14*helper_16 + helper_14 + 1640.0000000000341*helper_15*x + 17.333333333326934*helper_15 - 332.00000000001654*helper_16*helper_4 - 192.00000000000153*helper_16*x + 216.00000000002171*helper_3 + 265.33333333334559*helper_4 + 2968.0000000003274*helper_6 + 1442.6666666666974*helper_8 - 64.000000000000568*x - 98.666666666668206*y - 32.000000000000284*z + 32.000000000000284) + helper_11*y*(helper_7 + 2304.0000000002983*helper_8 + helper_9 + 1408.0000000000462*x - 98.666666666668618) + helper_13*(1152.0000000001492*helper_12 + 704.00000000002308*helper_2 + 216.00000000002171*helper_4 + helper_7*x + helper_9*x - 98.666666666668618*x - 66.666666666668846*y) + 1728.0000000003852*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 576.00000000012835*x;
double helper_6 = helper_4*y;
double helper_7 = pow(helper_0, 2);
double helper_8 = 1417.3333333334929*x;
double helper_9 = x*y;
double helper_10 = 1152.0000000001492*helper_4;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 4.5083936583981039e-12*helper_3;
double helper_14 = helper_0*x;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 792.00000000015007) + 2*helper_0*helper_6*(helper_5*y + 1152.0000000001492*x + 792.00000000015007*y - 66.666666666669911) + helper_1*(-helper_0*helper_13 + 820.00000000001705*helper_0*helper_4 - 3.2702729413360602e-12*helper_0*y - 1.2612133559739895e-13*helper_1 + 648.00000000006514*helper_11 + 115.99999999999504*helper_12 - helper_13 + 664.00000000003308*helper_14*y + 17.333333333326934*helper_14 + 704.00000000002308*helper_2 + 721.33333333334872*helper_4 + 2968.0000000003274*helper_6 - 3.2578384434602007e-12*helper_7*y + 1.1102230246274133e-14*helper_7 + 530.66666666669119*helper_9 - 98.666666666668206*x + 8.4909856923342049e-13*y + 1.2030376694839438e-12*z - 1.2030376694839438e-12) + helper_12*(helper_10*y + 792.00000000015007*helper_11 + 216.00000000002171*helper_3 + 704.00000000002308*helper_4 + helper_8*y - 98.666666666668618*x - 66.666666666668846*y) + 1728.0000000003852*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 1584.0000000003001*helper_9 + 432.00000000004343*y - 66.666666666668846))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 128.00000000000114*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = helper_2*x;
double helper_12 = pow(z, 5);
double helper_13 = helper_12*x;
double helper_14 = helper_6*y;
double helper_15 = helper_7*y;
double helper_16 = 512.00000000000455*helper_7;
double helper_17 = helper_9*z;
double helper_18 = helper_6*helper_9;
double helper_19 = 792.00000000015007*helper_6;
double helper_20 = helper_0*helper_6;
double helper_21 = helper_2*helper_6;
double helper_22 = 1152.0000000001492*helper_7*helper_9;
double helper_23 = helper_10*helper_7;
double helper_24 = helper_0*helper_9;
double helper_25 = helper_1*helper_9;
double helper_26 = helper_0*helper_10;
val[2] = (704.00000000002274*helper_0*helper_15 + 1417.3333333334922*helper_0*helper_18 + helper_0*helper_22 + 2215.9999999999404*helper_0*helper_3 - 768.00000000000682*helper_0*helper_7 - 448.00000000000398*helper_0*x - 1.3207213100952268e-12*helper_0*y - 6.011191544530807e-12*helper_1*helper_10 + 3280.0000000000682*helper_1*helper_14 + helper_1*helper_16 - 2250.6666666665933*helper_1*helper_3 - 2176.0000000000196*helper_1*helper_6 + 512.00000000000443*helper_1*x - 2.9771740628336457e-12*helper_1*y - helper_10*helper_19 + 1.5027978861327005e-12*helper_10*helper_2 - 432.00000000004343*helper_10*helper_4 + 216.00000000002171*helper_10*x - 6.0111915445308054e-12*helper_10*z + 1.5027978861327018e-12*helper_10 - 332.0000000000166*helper_11*helper_9 - 288.0000000000025*helper_11 - 256.00000000000227*helper_12*helper_6 + 3.2578384434602003e-12*helper_12*helper_9 - 2.2923885012457284e-12*helper_12*y - 231.99999999999011*helper_13*y + 64.000000000000568*helper_13 - 918.66666666668561*helper_14 + 704.00000000002296*helper_15 + helper_16*z - 2968.0000000003242*helper_17*helper_6 + 9.7486463346288815e-12*helper_17 + 1550.6666666668327*helper_18 + helper_19*helper_26 + 1142.6666666666229*helper_2*helper_3 - helper_2*helper_8 - 1.465405574663298e-11*helper_2*helper_9 + 4.5834447348617507e-12*helper_2*y - 5018.6666666667679*helper_20*y + 1984.000000000018*helper_20 - 820.00000000001728*helper_21*y + 1184.0000000000107*helper_21 - helper_22 + 1152.0000000002569*helper_23*z + 576.00000000012824*helper_23 - 2058.6666666667675*helper_24*x - 2.2767565610593824e-11*helper_24 + 1328.0000000000659*helper_25*x + 2.6037838551929886e-11*helper_25 + 216.00000000002149*helper_26*x + 9.0167873167962077e-12*helper_26 + 214.66666666666316*helper_3 + 1461.333333333404*helper_4*helper_9 - 1090.6666666666429*helper_4*y + 192.00000000000171*helper_4 + 3477.3333333334049*helper_5*helper_6 - 1408.0000000000455*helper_5*helper_7 + 2.4309443347198526e-12*helper_5 - 896.00000000000819*helper_6*z + 160.00000000000151*helper_6 - helper_8 - 398.66666666668544*helper_9*x - 1.6227019727921712e-12*helper_9 - 32.000000000000284*x + 3.7836400679219423e-13*y*pow(z, 6) - 8.0246920219919926e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_32(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 576.00000000006844*y;
double helper_6 = helper_4*x;
double helper_7 = 769.33333333339715*y;
double helper_8 = x*y;
double helper_9 = 576.00000000007469*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 2.494005002518137e-12*helper_2;
double helper_14 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 936.00000000006878) + 2*helper_0*helper_6*(helper_5*x + 936.00000000006878*x + 576.00000000007469*y - 66.666666666667794) + helper_1*(519.9999999999884*helper_0*helper_8 - 6.6666666666729952*helper_0*y + 1.662670001678758e-12*helper_10*x - 2.0783375020984475e-13*helper_10 + 27.999999999993193*helper_11 + 1080.0000000000064*helper_12 - helper_13*helper_14 + helper_13 - 155.99999999999937*helper_14*helper_4 - 1.2470025012590675e-12*helper_14*x + 128.00000000000642*helper_3 + 121.33333333333047*helper_4 + 1672.0000000001323*helper_6 + 386.66666666665554*helper_8 - 4.1566750041968951e-13*x - 34.666666666666238*y - 2.0783375020984475e-13*z + 2.0783375020984475e-13) + helper_11*x*(helper_7 + 1872.0000000001376*helper_8 + helper_9 + 720.00000000000421*x - 66.666666666666671) + helper_11*(936.00000000006878*helper_12 + 360.0000000000021*helper_2 + 128.00000000000642*helper_4 + helper_7*x + helper_9*x - 66.666666666666671*x - 34.66666666666697*y) + 1728.0000000002053*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 576.00000000006844*x;
double helper_6 = helper_4*y;
double helper_7 = 769.33333333339715*x;
double helper_8 = x*y;
double helper_9 = 936.00000000006878*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 2.2994939286037026e-12*helper_3;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 576.00000000007469) + 2*helper_0*helper_6*(helper_5*y + 936.00000000006878*x + 576.00000000007469*y - 66.666666666667794) + helper_1*(-helper_0*helper_13 + 259.9999999999942*helper_0*helper_4 + 311.99999999999875*helper_0*helper_8 - 6.6666666666729952*helper_0*x - 1.1604051053382669e-12*helper_0*y + 1.3233858453533416e-13*helper_1 - 1.2683187833318e-12*helper_10*y + 3.1530333899356117e-13*helper_10 + 27.999999999993193*helper_11 + 384.00000000001927*helper_12 - helper_13 + 360.0000000000021*helper_2 + 193.33333333332777*helper_4 + 1672.0000000001323*helper_6 + 242.66666666666094*helper_8 - 34.666666666666238*x + 7.8737016906420959e-13*y + 9.565681580170894e-13*z - 9.565681580170894e-13) + helper_11*y*(helper_7 + 1152.0000000001494*helper_8 + helper_9 + 256.00000000001285*y - 34.66666666666697) + helper_11*(576.00000000007469*helper_12 + 128.00000000000642*helper_3 + 360.0000000000021*helper_4 + helper_7*y + helper_9*y - 66.666666666666671*x - 34.66666666666697*y) + 1728.0000000002053*helper_2*helper_3)/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 8.3133500083937972e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 7.6649797620123433e-13*helper_10;
double helper_12 = helper_9*x;
double helper_13 = helper_10*x;
double helper_14 = 3.3253400033575157e-12*helper_1;
double helper_15 = pow(z, 5);
double helper_16 = helper_15*x;
double helper_17 = helper_7*y;
double helper_18 = helper_1*y;
double helper_19 = 3.0659919048049373e-12*helper_10;
double helper_20 = helper_6*helper_9;
double helper_21 = helper_0*helper_6;
double helper_22 = helper_2*helper_6;
double helper_23 = helper_10*helper_7;
double helper_24 = helper_0*helper_9;
double helper_25 = helper_1*helper_9;
val[2] = (4.5989878572074044e-12*helper_0*helper_10 + 128.00000000000654*helper_0*helper_13 + 360.00000000000227*helper_0*helper_17 + 599.99999999990189*helper_0*helper_3 - 4.9880100050362749e-12*helper_0*helper_7 - 2.9096725029378272e-12*helper_0*x - 5.3885784723213496e-12*helper_0*y - helper_1*helper_19 - 586.66666666655567*helper_1*helper_3 - 1.4132695014269446e-11*helper_1*helper_6 + 576.00000000007481*helper_10*helper_21 - 256.00000000001285*helper_10*helper_4 - 576.00000000007458*helper_10*helper_6 + helper_11*helper_2 + helper_11 - 155.99999999999937*helper_12*helper_2 - 190.66666666666634*helper_12 + 128.00000000000642*helper_13 + helper_14*helper_7 + helper_14*x - 1.6626700016787572e-12*helper_15*helper_6 + 1.2683187833318115e-12*helper_15*helper_9 + 1.7514878436488919e-12*helper_15*y - 55.999999999986372*helper_16*y + 4.156675004196889e-13*helper_16 + 360.00000000000216*helper_17 + 1039.9999999999768*helper_18*helper_6 + 5.4605209243171826e-12*helper_18 - helper_19*z + 286.66666666660478*helper_2*helper_3 - helper_2*helper_8 - 5.7613913639898689e-12*helper_2*helper_9 - 1.8705037518886031e-12*helper_2*x - 3.7587710721715124e-12*helper_2*y - 1672.0000000001298*helper_20*z + 902.66666666673268*helper_20 + 769.33333333339715*helper_21*helper_9 - 1626.6666666666324*helper_21*y + 1.288569251301038e-11*helper_21 - 259.9999999999942*helper_22*y + 7.6898487577642594e-12*helper_22 + 1152.0000000001369*helper_23*z + 576.00000000006844*helper_23 + 936.00000000006889*helper_24*helper_7 - 970.66666666666333*helper_24*x - 9.2019725173031992e-12*helper_24 + 623.99999999999773*helper_25*x + 1.0362377622641468e-11*helper_25 + 62.666666666659381*helper_3 + 693.33333333333167*helper_4*helper_9 - 306.66666666662394*helper_4*y + 1.2470025012590687e-12*helper_4 + 1173.3333333333107*helper_5*helper_6 - 720.00000000000455*helper_5*helper_7 + 3.055333763768762e-12*helper_5 - 326.66666666666083*helper_6*y - 5.8193450058756575e-12*helper_6*z + 1.0391687510492247e-12*helper_6 - 936.00000000006867*helper_7*helper_9 + 3.3253400033575169e-12*helper_7*z - helper_8 + 4.0207837059824677e-12*helper_9*z - 6.8811623066266546e-13*helper_9 - 2.0783375020984445e-13*x - 3.9701575360600253e-13*y*pow(z, 6) - 7.229772336359703e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_33(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 1.8950232008599917e-11*y;
double helper_6 = helper_4*x;
double helper_7 = 545.33333333335702*y;
double helper_8 = x*y;
double helper_9 = 2.4847774879325669e-11*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 1.9184653865529667e-13*helper_2;
double helper_14 = helper_0*y;
double helper_15 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 72.000000000026063) + 2*helper_0*helper_6*(helper_5*x + 72.000000000026063*x + 2.4847774879325669e-11*y + 189.33333333332934) + helper_1*(1.2789769243686446e-13*helper_10*x - 1.5987211554608057e-14*helper_10 + 155.99999999999557*helper_11 + 216.00000000002177*helper_12 - helper_13*helper_15 + helper_13 + 712.00000000000591*helper_14*x + 249.33333333332484*helper_14 - 156.00000000000125*helper_15*helper_4 - 9.5923269327648248e-14*helper_15*x + 5.6275366311369416e-12*helper_3 + 249.33333333333059*helper_4 + 712.00000000005525*helper_6 + 1090.6666666666649*helper_8 - 3.1974423109216114e-14*x + 93.33333333332935*y - 1.5987211554608057e-14*z + 1.5987211554608057e-14) + helper_11*x*(helper_7 + 144.00000000005213*helper_8 + helper_9 + 144.00000000001452*x + 189.33333333332939) + helper_11*(72.000000000026063*helper_12 + 72.000000000007262*helper_2 + 5.6275366311369416e-12*helper_4 + helper_7*x + helper_9*x + 189.33333333332939*x + 93.333333333329378*y) + 5.6850696025799754e-11*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 1.8950232008599917e-11*x;
double helper_6 = helper_4*y;
double helper_7 = 545.33333333335702*x;
double helper_8 = x*y;
double helper_9 = 72.000000000026063*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 9.4591001698068346e-13*helper_3;
double helper_14 = helper_0*x;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 2.4847774879325669e-11) + 2*helper_0*helper_6*(helper_5*y + 72.000000000026063*x + 2.4847774879325669e-11*y + 189.33333333332934) + helper_1*(-helper_0*helper_13 + 356.00000000000296*helper_0*helper_4 - 8.6730622683720975e-13*helper_0*y - 1.3411494137471626e-13*helper_1 - 8.988365607365547e-13*helper_10*y - 1.5187850976871768e-13*helper_10 + 155.99999999999557*helper_11 + 1.6882609893410823e-11*helper_12 - helper_13 + 312.0000000000025*helper_14*y + 249.33333333332484*helper_14 + 72.000000000007262*helper_2 + 545.33333333333246*helper_4 + 712.00000000005525*helper_6 + 498.66666666666117*helper_8 + 93.33333333332935*x + 3.9523939676671817e-14*y + 8.0824236192733194e-14*z - 8.0824236192733194e-14) + helper_11*y*(helper_7 + 4.9695549758651338e-11*helper_8 + helper_9 + 1.1255073262273883e-11*y + 93.333333333329378) + helper_11*(2.4847774879325669e-11*helper_12 + 5.6275366311369416e-12*helper_3 + 72.000000000007262*helper_4 + helper_7*y + helper_9*y + 189.33333333332939*x + 93.333333333329378*y) + 5.6850696025799754e-11*helper_2*helper_3)/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 6.3948846218431736e-14*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 3.1530333899356147e-13*helper_9;
double helper_11 = helper_8*x;
double helper_12 = 2.5579538487372992e-13*helper_1;
double helper_13 = pow(z, 5);
double helper_14 = helper_5*y;
double helper_15 = helper_6*y;
double helper_16 = helper_2*y;
double helper_17 = 1.2612133559742459e-12*helper_9;
double helper_18 = helper_5*helper_8;
double helper_19 = 2.4847774879325669e-11*helper_5;
double helper_20 = helper_0*helper_5;
double helper_21 = 72.000000000026063*helper_8;
double helper_22 = helper_6*helper_9;
double helper_23 = helper_0*helper_6;
double helper_24 = helper_1*helper_8;
double helper_25 = helper_0*helper_9;
val[2] = (-842.66666666667788*helper_0*helper_11 + 72.000000000007276*helper_0*helper_15 + 545.33333333335668*helper_0*helper_18 + 1623.9999999999623*helper_0*helper_3 - 6.3864469268539177e-12*helper_0*helper_8 - 2.2382096176451251e-13*helper_0*x + 2.5126567493314786e-12*helper_0*y + 1424.0000000000118*helper_1*helper_14 - helper_1*helper_17 - 2122.6666666666119*helper_1*helper_3 - 1.0871303857133469e-12*helper_1*helper_5 - 4.6860293423376895e-12*helper_1*y + helper_10*helper_2 + helper_10 - 156.00000000000125*helper_11*helper_2 - 62.666666666671915*helper_11 + helper_12*helper_6 + helper_12*x - 311.99999999999113*helper_13*helper_3 - 1.2789769243686347e-13*helper_13*helper_5 + 8.9883656073655521e-13*helper_13*helper_8 + 3.1974423109215868e-14*helper_13*x - 2.1103119252074378e-12*helper_13*y + 1045.333333333353*helper_14*z - 166.66666666667356*helper_14 - 144.00000000001455*helper_15*z + 72.000000000007248*helper_15 - 356.00000000000296*helper_16*helper_5 + 1310.6666666666308*helper_16*x + 4.435563027982322e-12*helper_16 - helper_17*z - 712.00000000005525*helper_18*z + 166.66666666669835*helper_18 + helper_19*helper_25 - helper_19*helper_9 + 5.9152682752049353e-13*helper_2*helper_5 - helper_2*helper_7 - 4.0605296902641687e-12*helper_2*helper_8 - 1.4388490399147141e-13*helper_2*x - 1946.6666666666888*helper_20*y + 9.9120711638570024e-13*helper_20 + helper_21*helper_23 - helper_21*helper_6 + 3.7900464017199847e-11*helper_22*z + 1.8950232008599907e-11*helper_22 - 3.8369307731059042e-13*helper_23 + 624.000000000005*helper_24*x + 7.2537531536911271e-12*helper_24 + 5.6275366311369456e-12*helper_25*x + 1.8918200339613685e-12*helper_25 - 562.66666666665628*helper_3*z + 62.666666666666288*helper_3 + 437.33333333334633*helper_4*helper_8 - 1.1255073262273882e-11*helper_4*helper_9 + 9.5923269327647604e-14*helper_4 - 4.4764192352902503e-13*helper_5*z + 7.9936057773040314e-14*helper_5 + 2.5579538487372992e-13*helper_6*z - helper_7 + 2.7595703500083549e-12*helper_8*z - 4.6518344731795261e-13*helper_8 + 5.6275366311369408e-12*helper_9*x - 1.5987211554608057e-14*x + 4.023448241241489e-13*y*pow(z, 6) - 5.7198690228678311e-13*y*z + 1.7763568393980189e-14*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_34(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 576.00000000006673*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 958.66666666672779*y;
double helper_8 = x*y;
double helper_9 = 576.00000000007469*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 2.1103119252076789e-12*helper_2;
double helper_14 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_13 + 1.0551559626038388e-12*helper_0*x + 1.4068746168051193e-12*helper_10*x - 1.7585932710063992e-13*helper_10 + 99.999999999991331*helper_11 + 648.00000000000705*helper_12 + helper_13 + 631.99999999998533*helper_14*x + 134.66666666665685*helper_14 + 128.00000000000813*helper_3 + 1784.0000000001296*helper_5*x + 262.66666666666219*helper_5 + 227.9999999999992*helper_6 + 765.33333333331586*helper_8 - 3.5171865420127983e-13*x + 34.666666666665378*y - 1.7585932710063992e-13*z + 1.7585932710063992e-13) + helper_11*x*(helper_7 + 1584.0000000001351*helper_8 + helper_9 + 432.00000000000472*x + 66.666666666665037) + helper_11*(792.00000000006753*helper_12 + 216.00000000000236*helper_2 + 128.00000000000813*helper_5 + helper_7*x + helper_9*x + 66.666666666665037*x + 34.666666666664753*y) + 1728.0000000002001*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 792.00000000006753) + 2*helper_6*x*(helper_4*x + 792.00000000006753*x + 576.00000000007469*y + 66.666666666663929))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 576.00000000006673*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 958.66666666672779*x;
double helper_10 = x*y;
double helper_11 = 792.00000000006753*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 1.9424462038842462e-12*helper_3;
val[1] = (2*helper_0*helper_7*(helper_4*y + 792.00000000006753*x + 576.00000000007469*y + 66.666666666663929) + helper_1*(455.99999999999841*helper_0*helper_10 - helper_0*helper_14 + 134.66666666665685*helper_0*x - 1.2163603457793736e-12*helper_0*y + 5.9507954119923575e-14*helper_1 + 525.33333333332439*helper_10 + 384.00000000002439*helper_12 + 99.999999999991331*helper_13 - helper_14 + 216.00000000000236*helper_2 + 382.66666666665793*helper_5 + 315.99999999999267*helper_6 + 1784.0000000001296*helper_7 - 1.175948227682984e-12*helper_8*y + 1.5010215292933692e-13*helper_8 + 34.666666666665378*x + 6.2305716141968389e-13*y + 7.5228712148605868e-13*z - 7.5228712148605868e-13) + helper_10*helper_8*(1152.0000000001494*helper_10 + helper_11 + helper_9 + 256.00000000001626*y + 34.666666666664753) + helper_13*(helper_11*y + 576.00000000007469*helper_12 + 128.00000000000813*helper_3 + 216.00000000000236*helper_5 + helper_9*y + 66.666666666665037*x + 34.666666666664753*y) + 1728.0000000002001*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 576.00000000007469))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 7.0343730840255896e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 6.4748206796141896e-13*helper_10;
double helper_12 = helper_9*x;
double helper_13 = helper_10*x;
double helper_14 = 2.8137492336102387e-12*helper_1;
double helper_15 = helper_2*x;
double helper_16 = pow(z, 5);
double helper_17 = helper_7*y;
double helper_18 = helper_1*y;
double helper_19 = helper_6*helper_9;
double helper_20 = helper_0*helper_6;
double helper_21 = helper_2*helper_6;
double helper_22 = helper_7*helper_9;
double helper_23 = helper_10*helper_7;
double helper_24 = helper_1*helper_9;
double helper_25 = helper_0*helper_10;
val[2] = -(-1333.3333333333305*helper_0*helper_12 + 128.00000000000801*helper_0*helper_13 + 216.00000000000227*helper_0*helper_17 + 958.66666666672745*helper_0*helper_19 + 792.00000000006776*helper_0*helper_22 + 1191.9999999998852*helper_0*helper_3 - 4.2206238504153594e-12*helper_0*helper_7 - 8.1104012394917178e-12*helper_0*helper_9 - 2.4620305794089576e-12*helper_0*x - 4.1895376057261757e-12*helper_0*y - 2.5899282718456625e-12*helper_1*helper_10 - 1461.3333333331989*helper_1*helper_3 - 1.1958434242843512e-11*helper_1*helper_6 - 256.00000000001626*helper_10*helper_4 - 576.00000000007458*helper_10*helper_6 - 2.5899282718456617e-12*helper_10*z + helper_11*helper_2 + helper_11 - 193.33333333333445*helper_12 + 128.00000000000813*helper_13 + helper_14*helper_7 + helper_14*x - 227.9999999999992*helper_15*helper_9 - 1.5827339439057597e-12*helper_15 - 199.99999999998266*helper_16*helper_3 - 1.4068746168051189e-12*helper_16*helper_6 + 1.1759482276829836e-12*helper_16*helper_9 + 3.5171865420128084e-13*helper_16*x + 7.7093886829995074e-13*helper_16*y + 216.00000000000233*helper_17 + 1263.9999999999714*helper_18*helper_6 + 3.577582674552911e-12*helper_18 - 1784.0000000001273*helper_19*z + 825.33333333339965*helper_19 + 865.33333333325618*helper_2*helper_3 - helper_2*helper_8 - 5.2715609655252313e-12*helper_2*helper_9 - 1.9291235275892504e-12*helper_2*y - 1829.3333333332912*helper_20*y + 1.0903278280239676e-11*helper_20 - 315.99999999999272*helper_21*y + 6.5067951027236764e-12*helper_21 - 792.00000000006753*helper_22 + 1152.0000000001332*helper_23*z + 576.00000000006685*helper_23 + 911.99999999999727*helper_24*x + 9.3267615852710926e-12*helper_24 + 576.00000000007481*helper_25*helper_6 + 3.884892407768494e-12*helper_25 + 65.333333333325811*helper_3 + 842.66666666666731*helper_4*helper_9 - 461.33333333328585*helper_4*y + 1.0551559626038392e-12*helper_4 + 1130.6666666666406*helper_5*helper_6 - 432.00000000000455*helper_5*helper_7 + 2.5792701308094892e-12*helper_5 - 249.33333333332763*helper_6*y - 4.9240611588179192e-12*helper_6*z + 8.7929663550320011e-13*helper_6 + 2.8137492336102387e-12*helper_7*z - helper_8 + 3.4470204468561719e-12*helper_9*z - 5.6776805479329708e-13*helper_9 - 1.7585932710064042e-13*x - 1.785238623597698e-13*y*pow(z, 6) - 6.3060667798715606e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_35(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 576.00000000006492*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*x;
double helper_8 = 958.66666666671927*y;
double helper_9 = x*y;
double helper_10 = 792.00000000007071*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*x;
double helper_13 = helper_2*y;
double helper_14 = helper_11*y;
double helper_15 = 1.5347723092419872e-12*helper_2;
double helper_16 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_15 - 1.2789769243683227e-13*helper_11 + 1.0231815394946582e-12*helper_12 + 383.99999999999983*helper_13 + 99.999999999989342*helper_14 + helper_15 + 455.99999999997578*helper_16*x + 134.66666666665424*helper_16 + 216.00000000000639*helper_3 + 1784.0000000001135*helper_5*x + 382.66666666665844*helper_5 + 315.99999999999557*helper_6 + 7.6738615462099302e-13*helper_7 + 525.33333333330461*helper_9 - 2.5579538487366454e-13*x + 34.666666666664653*y - 1.2789769243683227e-13*z + 1.2789769243683227e-13) + helper_12*y*(helper_10 + helper_8 + 1152.0000000001273*helper_9 + 255.99999999999989*x + 34.666666666664284) + helper_14*(helper_10*x + 576.00000000006366*helper_13 + 127.99999999999994*helper_2 + 216.00000000000639*helper_5 + helper_8*x + 34.666666666664284*x + 66.66666666666417*y) + 1728.0000000001946*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 576.00000000006366) + 2*helper_5*helper_7*(helper_4*x + 576.00000000006366*x + 792.00000000007071*y + 66.666666666663247))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 576.00000000006492*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = 958.66666666671927*x;
double helper_9 = x*y;
double helper_10 = 576.00000000006366*helper_5;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 1.4175327578414524e-12*helper_3;
double helper_14 = helper_0*x;
val[1] = (helper_1*(-helper_0*helper_13 - 1.6324719354088625e-12*helper_0*y - 5.50670620213907e-14*helper_1 + 648.00000000001921*helper_11 + 99.999999999989342*helper_12 - helper_13 + 631.99999999999113*helper_14*y + 134.66666666665424*helper_14 + 127.99999999999994*helper_2 + 1784.0000000001135*helper_5*y + 262.6666666666523*helper_5 + 227.99999999998789*helper_6 - 1.0551559626037498e-12*helper_7*y - 1.540989558179535e-13*helper_7 + 765.33333333331689*helper_9 + 34.666666666664653*x + 2.7711166694648365e-13*y + 4.5163872641756275e-13*z - 4.5163872641756275e-13) + helper_12*(helper_10*y + 792.00000000007071*helper_11 + 216.00000000000639*helper_3 + 127.99999999999994*helper_5 + helper_8*y + 34.666666666664284*x + 66.66666666666417*y) + 1728.0000000001946*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 792.00000000007071) + 2*helper_6*y*(helper_4*y + 576.00000000006366*x + 792.00000000007071*y + 66.666666666663247) + helper_7*helper_9*(helper_10 + helper_8 + 1584.0000000001414*helper_9 + 432.00000000001279*y + 66.66666666666417))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 5.1159076974732898e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 4.7251091928048328e-13*helper_10;
double helper_12 = helper_9*x;
double helper_13 = 2.0463630789893159e-12*helper_1;
double helper_14 = pow(z, 5);
double helper_15 = helper_7*y;
double helper_16 = helper_1*y;
double helper_17 = helper_2*y;
double helper_18 = helper_9*z;
double helper_19 = 1.8900436771219372e-12*helper_10;
double helper_20 = helper_6*helper_9;
double helper_21 = helper_0*helper_6;
double helper_22 = 576.00000000006366*helper_7*helper_9;
double helper_23 = helper_10*helper_7;
double helper_24 = helper_0*helper_9;
double helper_25 = helper_1*helper_9;
double helper_26 = helper_0*helper_10;
val[2] = -(127.99999999999983*helper_0*helper_15 + 958.66666666671927*helper_0*helper_20 + helper_0*helper_22 + 1191.9999999998613*helper_0*helper_3 - 3.0695446184839737e-12*helper_0*helper_7 - 1.7905676941156514e-12*helper_0*x - 3.3137936839018652e-12*helper_0*y - helper_1*helper_19 - 1461.3333333331702*helper_1*helper_3 - 8.6970430857045956e-12*helper_1*helper_6 - 432.00000000001273*helper_10*helper_4 - 792.00000000007071*helper_10*helper_6 + 216.00000000000642*helper_10*x + helper_11*helper_2 + helper_11 - 315.99999999999557*helper_12*helper_2 - 249.33333333333144*helper_12 + helper_13*helper_7 + helper_13*x - 199.99999999997868*helper_14*helper_3 - 1.023181539494658e-12*helper_14*helper_6 + 1.0551559626037498e-12*helper_14*helper_9 + 2.5579538487366449e-13*helper_14*x - 6.8300920474912569e-13*helper_14*y - 255.99999999999989*helper_15*z + 127.99999999999994*helper_15 + 911.99999999995134*helper_16*helper_6 + 1.5845103007458819e-12*helper_16 - 227.99999999998795*helper_17*helper_6 + 865.33333333323912*helper_17*x + 4.8538950636548271e-13*helper_17 - 1784.0000000001121*helper_18*helper_6 + 1130.6666666666538*helper_18*x + 2.0108359422010237e-12*helper_18 - helper_19*z + 4.7322146201627939e-12*helper_2*helper_6 - helper_2*helper_8 - 4.4595438453143175e-12*helper_2*helper_9 - 1.1510792319314901e-12*helper_2*x + 825.33333333339283*helper_20 - 1333.3333333332628*helper_21*y + 7.9296569310836018e-12*helper_21 - helper_22 + 1152.0000000001296*helper_23*z + 576.00000000006503*helper_23 - 1829.3333333333096*helper_24*x - 5.6541438198109132e-12*helper_24 + 1263.9999999999823*helper_25*x + 7.2866157552197725e-12*helper_25 + 792.00000000007049*helper_26*helper_6 + 216.00000000000648*helper_26*x + 2.8350655156829056e-12*helper_26 - 461.33333333327641*helper_3*z + 65.333333333324447*helper_3 + 7.6738615462099342e-13*helper_4 + 842.66666666662309*helper_5*helper_6 + 2.3563373474647541e-12*helper_5 - 193.33333333332359*helper_6*y - 3.5811353882313056e-12*helper_6*z + 6.3948846218416236e-13*helper_6 + 2.0463630789893159e-12*helper_7*z - helper_8 - 2.3891999489931864e-13*helper_9 - 1.2789769243683076e-13*x + 1.6520118606417159e-13*y*pow(z, 6) - 5.9463545198929826e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_36(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 2.2410008012700788e-11*y;
double helper_6 = helper_4*x;
double helper_7 = 545.3333333333627*y;
double helper_8 = x*y;
double helper_9 = 72.000000000029445*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 8.2108389936195306e-26*helper_2;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 3.0815276193386688e-11) + 2*helper_0*helper_6*(helper_5*x + 3.0815276193386688e-11*x + 72.000000000029445*y + 189.33333333332897) + helper_1*(-6.8423658280162755e-27*helper_10 + 5.4738926624130204e-26*helper_11 + 2.5246002608327689e-11*helper_12 + 155.99999999999574*helper_13 - helper_14*helper_16 + helper_14 + 312.00000000000807*helper_15*x + 249.33333333332479*helper_15 - 356.00000000000284*helper_16*helper_4 - 4.1054194968097624e-26*helper_16*x + 72.000000000007049*helper_3 + 545.33333333333212*helper_4 + 712.00000000006708*helper_6 + 498.66666666666617*helper_8 - 1.3684731656032551e-26*x + 93.333333333329023*y - 6.8423658280162755e-27*z + 6.8423658280162755e-27) + helper_11*y*(helper_7 + 6.1630552386773376e-11*helper_8 + helper_9 + 1.6830668405551794e-11*x + 93.333333333329008) + helper_13*(3.0815276193386688e-11*helper_12 + 8.4153342027758968e-12*helper_2 + 72.000000000007049*helper_4 + helper_7*x + helper_9*x + 93.333333333329008*x + 189.33333333332911*y) + 6.7230024038102367e-11*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 2.2410008012700788e-11*x;
double helper_6 = helper_4*y;
double helper_7 = pow(helper_0, 2);
double helper_8 = 545.3333333333627*x;
double helper_9 = x*y;
double helper_10 = 3.0815276193386688e-11*helper_4;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 4.1566750041971889e-13*helper_3;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 72.000000000029445) + 2*helper_0*helper_6*(helper_5*y + 3.0815276193386688e-11*x + 72.000000000029445*y + 189.33333333332897) + helper_1*(-helper_0*helper_13 + 156.00000000000404*helper_0*helper_4 + 712.00000000000568*helper_0*helper_9 + 249.33333333332479*helper_0*x - 5.9152682752032935e-13*helper_0*y - 1.0125233984581142e-13*helper_1 + 216.00000000002115*helper_11 + 155.99999999999574*helper_12 - helper_13 + 8.4153342027758968e-12*helper_2 + 249.33333333333309*helper_4 + 712.00000000006708*helper_6 - 4.7961634663810226e-13*helper_7*y - 1.5853984791646844e-13*helper_7 + 1090.6666666666642*helper_9 + 93.333333333329023*x - 6.3948846218389907e-14*y + 1.3766765505377729e-14*z - 1.3766765505377729e-14) + helper_12*(helper_10*y + 72.000000000029445*helper_11 + 72.000000000007049*helper_3 + 8.4153342027758968e-12*helper_4 + helper_8*y + 93.333333333329008*x + 189.33333333332911*y) + 6.7230024038102367e-11*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 144.00000000005889*helper_9 + 144.0000000000141*y + 189.33333333332911))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = pow(x, 2);
double helper_5 = pow(x, 3);
double helper_6 = 2.7369463312065102e-26*helper_5;
double helper_7 = pow(y, 2);
double helper_8 = pow(y, 3);
double helper_9 = 1.3855583347323923e-13*helper_8;
double helper_10 = helper_7*x;
double helper_11 = helper_8*x;
double helper_12 = 72.000000000007049*helper_11;
double helper_13 = helper_1*x;
double helper_14 = pow(z, 5);
double helper_15 = helper_4*y;
double helper_16 = helper_4*z;
double helper_17 = helper_5*y;
double helper_18 = 1.0947785324826041e-25*helper_5;
double helper_19 = helper_7*z;
double helper_20 = 5.542233338929569e-13*helper_8;
double helper_21 = helper_4*helper_7;
double helper_22 = 72.000000000029445*helper_8;
double helper_23 = helper_0*helper_4;
double helper_24 = helper_2*helper_4;
double helper_25 = 3.0815276193386688e-11*helper_5;
double helper_26 = helper_5*helper_8;
double helper_27 = helper_0*helper_7;
val[2] = (-1946.6666666666883*helper_0*helper_10 + helper_0*helper_12 + 8.415334202775892e-12*helper_0*helper_17 + 545.33333333336259*helper_0*helper_21 + 1623.9999999999663*helper_0*helper_3 - 1.6421677987239059e-25*helper_0*helper_5 + 8.3133500083943536e-13*helper_0*helper_8 - 9.5793121592227852e-26*helper_0*x + 1.3029577416998779e-12*helper_0*y + 624.00000000001614*helper_1*helper_15 + helper_1*helper_18 - helper_1*helper_20 - 2122.6666666666156*helper_1*helper_3 - 4.6528087630510677e-25*helper_1*helper_4 + 3.6131098113403635e-12*helper_1*helper_7 - 2.8492763703978055e-12*helper_1*y - 356.00000000000273*helper_10*helper_2 - 166.66666666667379*helper_10 - 144.0000000000141*helper_11*z + helper_12 + 1424.0000000000114*helper_13*helper_7 + 1.0947785324826043e-25*helper_13 - 311.99999999999142*helper_14*helper_3 - 5.4738926624130193e-26*helper_14*helper_4 + 4.7961634663810216e-13*helper_14*helper_7 + 1.3684731656032548e-26*helper_14*x - 1.5054624213916686e-12*helper_14*y - 62.666666666675042*helper_15 + 437.33333333335804*helper_16*y - 1.9158624318445577e-25*helper_16 - 1.6830668405551797e-11*helper_17*z + 8.4153342027758952e-12*helper_17 + helper_18*z + 1045.3333333333535*helper_19*x + 1.2150280781498525e-12*helper_19 + 1310.6666666666324*helper_2*helper_3 - helper_2*helper_6 - 2.1023183194303471e-12*helper_2*helper_7 + helper_2*helper_9 - 6.1581292452146477e-26*helper_2*x + 2.9571900483914519e-12*helper_2*y - helper_20*z - 712.00000000006764*helper_21*z + 166.66666666670483*helper_21 + helper_22*helper_23 - helper_22*helper_4 - 842.66666666669516*helper_23*y + 4.2422668133700916e-25*helper_23 - 156.00000000000404*helper_24*y + 2.5316753563660219e-25*helper_24 + helper_25*helper_27 - helper_25*helper_7 + 4.4820016025401582e-11*helper_26*z + 2.2410008012700785e-11*helper_26 - 3.0215829838200345e-12*helper_27 - 562.66666666665833*helper_3*z + 62.666666666666686*helper_3 + 3.4211829140081392e-26*helper_4 - helper_6 - 1.8385293287793779e-13*helper_7 + helper_9 + 4.1054194968097647e-26*x*z - 6.8423658280162755e-27*x + 3.037570195374342e-13*y*pow(z, 6) - 1.8207657603840948e-13*y*z - 2.7089441800880575e-14*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_37(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 576.00000000006753*y;
double helper_6 = helper_4*x;
double helper_7 = pow(helper_0, 2);
double helper_8 = 769.33333333338737*y;
double helper_9 = x*y;
double helper_10 = 936.00000000007094*helper_4;
double helper_11 = helper_2*y;
double helper_12 = helper_7*y;
double helper_13 = 1.5134560271691899e-12*helper_2;
double helper_14 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 576.00000000006582) + 2*helper_0*helper_6*(helper_5*x + 576.00000000006582*x + 936.00000000007094*y - 66.666666666669002) + helper_1*(311.99999999997561*helper_0*helper_9 - 6.6666666666770311*helper_0*y + 383.99999999999807*helper_11 + 27.999999999990237*helper_12 - helper_13*helper_14 + helper_13 - 259.99999999999454*helper_14*helper_4 - 7.5672801358459434e-13*helper_14*x + 360.00000000000443*helper_3 + 193.3333333333253*helper_4 + 1672.0000000001141*helper_6 + 1.00897068477946e-12*helper_7*x - 1.261213355974325e-13*helper_7 + 242.66666666663991*helper_9 - 2.52242671194865e-13*x - 34.666666666667552*y - 1.261213355974325e-13*z + 1.261213355974325e-13) + helper_12*(helper_10*x + 576.00000000006582*helper_11 + 127.99999999999935*helper_2 + 360.00000000000443*helper_4 + helper_8*x - 34.666666666667986*x - 66.666666666668007*y) + 1728.0000000002026*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 1152.0000000001316*helper_9 + 255.99999999999869*x - 34.666666666667986))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 576.00000000006753*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 769.33333333338737*x;
double helper_8 = x*y;
double helper_9 = 576.00000000006582*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_3*x;
double helper_12 = helper_10*x;
val[1] = -(2*helper_0*helper_5*y*(helper_4*y + 576.00000000006582*x + 936.00000000007094*y - 66.666666666669002) - helper_1*(1.2256862191862269e-12*helper_0*helper_3 - 519.99999999998909*helper_0*x*y + 6.6666666666770311*helper_0*x + 1.7923440509548814e-12*helper_0*y + 8.7041485130593719e-14*helper_1 + 9.9120711638533895e-13*helper_10*y + 2.5801583092286548e-13*helper_10 - 1080.0000000000132*helper_11 - 27.999999999990237*helper_12 - 127.99999999999935*helper_2 + 1.2256862191862269e-12*helper_3 - 1672.0000000001141*helper_5*y - 121.33333333331996*helper_5 - 155.99999999998781*helper_6 - 386.6666666666506*helper_8 + 34.666666666667552*x - 1.492139745096693e-13*y - 3.5127456499145123e-13*z + 3.5127456499145123e-13) + helper_10*x*y*(helper_7 + 1872.0000000001419*helper_8 + helper_9 + 720.00000000000887*y - 66.666666666668007) + helper_12*(936.00000000007094*helper_11 + 360.00000000000443*helper_3 + 127.99999999999935*helper_5 + helper_7*y + helper_9*y - 34.666666666667986*x - 66.666666666668007*y) + 1728.0000000002026*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 936.00000000007094))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 5.0448534238972929e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = 4.0856207306207154e-13*helper_10;
double helper_12 = helper_10*x;
double helper_13 = helper_0*x;
double helper_14 = 2.01794136955892e-12*helper_1;
double helper_15 = pow(z, 5);
double helper_16 = helper_7*y;
double helper_17 = helper_1*y;
double helper_18 = helper_2*y;
double helper_19 = 1.6342482922482868e-12*helper_10;
double helper_20 = helper_6*helper_9;
double helper_21 = 936.00000000007094*helper_10;
double helper_22 = helper_0*helper_6;
double helper_23 = helper_10*helper_7;
double helper_24 = helper_0*helper_7;
double helper_25 = helper_1*helper_9;
double helper_26 = helper_2*helper_9;
val[2] = (2.4513724383724305e-12*helper_0*helper_10 + 360.00000000000455*helper_0*helper_12 + 127.99999999999943*helper_0*helper_16 + 599.99999999986687*helper_0*helper_3 - 4.5350390109887433e-12*helper_0*helper_9 - 3.3510971775293007e-12*helper_0*y - helper_1*helper_19 - 586.6666666665127*helper_1*helper_3 - 8.5762508206254109e-12*helper_1*helper_6 + helper_11*helper_2 + helper_11 - 720.00000000000864*helper_12*z + 360.00000000000432*helper_12 - 1626.6666666666351*helper_13*helper_9 - 1.7656986983640549e-12*helper_13 + helper_14*helper_7 + helper_14*x - 55.99999999998046*helper_15*helper_3 - 1.0089706847794596e-12*helper_15*helper_6 + 9.9120711638533855e-13*helper_15*helper_9 + 2.5224267119486556e-13*helper_15*x - 1.0507150705049565e-12*helper_15*y + 127.99999999999937*helper_16 + 623.99999999995134*helper_17*helper_6 + 1.3429257705874959e-12*helper_17 - 155.99999999998784*helper_18*helper_6 + 9.85433956656611e-13*helper_18 - helper_19*z + 286.66666666657932*helper_2*helper_3 + 4.6664894171050016e-12*helper_2*helper_6 - helper_2*helper_8 - 1.1350920203768919e-12*helper_2*x - 1672.0000000001125*helper_20*z + 902.6666666667254*helper_20 + helper_21*helper_22 - helper_21*helper_6 + 769.33333333338669*helper_22*helper_9 - 970.66666666659455*helper_22*y + 7.819522807040817e-12*helper_22 + 1152.0000000001351*helper_23*z + 576.00000000006753*helper_23 + 576.00000000006571*helper_24*helper_9 - 3.026912054338379e-12*helper_24 + 1039.9999999999782*helper_25*x + 6.3273830619436308e-12*helper_25 - 259.99999999999454*helper_26*x - 4.0598635564492522e-12*helper_26 + 62.666666666657505*helper_3 + 1173.3333333333144*helper_4*helper_9 + 7.5672801358459455e-13*helper_4 + 693.3333333332871*helper_5*helper_6 - 255.99999999999886*helper_5*helper_7 - 306.66666666661047*helper_5*x + 2.4185098368437722e-12*helper_5 - 190.6666666666558*helper_6*y - 3.5313973967281103e-12*helper_6*z + 6.3060667798716323e-13*helper_6 - 576.00000000006594*helper_7*helper_9 + 2.01794136955892e-12*helper_7*z - helper_8 - 326.66666666666259*helper_9*x + 1.371347480016931e-12*helper_9*z - 9.5035090907898708e-14*helper_9 - 1.2612133559743306e-13*x + 2.6112445539178027e-13*y*pow(z, 6) - 6.0618177144540151e-13*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_38(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 576.00000000012483*y;
double helper_6 = helper_4*x;
double helper_7 = pow(helper_0, 2);
double helper_8 = 1417.3333333334722*y;
double helper_9 = x*y;
double helper_10 = 1152.0000000001389*helper_4;
double helper_11 = helper_2*y;
double helper_12 = helper_7*y;
double helper_13 = 2.1103119252078776e-12*helper_2;
double helper_14 = helper_0*y;
double helper_15 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 792.00000000014199) + 2*helper_0*helper_6*(helper_5*x + 792.00000000014199*x + 1152.0000000001389*y - 66.666666666671048) + helper_1*(648.00000000005571*helper_11 + 115.99999999999061*helper_12 - helper_13*helper_15 + helper_13 + 664.00000000001376*helper_14*x + 17.333333333321548*helper_14 - 820.00000000000637*helper_15*helper_4 - 1.055155962603938e-12*helper_15*x + 704.00000000001569*helper_3 + 721.33333333333508*helper_4 + 2968.0000000002879*helper_6 + 1.4068746168052518e-12*helper_7*x - 1.7585932710065648e-13*helper_7 + 530.66666666667425*helper_9 - 3.5171865420131295e-13*x - 98.666666666669329*y - 1.7585932710065648e-13*z + 1.7585932710065648e-13) + helper_12*(helper_10*x + 792.00000000014199*helper_11 + 216.00000000001859*helper_2 + 704.00000000001569*helper_4 + helper_8*x - 66.666666666669954*x - 98.66666666666967*y) + 1728.0000000003745*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 1584.000000000284*helper_9 + 432.00000000003718*x - 66.666666666669954))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 576.00000000012483*x;
double helper_6 = helper_4*y;
double helper_7 = pow(helper_0, 2);
double helper_8 = 1417.3333333334722*x;
double helper_9 = x*y;
double helper_10 = 792.00000000014199*helper_4;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 383.99999999999807*helper_3;
double helper_14 = helper_0*x;
double helper_15 = -helper_0;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 1152.0000000001389) + 2*helper_0*helper_6*(helper_5*y + 792.00000000014199*x + 1152.0000000001389*y - 66.666666666671048) + helper_1*(-2.0339285811130429e-13*helper_1 + 2112.0000000000473*helper_11 + 115.99999999999061*helper_12 - helper_13*helper_15 + helper_13 + 1640.0000000000127*helper_14*y + 17.333333333321548*helper_14 - 332.00000000000688*helper_15*helper_4 - 191.99999999999733*helper_15*y + 216.00000000001859*helper_2 + 265.33333333333712*helper_4 + 2968.0000000002879*helper_6 + 255.99999999999824*helper_7*y - 32.000000000000441*helper_7 + 1442.6666666666702*helper_9 - 98.666666666669329*x - 63.999999999999879*y - 31.999999999999496*z + 31.999999999999496) + helper_12*(helper_10*y + 1152.0000000001389*helper_11 + 704.00000000001569*helper_3 + 216.00000000001859*helper_4 + helper_8*y - 66.666666666669954*x - 98.66666666666967*y) + 1728.0000000003745*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 2304.0000000002779*helper_9 + 1408.0000000000314*y - 98.66666666666967))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 7.0343730840262581e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = helper_10*x;
double helper_12 = 2.8137492336105032e-12*helper_1;
double helper_13 = pow(z, 5);
double helper_14 = helper_13*x;
double helper_15 = helper_7*y;
double helper_16 = helper_1*y;
double helper_17 = helper_6*helper_9;
double helper_18 = helper_0*helper_6;
double helper_19 = helper_2*helper_6;
double helper_20 = helper_10*helper_7;
double helper_21 = helper_0*helper_9;
double helper_22 = helper_1*helper_9;
double helper_23 = helper_2*helper_9;
val[2] = (-767.99999999999613*helper_0*helper_10 + 704.00000000001637*helper_0*helper_11 + 216.00000000001864*helper_0*helper_15 + 2215.9999999998827*helper_0*helper_3 - 4.2206238504157552e-12*helper_0*helper_7 - 2.4620305794091858e-12*helper_0*x - 448.00000000000273*helper_0*y + 511.99999999999727*helper_1*helper_10 - 1.1958434242844643e-11*helper_1*helper_6 + 1152.0000000001387*helper_10*helper_18 - 127.99999999999932*helper_10*helper_2 - 1152.0000000001392*helper_10*helper_6 + 511.99999999999739*helper_10*z - 127.99999999999935*helper_10 - 1408.0000000000318*helper_11*z + 704.00000000001592*helper_11 + helper_12*helper_7 + helper_12*x - 1.4068746168052516e-12*helper_13*helper_6 - 255.99999999999829*helper_13*helper_9 + 63.9999999999972*helper_13*y - 231.99999999998124*helper_14*y + 3.5171865420131381e-13*helper_14 - 432.00000000003706*helper_15*z + 216.00000000001853*helper_15 + 1328.0000000000282*helper_16*helper_6 - 2250.6666666665265*helper_16*x + 511.99999999999864*helper_16 - 2968.0000000002879*helper_17*z + 1550.6666666668148*helper_17 + 1417.3333333334722*helper_18*helper_9 - 2058.6666666667115*helper_18*y + 1.0903278280240728e-11*helper_18 - 332.00000000000682*helper_19*y + 6.5067951027242888e-12*helper_19 + 1142.6666666665847*helper_2*helper_3 - helper_2*helper_8 - 1.5827339439059122e-12*helper_2*x - 287.99999999999579*helper_2*y + 1152.0000000002497*helper_20*z + 576.00000000012483*helper_20 + 792.00000000014211*helper_21*helper_7 - 5018.6666666667079*helper_21*x + 1983.9999999999907*helper_21 + 3280.0000000000255*helper_22*x - 2175.9999999999882*helper_22 - 820.00000000000637*helper_23*x + 1183.9999999999927*helper_23 + 214.66666666665967*helper_3 + 3477.3333333333649*helper_4*helper_9 - 1090.6666666666199*helper_4*y + 1.0551559626039366e-12*helper_4 + 1461.3333333333674*helper_5*helper_6 + 192.00000000000279*helper_5 - 398.66666666667675*helper_6*y - 4.9240611588183821e-12*helper_6*z + 8.7929663550328291e-13*helper_6 - 792.00000000014188*helper_7*helper_9 + 2.8137492336105032e-12*helper_7*z - helper_8 - 918.66666666667595*helper_9*x - 895.99999999999659*helper_9*z + 159.99999999999957*helper_9 - 1.7585932710065691e-13*x + 6.1017857433391323e-13*y*pow(z, 6) - 32.000000000000774*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_39(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 5.1216843266486259e-14*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*x;
double helper_8 = pow(helper_0, 2);
double helper_9 = 329.33333333332564*y;
double helper_10 = x*y;
double helper_11 = 1.3215132113608593e-12*helper_5;
double helper_12 = helper_2*y;
double helper_13 = helper_8*y;
double helper_14 = 1.9184653865522402e-13*helper_2;
double helper_15 = helper_0*y;
val[0] = (2*helper_0*helper_7*(helper_4*x + 71.999999999998806*x - 1.3215132113608593e-12*y - 189.33333333332956) - helper_1*(-helper_0*helper_14 - 9.5923269327611922e-14*helper_0*x + 658.66666666664821*helper_10 - 215.9999999999965*helper_12 + 339.99999999999335*helper_13 - helper_14 + 279.99999999998903*helper_15*x + 625.3333333333228*helper_15 + 1.2021997980449216e-12*helper_3 + 625.33333333332439*helper_5 + 339.99999999999449*helper_6 + 279.99999999999204*helper_7 - 1.2789769243681601e-13*helper_8*x + 1.5987211554602002e-14*helper_8 + 3.1974423109204003e-14*x + 285.33333333332951*y + 1.5987211554602002e-14*z - 1.5987211554602002e-14) - helper_10*helper_8*(-143.99999999999761*helper_10 + helper_11 + helper_9 - 143.99999999999767*x + 189.33333333332953) - helper_13*(helper_11*x - 71.999999999998806*helper_12 - 71.999999999998835*helper_2 + 1.2021997980449216e-12*helper_5 + helper_9*x + 189.33333333332953*x + 285.33333333332962*y) + 1.5365052979945878e-13*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 71.999999999998806))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 5.1216843266486259e-14*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_5*y;
double helper_8 = pow(helper_0, 2);
double helper_9 = 329.33333333332564*x;
double helper_10 = x*y;
double helper_11 = 71.999999999998806*helper_5;
double helper_12 = helper_3*x;
double helper_13 = helper_8*x;
double helper_14 = 3.7569947153314146e-13*helper_3;
val[1] = (2*helper_0*helper_7*(helper_4*y + 71.999999999998806*x - 1.3215132113608593e-12*y - 189.33333333332956) - helper_1*(679.99999999998897*helper_0*helper_10 - helper_0*helper_14 + 625.3333333333228*helper_0*x + 447.9999999999996*helper_0*y + 127.99999999999991*helper_1 + 1250.6666666666488*helper_10 + 3.6065993941347649e-12*helper_12 + 339.99999999999335*helper_13 - helper_14 - 71.999999999998835*helper_2 + 329.33333333332411*helper_5 + 139.99999999999451*helper_6 + 279.99999999999204*helper_7 + 255.99999999999957*helper_8*y + 223.99999999999989*helper_8 + 285.33333333332951*x + 191.99999999999994*y + 95.999999999999972*z - 95.999999999999972) - helper_10*helper_8*(2.6430264227217187e-12*helper_10 - helper_11 + helper_9 + 2.4043995960898432e-12*y + 285.33333333332962) - helper_13*(-helper_11*y + 1.3215132113608593e-12*helper_12 + 1.2021997980449216e-12*helper_3 - 71.999999999998835*helper_5 + helper_9*y + 189.33333333332953*x + 285.33333333332962*y) + 1.5365052979945878e-13*helper_2*helper_3 + helper_3*helper_6*(helper_4 - 1.3215132113608593e-12))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 6.3948846218407199e-14*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 1.252331571777142e-13*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_9*x;
double helper_13 = 2.5579538487363475e-13*helper_1;
double helper_14 = pow(z, 5);
double helper_15 = helper_14*x;
double helper_16 = helper_5*y;
double helper_17 = helper_6*y;
double helper_18 = helper_6*z;
double helper_19 = 1855.9999999999986*y;
double helper_20 = helper_2*y;
double helper_21 = 5.0093262871085498e-13*helper_9;
double helper_22 = helper_5*helper_8;
double helper_23 = 1.3215132113608591e-12*helper_5*helper_9;
double helper_24 = helper_0*helper_5;
double helper_25 = 71.999999999998806*helper_6*helper_8;
val[2] = -(1754.6666666666372*helper_0*helper_11 - 1.2021997980449238e-12*helper_0*helper_12 + 71.999999999998778*helper_0*helper_17 + helper_0*helper_19 - 329.33333333332575*helper_0*helper_22 - helper_0*helper_23 + helper_0*helper_25 - 3047.99999999993*helper_0*helper_3 - 3.8369307731045188e-13*helper_0*helper_6 - 1215.999999999997*helper_0*helper_8 - 7.5139894306628242e-13*helper_0*helper_9 - 2.2382096176442734e-13*helper_0*x - 1359.9999999999782*helper_1*helper_11 - 559.99999999997829*helper_1*helper_16 + helper_1*helper_21 + 4298.6666666665751*helper_1*helper_3 - 1.0871303857129367e-12*helper_1*helper_5 + 1663.9999999999968*helper_1*helper_8 - 3583.9999999999973*helper_1*y - helper_10*helper_2 - helper_10 + 339.99999999999454*helper_11*helper_2 - 789.33333333331848*helper_11*z + 54.666666666664923*helper_11 - 1.2021997980449202e-12*helper_12 + helper_13*helper_6 + helper_13*x - helper_14*helper_19 - 1.278976924368144e-13*helper_14*helper_5 + 255.9999999999996*helper_14*helper_8 + 679.99999999998681*helper_15*y + 3.19744231092036e-14*helper_15 - 181.33333333331893*helper_16*z - 49.333333333335077*helper_16 - 143.99999999999767*helper_17*z + 71.999999999998835*helper_17 + 1.0243368653297294e-13*helper_18*helper_9 + 2.5579538487363475e-13*helper_18 + 5.9152682752027482e-13*helper_2*helper_5 - helper_2*helper_7 - 1055.999999999998*helper_2*helper_8 - 1.4388490399141759e-13*helper_2*x + 139.99999999999454*helper_20*helper_5 - 2774.6666666666106*helper_20*x + 3615.9999999999973*helper_20 + helper_21*z + 279.99999999999227*helper_22*z + 49.333333333333371*helper_22 + helper_23 + 650.66666666663764*helper_24*y + 9.9120711638532421e-13*helper_24 - helper_25 - 54.6666666666639*helper_3 + 2.4043995960898428e-12*helper_4*helper_9 + 898.66666666664241*helper_4*y + 9.5923269327613096e-14*helper_4 - 4.4764192352885559e-13*helper_5*z + 7.9936057773009428e-14*helper_5 + 5.1216843266486468e-14*helper_6*helper_9 - helper_7 + 383.99999999999881*helper_8*z - 31.999999999999829*helper_8 - 1.5987211554602002e-14*x + 383.99999999999977*y*pow(z, 6) - 447.99999999999977*y*z + 32.0*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_40(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 576.00000000015234*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 2038.6666666668402*y;
double helper_8 = x*y;
double helper_9 = 1152.0000000001751*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 2.4940050025184335e-12*helper_2;
double helper_14 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_13 + 1.2470025012592157e-12*helper_0*x + 1.6626700016789557e-12*helper_10*x - 2.0783375020986947e-13*helper_10 + 587.99999999998431*helper_11 + 1080.0000000000884*helper_12 + helper_13 + 1640.0000000000232*helper_14*x + 686.66666666664332*helper_14 + 704.00000000002365*helper_3 + 3944.0000000003683*helper_5*x + 1390.6666666666647*helper_5 + 1292.0000000000082*helper_6 + 1773.3333333333399*helper_8 - 4.1566750041973893e-13*x + 98.666666666658799*y - 2.0783375020986947e-13*z + 2.0783375020986947e-13) + helper_11*x*(helper_7 + 1872.0000000003604*helper_8 + helper_9 + 720.000000000059*x + 66.666666666658159) + helper_11*(936.00000000018019*helper_12 + 360.0000000000295*helper_2 + 704.00000000002365*helper_5 + helper_7*x + helper_9*x + 66.666666666658159*x + 98.666666666658486*y) + 1728.000000000457*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 936.00000000018019) + 2*helper_6*x*(helper_4*x + 936.00000000018019*x + 1152.0000000001751*y + 66.66666666665698))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 576.00000000015234*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 2038.6666666668402*x;
double helper_8 = x*y;
double helper_9 = 936.00000000018019*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 383.99999999999613*helper_3;
double helper_14 = helper_0*x;
val[1] = (helper_1*(helper_0*helper_13 + 575.99999999999602*helper_0*y + 127.99999999999963*helper_1 + 511.99999999999676*helper_10*y + 159.99999999999943*helper_10 + 587.99999999998431*helper_11 + 2112.0000000000709*helper_12 + helper_13 + 2584.0000000000164*helper_14*y + 686.66666666664332*helper_14 + 360.0000000000295*helper_2 + 3944.0000000003683*helper_5*y + 886.66666666666993*helper_5 + 820.0000000000116*helper_6 + 2781.3333333333294*helper_8 + 98.666666666658799*x + 64.00000000000027*y + 32.00000000000076*z - 32.00000000000076) + helper_11*y*(helper_7 + 2304.0000000003502*helper_8 + helper_9 + 1408.0000000000473*y + 98.666666666658486) + helper_11*(1152.0000000001751*helper_12 + 704.00000000002365*helper_3 + 360.0000000000295*helper_5 + helper_7*y + helper_9*y + 66.666666666658159*x + 98.666666666658486*y) + 1728.000000000457*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 1152.0000000001751) + 2*helper_6*y*(helper_4*y + 936.00000000018019*x + 1152.0000000001751*y + 66.66666666665698))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = y*z;
double helper_6 = pow(x, 2);
double helper_7 = pow(x, 3);
double helper_8 = 8.3133500083947726e-13*helper_7;
double helper_9 = pow(y, 2);
double helper_10 = pow(y, 3);
double helper_11 = helper_10*x;
double helper_12 = 704.00000000002365*helper_11;
double helper_13 = 3.3253400033579127e-12*helper_1;
double helper_14 = pow(z, 5);
double helper_15 = helper_7*y;
double helper_16 = helper_1*y;
double helper_17 = helper_6*helper_9;
double helper_18 = 1152.0000000001751*helper_10;
double helper_19 = helper_0*helper_6;
double helper_20 = helper_2*helper_6;
double helper_21 = helper_10*helper_7;
double helper_22 = helper_0*helper_9;
double helper_23 = helper_1*helper_9;
double helper_24 = helper_2*helper_9;
val[2] = -(-767.99999999999227*helper_0*helper_10 + helper_0*helper_12 + 360.00000000002933*helper_0*helper_15 + 2038.6666666668389*helper_0*helper_17 + 7639.9999999998272*helper_0*helper_3 - 4.9880100050368678e-12*helper_0*helper_7 - 2.9096725029381729e-12*helper_0*x - 2751.9999999999991*helper_0*y + 511.99999999999477*helper_1*helper_10 - 9013.3333333331138*helper_1*helper_3 - 1.4132695014271126e-11*helper_1*helper_6 - 127.99999999999875*helper_10*helper_2 + 511.99999999999488*helper_10*z - 127.99999999999872*helper_10 - 1408.0000000000473*helper_11*z + helper_12 + helper_13*helper_7 + helper_13*x - 1175.9999999999688*helper_14*helper_3 - 1.6626700016789545e-12*helper_14*helper_6 - 511.9999999999967*helper_14*helper_9 + 4.1566750041973863e-13*helper_14*x + 1983.9999999999941*helper_14*y - 720.00000000005866*helper_15*z + 360.00000000002944*helper_15 + 3280.0000000000464*helper_16*helper_6 + 4607.9999999999909*helper_16 - 3944.0000000003656*helper_17*z + 1905.3333333335258*helper_17 + helper_18*helper_19 - helper_18*helper_6 - 4853.3333333334112*helper_19*y + 1.2885692513011906e-11*helper_19 + 5193.3333333332012*helper_2*helper_3 - helper_2*helper_8 - 1.8705037518888176e-12*helper_2*x - 4191.9999999999882*helper_2*y - 820.00000000001182*helper_20*y + 7.689848757765169e-12*helper_20 + 1152.0000000003047*helper_21*z + 576.00000000015234*helper_21 + 936.00000000018031*helper_22*helper_7 - 7653.3333333333894*helper_22*x + 3391.9999999999795*helper_22 + 5168.0000000000364*helper_23*x - 3967.9999999999754*helper_23 - 1292.0000000000082*helper_24*x + 2271.9999999999854*helper_24 - 3133.3333333332703*helper_3*z + 489.3333333333253*helper_3 + 4970.6666666667152*helper_4*helper_9 + 1.2470025012592172e-12*helper_4 + 3146.6666666667297*helper_5*helper_6 + 832.00000000000193*helper_5 - 753.33333333335349*helper_6*y - 5.819345005876349e-12*helper_6*z + 1.0391687510493491e-12*helper_6 - 936.00000000018008*helper_7*helper_9 + 3.3253400033579127e-12*helper_7*z - helper_8 - 1193.3333333333496*helper_9*x - 1407.999999999992*helper_9*z + 223.99999999999886*helper_9 - 2.0783375020986843e-13*x - 383.99999999999886*y*pow(z, 6) - 96.000000000000739*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_41(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 9443.555555555793*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 595.34339277524862*y;
double helper_11 = x*y;
double helper_12 = 7022.6172839507499*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
val[0] = (4096.0000000001201*helper_0*helper_2*helper_3 + helper_1*(2672.3731138544617*helper_0*helper_11 + 1336.1865569273252*helper_0*helper_6 + 1.1287102939538151e-11*helper_0*x + 89.664380429872352*helper_0*y + 457.45221764957523*helper_11 + 1.4173660873232414e-11*helper_13 + 5025.1851851850552*helper_14 - 2.3242642378642532e-12*helper_2 + 1675.0617283950901*helper_3 + 1.4150327000675699e-11*helper_4 + 1.5661414552058873e-11*helper_5 + 10801.163237311257*helper_6*x + 228.72610882489857*helper_6 + 343.79149519893139*helper_7*y - 7.1536740865733855e-13*helper_7 + 4.7369515717340324e-14*helper_9 - 1.6470556057572041e-12*x + 1.8728852309527233*y - 8.2352780287860203e-13*z + 8.2352780287860203e-13) + helper_11*helper_9*(helper_10 + 14045.23456790138*helper_11 + helper_12 + 2048.0000000001237*helper_4 + 278.12345678985304*x + 1.8728852309570856) + 2*helper_13*helper_6*(2958.2222222223568*helper_4 + 2958.2222222223018*helper_6 + helper_8*x + 366.61728395038023*x + 366.61728395049875*y + 1.8728852309417623) + 7281.7777777778665*helper_2*pow(y, 4) + 3*helper_3*helper_5*(4096.0000000001201*x + 4096.0000000000809*y + 227.55555555547679) + helper_4*helper_6*helper_7*(helper_8 + 5916.4444444447136*x + 366.61728395038023) + helper_9*y*(helper_10*x + helper_12*x + 7022.6172839506899*helper_14 + 682.6666666667079*helper_2 + 682.66666666667777*helper_3 + 139.06172839492652*helper_4 + 139.06172839504833*helper_6 + 1.8728852309570856*x + 1.8728852309482167*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 9443.555555555793*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 595.34339277524862*x;
double helper_11 = x*y;
double helper_12 = 7022.6172839506899*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_4*x;
val[1] = (4096.0000000000809*helper_0*helper_2*helper_3 + helper_1*(2672.3731138546505*helper_0*helper_11 + 1336.1865569272309*helper_0*helper_6 + 89.664380429872352*helper_0*x - 1.1700767469693759e-11*helper_0*y + 457.45221764979715*helper_11 + 7.9145103822325637e-13*helper_13 + 5025.1851851852698*helper_14 - 1.4921742278314073e-27*helper_2 + 1675.0617283950185*helper_3 + 1.0466060574212607e-11*helper_4 + 1.0466060574212645e-11*helper_5 + 10801.163237311257*helper_6*y + 228.72610882478762*helper_6 + 343.79149519893139*helper_7*x - 9.2796601069371826e-12*helper_7 - 3.1687525641069935e-12*helper_9 + 1.8728852309527233*x - 6.617696788364521e-13*y + 6.0347763975448381e-13*z - 6.0347763975448381e-13) + helper_11*helper_9*(helper_10 + 14045.2345679015*helper_11 + helper_12 + 2048.0000000000332*helper_4 + 278.12345679009667*y + 1.8728852309482167) + 2*helper_13*helper_6*(2958.2222222223018*helper_4 + 2958.2222222223568*helper_6 + helper_8*y + 366.61728395038023*x + 366.61728395049875*y + 1.8728852309417623) + 7281.7777777778665*helper_2*pow(x, 4) + 3*helper_3*helper_5*(4096.0000000001201*x + 4096.0000000000809*y + 227.55555555547679) + helper_4*helper_6*helper_7*(helper_8 + 5916.4444444446035*y + 366.61728395049875) + helper_9*x*(helper_10*y + helper_12*y + 7022.6172839507499*helper_14 + 682.66666666667777*helper_2 + 682.6666666667079*helper_3 + 139.06172839504833*helper_4 + 139.06172839492652*helper_6 + 1.8728852309570856*x + 1.8728852309482167*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(4096.0000000001201*x + 4096.0000000000809*y + 227.55555555547679);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_6*y;
double helper_13 = 7022.6172839507499*helper_11 + 7022.6172839506899*helper_12 + 682.6666666667079*helper_3 + 682.66666666667777*helper_4 + 139.06172839492652*helper_6 + 595.34339277524862*helper_7 + 139.06172839504833*helper_8 + 1.8728852309570856*x + 1.8728852309482167*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = pow(helper_0, 4);
double helper_17 = -helper_0;
double helper_18 = 1.3954747432283526e-11*helper_4;
double helper_19 = 2.088188606941183e-11*helper_3;
double helper_20 = 2.2574205879076302e-11*helper_6;
double helper_21 = helper_0*y;
double helper_22 = helper_6*helper_8;
double helper_23 = helper_0*helper_8;
double helper_24 = helper_9*y;
double helper_25 = helper_0*x;
double helper_26 = helper_9*x;
double helper_27 = helper_21*x;
double helper_28 = helper_7*helper_9;
double helper_29 = 5344.7462277093009*helper_0*helper_11 + 5344.7462277089235*helper_0*helper_12 - 2.3242642378642532e-12*helper_1 + 1.5829020764465127e-12*helper_10 + 914.9044352995943*helper_11 + 914.90443529915046*helper_12 - 1.2675010256427974e-11*helper_14*y + 1.8947806286936129e-13*helper_15 - 1.4921742278314073e-27*helper_2 + 2.4139105590179352e-12*helper_21 + 21602.326474622514*helper_22 - 2.3401534939387519e-11*helper_23 - 3.7118640427748731e-11*helper_24 - 3.2941112115144081e-12*helper_25 - 2.8614696346293542e-12*helper_26 + 358.65752171948941*helper_27 + 1375.1659807957255*helper_28 + 6700.2469135800739*helper_3*y + 1.8867102667567599e-11*helper_3 + 6700.2469135803603*helper_4*x + 1.3954747432283475e-11*helper_4 + 2.8347321746464828e-11*helper_6*helper_9 - 3.2941112115144081e-12*helper_6 + 7.4915409238108932*helper_7 - 1.3235393576729042e-12*helper_8;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_22*(2958.2222222223568*helper_6 + 9443.555555555793*helper_7 + 2958.2222222223018*helper_8 + 366.61728395038023*x + 366.61728395049875*y + 1.8728852309417623) + 3*helper_13*helper_28 + helper_14*(helper_0*helper_18 + helper_0*helper_19 + helper_0*helper_20 + helper_29) + helper_16*(1.4173660873232414e-11*helper_0*helper_6 + 1336.1865569273252*helper_11 + 1336.1865569272309*helper_12 - 1.8559320213874365e-11*helper_21 + 7.9145103822325637e-13*helper_23 - 9.5062576923209813e-12*helper_24 - 1.4307348173146771e-12*helper_25 + 1.4210854715202097e-13*helper_26 + 687.58299039786277*helper_27 + 5.2204715173529575e-12*helper_3 + 3.4886868580708816e-12*helper_4 + 5.6435514697690756e-12*helper_6 + 89.664380429872352*helper_7 - 5.8503837348468797e-12*helper_8 - 8.2352780287860203e-13*x + 6.0347763975448381e-13*y) + helper_5) + 7281.7777777778665*helper_1*helper_2 + helper_10*helper_6*(11832.888888889427*helper_6 + 37774.222222223172*helper_7 + 11832.888888889207*helper_8 + 1466.4691358015209*x + 1466.469135801995*y + 7.4915409237670492) + 4*helper_13*helper_15*y + helper_16*(-helper_17*helper_18 - helper_17*helper_19 - helper_17*helper_20 + helper_29))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_42(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 12288.000000000078*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 85.157750343411905*y;
double helper_11 = x*y;
double helper_12 = 8874.666666666657*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
val[0] = -(5461.3333333334303*helper_0*helper_2*helper_3 - helper_1*(-1181.7613168724481*helper_0*helper_6 - 1500.7078189297983*helper_0*x*y - 1.0616297964185357e-11*helper_0*x + 37.750342935457439*helper_0*y - 1.370031660136059e-11*helper_13 - 3313.7777777774836*helper_14 + 2.6526928801711407e-12*helper_2 - 2048.0000000000236*helper_3 - 1.1007885960782597e-11*helper_4 - 1.3089776176559707e-11*helper_5 - 10375.374485596158*helper_6*x + 37.750342935513615*helper_6 - 157.76131687246246*helper_7*y + 3.5066599829652242e-13*helper_7 - 6.1580370432542454e-14*helper_9 + 170.31550068615951*x*y + 9.5949585725255556e-13*x + 3.5116598079075501*y + 4.7974792862627778e-13*z - 4.7974792862627778e-13) + helper_11*helper_9*(-helper_10 + 14497.185185184931*helper_11 + helper_12 + 1536.0000000001303*helper_4 - 94.814814815211733*x - 3.5116598079009647) + 2*helper_13*helper_6*(3242.6666666667984*helper_4 + 4437.3333333334131*helper_6 + helper_8*x - 47.407407407859139*x - 2.8492763703980017e-10*y - 3.5116598079220265) + 10922.666666666682*helper_2*pow(y, 4) + 3*helper_3*helper_5*(5461.3333333334303*x + 6144.0000000000591*y - 2.2435444914549252e-10) + helper_4*helper_6*helper_7*(helper_8 + 6485.3333333335968*x - 47.407407407859139) - helper_9*y*(helper_10*x - helper_12*x - 7248.5925925924657*helper_14 - 512.00000000004343*helper_2 - 1024.0000000000123*helper_3 + 47.407407407605866*helper_4 + 2.5565327632648405e-11*helper_6 + 3.5116598079009647*x + 3.5116598079144898*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 12288.000000000078*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 85.157750343411905*x;
double helper_11 = x*y;
double helper_12 = 7248.5925925924657*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
val[1] = -(6144.0000000000591*helper_0*helper_2*helper_3 - helper_1*(-750.35390946489917*helper_0*helper_6 - 2363.5226337448962*helper_0*x*y + 37.750342935457439*helper_0*x + 1.1269073465875658e-11*helper_0*y - 10375.374485596158*helper_13 - 6144.0000000000709*helper_14 + 2.2270419866506113e-27*helper_2 - 1104.5925925924946*helper_3 - 2.1633257599563126e-11*helper_4 - 2.1633257599563175e-11*helper_5 + 85.157750343079755*helper_6 - 157.76131687246246*helper_7*x - 6.7599369175037632e-12*helper_7*y + 1.2336390254095628e-11*helper_7 + 3.9448042454907904e-12*helper_9 + 75.50068587102723*x*y + 3.5116598079075501*x + 2.2081184617401405e-12*y + 1.071782351941171e-12*z - 1.071782351941171e-12) + helper_11*helper_9*(-helper_10 + 17749.333333333314*helper_11 + helper_12 + 3072.0000000000368*helper_4 - 5.1130655265296809e-11*y - 3.5116598079144898) + 2*helper_13*helper_7*(4437.3333333334131*helper_4 + 3242.6666666667984*helper_6 + helper_8*y - 47.407407407859139*x - 2.8492763703980017e-10*y - 3.5116598079220265) + 10922.666666666682*helper_2*pow(x, 4) + 3*helper_3*helper_5*(5461.3333333334303*x + 6144.0000000000591*y - 2.2435444914549252e-10) + helper_4*helper_6*helper_7*(helper_8 + 8874.6666666668261*y - 2.8492763703980017e-10) - helper_9*x*(helper_10*y - helper_12*y - 8874.666666666657*helper_14 - 1024.0000000000123*helper_2 - 512.00000000004343*helper_3 + 2.5565327632648405e-11*helper_4 + 47.407407407605866*helper_6 + 3.5116598079009647*x + 3.5116598079144898*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 3);
double helper_6 = helper_5*(5461.3333333334303*x + 6144.0000000000591*y - 2.2435444914549252e-10);
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_7*x;
double helper_13 = helper_9*y;
double helper_14 = -8874.666666666657*helper_12 - 7248.5925925924657*helper_13 - 512.00000000004343*helper_3 - 1024.0000000000123*helper_5 + 2.5565327632648405e-11*helper_7 + 85.157750343411905*helper_8 + 47.407407407605866*helper_9 + 3.5116598079009647*x + 3.5116598079144898*y;
double helper_15 = pow(helper_0, 3);
double helper_16 = helper_15*y;
double helper_17 = pow(helper_0, 4);
double helper_18 = helper_0*x;
double helper_19 = helper_0*y;
double helper_20 = helper_19*x;
double helper_21 = helper_10*x;
double helper_22 = helper_0*helper_9;
double helper_23 = helper_10*y;
double helper_24 = helper_0*helper_7;
double helper_25 = -4727.0452674897924*helper_0*helper_12 - 2.8844343466084232e-11*helper_0*helper_5 + 2.6526928801711407e-12*helper_1 - 1.3519873835007526e-11*helper_10*helper_7 - 2.7400633202721179e-11*helper_11 + 151.00137174205446*helper_12 + 340.63100137231902*helper_13 - 2.4632148173016981e-13*helper_15*x + 1.5779216981963162e-11*helper_16 + 1.9189917145051111e-12*helper_18 - 3001.4156378595967*helper_19*helper_9 + 4.287129407764684e-12*helper_19 + 2.2270419866506113e-27*helper_2 + 151.00137174182976*helper_20 + 1.4026639931860897e-12*helper_21 - 2.1232595928370715e-11*helper_22 - 631.04526748984983*helper_23*x + 4.9345561016382512e-11*helper_23 + 2.2538146931751315e-11*helper_24 - 4418.3703703699784*helper_3*y - 1.4677181281043463e-11*helper_3 - 1.7453034902079609e-11*helper_4 - 8192.0000000000946*helper_5*x - 2.8844343466084171e-11*helper_5 - 20750.748971192315*helper_7*helper_9 + 4.416236923480281e-12*helper_7 + 14.0466392316302*helper_8 + 1.9189917145051111e-12*helper_9;
val[2] = (helper_0*(3*helper_10*helper_14*x*y + helper_15*helper_25 - helper_17*(1181.7613168724481*helper_12 + 750.35390946489917*helper_13 - 7.0133199659304485e-13*helper_18 - 2.4672780508191256e-11*helper_19 + 315.52263374492492*helper_20 + 1.8474111129762736e-13*helper_21 + 1.370031660136059e-11*helper_22 - 1.1834412736472371e-11*helper_23 + 6.7599369175037632e-12*helper_24 + 4.3632587255199022e-12*helper_3 + 7.211085866521058e-12*helper_5 - 5.6345367329378288e-12*helper_7 - 37.750342935457439*helper_8 + 5.3081489820926787e-12*helper_9 - 4.7974792862627778e-13*x - 1.071782351941171e-12*y) - 2*helper_24*helper_9*(4437.3333333334131*helper_7 + 12288.000000000078*helper_8 + 3242.6666666667984*helper_9 - 47.407407407859139*x - 2.8492763703980017e-10*y - 3.5116598079220265) - helper_3*helper_6) + 10922.666666666682*helper_1*helper_2 + helper_11*helper_7*(17749.333333333652*helper_7 + 49152.000000000313*helper_8 + 12970.666666667194*helper_9 - 189.62962963143656*x - 1.1397105481592007e-9*y - 14.046639231688106) - 4*helper_14*helper_16*x - helper_17*helper_25 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_43(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 6940.4444444444007*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 504.50845907669674*y;
double helper_11 = x*y;
double helper_12 = 4810.2716049381979*helper_6;
double helper_13 = helper_6*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (3185.7777777778147*helper_0*helper_2*helper_3 + helper_1*(407.00137174212671*helper_0*helper_6 - 49.397347965212845*helper_0*y - 275.78235025167538*helper_11 + 4164.1262002738958*helper_13 + 1270.5185185183145*helper_14 + 460.02743484209213*helper_15*y + 4.0048293899252682e-12*helper_15 - 1.3137145692276102e-12*helper_2 + 1055.6049382716214*helper_3 + 2.4216349090615387e-12*helper_4 + 3.7621922038622628e-12*helper_5 - 188.4590763603037*helper_6 + 5.4071424978094138e-12*helper_7*x + 34.063100137194084*helper_7*y + 1.088621648244332e-13*helper_7 + 3.4737644859382956e-14*helper_9 + 8.3510701783065676e-14*x + 1.8728852309324617*y + 4.1755350891532838e-14*z - 4.1755350891532838e-14) + helper_11*helper_9*(-helper_10 + 6535.9012345675856*helper_11 + helper_12 + 682.66666666673382*helper_4 - 176.98765432123056*x + 1.8728852309368427) + 2*helper_13*helper_7*(1592.8888888889512*helper_4 + 2958.2222222222599*helper_6 + helper_8*x - 316.04938271637133*x - 366.61728395083333*y + 1.8728852309241883) + 7281.7777777777519*helper_2*pow(y, 4) + 3*helper_3*helper_5*(3185.7777777778147*x + 4096.0000000000182*y - 227.55555555573991) + helper_4*helper_6*helper_7*(helper_8 + 3185.7777777779024*x - 316.04938271637133) + helper_9*y*(-helper_10*x + helper_12*x + 3267.9506172837928*helper_14 + 227.55555555557794*helper_2 + 682.66666666667311*helper_3 - 88.493827160615282*helper_4 - 139.06172839507292*helper_6 + 1.8728852309368427*x + 1.8728852309279933*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 6940.4444444444007*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 504.50845907669674*x;
double helper_11 = x*y;
double helper_12 = 3267.9506172837928*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_4*x;
val[1] = (4096.0000000000182*helper_0*helper_2*helper_3 + helper_1*(814.00274348425341*helper_0*helper_11 + 230.01371742104607*helper_0*helper_6 - 49.397347965212845*helper_0*x - 4.4424528070816834e-12*helper_0*y - 376.9181527206074*helper_11 + 6.7088684858984509e-12*helper_13 + 3166.8148148148643*helper_14 - 1.4772150877027493e-27*helper_2 + 423.50617283943814*helper_3 + 1.6681401573285923e-11*helper_4 + 1.6681401573285949e-11*helper_5 + 4164.1262002738958*helper_6*y - 137.89117512583769*helper_6 + 34.063100137194084*helper_7*x - 7.4072100381412386e-12*helper_7 - 2.2818241732938338e-12*helper_9 + 1.8728852309324617*x - 2.0514618810283349e-12*y - 1.5012475335415458e-12*z + 1.5012475335415458e-12) + helper_11*helper_9*(-helper_10 + 9620.5432098763958*helper_11 + helper_12 + 2048.0000000000191*helper_4 - 278.12345679014584*y + 1.8728852309279933) + 2*helper_13*helper_6*(2958.2222222222599*helper_4 + 1592.8888888889512*helper_6 + helper_8*y - 316.04938271637133*x - 366.61728395083333*y + 1.8728852309241883) + 7281.7777777777519*helper_2*pow(x, 4) + 3*helper_3*helper_5*(3185.7777777778147*x + 4096.0000000000182*y - 227.55555555573991) + helper_4*helper_6*helper_7*(helper_8 + 5916.4444444445198*y - 366.61728395083333) + helper_9*x*(-helper_10*y + helper_12*y + 4810.2716049381979*helper_14 + 682.66666666667311*helper_2 + 227.55555555557794*helper_3 - 139.06172839507292*helper_4 - 88.493827160615282*helper_6 + 1.8728852309368427*x + 1.8728852309279933*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(3185.7777777778147*x + 4096.0000000000182*y - 227.55555555573991);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_6*y;
double helper_13 = 4810.2716049381979*helper_11 + 3267.9506172837928*helper_12 + 227.55555555557794*helper_3 + 682.66666666667311*helper_4 - 88.493827160615282*helper_6 - 504.50845907669674*helper_7 - 139.06172839507292*helper_8 + 1.8728852309368427*x + 1.8728852309279933*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = pow(helper_0, 4);
double helper_17 = -helper_0;
double helper_18 = 8.0096587798505363e-12*helper_6;
double helper_19 = 5.0162562718163504e-12*helper_3;
double helper_20 = 2.2241868764381265e-11*helper_4;
double helper_21 = helper_0*x;
double helper_22 = helper_9*x;
double helper_23 = helper_6*helper_8;
double helper_24 = helper_9*y;
double helper_25 = helper_0*y;
double helper_26 = helper_0*helper_8;
double helper_27 = helper_22*y;
double helper_28 = helper_21*y;
double helper_29 = -1.3137145692276102e-12*helper_1 + 1.3417736971796902e-11*helper_10 - 753.83630544121479*helper_11 - 551.56470050335076*helper_12 - 9.1272966931753353e-12*helper_14*y + 1.3895057943753182e-13*helper_15 - 1.4772150877027493e-27*helper_2 + 1628.0054869685068*helper_21*helper_8 + 1.6702140356613135e-13*helper_21 + 4.3544865929773279e-13*helper_22 + 8328.2524005477917*helper_23 - 2.9628840152564954e-11*helper_24 + 920.05486968418427*helper_25*helper_6 - 6.0049901341661834e-12*helper_25 - 8.8849056141633668e-12*helper_26 + 136.25240054877634*helper_27 - 197.58939186085138*helper_28 + 1694.0246913577525*helper_3*y + 3.2288465454153849e-12*helper_3 + 4222.4197530864858*helper_4*x + 2.224186876438123e-11*helper_4 + 1.0814284995618828e-11*helper_6*helper_9 + 1.6702140356613135e-13*helper_6 + 7.4915409237298469*helper_7 - 4.1029237620566698e-12*helper_8;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_23*(1592.8888888889512*helper_6 + 6940.4444444444007*helper_7 + 2958.2222222222599*helper_8 - 316.04938271637133*x - 366.61728395083333*y + 1.8728852309241883) + 3*helper_13*helper_27 + helper_14*(helper_0*helper_18 + helper_0*helper_19 + helper_0*helper_20 + helper_29) + helper_16*(5.4071424978094138e-12*helper_0*helper_6 + 407.00137174212671*helper_11 + 230.01371742104607*helper_12 + 2.1772432964886639e-13*helper_21 + 1.0421293457814887e-13*helper_22 - 6.8454725198815014e-12*helper_24 - 1.4814420076282477e-11*helper_25 + 6.7088684858984509e-12*helper_26 + 68.126200274388168*helper_28 + 1.2540640679540876e-12*helper_3 + 5.5604671910953163e-12*helper_4 + 2.0024146949626341e-12*helper_6 - 49.397347965212845*helper_7 - 2.2212264035408417e-12*helper_8 + 4.1755350891532838e-14*x - 1.5012475335415458e-12*y) + helper_5) + 7281.7777777777519*helper_1*helper_2 + helper_10*helper_6*(6371.5555555558049*helper_6 + 27761.777777777603*helper_7 + 11832.88888888904*helper_8 - 1264.1975308654853*x - 1466.4691358033333*y + 7.4915409236967534) + 4*helper_13*helper_15*y + helper_16*(-helper_17*helper_18 - helper_17*helper_19 - helper_17*helper_20 + helper_29))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_44(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 12288.000000000035*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 85.15775034373506*y;
double helper_11 = x*y;
double helper_12 = 7248.5925925924003*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
val[0] = -(6144.0000000001628*helper_0*helper_2*helper_3 - helper_1*(-750.35390946494999*helper_0*helper_6 - 2363.5226337444687*helper_0*x*y - 1.440033277807212e-11*helper_0*x + 37.750342935460864*helper_0*y - 1.7886729134868485e-11*helper_13 - 6143.9999999995634*helper_14 + 3.486396356796368e-12*helper_2 - 1104.5925925925426*helper_3 - 1.648459146963542e-11*helper_4 - 1.864464138634616e-11*helper_5 - 10375.374485595587*helper_6*x + 85.157750343002007*helper_6 - 157.7613168724487*helper_7*y + 7.5791225147751091e-13*helper_7 + 75.500685871489253*x*y + 1.7337242752547955e-12*x + 3.5116598078923267*y + 8.6686213762739776e-13*z - 8.6686213762739776e-13) + helper_11*helper_9*(-helper_10 + 17749.33333333279*helper_11 + helper_12 + 3072.000000000171*helper_4 - 5.4509049126258929e-10*x - 3.5116598078852022) + 2*helper_13*helper_6*(4437.3333333335195*helper_4 + 3242.6666666667738*helper_6 + helper_8*x - 6.9849193096162502e-10*x - 47.407407407944319*y - 3.5116598079059691) + 10922.666666666777*helper_2*pow(y, 4) + 3*helper_3*helper_5*(6144.0000000001628*x + 5461.3333333334394*y - 3.9578177772152741e-10) + helper_4*helper_6*helper_7*(helper_8 + 8874.666666667039*x - 6.9849193096162502e-10) - helper_9*y*(helper_10*x - helper_12*x - 8874.666666666395*helper_14 - 1024.0000000000571*helper_2 - 512.00000000001455*helper_3 + 2.7254524563129465e-10*helper_4 + 47.407407407513368*helper_6 + 3.5116598078852022*x + 3.5116598078982761*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 12288.000000000035*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 85.15775034373506*x;
double helper_11 = x*y;
double helper_12 = 8874.666666666395*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_4*x;
val[1] = -(5461.3333333334394*helper_0*helper_2*helper_3 - helper_1*(-1181.7613168722344*helper_0*helper_6 - 1500.7078189299*helper_0*x*y + 37.750342935460864*helper_0*x + 6.7977448087513338e-12*helper_0*y - 6.3304533083336773e-12*helper_13 - 3313.7777777776278*helper_14 + 2.1541161785235313e-27*helper_2 - 2047.9999999998545*helper_3 - 1.7325444234281644e-11*helper_4 - 1.7325444234281699e-11*helper_5 - 10375.374485595587*helper_6*y + 37.750342935744627*helper_6 - 157.7613168724487*helper_7*x + 9.6480756625827057e-12*helper_7 + 2.6667648427878658e-12*helper_9 + 170.31550068600401*x*y + 3.5116598078923267*x - 7.1001640780728007e-13*y - 1.0775888640473118e-12*z + 1.0775888640473118e-12) + helper_11*helper_9*(-helper_10 + 14497.185185184801*helper_11 + helper_12 + 1536.0000000000437*helper_4 - 94.814814815026736*y - 3.5116598078982761) + 2*helper_13*helper_6*(3242.6666666667738*helper_4 + 4437.3333333335195*helper_6 + helper_8*y - 6.9849193096162502e-10*x - 47.407407407944319*y - 3.5116598079059691) + 10922.666666666777*helper_2*pow(x, 4) + 3*helper_3*helper_5*(6144.0000000001628*x + 5461.3333333334394*y - 3.9578177772152741e-10) + helper_4*helper_6*helper_7*(helper_8 + 6485.3333333335477*y - 47.407407407944319) - helper_9*x*(helper_10*y - helper_12*y - 7248.5925925924003*helper_14 - 512.00000000001455*helper_2 - 1024.0000000000571*helper_3 + 47.407407407513368*helper_4 + 2.7254524563129465e-10*helper_6 + 3.5116598078852022*x + 3.5116598078982761*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 3);
double helper_6 = helper_5*(6144.0000000001628*x + 5461.3333333334394*y - 3.9578177772152741e-10);
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_7*x;
double helper_13 = helper_9*y;
double helper_14 = -7248.5925925924003*helper_12 - 8874.666666666395*helper_13 - 1024.0000000000571*helper_3 - 512.00000000001455*helper_5 + 47.407407407513368*helper_7 + 85.15775034373506*helper_8 + 2.7254524563129465e-10*helper_9 + 3.5116598078852022*x + 3.5116598078982761*y;
double helper_15 = pow(helper_0, 3);
double helper_16 = helper_15*y;
double helper_17 = pow(helper_0, 4);
double helper_18 = helper_0*x;
double helper_19 = helper_0*y;
double helper_20 = helper_19*x;
double helper_21 = helper_7*helper_9;
double helper_22 = helper_0*helper_9;
double helper_23 = helper_10*y;
double helper_24 = helper_0*helper_7;
double helper_25 = 2.31005923123756e-11*helper_0*helper_5 - 3.486396356796368e-12*helper_1 + 1.2660906616667355e-11*helper_10*helper_7 - 3.0316490059100437e-12*helper_10*x + 3.577345826973697e-11*helper_11 - 340.63100137200803*helper_12 - 151.00137174297851*helper_13 - 1.0667059371151463e-11*helper_16 - 3.467448550509591e-12*helper_18 + 4727.0452674889375*helper_19*helper_9 + 4.3103554561892473e-12*helper_19 - 2.1541161785235313e-27*helper_2 - 151.00137174184346*helper_20 + 20750.748971191173*helper_21 + 2.8800665556144241e-11*helper_22 + 631.04526748979481*helper_23*x - 3.8592302650330823e-11*helper_23 + 3001.4156378598*helper_24*x - 1.3595489617502668e-11*helper_24 + 8191.9999999994179*helper_3*y + 2.1979455292847227e-11*helper_3 + 2.4859521848461546e-11*helper_4 + 4418.3703703701703*helper_5*x + 2.3100592312375526e-11*helper_5 + 1.4200328156145601e-12*helper_7 - 14.046639231569307*helper_8 - 3.467448550509591e-12*helper_9;
val[2] = (helper_0*(-2*helper_0*helper_21*(3242.6666666667738*helper_7 + 12288.000000000035*helper_8 + 4437.3333333335195*helper_9 - 6.9849193096162502e-10*x - 47.407407407944319*y - 3.5116598079059691) + 3*helper_10*helper_14*x*y - helper_15*helper_25 - helper_17*(750.35390946494999*helper_12 + 1181.7613168722344*helper_13 - 1.5158245029550218e-12*helper_18 - 1.9296151325165411e-11*helper_19 + 315.5226337448974*helper_20 + 1.7886729134868485e-11*helper_22 - 8.0002945283635969e-12*helper_23 + 6.3304533083336773e-12*helper_24 + 6.2148804621153866e-12*helper_3 + 5.7751480780939e-12*helper_5 - 3.3988724043756669e-12*helper_7 - 37.750342935460864*helper_8 + 7.2001663890360602e-12*helper_9 - 8.6686213762739776e-13*x + 1.0775888640473118e-12*y) - helper_3*helper_6) + 10922.666666666777*helper_1*helper_2 + helper_11*helper_7*(12970.666666667095*helper_7 + 49152.000000000138*helper_8 + 17749.333333334078*helper_9 - 2.7939677238465001e-9*x - 189.62962963177728*y - 14.046639231623876) - 4*helper_14*helper_16*x + helper_17*helper_25 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_45(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 16383.999999999827*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 13.004115225204941*y;
double helper_11 = x*y;
double helper_12 = 9727.9999999996198*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
val[0] = (8192.0000000001255*helper_0*helper_2*helper_3 + helper_1*(1847.5061728389283*helper_0*helper_11 + 923.75308641964921*helper_0*helper_6 + 1.3983481039759542e-11*helper_0*x + 13.00411522640205*helper_0*y + 26.008230452036486*helper_11 + 1.7962520360016231e-11*helper_13 + 4607.999999999317*helper_14 - 3.9790393202566968e-12*helper_2 + 1535.9999999999354*helper_3 + 1.3045564628556612e-11*helper_4 + 1.6029844118749049e-11*helper_5 + 11575.50617283778*helper_6*x + 13.004115226231306*helper_6 + 155.75308641976756*helper_7*y - 3.4106051316492766e-13*helper_7 - 9.3081098384590007e-13*x + 1.2510288066526334*y - 4.6540549192295004e-13*z + 4.6540549192295004e-13) + helper_11*helper_9*(helper_10 + 19455.999999998916*helper_11 + helper_12 + 2304.0000000001774*helper_4 - 7.307789928745587e-10*x + 1.2510288066630073) + 2*helper_13*helper_6*(4864.0000000001764*helper_4 + 4864.0000000001037*helper_6 + helper_8*x - 9.7907104645856842e-10*x - 7.5138473221159074e-10*y + 1.2510288066333635) + 16383.999999999996*helper_2*pow(y, 4) + 3*helper_3*helper_5*(8192.0000000001255*x + 8192.0000000000691*y - 5.7445959100733086e-10) + helper_4*helper_6*helper_7*(helper_8 + 9728.0000000003529*x - 9.7907104645856842e-10) + helper_9*y*(helper_10*x + helper_12*x + 9727.9999999994579*helper_14 + 768.00000000005912*helper_2 + 768.00000000001558*helper_3 - 3.6538949643727935e-10*helper_4 - 1.3008616406295914e-10*helper_6 + 1.2510288066630073*x + 1.2510288066431121*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 16383.999999999827*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 13.004115225204941*x;
double helper_11 = x*y;
double helper_12 = 9727.9999999994579*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
val[1] = (8192.0000000000691*helper_0*helper_2*helper_3 + helper_1*(1847.5061728392984*helper_0*helper_11 + 923.75308641946413*helper_0*helper_6 + 13.00411522640205*helper_0*x - 5.8964027076577107e-12*helper_0*y + 26.008230452462612*helper_11 + 11575.50617283778*helper_13 + 4607.9999999998063*helper_14 - 3.2311742677852636e-27*helper_2 + 1535.9999999997724*helper_3 + 3.0514915697160364e-11*helper_4 + 3.0514915697160428e-11*helper_5 + 13.004115226018243*helper_6 + 155.75308641976756*helper_7*x + 1.3805615087340737e-11*helper_7*y - 1.3052421969040149e-11*helper_7 - 3.3540961503409836e-12*helper_9 + 1.2510288066526334*x - 4.6762593797272905e-13*y - 3.9826441184798304e-13*z + 3.9826441184798304e-13) + helper_11*helper_9*(helper_10 + 19455.99999999924*helper_11 + helper_12 + 2304.0000000000468*helper_4 - 2.6017232812591828e-10*y + 1.2510288066431121) + 2*helper_13*helper_7*(4864.0000000001037*helper_4 + 4864.0000000001764*helper_6 + helper_8*y - 9.7907104645856842e-10*x - 7.5138473221159074e-10*y + 1.2510288066333635) + 16383.999999999996*helper_2*pow(x, 4) + 3*helper_3*helper_5*(8192.0000000001255*x + 8192.0000000000691*y - 5.7445959100733086e-10) + helper_4*helper_6*helper_7*(helper_8 + 9728.0000000002074*y - 7.5138473221159074e-10) + helper_9*x*(helper_10*y + helper_12*y + 9727.9999999996198*helper_14 + 768.00000000001558*helper_2 + 768.00000000005912*helper_3 - 1.3008616406295914e-10*helper_4 - 3.6538949643727935e-10*helper_6 + 1.2510288066630073*x + 1.2510288066431121*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(8192.0000000001255*x + 8192.0000000000691*y - 5.7445959100733086e-10);
double helper_6 = pow(y, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_6*x;
double helper_13 = helper_8*y;
double helper_14 = 9727.9999999996198*helper_12 + 9727.9999999994579*helper_13 + 768.00000000005912*helper_3 + 768.00000000001558*helper_4 - 1.3008616406295914e-10*helper_6 + 13.004115225204941*helper_7 - 3.6538949643727935e-10*helper_8 + 1.2510288066630073*x + 1.2510288066431121*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = -helper_0;
double helper_17 = 2.1373125491665397e-11*helper_3;
double helper_18 = 4.0686554262880569e-11*helper_4;
double helper_19 = 2.7966962079519085e-11*helper_8;
double helper_20 = helper_6*helper_8;
double helper_21 = helper_0*y;
double helper_22 = helper_0*x;
double helper_23 = helper_0*helper_6;
double helper_24 = helper_9*y;
double helper_25 = helper_0*helper_7;
double helper_26 = helper_7*helper_9;
double helper_27 = 3695.0123456785968*helper_0*helper_12 + 3695.0123456778565*helper_0*helper_13 - 3.9790393202566968e-12*helper_1 + 3.5925040720032462e-11*helper_10 - 1.3416384601363934e-11*helper_11*y + 52.016460904925225*helper_12 + 52.016460904072972*helper_13 - 3.2311742677852636e-27*helper_2 + 23151.012345675561*helper_20 - 1.5930576473919322e-12*helper_21 - 1.8616219676918001e-12*helper_22 - 1.1792805415315421e-11*helper_23 - 5.2209687876160594e-11*helper_24 + 52.016460905608199*helper_25 + 623.01234567907022*helper_26 + 6143.9999999990896*helper_3*y + 1.7394086171408816e-11*helper_3 + 6143.9999999997417*helper_4*x + 4.0686554262880485e-11*helper_4 + 2.7611230174681473e-11*helper_6*helper_9 - 9.352518759454581e-13*helper_6 + 5.0041152266105335*helper_7 - 1.8616219676918001e-12*helper_8 - 1.3642420526597106e-12*helper_9*x;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_20*(4864.0000000001037*helper_6 + 16383.999999999827*helper_7 + 4864.0000000001764*helper_8 - 9.7907104645856842e-10*x - 7.5138473221159074e-10*y + 1.2510288066333635) + helper_11*(helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_27) + 3*helper_14*helper_26 + helper_15*(1.7962520360016231e-11*helper_0*helper_8 + 923.75308641964921*helper_12 + 923.75308641946413*helper_13 - 2.6104843938080297e-11*helper_21 - 6.8212102632985531e-13*helper_22 + 1.3805615087340737e-11*helper_23 - 1.0062288451022951e-11*helper_24 + 311.50617283953511*helper_25 + 5.3432813729163492e-12*helper_3 + 1.0171638565720142e-11*helper_4 - 2.9482013538288553e-12*helper_6 + 13.00411522640205*helper_7 + 6.9917405198797711e-12*helper_8 - 4.6540549192295004e-13*x - 3.9826441184798304e-13*y) + helper_5) + 16383.999999999996*helper_1*helper_2 + helper_10*helper_6*(19456.000000000415*helper_6 + 65535.999999999309*helper_7 + 19456.000000000706*helper_8 - 3.9162841858342737e-9*x - 3.005538928846363e-9*y + 5.0041152265334539) + 4*helper_11*helper_14*helper_7 + helper_15*(-helper_16*helper_17 - helper_16*helper_18 - helper_16*helper_19 + helper_27))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_46(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 9557.3333333331739*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 57.064471878612494*y;
double helper_11 = x*y;
double helper_12 = 5722.0740740738311*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_5*y;
double helper_15 = -helper_0;
val[0] = -(helper_1*(837.00411522598301*helper_0*helper_11 + 9.6570644719053078*helper_0*y + 19.314128943359506*helper_11 + 7.5791225147748517e-12*helper_13 + 2047.999999999578*helper_14 - 5.0022208597517176e-12*helper_15*helper_5 - 508.57613168719013*helper_15*helper_6 - 5.6085506609334463e-12*helper_15*x - 1.9705718538414086e-12*helper_2 + 943.40740740738261*helper_3 + 3.0695446184842459e-12*helper_5 + 5795.8189300401073*helper_6*x + 57.064471879225493*helper_6 + 77.16872427983931*helper_7*y + 1.5158245029544509e-13*helper_7 + 1.8000415972580124e-13*x - 3.5116598079251702*y + 9.0002079862900619e-14*z - 9.0002079862900619e-14) + helper_11*helper_9*(helper_10 + 9557.3333333326118*helper_11 + helper_12 + 1024.0000000000914*helper_5 - 4.2746250983328041e-10*x - 3.5116598079185004) + 2*helper_13*helper_6*(2389.3333333334185*helper_5 + 3242.666666666717*helper_6 + helper_8*x - 5.8450192833940567e-10*x + 47.407407406969142*y - 3.5116598079369936) + 4778.6666666667152*helper_2*helper_4 + 10922.666666666622*helper_2*pow(y, 4) + 3*helper_4*helper_5*(4778.6666666667152*x + 5461.3333333333539*y - 3.4818488832873741e-10) + helper_5*helper_6*helper_7*(helper_8 + 4778.666666666837*x - 5.8450192833940567e-10) + helper_9*y*(helper_10*x + helper_12*x + 4778.6666666663059*helper_14 + 341.33333333336384*helper_2 + 512.00000000000853*helper_3 - 2.137312549166402e-10*helper_5 + 47.407407407345211*helper_6 - 3.5116598079185004*x - 3.5116598079314989*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 9557.3333333331739*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 57.064471878612494*x;
double helper_11 = x*y;
double helper_12 = 4778.6666666663059*helper_6;
double helper_13 = helper_7*y;
double helper_14 = helper_5*x;
double helper_15 = -helper_0;
val[1] = -(helper_1*(1017.1522633743803*helper_0*helper_11 + 9.6570644719053078*helper_0*x - 1.6607839931820349e-12*helper_0*y + 114.12894375845099*helper_11 + 1.1105563656617523e-11*helper_13 + 2830.222222222148*helper_14 - 2.2378982069392742e-11*helper_15*helper_5 - 418.50205761299151*helper_15*helper_6 - 2.1541161785235008e-27*helper_2 + 682.666666666526*helper_3 + 2.2378982069392703e-11*helper_5 + 5795.8189300401073*helper_6*y + 9.6570644716797531*helper_6 + 77.16872427983931*helper_7*x - 8.0072811664261341e-12*helper_7 - 1.9637222803496256e-12*helper_9 - 3.5116598079251702*x - 1.1766061376238719e-12*y - 1.1278897340561557e-12*z + 1.1278897340561557e-12) + helper_11*helper_9*(helper_10 + 11444.148148147662*helper_11 + helper_12 + 1536.0000000000255*helper_5 + 94.814814814690422*y - 3.5116598079314989) + 2*helper_13*helper_6*(3242.666666666717*helper_5 + 2389.3333333334185*helper_6 + helper_8*y - 5.8450192833940567e-10*x + 47.407407406969142*y - 3.5116598079369936) + 5461.3333333333539*helper_2*helper_4 + 10922.666666666622*helper_2*pow(x, 4) + 3*helper_4*helper_5*(4778.6666666667152*x + 5461.3333333333539*y - 3.4818488832873741e-10) + helper_5*helper_6*helper_7*(helper_8 + 6485.333333333434*y + 47.407407406969142) + helper_9*x*(helper_10*y + helper_12*y + 5722.0740740738311*helper_14 + 512.00000000000853*helper_2 + 341.33333333336384*helper_3 + 47.407407407345211*helper_5 - 2.137312549166402e-10*helper_6 - 3.5116598079185004*x - 3.5116598079314989*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(4778.6666666667152*x + 5461.3333333333539*y - 3.4818488832873741e-10);
double helper_6 = pow(x, 2);
double helper_7 = x*y;
double helper_8 = pow(y, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_8*x;
double helper_13 = helper_6*y;
double helper_14 = 5722.0740740738311*helper_12 + 4778.6666666663059*helper_13 + 341.33333333336384*helper_3 + 512.00000000000853*helper_4 - 2.137312549166402e-10*helper_6 + 57.064471878612494*helper_7 + 47.407407407345211*helper_8 - 3.5116598079185004*x - 3.5116598079314989*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = 6.6696278130022902e-12*helper_3;
double helper_17 = 2.9838642759190323e-11*helper_4;
double helper_18 = 1.1217101321866893e-11*helper_6;
double helper_19 = helper_9*x;
double helper_20 = helper_0*x;
double helper_21 = helper_6*helper_8;
double helper_22 = helper_0*y;
double helper_23 = helper_9*y;
double helper_24 = helper_20*y;
double helper_25 = helper_19*y;
double helper_26 = 2034.3045267487605*helper_0*helper_12 + 1674.008230451966*helper_0*helper_13 - 3.3215679863640698e-12*helper_0*helper_8 - 1.9705718538414086e-12*helper_1 + 2.2211127313235046e-11*helper_10 - 7.8548891213985022e-12*helper_11*y + 228.25788751690197*helper_12 + 38.628257886719013*helper_13 + 6.0632980118178037e-13*helper_19 - 2.1541161785235008e-27*helper_2 + 3.6000831945160248e-13*helper_20 + 11591.637860080215*helper_21 - 4.5115589362246228e-12*helper_22 - 3.2029124665704536e-11*helper_23 + 38.628257887621231*helper_24 + 308.67489711935724*helper_25 + 2730.666666666104*helper_3*y + 4.0927261579789946e-12*helper_3 + 3773.6296296295304*helper_4*x + 2.9838642759190271e-11*helper_4 + 1.5158245029549703e-11*helper_6*helper_9 + 3.6000831945160248e-13*helper_6 - 14.046639231700681*helper_7 - 2.3532122752477438e-12*helper_8;
double helper_27 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_21*(2389.3333333334185*helper_6 + 9557.3333333331739*helper_7 + 3242.666666666717*helper_8 - 5.8450192833940567e-10*x + 47.407407406969142*y - 3.5116598079369936) + helper_11*(-helper_16*helper_27 - helper_17*helper_27 - helper_18*helper_27 + helper_26) + 3*helper_14*helper_25 + helper_15*(508.57613168719013*helper_12 + 418.50205761299151*helper_13 + 3.0316490059089018e-13*helper_20 - 1.6014562332852268e-11*helper_22 - 5.8911668410488767e-12*helper_23 + 154.33744855967862*helper_24 - 7.5791225147748517e-12*helper_27*helper_6 - 1.1105563656617523e-11*helper_27*helper_8 + 1.6674069532505725e-12*helper_3 + 7.4596606897975807e-12*helper_4 + 2.8042753304667231e-12*helper_6 + 9.6570644719053078*helper_7 - 8.3039199659101745e-13*helper_8 + 9.0002079862900619e-14*x - 1.1278897340561557e-12*y) + helper_5) + 10922.666666666622*helper_1*helper_2 + helper_10*helper_6*(9557.3333333336741*helper_6 + 38229.333333332695*helper_7 + 12970.666666666868*helper_8 - 2.3380077133576227e-9*x + 189.62962962787657*y - 14.046639231747974) + 4*helper_11*helper_14*helper_7 + helper_15*(helper_0*helper_16 + helper_0*helper_17 + helper_0*helper_18 + helper_26))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_47(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 6940.4444444443361*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 504.50845907704513*y;
double helper_11 = x*y;
double helper_12 = 3267.9506172836655*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (4096.0000000000973*helper_0*helper_2*helper_3 + helper_1*(230.01371742103106*helper_0*helper_6 - 49.39734796521801*helper_0*y - 376.91815272095789*helper_11 + 9.637415694018566e-12*helper_13 + 3166.8148148144646*helper_14 + 814.00274348391872*helper_15*y + 7.8849190551277594e-12*helper_15 - 2.324264237864237e-12*helper_2 + 423.50617283943399*helper_3 + 7.8007065827414329e-12*helper_4 + 9.1602116838291118e-12*helper_5 + 4164.1262002732647*helper_6*x - 137.89117512583158*helper_6 + 34.06310013717794*helper_7*y - 2.9991921155110236e-13*helper_7 - 5.3685451146319007e-14*helper_9 - 6.6457676124924135e-13*x + 1.8728852309484418*y - 3.3228838062462068e-13*z + 3.3228838062462068e-13) + helper_11*helper_9*(-helper_10 + 9620.5432098759047*helper_11 + helper_12 + 2048.0000000001037*helper_4 - 278.12345679051163*x + 1.8728852309531629) + 2*helper_13*helper_6*(2958.2222222223354*helper_4 + 1592.8888888889528*helper_6 + helper_8*x - 366.61728395119536*x - 316.04938271654316*y + 1.8728852309409447) + 7281.7777777778374*helper_2*pow(y, 4) + 3*helper_3*helper_5*(4096.0000000000973*x + 3185.7777777778388*y - 227.55555555592261) + helper_4*helper_6*helper_7*(helper_8 + 5916.4444444446708*x - 366.61728395119536) + helper_9*y*(-helper_10*x + helper_12*x + 4810.2716049379524*helper_14 + 682.6666666667013*helper_2 + 227.55555555556398*helper_3 - 139.06172839525581*helper_4 - 88.493827160599182*helper_6 + 1.8728852309531629*x + 1.8728852309452115*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 6940.4444444443361*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 504.50845907704513*x;
double helper_11 = x*y;
double helper_12 = 4810.2716049379524*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (3185.7777777778388*helper_0*helper_2*helper_3 + helper_1*(407.00137174195936*helper_0*helper_6 - 49.39734796521801*helper_0*x - 275.78235025166316*helper_11 + 4164.1262002732647*helper_13 + 1270.518518518302*helper_14 + 460.02743484206212*helper_15*x - 2.7494970178022797e-13*helper_15 - 1.3725011068023168e-27*helper_2 + 1055.6049382714882*helper_3 + 1.1744412484137979e-11*helper_4 + 1.1744412484138014e-11*helper_5 - 188.45907636047895*helper_6 + 34.06310013717794*helper_7*x + 5.8026633338285921e-12*helper_7*y - 4.7149904770021803e-12*helper_7 - 1.0387362361809544e-12*helper_9 + 1.8728852309484418*x + 8.6500354204502473e-13*y + 4.5175538359635091e-13*z - 4.5175538359635091e-13) + helper_11*helper_9*(-helper_10 + 6535.9012345673309*helper_11 + helper_12 + 682.66666666669198*helper_4 - 176.98765432119836*y + 1.8728852309452115) + 2*helper_13*helper_7*(1592.8888888889528*helper_4 + 2958.2222222223354*helper_6 + helper_8*y - 366.61728395119536*x - 316.04938271654316*y + 1.8728852309409447) + 7281.7777777778374*helper_2*pow(x, 4) + 3*helper_3*helper_5*(4096.0000000000973*x + 3185.7777777778388*y - 227.55555555592261) + helper_4*helper_6*helper_7*(helper_8 + 3185.7777777779056*y - 316.04938271654316) + helper_9*x*(-helper_10*y + helper_12*y + 3267.9506172836655*helper_14 + 227.55555555556398*helper_2 + 682.6666666667013*helper_3 - 88.493827160599182*helper_4 - 139.06172839525581*helper_6 + 1.8728852309531629*x + 1.8728852309452115*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 3);
double helper_6 = helper_5*(4096.0000000000973*x + 3185.7777777778388*y - 227.55555555592261);
double helper_7 = pow(y, 2);
double helper_8 = x*y;
double helper_9 = pow(x, 2);
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*helper_9;
double helper_12 = helper_7*x;
double helper_13 = helper_9*y;
double helper_14 = 3267.9506172836655*helper_12 + 4810.2716049379524*helper_13 + 227.55555555556398*helper_3 + 682.6666666667013*helper_5 - 88.493827160599182*helper_7 - 504.50845907704513*helper_8 - 139.06172839525581*helper_9 + 1.8728852309531629*x + 1.8728852309452115*y;
double helper_15 = pow(helper_0, 3);
double helper_16 = pow(helper_0, 4);
double helper_17 = helper_0*y;
double helper_18 = helper_0*helper_9;
double helper_19 = helper_10*y;
double helper_20 = helper_19*x;
double helper_21 = helper_0*helper_7;
double helper_22 = 1628.0054869678374*helper_0*helper_13 + 1.2213615578438816e-11*helper_0*helper_5 - 5.4989940356045593e-13*helper_0*helper_7 - 197.58939186087204*helper_0*x*y - 1.3291535224984827e-12*helper_0*x - 2.324264237864237e-12*helper_1 + 1.1605326667657184e-11*helper_10*helper_7 - 1.1996768462044095e-12*helper_10*x - 1.8859961908008721e-11*helper_10*y + 1.9274831388037132e-11*helper_11 - 2.1474180458527603e-13*helper_15*x - 4.1549449447238177e-12*helper_15*y + 1.8070215343854037e-12*helper_17 + 1.5769838110255519e-11*helper_18 - 1.3725011068023168e-27*helper_2 + 136.25240054871176*helper_20 + 920.05486968412424*helper_21*x + 1694.0246913577359*helper_3*x + 1.5659216645517304e-11*helper_3 + 1.5659216645517352e-11*helper_4 + 4222.4197530859528*helper_5*y + 1.0400942110321911e-11*helper_5 + 8328.2524005465293*helper_7*helper_9 - 551.56470050332632*helper_7*x + 1.7300070840900495e-12*helper_7 + 7.4915409237937673*helper_8 - 753.83630544191578*helper_9*y - 1.3291535224984827e-12*helper_9;
double helper_23 = helper_0*x;
val[2] = -(-helper_0*(3*helper_14*helper_20 + helper_15*helper_22 + helper_16*(-1.6105635343895704e-13*helper_10*x + 230.01371742103106*helper_12 + 407.00137174195936*helper_13 - 9.4299809540043607e-12*helper_17 + 9.637415694018566e-12*helper_18 - 3.116208708542863e-12*helper_19 + 5.8026633338285921e-12*helper_21 + 68.126200274355881*helper_23*y - 5.9983842310220473e-13*helper_23 + 3.9148041613793381e-12*helper_3 + 3.0534038946097039e-12*helper_5 - 1.3747485089011398e-13*helper_7 - 49.39734796521801*helper_8 + 3.9424595275638797e-12*helper_9 - 3.3228838062462068e-13*x + 4.5175538359635091e-13*y) + 2*helper_21*helper_9*(1592.8888888889528*helper_7 + 6940.4444444443361*helper_8 + 2958.2222222223354*helper_9 - 366.61728395119536*x - 316.04938271654316*y + 1.8728852309409447) + helper_3*helper_6) + 7281.7777777778374*helper_1*helper_2 + helper_11*helper_7*(6371.5555555558112*helper_7 + 27761.777777777344*helper_8 + 11832.888888889342*helper_9 - 1466.4691358047814*x - 1264.1975308661727*y + 7.4915409237637789) + 4*helper_14*helper_15*x*y + helper_16*helper_22 + 4*helper_4*helper_6)/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_48(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(x, 2);
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 9557.3333333331084*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 57.06447187840142*y;
double helper_11 = x*y;
double helper_12 = 4778.6666666662768*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_5*y;
double helper_15 = -helper_0;
val[0] = -(helper_1*(1017.152263374046*helper_0*helper_11 + 9.6570644719010446*helper_0*y + 114.12894375809424*helper_11 + 1.019286712646712e-11*helper_13 + 2830.2222222217142*helper_14 - 8.2265058962452005e-12*helper_15*helper_5 - 418.50205761304238*helper_15*helper_6 - 7.9857108580152481e-12*helper_15*x - 2.6526928801711209e-12*helper_2 + 682.66666666658023*helper_3 + 6.3435676464809286e-12*helper_5 + 5795.8189300397025*helper_6*x + 9.6570644717534861*helper_6 + 77.168724279828524*helper_7*y - 1.1118677994767512e-13*helper_7 - 7.1054273576010473e-14*helper_9 - 2.8158545454203058e-13*x - 3.5116598079158297*y - 1.4079272727101529e-13*z + 1.4079272727101529e-13) + helper_11*helper_9*(helper_10 + 11444.148148147202*helper_11 + helper_12 + 1536.0000000001057*helper_5 + 94.814814814311305*x - 3.5116598079091266) + 2*helper_13*helper_6*(3242.6666666667716*helper_5 + 2389.3333333333935*helper_6 + helper_8*x + 47.407407406674288*x - 6.1127281014705659e-10*y - 3.5116598079273431) + 5461.333333333404*helper_2*helper_4 + 10922.66666666665*helper_2*pow(y, 4) + 3*helper_4*helper_5*(5461.333333333404*x + 4778.6666666667024*y - 4.5831795971564993e-10) + helper_5*helper_6*helper_7*(helper_8 + 6485.3333333335431*x + 47.407407406674288) + helper_9*y*(helper_10*x + helper_12*x + 5722.074074073601*helper_14 + 512.00000000003524*helper_2 + 341.33333333334213*helper_3 + 47.407407407155652*helper_5 - 1.2461498499760637e-10*helper_6 - 3.5116598079091266*x - 3.5116598079214634*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = helper_0*helper_3;
double helper_5 = pow(y, 2);
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 9557.3333333331084*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 57.06447187840142*x;
double helper_11 = x*y;
double helper_12 = 5722.074074073601*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_5*x;
double helper_15 = -helper_0;
val[1] = -(helper_1*(837.00411522608476*helper_0*helper_11 + 9.6570644719010446*helper_0*x + 19.314128943506972*helper_11 + 5795.8189300397025*helper_13 + 2047.9999999997408*helper_14 - 1.8372705277599101e-11*helper_15*helper_5 - 508.57613168702301*helper_15*helper_6 - 2.2468720985353936e-13*helper_15*y - 2.069971015299931e-27*helper_2 + 943.407407407238*helper_3 + 1.8372705277599055e-11*helper_5 + 57.064471879047119*helper_6 + 77.168724279828524*helper_7*x + 9.7127025215458309e-12*helper_7*y - 6.4164960496234201e-12*helper_7 - 1.3057794443339879e-12*helper_9 - 3.5116598079158297*x + 6.732721376292773e-13*y + 4.5484420984428143e-14*z - 4.5484420984428143e-14) + helper_11*helper_9*(helper_10 + 9557.3333333325536*helper_11 + helper_12 + 1024.0000000000264*helper_5 - 2.4922996999521274e-10*y - 3.5116598079214634) + 2*helper_13*helper_7*(2389.3333333333935*helper_5 + 3242.6666666667716*helper_6 + helper_8*y + 47.407407406674288*x - 6.1127281014705659e-10*y - 3.5116598079273431) + 4778.6666666667024*helper_2*helper_4 + 10922.66666666665*helper_2*pow(x, 4) + 3*helper_4*helper_5*(5461.333333333404*x + 4778.6666666667024*y - 4.5831795971564993e-10) + helper_5*helper_6*helper_7*(helper_8 + 4778.666666666787*y - 6.1127281014705659e-10) + helper_9*x*(helper_10*y + helper_12*y + 4778.6666666662768*helper_14 + 341.33333333334213*helper_2 + 512.00000000003524*helper_3 - 1.2461498499760637e-10*helper_5 + 47.407407407155652*helper_6 - 3.5116598079091266*x - 3.5116598079214634*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(5461.333333333404*x + 4778.6666666667024*y - 4.5831795971564993e-10);
double helper_6 = pow(y, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = helper_6*x;
double helper_12 = helper_8*y;
double helper_13 = 4778.6666666662768*helper_11 + 5722.074074073601*helper_12 + 512.00000000003524*helper_3 + 341.33333333334213*helper_4 - 1.2461498499760637e-10*helper_6 + 57.06447187840142*helper_7 + 47.407407407155652*helper_8 - 3.5116598079091266*x - 3.5116598079214634*y;
double helper_14 = pow(helper_0, 3);
double helper_15 = helper_14*x;
double helper_16 = pow(helper_0, 4);
double helper_17 = 4.4937441970707872e-13*helper_6;
double helper_18 = 2.4496940370132134e-11*helper_4;
double helper_19 = 1.5971421716030496e-11*helper_8;
double helper_20 = 1.0968674528326935e-11*helper_3;
double helper_21 = helper_0*y;
double helper_22 = helper_6*helper_8;
double helper_23 = helper_0*x;
double helper_24 = helper_9*y;
double helper_25 = helper_9*x;
double helper_26 = helper_21*x;
double helper_27 = helper_24*x;
double helper_28 = 1674.0082304521695*helper_0*helper_11 - 2.6526928801711209e-12*helper_1 + 2.0385734252934241e-11*helper_10 + 38.628257887013945*helper_11 + 228.25788751618848*helper_12 - 5.2231177773359516e-12*helper_14*y - 2.8421709430404189e-13*helper_15 - 2.069971015299931e-27*helper_2 + 2034.304526748092*helper_21*helper_8 + 1.8193768393771257e-13*helper_21 + 11591.637860079405*helper_22 - 5.6317090908406116e-13*helper_23 - 2.566598419849368e-11*helper_24 - 4.447471197907005e-13*helper_25 + 38.628257887604178*helper_26 + 308.6748971193141*helper_27 + 3773.629629628952*helper_3*y + 8.4580901953079053e-12*helper_3 + 2730.6666666663209*helper_4*x + 2.4496940370132073e-11*helper_4 + 1.9425405043091662e-11*helper_6*helper_9 + 1.3465442752585546e-12*helper_6 - 14.046639231663319*helper_7 - 5.6317090908406116e-13*helper_8;
double helper_29 = -helper_0;
val[2] = (4*helper_0*helper_5 - helper_0*(2*helper_0*helper_22*(2389.3333333333935*helper_6 + 9557.3333333331084*helper_7 + 3242.6666666667716*helper_8 + 47.407407406674288*x - 6.1127281014705659e-10*y - 3.5116598079273431) + 3*helper_13*helper_27 + helper_14*(-helper_17*helper_29 - helper_18*helper_29 - helper_19*helper_29 - helper_20*helper_29 + helper_28) + helper_16*(418.50205761304238*helper_11 + 508.57613168702301*helper_12 - 1.283299209924684e-11*helper_21 - 2.2237355989535025e-13*helper_23 - 3.9173383330019637e-12*helper_24 - 2.1316282072803142e-13*helper_25 + 154.33744855965705*helper_26 - 9.7127025215458309e-12*helper_29*helper_6 - 1.019286712646712e-11*helper_29*helper_8 + 2.7421686320817338e-12*helper_3 + 6.1242350925330335e-12*helper_4 + 1.1234360492676968e-13*helper_6 + 9.6570644719010446*helper_7 + 3.992855429007624e-12*helper_8 - 1.4079272727101529e-13*x + 4.5484420984428143e-14*y) + helper_5) + 10922.66666666665*helper_1*helper_2 + helper_10*helper_6*(9557.333333333574*helper_6 + 38229.333333332434*helper_7 + 12970.666666667086*helper_8 + 189.62962962669715*x - 2.4450912405882264e-9*y - 14.046639231709372) + 4*helper_13*helper_15*y + helper_16*(helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_0*helper_20 + helper_28))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_49(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 3);
double helper_4 = pow(x, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(y, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 5802.6666666665324*y;
double helper_9 = pow(helper_0, 3);
double helper_10 = 443.63968907129669*y;
double helper_11 = x*y;
double helper_12 = 3103.6049382714004*helper_6;
double helper_13 = helper_7*x;
double helper_14 = helper_4*y;
double helper_15 = helper_0*x;
val[0] = (3185.7777777778037*helper_0*helper_2*helper_3 + helper_1*(324.82853223587227*helper_0*helper_6 + 39.096479195247909*helper_0*y + 255.18061271117946*helper_11 + 4.6604585759832735e-12*helper_13 + 1460.1481481478427*helper_14 + 649.65706447160994*helper_15*y + 3.4448164485556569e-12*helper_15 - 1.313714569227601e-12*helper_2 + 486.71604938267774*helper_3 + 1.6426695394871932e-12*helper_4 + 2.8695399965662968e-12*helper_5 + 3671.0891632365115*helper_6*x + 127.59030635566631*helper_6 + 65.66803840876986*helper_7*y + 8.8510817331080975e-14*helper_7 - 4.1053580288361602e-14*helper_9 + 1.5649484451797888e-13*x + 1.8728852309280564*y + 7.8247422258989439e-14*z - 7.8247422258989439e-14) + helper_11*helper_9*(helper_10 + 6207.2098765426681*helper_11 + helper_12 + 682.66666666672052*helper_4 + 176.98765432069737*x + 1.872885230932273) + 2*helper_13*helper_6*(1592.8888888889385*helper_4 + 1592.8888888889178*helper_6 + helper_8*x + 316.04938271564748*x + 316.04938271572934*y + 1.8728852309204287) + 7281.7777777777428*helper_2*pow(y, 4) + 3*helper_3*helper_5*(3185.7777777778037*x + 3185.7777777777869*y + 227.55555555531257) + helper_4*helper_6*helper_7*(helper_8 + 3185.777777777877*x + 316.04938271564748) + helper_9*y*(helper_10*x + helper_12*x + 3103.604938271334*helper_14 + 227.55555555557351*helper_2 + 227.55555555556037*helper_3 + 88.493827160348687*helper_4 + 88.493827160433767*helper_6 + 1.872885230932273*x + 1.8728852309240795*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 4);
double helper_2 = pow(y, 3);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 2);
double helper_5 = helper_0*helper_4;
double helper_6 = pow(x, 2);
double helper_7 = pow(helper_0, 2);
double helper_8 = 5802.6666666665324*x;
double helper_9 = pow(helper_0, 3);
double helper_10 = 443.63968907129669*x;
double helper_11 = x*y;
double helper_12 = 3103.604938271334*helper_6;
double helper_13 = helper_6*y;
double helper_14 = helper_4*x;
double helper_15 = helper_0*y;
val[1] = (3185.7777777777869*helper_0*helper_2*helper_3 + helper_1*(324.82853223580497*helper_0*helper_6 + 39.096479195247909*helper_0*x + 255.18061271133263*helper_11 + 3671.0891632365115*helper_13 + 1460.1481481480332*helper_14 + 649.65706447174455*helper_15*x + 6.3861307645809364e-13*helper_15 - 1.3874602469309339e-27*helper_2 + 486.71604938261419*helper_3 + 1.2841477309952282e-11*helper_4 + 1.2841477309952308e-11*helper_5 + 127.59030635558973*helper_6 + 65.66803840876986*helper_7*x + 7.0905001120621918e-12*helper_7*y - 3.9831588720833393e-12*helper_7 - 7.6050612233558798e-13*helper_9 + 1.8728852309280564*x + 2.282399235290374e-14*y - 2.6129289252044484e-13*z + 2.6129289252044484e-13) + helper_11*helper_9*(helper_10 + 6207.2098765428009*helper_11 + helper_12 + 682.66666666668107*helper_4 + 176.98765432086753*y + 1.8728852309240795) + 2*helper_13*helper_7*(1592.8888888889178*helper_4 + 1592.8888888889385*helper_6 + helper_8*y + 316.04938271564748*x + 316.04938271572934*y + 1.8728852309204287) + 7281.7777777777428*helper_2*pow(x, 4) + 3*helper_3*helper_5*(3185.7777777778037*x + 3185.7777777777869*y + 227.55555555531257) + helper_4*helper_6*helper_7*(helper_8 + 3185.7777777778356*y + 316.04938271572934) + helper_9*x*(helper_10*y + helper_12*y + 3103.6049382714004*helper_14 + 227.55555555556037*helper_2 + 227.55555555557351*helper_3 + 88.493827160433767*helper_4 + 88.493827160348687*helper_6 + 1.872885230932273*x + 1.8728852309240795*y))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(x, 4);
double helper_2 = pow(y, 4);
double helper_3 = pow(x, 3);
double helper_4 = pow(y, 3);
double helper_5 = helper_3*helper_4*(3185.7777777778037*x + 3185.7777777777869*y + 227.55555555531257);
double helper_6 = pow(y, 2);
double helper_7 = x*y;
double helper_8 = pow(x, 2);
double helper_9 = pow(helper_0, 2);
double helper_10 = helper_8*helper_9;
double helper_11 = pow(helper_0, 3);
double helper_12 = helper_6*x;
double helper_13 = helper_8*y;
double helper_14 = 3103.6049382714004*helper_12 + 3103.604938271334*helper_13 + 227.55555555557351*helper_3 + 227.55555555556037*helper_4 + 88.493827160433767*helper_6 + 443.63968907129669*helper_7 + 88.493827160348687*helper_8 + 1.872885230932273*x + 1.8728852309240795*y;
double helper_15 = pow(helper_0, 4);
double helper_16 = -helper_0;
double helper_17 = 1.2772261529161873e-12*helper_6;
double helper_18 = 6.8896328971113137e-12*helper_8;
double helper_19 = 3.8260533287550625e-12*helper_3;
double helper_20 = 1.7121969746603079e-11*helper_4;
double helper_21 = helper_9*x;
double helper_22 = helper_0*x;
double helper_23 = helper_6*helper_8;
double helper_24 = helper_9*y;
double helper_25 = helper_0*y;
double helper_26 = helper_21*y;
double helper_27 = helper_22*y;
double helper_28 = 1299.3141289432199*helper_0*helper_13 - 1.313714569227601e-12*helper_1 + 9.320917151966547e-12*helper_10 - 1.6421432115344641e-13*helper_11*x - 3.0420244893423519e-12*helper_11*y + 510.36122542266526*helper_12 + 510.36122542235893*helper_13 - 1.3874602469309339e-27*helper_2 + 3.540432693243239e-13*helper_21 + 1299.3141289434891*helper_22*helper_6 + 3.1298968903595776e-13*helper_22 + 7342.1783264730229*helper_23 - 1.5932635488333357e-11*helper_24 - 1.0451715700817793e-12*helper_25 + 262.67215363507944*helper_26 + 156.38591678099164*helper_27 + 1946.8641975304567*helper_3*y + 2.190226052649591e-12*helper_3 + 1946.864197530711*helper_4*x + 1.7121969746603043e-11*helper_4 + 1.4181000224124384e-11*helper_6*helper_9 + 4.564798470580748e-14*helper_6 + 7.4915409237122255*helper_7 + 3.1298968903595776e-13*helper_8;
val[2] = -(4*helper_0*helper_5 - helper_0*(2*helper_0*helper_23*(1592.8888888889178*helper_6 + 5802.6666666665324*helper_7 + 1592.8888888889385*helper_8 + 316.04938271564748*x + 316.04938271572934*y + 1.8728852309204287) + helper_11*(helper_0*helper_17 + helper_0*helper_18 + helper_0*helper_19 + helper_0*helper_20 + helper_28) + 3*helper_14*helper_26 + helper_15*(7.0905001120621918e-12*helper_0*helper_6 + 4.6604585759832735e-12*helper_0*helper_8 + 324.82853223587227*helper_12 + 324.82853223580497*helper_13 - 1.2316074086508481e-13*helper_21 + 1.7702163466216195e-13*helper_22 - 2.2815183670067639e-12*helper_24 - 7.9663177441666785e-12*helper_25 + 131.33607681753972*helper_27 + 9.5651333218876562e-13*helper_3 + 4.2804924366507697e-12*helper_4 + 3.1930653822904682e-13*helper_6 + 39.096479195247909*helper_7 + 1.7224082242778284e-12*helper_8 + 7.8247422258989439e-14*x - 2.6129289252044484e-13*y) + helper_5) + 7281.7777777777428*helper_1*helper_2 + helper_10*helper_6*(6371.5555555556712*helper_6 + 23210.666666666129*helper_7 + 6371.5555555557539*helper_8 + 1264.1975308625899*x + 1264.1975308629173*y + 7.4915409236817148) + 4*helper_11*helper_14*helper_7 + helper_15*(-helper_16*helper_17 - helper_16*helper_18 - helper_16*helper_19 - helper_16*helper_20 + helper_28))/pow(helper_0, 5);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_50(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 2.6651694767140311e-11*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = 1080.0000000000221*y;
double helper_9 = x*y;
double helper_10 = 3.6936433404046677e-11*helper_5;
double helper_11 = helper_2*y;
double helper_12 = helper_7*y;
double helper_13 = 1.0204584621911641e-25*helper_2;
val[0] = (helper_1*(helper_0*helper_13 + 1295.9999999999939*helper_0*helper_9 + 5.1022923109558165e-26*helper_0*x + 1079.9999999999734*helper_0*y + 3.5226094258341906e-11*helper_11 + 647.99999999998488*helper_12 + helper_13 + 9.7091990209407227e-12*helper_3 + 1296.0000000000675*helper_5*x + 1079.9999999999832*helper_5 + 647.99999999999466*helper_6 + 6.8030564146077603e-26*helper_7*x - 8.5038205182597004e-27*helper_7 + 2159.9999999999709*helper_9 - 1.7007641036519401e-26*x + 431.99999999998823*y - 8.5038205182597004e-27*z + 8.5038205182597004e-27) + helper_12*(helper_10*x + 3.8297802917259932e-11*helper_11 + 1.1742031419447303e-11*helper_2 + 9.7091990209407227e-12*helper_5 + helper_8*x + 431.99999999998835*x + 431.99999999998846*y) + 7.9955084301420925e-11*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 3.8297802917259932e-11) + 2*helper_6*x*(helper_4*x + 3.8297802917259932e-11*x + 3.6936433404046677e-11*y + 431.99999999998829) + helper_7*helper_9*(helper_10 + helper_8 + 7.6595605834519865e-11*helper_9 + 2.3484062838894606e-11*x + 431.99999999998835))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 2.6651694767140311e-11*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = pow(helper_0, 2);
double helper_8 = 1080.0000000000221*x;
double helper_9 = x*y;
double helper_10 = 3.8297802917259932e-11*helper_5;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 1.7266188478970984e-12*helper_3;
val[1] = (helper_1*(-helper_0*helper_13 + 1295.9999999999893*helper_0*helper_9 + 1079.9999999999734*helper_0*x - 1.7266188478970816e-12*helper_0*y - 2.8776980798283396e-13*helper_1 + 2.9127597062822167e-11*helper_11 + 647.99999999998488*helper_12 - helper_13 + 1.1742031419447303e-11*helper_2 + 1296.0000000000675*helper_5*y + 1079.9999999999854*helper_5 + 647.99999999999693*helper_6 - 1.7266188478970669e-12*helper_7*y - 3.5971225997854219e-13*helper_7 + 2159.9999999999663*helper_9 + 431.99999999998823*x + 2.2648027683336887e-26*y + 7.1942451995741004e-14*z - 7.1942451995741004e-14) + helper_12*(helper_10*y + 3.6936433404046677e-11*helper_11 + 9.7091990209407227e-12*helper_3 + 1.1742031419447303e-11*helper_5 + helper_8*y + 431.99999999998835*x + 431.99999999998846*y) + 7.9955084301420925e-11*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 3.6936433404046677e-11) + 2*helper_6*y*(helper_4*y + 3.8297802917259932e-11*x + 3.6936433404046677e-11*y + 431.99999999998829) + helper_7*helper_9*(helper_10 + helper_8 + 7.3872866808093355e-11*helper_9 + 1.9418398041881445e-11*y + 431.99999999998846))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = x*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = pow(y, 2);
double helper_8 = pow(y, 3);
double helper_9 = 5.7553961596569933e-13*helper_8;
double helper_10 = helper_7*x;
double helper_11 = helper_8*x;
double helper_12 = helper_1*x;
double helper_13 = helper_2*x;
double helper_14 = pow(z, 5);
double helper_15 = helper_5*y;
double helper_16 = helper_6*y;
double helper_17 = helper_0*y;
double helper_18 = helper_2*y;
double helper_19 = helper_5*helper_7;
double helper_20 = 3.6936433404046677e-11*helper_5;
double helper_21 = 3.8297802917259932e-11*helper_6*helper_7;
double helper_22 = helper_6*helper_8;
double helper_23 = helper_0*helper_8;
val[2] = -(-3455.99999999998*helper_0*helper_10 + 9.7091990209407114e-12*helper_0*helper_11 + 1080.0000000000227*helper_0*helper_19 + helper_0*helper_21 + 6479.9999999998563*helper_0*helper_3 + 5.2723687213210162e-25*helper_0*helper_5 - 2.0409169243823282e-25*helper_0*helper_6 - 1.2086331935279424e-11*helper_0*helper_7 - 1.190534872556358e-25*helper_0*x + 2591.9999999999873*helper_1*helper_15 - 8639.9999999998035*helper_1*helper_3 - 5.7825979524165969e-25*helper_1*helper_5 + 1.3606112829215518e-25*helper_1*helper_6 + 1.3812950783176506e-11*helper_1*helper_7 - 2.3021584638627973e-12*helper_1*helper_8 - 9.7841734714162257e-12*helper_1*y - 216.00000000000614*helper_10 + 9.7091990209407276e-12*helper_11 + 2591.9999999999782*helper_12*helper_7 + 1.3606112829215523e-25*helper_12 - 647.99999999999454*helper_13*helper_7 - 7.653438466433733e-26*helper_13 - 1295.99999999997*helper_14*helper_3 - 6.8030564146077591e-26*helper_14*helper_5 + 1.7266188478970685e-12*helper_14*helper_7 + 1.7007641036519398e-26*helper_14*x - 4.4604320237339279e-12*helper_14*y + 1728.0000000000118*helper_15*z - 216.00000000000864*helper_15 - 2.34840628388946e-11*helper_16*z + 1.17420314194473e-11*helper_16 - 3455.9999999999932*helper_17*helper_5 + 1.1742031419447306e-11*helper_17*helper_6 + 5.3237414476822372e-12*helper_17 - 647.99999999999682*helper_18*helper_5 + 9.2805763074463662e-12*helper_18 - 1296.0000000000682*helper_19*z + 216.00000000004547*helper_19 + 5399.9999999998745*helper_2*helper_3 + 3.1464135917560892e-25*helper_2*helper_5 - 3.4015282073038796e-26*helper_2*helper_6 - 7.7697848155367941e-12*helper_2*helper_7 + helper_2*helper_9 + helper_20*helper_23 - helper_20*helper_8 - helper_21 + 5.3303389534280615e-11*helper_22*z + 2.6651694767140317e-11*helper_22 + 3.453237695794196e-12*helper_23 - 2159.9999999999545*helper_3*z + 215.99999999999636*helper_3 + 1728.0000000000018*helper_4*helper_7 - 1.9418398041881436e-11*helper_4*helper_8 + 5.1022923109558205e-26*helper_4 - 2.3810697451127168e-25*helper_5*z + 4.2519102591298523e-26*helper_5 + 1.3606112829215523e-25*helper_6*z - 3.4015282073038801e-26*helper_6 + 5.1798565436911707e-12*helper_7*z - 8.6330942394852627e-13*helper_7 - 2.3021584638627981e-12*helper_8*z + helper_9 - 8.5038205182597004e-27*x + 8.6330942394850234e-13*y*pow(z, 6) - 1.2949641359226249e-12*y*z + 7.1942451995676179e-14*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_51(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 1845.2812500004031*y;
double helper_6 = helper_4*x;
double helper_7 = 5211.2109375004611*y;
double helper_8 = x*y;
double helper_9 = 3075.468750000467*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 8.1946949226376042e-12*helper_2;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 3075.4687500004652) + 2*helper_0*helper_6*(helper_5*x + 3075.4687500004652*x + 3075.468750000467*y + 34.171874999978037) + helper_1*(-6.8289124355313368e-13*helper_10 + 5.4631299484250695e-12*helper_11 + 3690.5625000002019*helper_12 + 871.38281249996476*helper_13 - helper_14*helper_16 + helper_14 + 4203.1406250000537*helper_15*x + 905.55468749994839*helper_15 - 2101.5703125000305*helper_16*helper_4 - 4.0973474613187989e-12*helper_16*x + 1230.1875000000659*helper_3 + 2135.7421875000064*helper_4 + 10354.078125000971*helper_6 + 4271.4843750000191*helper_8 - 1.3657824871062674e-12*x + 34.171874999983295*y - 6.8289124355313368e-13*z + 6.8289124355313368e-13) + helper_11*y*(helper_7 + 6150.9375000009304*helper_8 + helper_9 + 2460.3750000001346*x + 34.171874999981746) + helper_13*(3075.4687500004652*helper_12 + 1230.1875000000673*helper_2 + 1230.1875000000659*helper_4 + helper_7*x + helper_9*x + 34.171874999981746*x + 34.171874999981839*y) + 5535.8437500012096*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 1845.2812500004031*x;
double helper_6 = helper_4*y;
double helper_7 = pow(helper_0, 2);
double helper_8 = 5211.2109375004611*x;
double helper_9 = x*y;
double helper_10 = 3075.4687500004652*helper_4;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 1.0926259896849237e-11*helper_3;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 3075.468750000467) + 2*helper_0*helper_6*(helper_5*y + 3075.4687500004652*x + 3075.468750000467*y + 34.171874999978037) + helper_1*(-helper_0*helper_13 + 2101.5703125000268*helper_0*helper_4 + 4203.1406250000609*helper_0*helper_9 + 905.55468749994839*helper_0*x - 1.0106790404585388e-11*helper_0*y - 6.6012820210119113e-13*helper_1 + 3690.5625000001978*helper_11 + 871.38281249996476*helper_12 - helper_13 + 1230.1875000000673*helper_2 + 2135.7421875000095*helper_4 + 10354.078125000971*helper_6 - 8.6044296687685416e-12*helper_7*y - 8.5930481480414303e-13*helper_7 + 4271.4843750000127*helper_9 + 34.171874999983295*x + 1.5023607358169828e-12*y + 2.6575850894942591e-12*z - 2.6575850894942591e-12) + helper_12*(helper_10*y + 3075.468750000467*helper_11 + 1230.1875000000659*helper_3 + 1230.1875000000673*helper_4 + helper_8*y + 34.171874999981746*x + 34.171874999981839*y) + 5535.8437500012096*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 6150.9375000009341*helper_9 + 2460.3750000001319*y + 34.171874999981839))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 2.7315649742125343e-12*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = helper_9*x;
double helper_11 = helper_0*x;
double helper_12 = 1.0926259896850139e-11*helper_1;
double helper_13 = pow(z, 5);
double helper_14 = helper_13*x;
double helper_15 = 1230.1875000000673*helper_6*y;
double helper_16 = helper_6*z;
double helper_17 = helper_8*z;
double helper_18 = helper_5*helper_8;
double helper_19 = helper_0*helper_5;
double helper_20 = helper_1*helper_5;
double helper_21 = helper_2*helper_5;
double helper_22 = helper_6*helper_8;
double helper_23 = helper_1*helper_8;
double helper_24 = helper_2*helper_8;
val[2] = (1230.1875000000668*helper_0*helper_10 + helper_0*helper_15 + 3075.4687500004648*helper_0*helper_22 + 11994.328124999603*helper_0*helper_3 - 1.6389389845275212e-11*helper_0*helper_6 - 5.5723925473929242e-11*helper_0*helper_8 + 2.1852519793698475e-11*helper_0*helper_9 - 3.4258377384948141e-12*helper_0*y - 13805.437499999502*helper_1*helper_3 - 1.4568346529132312e-11*helper_1*helper_9 - 1.179125547201157e-11*helper_1*y - 2460.3750000001319*helper_10*z + 1230.1875000000659*helper_10 - 12575.2500000002*helper_11*helper_8 - 9.5604774097438691e-12*helper_11 + helper_12*helper_6 + helper_12*x - 5.4631299484250687e-12*helper_13*helper_5 + 8.6044296687685368e-12*helper_13*helper_8 - 1.0163698008213156e-11*helper_13*y - 1742.7656249999291*helper_14*y + 1.3657824871062682e-12*helper_14 + helper_15 + 3690.5625000008067*helper_16*helper_9 + 1.0926259896850139e-11*helper_16 + 8337.9375000001564*helper_17*x + 2.2808567534671928e-11*helper_17 - 10354.078125000964*helper_18*z + 5142.8671875005039*helper_18 + 5211.2109375004584*helper_19*helper_8 + 3075.4687500004666*helper_19*helper_9 - 12575.250000000178*helper_19*y + 4.2339257100294289e-11*helper_19 + 7808.2734374996962*helper_2*helper_3 - helper_2*helper_7 + 3.6420866322830781e-12*helper_2*helper_9 - 6.1460211919782068e-12*helper_2*x + 1.8455135857017915e-11*helper_2*y + 8406.2812500001091*helper_20*y - 4.6436604561613089e-11*helper_20 - 2101.5703125000273*helper_21*y + 2.5266976011465944e-11*helper_21 - 3075.4687500004657*helper_22 + 8406.2812500001201*helper_23*x + 6.5830715878514653e-11*helper_23 - 2101.5703125000309*helper_24*x - 3.7968753141550012e-11*helper_24 + 837.21093749998113*helper_3 + 8337.9375000001437*helper_4*helper_5 - 2460.3750000001346*helper_4*helper_6 - 5091.6093749998536*helper_4*x + 7.3410808681970228e-12*helper_4 - 3075.4687500004675*helper_5*helper_9 - 2067.398437500045*helper_5*y - 1.9120954819487745e-11*helper_5*z + 3.4144562177656704e-12*helper_5 + 1845.2812500004029*helper_6*helper_9 - helper_7 - 2067.3984375000487*helper_8*x - 3.5510344664758463e-12*helper_8 - 1.4568346529132315e-11*helper_9*z + 3.6420866322830789e-12*helper_9 + 4.0973474613188005e-12*x*z - 6.8289124355313409e-13*x + 1.980384606303573e-12*y*pow(z, 6) - 2.395810112798971e-12*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_52(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 1845.2812500003249*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 4015.1953125003497*y;
double helper_8 = x*y;
double helper_9 = 3075.4687500003683*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*y;
double helper_12 = helper_2*y;
double helper_13 = 6.5557559381100497e-12*helper_2;
double helper_14 = helper_0*y;
val[0] = (helper_1*(helper_0*helper_13 + 3.2778779690550224e-12*helper_0*x + 4.3705039587400334e-12*helper_10*x - 5.4631299484250418e-13*helper_10 + 358.80468749997112*helper_11 + 1845.2812500001166*helper_12 + helper_13 + 1947.7968750000086*helper_14*x + 324.63281249996351*helper_14 + 1230.1875000000464*helper_3 + 8098.7343750007321*helper_5*x + 1554.8203125000025*helper_5 + 1588.9921875000173*helper_6 + 1879.4531249999902*helper_8 - 1.0926259896850084e-12*x - 34.171875000008392*y - 5.4631299484250418e-13*z + 5.4631299484250418e-13) + helper_11*x*(helper_7 + 4920.7500000007185*helper_8 + helper_9 + 1230.1875000000778*x - 34.171875000009962) + helper_11*(2460.3750000003593*helper_12 + 615.09375000003888*helper_2 + 1230.1875000000464*helper_5 + helper_7*x + helper_9*x - 34.171875000009962*x - 34.17187500000972*y) + 5535.843750000975*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 2460.3750000003593) + 2*helper_6*x*(helper_4*x + 2460.3750000003593*x + 3075.4687500003683*y - 34.171875000013337))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 1845.2812500003249*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 4015.1953125003497*x;
double helper_8 = x*y;
double helper_9 = 2460.3750000003593*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 6.4874668137543122e-12*helper_3;
double helper_14 = helper_0*x;
val[1] = (helper_1*(-helper_0*helper_13 - 7.0679243707742317e-12*helper_0*y - 3.6420866322821914e-13*helper_1 - 5.0533952022926473e-12*helper_10*y - 7.4548960754531664e-13*helper_10 + 358.80468749997112*helper_11 + 3690.5625000001392*helper_12 - helper_13 + 3177.9843750000346*helper_14*y + 324.63281249996351*helper_14 + 615.09375000003888*helper_2 + 8098.7343750007321*helper_5*y + 939.72656249999511*helper_5 + 973.89843750000432*helper_6 + 3109.640625000005*helper_8 - 34.171875000008392*x + 9.2190317879683853e-13*y + 1.8153525557787645e-12*z - 1.8153525557787645e-12) + helper_11*y*(helper_7 + 6150.9375000007367*helper_8 + helper_9 + 2460.3750000000928*y - 34.17187500000972) + helper_11*(3075.4687500003683*helper_12 + 1230.1875000000464*helper_3 + 615.09375000003888*helper_5 + helper_7*y + helper_9*y - 34.171875000009962*x - 34.17187500000972*y) + 5535.843750000975*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 3075.4687500003683) + 2*helper_6*y*(helper_4*y + 2460.3750000003593*x + 3075.4687500003683*y - 34.171875000013337))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 2.1852519793700167e-12*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 2.1624889379181041e-12*helper_9;
double helper_11 = helper_9*x;
double helper_12 = helper_0*x;
double helper_13 = helper_1*x;
double helper_14 = pow(z, 5);
double helper_15 = helper_6*y;
double helper_16 = helper_1*y;
double helper_17 = helper_8*z;
double helper_18 = helper_0*helper_5;
double helper_19 = helper_2*helper_5;
double helper_20 = 2460.3750000003593*helper_6*helper_8;
double helper_21 = helper_6*helper_9;
double helper_22 = helper_2*helper_8;
val[2] = -(1230.1875000000446*helper_0*helper_11 + 615.09375000003843*helper_0*helper_15 + helper_0*helper_20 + 5228.2968749996407*helper_0*helper_3 - 1.3111511876220098e-11*helper_0*helper_6 - 2.9330178910604062e-11*helper_0*helper_8 + 1.2974933627508621e-11*helper_0*helper_9 - 9.4125176403090563e-12*helper_0*y - 3.7149283649290293e-11*helper_1*helper_5 + 8.7410079174800685e-12*helper_1*helper_6 + 3.6398103281378005e-11*helper_1*helper_8 - 8.6499557516724162e-12*helper_1*helper_9 + helper_10*helper_2 + helper_10 - 2460.3750000000919*helper_11*z + 1230.1875000000464*helper_11 - 9568.1250000001091*helper_12*helper_8 - 7.6483819277950591e-12*helper_12 + 6355.9687500000691*helper_13*helper_8 - 5877.5624999995689*helper_13*y + 8.7410079174800653e-12*helper_13 - 717.60937499994225*helper_14*helper_3 - 4.3705039587400342e-12*helper_14*helper_5 + 5.0533952022926457e-12*helper_14*helper_8 + 1.0926259896850084e-12*helper_14*x - 5.0647767230173106e-12*helper_14*y + 615.09375000003888*helper_15 + 3895.5937500000182*helper_16*helper_5 + 3.1868258032824331e-13*helper_16 - 8098.7343750007276*helper_17*helper_5 + 6424.3125000000873*helper_17*x + 1.1131127269914775e-11*helper_17 + 4015.1953125003511*helper_18*helper_8 + 3075.4687500003693*helper_18*helper_9 - 5877.5625000000355*helper_18*y + 3.3871405680235271e-11*helper_18 - 973.89843750000455*helper_19*y + 2.0213580809172657e-11*helper_19 + 3263.4140624997481*helper_2*helper_3 - helper_2*helper_7 - 4.9168169535825381e-12*helper_2*x + 7.1191412140379301e-12*helper_2*y - helper_20 + 3690.5625000006503*helper_21*z + 1845.2812500003247*helper_21 - 1588.9921875000173*helper_22*x - 2.1733013826076119e-11*helper_22 + 392.97656249997874*helper_3 + 3963.9375000000368*helper_4*helper_5 - 1230.1875000000778*helper_4*helper_6 - 2289.5156249998572*helper_4*x + 8.1605503604602742e-12*helper_4 + 4083.5390625003774*helper_5*helper_8 - 3075.4687500003683*helper_5*helper_9 - 1008.0703125000143*helper_5*y - 1.5296763855590125e-11*helper_5*z + 2.7315649742125226e-12*helper_5 + 8.7410079174800653e-12*helper_6*z - helper_7 - 1623.1640625000268*helper_8*x - 1.5194330169055311e-12*helper_8 - 8.6499557516724178e-12*helper_9*z + 3.2778779690550253e-12*x*z - 5.4631299484250418e-13*x + 1.0926259896846574e-12*y*pow(z, 6) - 2.21370578118474e-12*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_53(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = 1845.2812500003135*y;
double helper_5 = pow(y, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = helper_0*x;
double helper_8 = 4015.1953125003411*y;
double helper_9 = x*y;
double helper_10 = 2460.3750000003565*helper_5;
double helper_11 = pow(helper_0, 2);
double helper_12 = helper_11*y;
double helper_13 = helper_2*y;
double helper_14 = 8.1946949226372859e-12*helper_2;
val[0] = (helper_1*(helper_0*helper_14 + 324.63281249996822*helper_0*y + 5.463129948424857e-12*helper_11*x - 6.8289124355310713e-13*helper_11 + 358.80468749997385*helper_12 + 3690.5625000001119*helper_13 + helper_14 + 615.09375000004411*helper_3 + 8098.7343750007103*helper_5*x + 939.72656250000387*helper_5 + 973.89843750001785*helper_6 + 3177.9843750000141*helper_7*y + 4.0973474613186397e-12*helper_7 + 3109.6406250000014*helper_9 - 1.3657824871062143e-12*x - 34.171875000005791*y - 6.8289124355310713e-13*z + 6.8289124355310713e-13) + helper_12*x*(helper_10 + helper_8 + 6150.9375000006912*helper_9 + 2460.3750000000746*x - 34.171875000007191) + helper_12*(helper_10*x + 3075.4687500003456*helper_13 + 1230.1875000000373*helper_2 + 615.09375000004411*helper_5 + helper_8*x - 34.171875000007191*x - 34.171875000007738*y) + 5535.8437500009404*helper_2*helper_3 + helper_2*helper_6*(helper_4 + 3075.4687500003456) + 2*helper_5*helper_7*(helper_4*x + 3075.4687500003456*x + 2460.3750000003565*y - 34.171875000010878))/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = 1845.2812500003135*x;
double helper_5 = pow(x, 2);
double helper_6 = helper_0*helper_5;
double helper_7 = 4015.1953125003411*x;
double helper_8 = x*y;
double helper_9 = 3075.4687500003456*helper_5;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_3*x;
double helper_13 = 9.5604774097429353e-12*helper_3;
val[1] = (helper_1*(-helper_0*helper_13 + 1947.7968750000357*helper_0*helper_8 + 324.63281249996822*helper_0*x - 7.2386471816624867e-12*helper_0*y - 1.5934129016231449e-13*helper_1 - 6.6923341868199198e-12*helper_10*y + 3.9835322540661365e-14*helper_10 + 358.80468749997385*helper_11 + 1845.2812500001323*helper_12 - helper_13 + 1230.1875000000373*helper_2 + 8098.7343750007103*helper_5*y + 1554.8203125000007*helper_5 + 1588.992187500007*helper_6 + 1879.4531250000077*helper_8 - 34.171875000005791*x + 2.0486737306593271e-12*y + 2.8738339832859489e-12*z - 2.8738339832859489e-12) + helper_11*y*(helper_7 + 4920.750000000713*helper_8 + helper_9 + 1230.1875000000882*y - 34.171875000007738) + helper_11*(2460.3750000003565*helper_12 + 615.09375000004411*helper_3 + 1230.1875000000373*helper_5 + helper_7*y + helper_9*y - 34.171875000007191*x - 34.171875000007738*y) + 5535.8437500009404*helper_2*helper_3 + helper_3*helper_6*(helper_4 + 2460.3750000003565) + 2*helper_6*y*(helper_4*y + 3075.4687500003456*x + 2460.3750000003565*y - 34.171875000010878))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 2.7315649742124277e-12*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 3.1868258032476444e-12*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_9*x;
double helper_13 = pow(z, 5);
double helper_14 = helper_13*x;
double helper_15 = helper_6*y;
double helper_16 = 1.0926259896849716e-11*helper_6;
double helper_17 = helper_1*y;
double helper_18 = helper_2*y;
double helper_19 = helper_8*z;
double helper_20 = helper_5*helper_8;
double helper_21 = 2460.3750000003565*helper_5;
double helper_22 = helper_0*helper_5;
double helper_23 = helper_2*helper_5;
double helper_24 = 3075.4687500003456*helper_6*helper_8;
double helper_25 = helper_6*helper_9;
double helper_26 = helper_1*helper_8;
double helper_27 = helper_0*helper_9;
val[2] = -(-5877.5625000001164*helper_0*helper_11 + 1230.1875000000391*helper_0*helper_15 + 4015.1953125003383*helper_0*helper_20 + helper_0*helper_24 + 5228.296874999668*helper_0*helper_3 - 1.6389389845274575e-11*helper_0*helper_6 - 4.5207400323211741e-11*helper_0*helper_8 - 9.5604774097435008e-12*helper_0*x - 9.2759393915983065e-12*helper_0*y + helper_1*helper_16 - 4.6436604561611279e-11*helper_1*helper_5 - 1.2747303212990578e-11*helper_1*helper_9 + 1.0926259896849717e-11*helper_1*x + helper_10*helper_2 + helper_10 - 973.89843750001774*helper_11*helper_2 - 1008.0703125000255*helper_11 - 1230.1875000000887*helper_12*z + 615.09375000004411*helper_12 - 5.4631299484248554e-12*helper_13*helper_5 + 6.6923341868199206e-12*helper_13*helper_8 - 2.9478138680029842e-12*helper_13*y - 717.6093749999477*helper_14*y + 1.3657824871062138e-12*helper_14 + 1230.1875000000373*helper_15 + helper_16*z + 6355.9687500000327*helper_17*helper_5 - 5877.5624999996053*helper_17*x + 1.1381520725916959e-12*helper_17 + 3263.4140624997708*helper_18*x + 4.6948772994248162e-12*helper_18 - 8098.7343750006985*helper_19*helper_5 + 3963.9375000000873*helper_19*x + 1.8984376570774735e-11*helper_19 - helper_2*helper_7 - 2.9842347343268349e-11*helper_2*helper_8 - 6.1460211919779652e-12*helper_2*x + 4083.539062500362*helper_20 + helper_21*helper_27 - helper_21*helper_9 - 9568.1250000000509*helper_22*y + 4.2339257100292648e-11*helper_22 - 1588.9921875000073*helper_23*y + 2.5266976011464962e-11*helper_23 - helper_24 + 3690.5625000006266*helper_25*z + 1845.2812500003138*helper_25 + 3895.5937500000728*helper_26*x + 5.244604750487422e-11*helper_26 + 615.09375000004502*helper_27*x + 1.9120954819485871e-11*helper_27 + 392.97656249997948*helper_3 + 6424.3125000000409*helper_4*helper_5 - 2460.3750000000755*helper_4*helper_6 - 2289.5156249998654*helper_4*x + 8.2288394848155223e-12*helper_4 - 1623.1640625000143*helper_5*y - 1.9120954819487008e-11*helper_5*z + 3.4144562177655375e-12*helper_5 - helper_7 - 3.0730105959886765e-12*helper_8 - 1.2747303212990579e-11*helper_9*z + 4.097347461318643e-12*x*z - 6.828912435531046e-13*x + 4.7802387048694361e-13*y*pow(z, 6) - 2.3161394677176832e-12*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void pyramid_4_basis_grad_value_3d_single_54(double x, double y, double z, double *val) {
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 3);
double helper_4 = pow(y, 2);
double helper_5 = 1845.2812500002565*y;
double helper_6 = helper_4*x;
double helper_7 = 3365.9296875002578*y;
double helper_8 = x*y;
double helper_9 = 2460.3750000002874*helper_4;
double helper_10 = pow(helper_0, 2);
double helper_11 = helper_10*x;
double helper_12 = helper_2*y;
double helper_13 = helper_10*y;
double helper_14 = 6.5557559381098114e-12*helper_2;
double helper_15 = helper_0*y;
double helper_16 = -helper_0;
val[0] = -(helper_0*helper_2*helper_4*(helper_5 + 2460.3750000002701) + 2*helper_0*helper_6*(helper_5*x + 2460.3750000002701*x + 2460.3750000002874*y + 34.171874999992276) + helper_1*(-5.4631299484248429e-13*helper_10 + 4.3705039587398743e-12*helper_11 + 1845.2812500000539*helper_12 + 256.2890624999747*helper_13 - helper_14*helper_16 + helper_14 + 1742.7656249999766*helper_15*x + 290.46093749997232*helper_15 - 871.38281250000682*helper_16*helper_4 - 3.2778779690549029e-12*helper_16*x + 615.09375000003263*helper_3 + 905.55468749999682*helper_4 + 6663.5156250005366*helper_6 + 1811.1093749999693*helper_8 - 1.0926259896849686e-12*x + 34.171874999996973*y - 5.4631299484248429e-13*z + 5.4631299484248429e-13) + helper_11*y*(helper_7 + 4920.7500000005402*helper_8 + helper_9 + 1230.1875000000359*x + 34.171874999995673) + helper_13*(2460.3750000002701*helper_12 + 615.09375000001796*helper_2 + 615.09375000003263*helper_4 + helper_7*x + helper_9*x + 34.171874999995673*x + 34.171874999995268*y) + 5535.8437500007694*helper_2*helper_3)/helper_1;}
{double helper_0 = z - 1;
double helper_1 = pow(helper_0, 3);
double helper_2 = pow(x, 3);
double helper_3 = pow(y, 2);
double helper_4 = pow(x, 2);
double helper_5 = 1845.2812500002565*x;
double helper_6 = helper_4*y;
double helper_7 = pow(helper_0, 2);
double helper_8 = 3365.9296875002578*x;
double helper_9 = x*y;
double helper_10 = 2460.3750000002701*helper_4;
double helper_11 = helper_3*x;
double helper_12 = helper_7*x;
double helper_13 = 6.2143103163329177e-12*helper_3;
val[1] = -(helper_0*helper_3*helper_4*(helper_5 + 2460.3750000002874) + 2*helper_0*helper_6*(helper_5*y + 2460.3750000002701*x + 2460.3750000002874*y + 34.171874999992276) + helper_1*(-helper_0*helper_13 + 871.38281249998829*helper_0*helper_4 + 1742.7656250000136*helper_0*helper_9 + 290.46093749997232*helper_0*x - 5.2924071375362271e-12*helper_0*y - 4.5526082903481456e-14*helper_1 + 1845.2812500000978*helper_11 + 256.2890624999747*helper_12 - helper_13 + 615.09375000001796*helper_2 + 905.55468749998465*helper_4 + 6663.5156250005366*helper_6 - 4.233925710028908e-12*helper_7*y - 7.3979884718185834e-14*helper_7 + 1811.1093749999936*helper_9 + 34.171874999996973*x + 1.4682161736392004e-12*y + 2.0771275324740145e-12*z - 2.0771275324740145e-12) + helper_12*(helper_10*y + 2460.3750000002874*helper_11 + 615.09375000003263*helper_3 + 615.09375000001796*helper_4 + helper_8*y + 34.171874999995673*x + 34.171874999995268*y) + 5535.8437500007694*helper_2*helper_3 + helper_7*helper_9*(helper_10 + helper_8 + 4920.7500000005748*helper_9 + 1230.1875000000653*y + 34.171874999995268))/helper_1;}
{double helper_0 = pow(z, 2);
double helper_1 = pow(z, 3);
double helper_2 = pow(z, 4);
double helper_3 = x*y;
double helper_4 = y*z;
double helper_5 = pow(x, 2);
double helper_6 = pow(x, 3);
double helper_7 = 2.1852519793699376e-12*helper_6;
double helper_8 = pow(y, 2);
double helper_9 = pow(y, 3);
double helper_10 = 2.0714367721109724e-12*helper_9;
double helper_11 = helper_8*x;
double helper_12 = helper_9*x;
double helper_13 = helper_0*x;
double helper_14 = pow(z, 5);
double helper_15 = helper_6*y;
double helper_16 = 8.7410079174797486e-12*helper_6;
double helper_17 = helper_1*y;
double helper_18 = helper_8*z;
double helper_19 = 8.2857470884438897e-12*helper_9;
double helper_20 = 2460.3750000002874*helper_9;
double helper_21 = helper_0*helper_5;
double helper_22 = helper_2*helper_5;
double helper_23 = 2460.3750000002701*helper_6*helper_8;
double helper_24 = helper_6*helper_9;
double helper_25 = helper_1*helper_8;
val[2] = (615.09375000003229*helper_0*helper_12 + 615.09375000001751*helper_0*helper_15 + helper_0*helper_23 + 3383.0156249996603*helper_0*helper_3 - 1.3111511876219626e-11*helper_0*helper_6 - 2.6462035687680405e-11*helper_0*helper_8 + 1.2428620632665839e-11*helper_0*helper_9 - 1.1893689158551137e-11*helper_0*y + helper_1*helper_16 - helper_1*helper_19 - 3.7149283649288942e-11*helper_1*helper_5 + 8.7410079174797502e-12*helper_1*x + helper_10*helper_2 + helper_10 - 871.38281250000682*helper_11*helper_2 - 837.2109375000116*helper_11 - 1230.1875000000655*helper_12*z + 615.09375000003251*helper_12 - 5194.1250000000446*helper_13*helper_8 - 7.6483819277947812e-12*helper_13 - 512.57812499994952*helper_14*helper_3 - 4.3705039587398751e-12*helper_14*helper_5 + 4.2339257100289089e-12*helper_14*helper_8 + 1.0926259896849688e-12*helper_14*x - 6.7150972282629544e-13*helper_14*y + 615.09375000001796*helper_15 + helper_16*z + 3485.5312499999527*helper_17*helper_5 - 3963.9374999996035*helper_17*x + 7.0565428500508932e-12*helper_17 - 6663.5156250005284*helper_18*helper_5 + 3417.1875000000368*helper_18*x + 1.0584814275072086e-11*helper_18 - helper_19*z + 2272.4296874997744*helper_2*helper_3 - helper_2*helper_7 - 1.852342498137643e-11*helper_2*helper_8 - 4.9168169535823588e-12*helper_2*x - 7.6825264899920536e-13*helper_2*y + helper_20*helper_21 - helper_20*helper_5 + 3365.9296875002565*helper_21*helper_8 - 5194.1249999999327*helper_21*y + 3.387140568023403e-11*helper_21 - 871.38281249998818*helper_22*y + 2.0213580809171917e-11*helper_22 - helper_23 + 3690.562500000513*helper_24*z + 1845.2812500002565*helper_24 + 3485.5312500000264*helper_25*x + 3.1754442825216626e-11*helper_25 + 222.11718749997709*helper_3 + 3417.1874999999618*helper_4*helper_5 - 1230.1875000000359*helper_4*helper_6 - 1401.0468749998581*helper_4*x + 8.2288394848152444e-12*helper_4 + 3297.5859375002733*helper_5*helper_8 - 837.21093749999261*helper_5*y - 1.5296763855589566e-11*helper_5*z + 2.7315649742124237e-12*helper_5 - helper_7 - 1.5877221412607953e-12*helper_8 + 3.2778779690549065e-12*x*z - 5.4631299484248318e-13*x + 1.3657824871044482e-13*y*pow(z, 6) - 2.0885090531999415e-12*y)/(6.0*helper_0 - 4.0*helper_1 + 1.0*helper_2 - 4.0*z + 1.0);}
}



POLYFEM_BOTH void pyramid_4_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_20(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_21(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_22(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_23(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_24(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_25(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_26(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_27(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_28(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_29(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_30(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_31(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_32(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_33(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_34(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 35:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_35(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 36:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_36(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 37:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_37(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 38:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_38(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 39:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_39(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 40:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_40(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 41:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_41(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 42:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_42(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 43:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_43(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 44:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_44(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 45:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_45(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 46:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_46(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 47:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_47(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 48:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_48(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 49:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_49(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 50:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_50(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 51:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_51(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 52:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_52(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 53:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_53(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 54:
		for (std::size_t i = 0; i < x.size(); ++i) {
			pyramid_4_basis_grad_value_3d_single_54(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


POLYFEM_BOTH int pyramid_basis_count_3d(const int pyramid) {
switch(pyramid) {
	case 0: return 1;
	case 1: return 5;
	case 2: return 14;
	case 3: return 30;
	case 4: return 55;
	default: assert(false); return 0;
}}

POLYFEM_BOTH void pyramid_basis_value_3d(const int pyramid, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){

switch(pyramid){
	case 0: pyramid_0_basis_value_3d(local_index, x, y, z, val); break;
	case 1: pyramid_1_basis_value_3d(local_index, x, y, z, val); break;
	case 2: pyramid_2_basis_value_3d(local_index, x, y, z, val); break;
	case 3: pyramid_3_basis_value_3d(local_index, x, y, z, val); break;
	case 4: pyramid_4_basis_value_3d(local_index, x, y, z, val); break;
	default: assert(false); 
}}

POLYFEM_BOTH void pyramid_grad_basis_value_3d(const int pyramid, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){

switch(pyramid){
	case 0: pyramid_0_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: pyramid_1_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: pyramid_2_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: pyramid_3_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 4: pyramid_4_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default: assert(false); 
}}


}}
