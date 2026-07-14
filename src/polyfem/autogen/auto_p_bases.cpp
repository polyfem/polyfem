#include "auto_p_bases.hpp"
#include "p_n_bases.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH double p_0_basis_value_2d_single_0(double x, double y) {
double result;
result = 1;
return result;
}



POLYFEM_BOTH void p_0_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_0_basis_value_2d_single_0(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_0_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 0;}
}



POLYFEM_BOTH void p_0_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_0_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_1_basis_value_2d_single_0(double x, double y) {
double result;
result = -x - y + 1;
return result;
}

inline POLYFEM_BOTH double p_1_basis_value_2d_single_1(double x, double y) {
double result;
result = x;
return result;
}

inline POLYFEM_BOTH double p_1_basis_value_2d_single_2(double x, double y) {
double result;
result = y;
return result;
}



POLYFEM_BOTH void p_1_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_2d_single_2(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_1_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -1;}
{val[1] = -1;}
}

inline POLYFEM_BOTH void p_1_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 1;}
{val[1] = 0;}
}

inline POLYFEM_BOTH void p_1_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 1;}
}



POLYFEM_BOTH void p_1_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_2_basis_value_2d_single_0(double x, double y) {
double result;
result = (x + y - 1)*(2*x + 2*y - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_2d_single_1(double x, double y) {
double result;
result = x*(2*x - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_2d_single_2(double x, double y) {
double result;
result = y*(2*y - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_2d_single_3(double x, double y) {
double result;
result = -4*x*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_2d_single_4(double x, double y) {
double result;
result = 4*x*y;
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_2d_single_5(double x, double y) {
double result;
result = -4*y*(x + y - 1);
return result;
}



POLYFEM_BOTH void p_2_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_2d_single_5(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 4*x + 4*y - 3;}
{val[1] = 4*x + 4*y - 3;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 4*x - 1;}
{val[1] = 0;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 4*y - 1;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = 4*(-2*x - y + 1);}
{val[1] = -4*x;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = 4*y;}
{val[1] = 4*x;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -4*y;}
{val[1] = 4*(-x - 2*y + 1);}
}



POLYFEM_BOTH void p_2_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_3_basis_value_2d_single_0(double x, double y) {
double result;
double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
result = -27.0/2.0*helper_0*y + 9*helper_0 - 27.0/2.0*helper_1*x + 9*helper_1 - 9.0/2.0*pow(x, 3) + 18*x*y - 11.0/2.0*x - 9.0/2.0*pow(y, 3) - 11.0/2.0*y + 1;
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_1(double x, double y) {
double result;
result = (1.0/2.0)*x*(9*pow(x, 2) - 9*x + 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_2(double x, double y) {
double result;
result = (1.0/2.0)*y*(9*pow(y, 2) - 9*y + 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_3(double x, double y) {
double result;
result = (9.0/2.0)*x*(x + y - 1)*(3*x + 3*y - 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_4(double x, double y) {
double result;
result = -9.0/2.0*x*(3*pow(x, 2) + 3*x*y - 4*x - y + 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_5(double x, double y) {
double result;
result = (9.0/2.0)*x*y*(3*x - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_6(double x, double y) {
double result;
result = (9.0/2.0)*x*y*(3*y - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_7(double x, double y) {
double result;
result = -9.0/2.0*y*(3*x*y - x + 3*pow(y, 2) - 4*y + 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_8(double x, double y) {
double result;
result = (9.0/2.0)*y*(x + y - 1)*(3*x + 3*y - 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_2d_single_9(double x, double y) {
double result;
result = -27*x*y*(x + y - 1);
return result;
}



POLYFEM_BOTH void p_3_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_7(x[i], y[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_8(x[i], y[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_2d_single_9(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -27.0/2.0*pow(x, 2) - 27*x*y + 18*x - 27.0/2.0*pow(y, 2) + 18*y - 11.0/2.0;}
{val[1] = -27.0/2.0*pow(x, 2) - 27*x*y + 18*x - 27.0/2.0*pow(y, 2) + 18*y - 11.0/2.0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = (27.0/2.0)*pow(x, 2) - 9*x + 1;}
{val[1] = 0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = (27.0/2.0)*pow(y, 2) - 9*y + 1;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = 9*((9.0/2.0)*pow(x, 2) + 6*x*y - 5*x + (3.0/2.0)*pow(y, 2) - 5.0/2.0*y + 1);}
{val[1] = (9.0/2.0)*x*(6*x + 6*y - 5);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = 9*(-9.0/2.0*pow(x, 2) - 3*x*y + 4*x + (1.0/2.0)*y - 1.0/2.0);}
{val[1] = -9.0/2.0*x*(3*x - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = (9.0/2.0)*y*(6*x - 1);}
{val[1] = (9.0/2.0)*x*(3*x - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = (9.0/2.0)*y*(3*y - 1);}
{val[1] = (9.0/2.0)*x*(6*y - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = -9.0/2.0*y*(3*y - 1);}
{val[1] = 9*(-3*x*y + (1.0/2.0)*x - 9.0/2.0*pow(y, 2) + 4*y - 1.0/2.0);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = (9.0/2.0)*y*(6*x + 6*y - 5);}
{val[1] = 9*((3.0/2.0)*pow(x, 2) + 6*x*y - 5.0/2.0*x + (9.0/2.0)*pow(y, 2) - 5*y + 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_2d_single_9(double x, double y, double *val) {
{val[0] = -27*y*(2*x + y - 1);}
{val[1] = -27*x*(x + 2*y - 1);}
}



POLYFEM_BOTH void p_3_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_8(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_2d_single_9(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_4_basis_value_2d_single_0(double x, double y) {
double result;
double helper_0 = pow(x, 2);
double helper_1 = pow(x, 3);
double helper_2 = pow(y, 2);
double helper_3 = pow(y, 3);
result = 64*helper_0*helper_2 - 80*helper_0*y + (70.0/3.0)*helper_0 + (128.0/3.0)*helper_1*y - 80.0/3.0*helper_1 - 80*helper_2*x + (70.0/3.0)*helper_2 + (128.0/3.0)*helper_3*x - 80.0/3.0*helper_3 + (32.0/3.0)*pow(x, 4) + (140.0/3.0)*x*y - 25.0/3.0*x + (32.0/3.0)*pow(y, 4) - 25.0/3.0*y + 1;
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_1(double x, double y) {
double result;
result = (1.0/3.0)*x*(32*pow(x, 3) - 48*pow(x, 2) + 22*x - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_2(double x, double y) {
double result;
result = (1.0/3.0)*y*(32*pow(y, 3) - 48*pow(y, 2) + 22*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_3(double x, double y) {
double result;
double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
result = -16.0/3.0*x*(24*helper_0*y - 18*helper_0 + 24*helper_1*x - 18*helper_1 + 8*pow(x, 3) - 36*x*y + 13*x + 8*pow(y, 3) + 13*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_4(double x, double y) {
double result;
double helper_0 = 32*pow(x, 2);
double helper_1 = pow(y, 2);
result = 4*x*(helper_0*y - helper_0 + 16*helper_1*x - 4*helper_1 + 16*pow(x, 3) - 36*x*y + 19*x + 7*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_5(double x, double y) {
double result;
double helper_0 = pow(x, 2);
result = -16.0/3.0*x*(8*helper_0*y - 14*helper_0 + 8*pow(x, 3) - 6*x*y + 7*x + y - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_6(double x, double y) {
double result;
result = (16.0/3.0)*x*y*(8*pow(x, 2) - 6*x + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_7(double x, double y) {
double result;
double helper_0 = 4*x;
result = helper_0*y*(-helper_0 + 16*x*y - 4*y + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_8(double x, double y) {
double result;
result = (16.0/3.0)*x*y*(8*pow(y, 2) - 6*y + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_9(double x, double y) {
double result;
double helper_0 = pow(y, 2);
result = -16.0/3.0*y*(8*helper_0*x - 14*helper_0 - 6*x*y + x + 8*pow(y, 3) + 7*y - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_10(double x, double y) {
double result;
double helper_0 = pow(x, 2);
double helper_1 = 32*pow(y, 2);
result = 4*y*(16*helper_0*y - 4*helper_0 + helper_1*x - helper_1 - 36*x*y + 7*x + 16*pow(y, 3) + 19*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_11(double x, double y) {
double result;
double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
result = -16.0/3.0*y*(24*helper_0*y - 18*helper_0 + 24*helper_1*x - 18*helper_1 + 8*pow(x, 3) - 36*x*y + 13*x + 8*pow(y, 3) + 13*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_12(double x, double y) {
double result;
result = 32*x*y*(x + y - 1)*(4*x + 4*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_13(double x, double y) {
double result;
result = -32*x*y*(4*y - 1)*(x + y - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_2d_single_14(double x, double y) {
double result;
result = -32*x*y*(4*x - 1)*(x + y - 1);
return result;
}



POLYFEM_BOTH void p_4_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_7(x[i], y[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_8(x[i], y[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_9(x[i], y[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_10(x[i], y[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_11(x[i], y[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_12(x[i], y[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_13(x[i], y[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_2d_single_14(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_0(double x, double y, double *val) {
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
val[0] = 128*helper_0*y - 80*helper_0 + 128*helper_1*x - 80*helper_1 + (128.0/3.0)*pow(x, 3) - 160*x*y + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y - 25.0/3.0;}
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
val[1] = 128*helper_0*y - 80*helper_0 + 128*helper_1*x - 80*helper_1 + (128.0/3.0)*pow(x, 3) - 160*x*y + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y - 25.0/3.0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = (128.0/3.0)*pow(x, 3) - 48*pow(x, 2) + (44.0/3.0)*x - 1;}
{val[1] = 0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = (128.0/3.0)*pow(y, 3) - 48*pow(y, 2) + (44.0/3.0)*y - 1;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_3(double x, double y, double *val) {
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
val[0] = -384*helper_0*y + 288*helper_0 - 256*helper_1*x + 96*helper_1 - 512.0/3.0*pow(x, 3) + 384*x*y - 416.0/3.0*x - 128.0/3.0*pow(y, 3) - 208.0/3.0*y + 16;}
{val[1] = -16.0/3.0*x*(24*pow(x, 2) + 48*x*y - 36*x + 24*pow(y, 2) - 36*y + 13);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = 96*pow(x, 2);
double helper_1 = pow(y, 2);
val[0] = 4*helper_0*y - 4*helper_0 + 128*helper_1*x - 16*helper_1 + 256*pow(x, 3) - 288*x*y + 152*x + 28*y - 12;}
{val[1] = 4*x*(32*pow(x, 2) + 32*x*y - 36*x - 8*y + 7);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_5(double x, double y, double *val) {
{double helper_0 = pow(x, 2);
val[0] = -128*helper_0*y + 224*helper_0 - 512.0/3.0*pow(x, 3) + 64*x*y - 224.0/3.0*x - 16.0/3.0*y + 16.0/3.0;}
{val[1] = -16.0/3.0*x*(8*pow(x, 2) - 6*x + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = (16.0/3.0)*y*(24*pow(x, 2) - 12*x + 1);}
{val[1] = (16.0/3.0)*x*(8*pow(x, 2) - 6*x + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_7(double x, double y, double *val) {
{double helper_0 = 4*y;
val[0] = helper_0*(-helper_0 + 32*x*y - 8*x + 1);}
{double helper_0 = 4*x;
val[1] = helper_0*(-helper_0 + 32*x*y - 8*y + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = (16.0/3.0)*y*(8*pow(y, 2) - 6*y + 1);}
{val[1] = (16.0/3.0)*x*(24*pow(y, 2) - 12*y + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_9(double x, double y, double *val) {
{val[0] = -16.0/3.0*y*(8*pow(y, 2) - 6*y + 1);}
{double helper_0 = pow(y, 2);
val[1] = -128*helper_0*x + 224*helper_0 + 64*x*y - 16.0/3.0*x - 512.0/3.0*pow(y, 3) - 224.0/3.0*y + 16.0/3.0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_10(double x, double y, double *val) {
{val[0] = 4*y*(32*x*y - 8*x + 32*pow(y, 2) - 36*y + 7);}
{double helper_0 = pow(x, 2);
double helper_1 = 96*pow(y, 2);
val[1] = 128*helper_0*y - 16*helper_0 + 4*helper_1*x - 4*helper_1 - 288*x*y + 28*x + 256*pow(y, 3) + 152*y - 12;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_11(double x, double y, double *val) {
{val[0] = -16.0/3.0*y*(24*pow(x, 2) + 48*x*y - 36*x + 24*pow(y, 2) - 36*y + 13);}
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
val[1] = -256*helper_0*y + 96*helper_0 - 384*helper_1*x + 288*helper_1 - 128.0/3.0*pow(x, 3) + 384*x*y - 208.0/3.0*x - 512.0/3.0*pow(y, 3) - 416.0/3.0*y + 16;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_12(double x, double y, double *val) {
{val[0] = 32*y*(12*pow(x, 2) + 16*x*y - 14*x + 4*pow(y, 2) - 7*y + 3);}
{val[1] = 32*x*(4*pow(x, 2) + 16*x*y - 7*x + 12*pow(y, 2) - 14*y + 3);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_13(double x, double y, double *val) {
{val[0] = -32*y*(4*y - 1)*(2*x + y - 1);}
{val[1] = -32*x*(8*x*y - x + 12*pow(y, 2) - 10*y + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_2d_single_14(double x, double y, double *val) {
{val[0] = -32*y*(12*pow(x, 2) + 8*x*y - 10*x - y + 1);}
{val[1] = -32*x*(4*x - 1)*(x + 2*y - 1);}
}



POLYFEM_BOTH void p_4_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_8(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_9(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_10(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_11(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_12(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_13(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_2d_single_14(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}


POLYFEM_BOTH int p_basis_count_2d(const int p) {
switch(p) {
	case 0: return 1;
	case 1: return 3;
	case 2: return 6;
	case 3: return 10;
	case 4: return 15;
	default: return (p + 1) * (p + 2) / 2;
}}

POLYFEM_BOTH void p_basis_value_2d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
if(bernstein) { b_basis_value_2d(p, local_index, x, y, z, val); return; }


switch(p){
	case 0: p_0_basis_value_2d(local_index, x, y, z, val); break;
	case 1: p_1_basis_value_2d(local_index, x, y, z, val); break;
	case 2: p_2_basis_value_2d(local_index, x, y, z, val); break;
	case 3: p_3_basis_value_2d(local_index, x, y, z, val); break;
	case 4: p_4_basis_value_2d(local_index, x, y, z, val); break;
	default:
#ifdef __CUDA_ARCH__
		assert(false && "generic Pn evaluation is host-only");
#else
		p_n_basis_value_2d(p, local_index, x, y, val);
#endif
}}

POLYFEM_BOTH void p_grad_basis_value_2d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
if(bernstein) { b_grad_basis_value_2d(p, local_index, x, y, z, grad_x, grad_y, grad_z); return; }


switch(p){
	case 0: p_0_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: p_1_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: p_2_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: p_3_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 4: p_4_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default:
#ifdef __CUDA_ARCH__
		assert(false && "generic Pn evaluation is host-only");
#else
		p_n_basis_grad_value_2d(p, local_index, x, y, grad_x, grad_y);
#endif
}}

inline POLYFEM_BOTH double p_0_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = 1;
return result;
}



POLYFEM_BOTH void p_0_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_0_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_0_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 0;}
}



POLYFEM_BOTH void p_0_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_0_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_1_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = -x - y - z + 1;
return result;
}

inline POLYFEM_BOTH double p_1_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = x;
return result;
}

inline POLYFEM_BOTH double p_1_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = y;
return result;
}

inline POLYFEM_BOTH double p_1_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = z;
return result;
}



POLYFEM_BOTH void p_1_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_1_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_1_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -1;}
{val[1] = -1;}
{val[2] = -1;}
}

inline POLYFEM_BOTH void p_1_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 1;}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_1_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 1;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_1_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 1;}
}



POLYFEM_BOTH void p_1_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_1_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_2_basis_value_3d_single_0(double x, double y, double z) {
double result;
result = (x + y + z - 1)*(2*x + 2*y + 2*z - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = x*(2*x - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = y*(2*y - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = z*(2*z - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = -4*x*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_5(double x, double y, double z) {
double result;
result = 4*x*y;
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = -4*y*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = -4*z*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_8(double x, double y, double z) {
double result;
result = 4*x*z;
return result;
}

inline POLYFEM_BOTH double p_2_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = 4*y*z;
return result;
}



POLYFEM_BOTH void p_2_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_2_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 4*x + 4*y + 4*z - 3;}
{val[1] = 4*x + 4*y + 4*z - 3;}
{val[2] = 4*x + 4*y + 4*z - 3;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 4*x - 1;}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 4*y - 1;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 4*z - 1;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 4*(-2*x - y - z + 1);}
{val[1] = -4*x;}
{val[2] = -4*x;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = 4*y;}
{val[1] = 4*x;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = -4*y;}
{val[1] = 4*(-x - 2*y - z + 1);}
{val[2] = -4*y;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = -4*z;}
{val[1] = -4*z;}
{val[2] = 4*(-x - y - 2*z + 1);}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = 4*z;}
{val[1] = 0;}
{val[2] = 4*x;}
}

inline POLYFEM_BOTH void p_2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 4*z;}
{val[2] = 4*y;}
}



POLYFEM_BOTH void p_2_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_2_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_3_basis_value_3d_single_0(double x, double y, double z) {
double result;
double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
double helper_2 = pow(z, 2);
double helper_3 = (27.0/2.0)*x;
double helper_4 = (27.0/2.0)*y;
double helper_5 = (27.0/2.0)*z;
result = -helper_0*helper_4 - helper_0*helper_5 + 9*helper_0 - helper_1*helper_3 - helper_1*helper_5 + 9*helper_1 - helper_2*helper_3 - helper_2*helper_4 + 9*helper_2 - 9.0/2.0*pow(x, 3) - 27*x*y*z + 18*x*y + 18*x*z - 11.0/2.0*x - 9.0/2.0*pow(y, 3) + 18*y*z - 11.0/2.0*y - 9.0/2.0*pow(z, 3) - 11.0/2.0*z + 1;
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = (1.0/2.0)*x*(9*pow(x, 2) - 9*x + 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = (1.0/2.0)*y*(9*pow(y, 2) - 9*y + 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = (1.0/2.0)*z*(9*pow(z, 2) - 9*z + 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_4(double x, double y, double z) {
double result;
result = (9.0/2.0)*x*(x + y + z - 1)*(3*x + 3*y + 3*z - 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_5(double x, double y, double z) {
double result;
double helper_0 = 3*x;
result = -9.0/2.0*x*(helper_0*y + helper_0*z + 3*pow(x, 2) - 4*x - y - z + 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_6(double x, double y, double z) {
double result;
result = (9.0/2.0)*x*y*(3*x - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = (9.0/2.0)*x*y*(3*y - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_8(double x, double y, double z) {
double result;
double helper_0 = 3*y;
result = -9.0/2.0*y*(helper_0*x + helper_0*z - x + 3*pow(y, 2) - 4*y - z + 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = (9.0/2.0)*y*(x + y + z - 1)*(3*x + 3*y + 3*z - 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_10(double x, double y, double z) {
double result;
result = (9.0/2.0)*z*(x + y + z - 1)*(3*x + 3*y + 3*z - 2);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_11(double x, double y, double z) {
double result;
double helper_0 = 3*z;
result = -9.0/2.0*z*(helper_0*x + helper_0*y - x - y + 3*pow(z, 2) - 4*z + 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_12(double x, double y, double z) {
double result;
result = (9.0/2.0)*x*z*(3*x - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_13(double x, double y, double z) {
double result;
result = (9.0/2.0)*x*z*(3*z - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_14(double x, double y, double z) {
double result;
result = (9.0/2.0)*y*z*(3*y - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_15(double x, double y, double z) {
double result;
result = (9.0/2.0)*y*z*(3*z - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = -27*x*y*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_17(double x, double y, double z) {
double result;
result = -27*x*z*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = 27*x*y*z;
return result;
}

inline POLYFEM_BOTH double p_3_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = -27*y*z*(x + y + z - 1);
return result;
}



POLYFEM_BOTH void p_3_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_14(x[i], y[i], z[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_15(x[i], y[i], z[i]);
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_16(x[i], y[i], z[i]);
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_17(x[i], y[i], z[i]);
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_18(x[i], y[i], z[i]);
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_3_basis_value_3d_single_19(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = 27*x;
val[0] = -helper_0*y - helper_0*z - 27.0/2.0*pow(x, 2) + 18*x - 27.0/2.0*pow(y, 2) - 27*y*z + 18*y - 27.0/2.0*pow(z, 2) + 18*z - 11.0/2.0;}
{double helper_0 = 27*x;
val[1] = -helper_0*y - helper_0*z - 27.0/2.0*pow(x, 2) + 18*x - 27.0/2.0*pow(y, 2) - 27*y*z + 18*y - 27.0/2.0*pow(z, 2) + 18*z - 11.0/2.0;}
{double helper_0 = 27*x;
val[2] = -helper_0*y - helper_0*z - 27.0/2.0*pow(x, 2) + 18*x - 27.0/2.0*pow(y, 2) - 27*y*z + 18*y - 27.0/2.0*pow(z, 2) + 18*z - 11.0/2.0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = (27.0/2.0)*pow(x, 2) - 9*x + 1;}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = (27.0/2.0)*pow(y, 2) - 9*y + 1;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = (27.0/2.0)*pow(z, 2) - 9*z + 1;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = 6*x;
val[0] = 9*helper_0*y + 9*helper_0*z + (81.0/2.0)*pow(x, 2) - 45*x + (27.0/2.0)*pow(y, 2) + 27*y*z - 45.0/2.0*y + (27.0/2.0)*pow(z, 2) - 45.0/2.0*z + 9;}
{val[1] = (9.0/2.0)*x*(6*x + 6*y + 6*z - 5);}
{val[2] = (9.0/2.0)*x*(6*x + 6*y + 6*z - 5);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = 3*x;
val[0] = -9*helper_0*y - 9*helper_0*z - 81.0/2.0*pow(x, 2) + 36*x + (9.0/2.0)*y + (9.0/2.0)*z - 9.0/2.0;}
{val[1] = -9.0/2.0*x*(3*x - 1);}
{val[2] = -9.0/2.0*x*(3*x - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = (9.0/2.0)*y*(6*x - 1);}
{val[1] = (9.0/2.0)*x*(3*x - 1);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = (9.0/2.0)*y*(3*y - 1);}
{val[1] = (9.0/2.0)*x*(6*y - 1);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = -9.0/2.0*y*(3*y - 1);}
{double helper_0 = 3*y;
val[1] = -9*helper_0*x - 9*helper_0*z + (9.0/2.0)*x - 81.0/2.0*pow(y, 2) + 36*y + (9.0/2.0)*z - 9.0/2.0;}
{val[2] = -9.0/2.0*y*(3*y - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = (9.0/2.0)*y*(6*x + 6*y + 6*z - 5);}
{double helper_0 = 6*y;
val[1] = 9*helper_0*x + 9*helper_0*z + (27.0/2.0)*pow(x, 2) + 27*x*z - 45.0/2.0*x + (81.0/2.0)*pow(y, 2) - 45*y + (27.0/2.0)*pow(z, 2) - 45.0/2.0*z + 9;}
{val[2] = (9.0/2.0)*y*(6*x + 6*y + 6*z - 5);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = (9.0/2.0)*z*(6*x + 6*y + 6*z - 5);}
{val[1] = (9.0/2.0)*z*(6*x + 6*y + 6*z - 5);}
{double helper_0 = 6*z;
val[2] = 9*helper_0*x + 9*helper_0*y + (27.0/2.0)*pow(x, 2) + 27*x*y - 45.0/2.0*x + (27.0/2.0)*pow(y, 2) - 45.0/2.0*y + (81.0/2.0)*pow(z, 2) - 45*z + 9;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -9.0/2.0*z*(3*z - 1);}
{val[1] = -9.0/2.0*z*(3*z - 1);}
{double helper_0 = 3*z;
val[2] = -9*helper_0*x - 9*helper_0*y + (9.0/2.0)*x + (9.0/2.0)*y - 81.0/2.0*pow(z, 2) + 36*z - 9.0/2.0;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = (9.0/2.0)*z*(6*x - 1);}
{val[1] = 0;}
{val[2] = (9.0/2.0)*x*(3*x - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = (9.0/2.0)*z*(3*z - 1);}
{val[1] = 0;}
{val[2] = (9.0/2.0)*x*(6*z - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = (9.0/2.0)*z*(6*y - 1);}
{val[2] = (9.0/2.0)*y*(3*y - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = (9.0/2.0)*z*(3*z - 1);}
{val[2] = (9.0/2.0)*y*(6*z - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = -27*y*(2*x + y + z - 1);}
{val[1] = -27*x*(x + 2*y + z - 1);}
{val[2] = -27*x*y;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = -27*z*(2*x + y + z - 1);}
{val[1] = -27*x*z;}
{val[2] = -27*x*(x + y + 2*z - 1);}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = 27*y*z;}
{val[1] = 27*x*z;}
{val[2] = 27*x*y;}
}

inline POLYFEM_BOTH void p_3_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = -27*y*z;}
{val[1] = -27*z*(x + 2*y + z - 1);}
{val[2] = -27*y*(x + y + 2*z - 1);}
}



POLYFEM_BOTH void p_3_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_3_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


inline POLYFEM_BOTH double p_4_basis_value_3d_single_0(double x, double y, double z) {
double result;
double helper_0 = x + y + z - 1;
double helper_1 = x*y;
double helper_2 = pow(y, 2);
double helper_3 = 9*x;
double helper_4 = pow(z, 2);
double helper_5 = pow(x, 2);
double helper_6 = 9*y;
double helper_7 = 9*z;
double helper_8 = 26*helper_0;
double helper_9 = helper_8*z;
double helper_10 = 13*pow(helper_0, 2);
double helper_11 = 13*helper_0;
result = (1.0/3.0)*helper_0*(3*pow(helper_0, 3) + helper_1*helper_8 + 18*helper_1*z + helper_10*x + helper_10*y + helper_10*z + helper_11*helper_2 + helper_11*helper_4 + helper_11*helper_5 + helper_2*helper_3 + helper_2*helper_7 + helper_3*helper_4 + helper_4*helper_6 + helper_5*helper_6 + helper_5*helper_7 + helper_9*x + helper_9*y + 3*pow(x, 3) + 3*pow(y, 3) + 3*pow(z, 3));
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_1(double x, double y, double z) {
double result;
result = (1.0/3.0)*x*(32*pow(x, 3) - 48*pow(x, 2) + 22*x - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_2(double x, double y, double z) {
double result;
result = (1.0/3.0)*y*(32*pow(y, 3) - 48*pow(y, 2) + 22*y - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_3(double x, double y, double z) {
double result;
result = (1.0/3.0)*z*(32*pow(z, 3) - 48*pow(z, 2) + 22*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_4(double x, double y, double z) {
double result;
double helper_0 = 36*x;
double helper_1 = y*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 24*x;
double helper_6 = 24*y;
double helper_7 = 24*z;
result = -16.0/3.0*x*(-helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 18*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4 + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3) + 13*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_5(double x, double y, double z) {
double result;
double helper_0 = 2*y;
double helper_1 = 2*z;
double helper_2 = x + y + z - 1;
double helper_3 = helper_2*x;
result = 4*helper_3*(-helper_0*helper_2 + helper_0*x - helper_0*z - helper_1*helper_2 + helper_1*x + 3*pow(helper_2, 2) + 10*helper_3 + 3*pow(x, 2) - pow(y, 2) - pow(z, 2));
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_6(double x, double y, double z) {
double result;
double helper_0 = 6*x;
double helper_1 = pow(x, 2);
double helper_2 = 8*helper_1;
result = -16.0/3.0*x*(-helper_0*y - helper_0*z - 14*helper_1 + helper_2*y + helper_2*z + 8*pow(x, 3) + 7*x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_7(double x, double y, double z) {
double result;
result = (16.0/3.0)*x*y*(8*pow(x, 2) - 6*x + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_8(double x, double y, double z) {
double result;
double helper_0 = 4*x;
result = helper_0*y*(-helper_0 + 16*x*y - 4*y + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_9(double x, double y, double z) {
double result;
result = (16.0/3.0)*x*y*(8*pow(y, 2) - 6*y + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_10(double x, double y, double z) {
double result;
double helper_0 = 6*y;
double helper_1 = pow(y, 2);
double helper_2 = 8*helper_1;
result = -16.0/3.0*y*(-helper_0*x - helper_0*z - 14*helper_1 + helper_2*x + helper_2*z + x + 8*pow(y, 3) + 7*y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_11(double x, double y, double z) {
double result;
double helper_0 = 2*y;
double helper_1 = 2*x;
double helper_2 = x + y + z - 1;
double helper_3 = helper_2*y;
result = -4*helper_3*(-helper_0*x - helper_0*z + helper_1*helper_2 + helper_1*z - 3*pow(helper_2, 2) + 2*helper_2*z - 10*helper_3 + pow(x, 2) - 3*pow(y, 2) + pow(z, 2));
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_12(double x, double y, double z) {
double result;
double helper_0 = 36*x;
double helper_1 = y*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 24*x;
double helper_6 = 24*y;
double helper_7 = 24*z;
result = -16.0/3.0*y*(-helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 18*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4 + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3) + 13*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_13(double x, double y, double z) {
double result;
double helper_0 = 36*x;
double helper_1 = y*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 24*x;
double helper_6 = 24*y;
double helper_7 = 24*z;
result = -16.0/3.0*z*(-helper_0*y - helper_0*z + 48*helper_1*x - 36*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 18*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 18*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 18*helper_4 + 8*pow(x, 3) + 13*x + 8*pow(y, 3) + 13*y + 8*pow(z, 3) + 13*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_14(double x, double y, double z) {
double result;
double helper_0 = 2*x;
double helper_1 = 2*z;
double helper_2 = x + y + z - 1;
double helper_3 = helper_2*z;
result = -4*helper_3*(helper_0*helper_2 + helper_0*y - helper_1*x - helper_1*y - 3*pow(helper_2, 2) + 2*helper_2*y - 10*helper_3 + pow(x, 2) + pow(y, 2) - 3*pow(z, 2));
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_15(double x, double y, double z) {
double result;
double helper_0 = 6*z;
double helper_1 = pow(z, 2);
double helper_2 = 8*helper_1;
result = -16.0/3.0*z*(-helper_0*x - helper_0*y - 14*helper_1 + helper_2*x + helper_2*y + x + y + 8*pow(z, 3) + 7*z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_16(double x, double y, double z) {
double result;
result = (16.0/3.0)*x*z*(8*pow(x, 2) - 6*x + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_17(double x, double y, double z) {
double result;
double helper_0 = 4*x;
result = helper_0*z*(-helper_0 + 16*x*z - 4*z + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_18(double x, double y, double z) {
double result;
result = (16.0/3.0)*x*z*(8*pow(z, 2) - 6*z + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_19(double x, double y, double z) {
double result;
result = (16.0/3.0)*y*z*(8*pow(y, 2) - 6*y + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_20(double x, double y, double z) {
double result;
double helper_0 = 4*y;
result = helper_0*z*(-helper_0 + 16*y*z - 4*z + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_21(double x, double y, double z) {
double result;
result = (16.0/3.0)*y*z*(8*pow(z, 2) - 6*z + 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_22(double x, double y, double z) {
double result;
result = 32*x*y*(x + y + z - 1)*(4*x + 4*y + 4*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_23(double x, double y, double z) {
double result;
result = -32*x*y*(4*y - 1)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_24(double x, double y, double z) {
double result;
result = -32*x*y*(4*x - 1)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_25(double x, double y, double z) {
double result;
result = 32*x*z*(x + y + z - 1)*(4*x + 4*y + 4*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_26(double x, double y, double z) {
double result;
result = -32*x*z*(4*z - 1)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_27(double x, double y, double z) {
double result;
result = -32*x*z*(4*x - 1)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_28(double x, double y, double z) {
double result;
result = 32*x*y*z*(4*x - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_29(double x, double y, double z) {
double result;
result = 32*x*y*z*(4*z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_30(double x, double y, double z) {
double result;
result = 32*x*y*z*(4*y - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_31(double x, double y, double z) {
double result;
result = -32*y*z*(4*y - 1)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_32(double x, double y, double z) {
double result;
result = -32*y*z*(4*z - 1)*(x + y + z - 1);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_33(double x, double y, double z) {
double result;
result = 32*y*z*(x + y + z - 1)*(4*x + 4*y + 4*z - 3);
return result;
}

inline POLYFEM_BOTH double p_4_basis_value_3d_single_34(double x, double y, double z) {
double result;
result = -256*x*y*z*(x + y + z - 1);
return result;
}



POLYFEM_BOTH void p_4_basis_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
assert(z.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_0(x[i], y[i], z[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_1(x[i], y[i], z[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_2(x[i], y[i], z[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_3(x[i], y[i], z[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_4(x[i], y[i], z[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_5(x[i], y[i], z[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_6(x[i], y[i], z[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_7(x[i], y[i], z[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_8(x[i], y[i], z[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_9(x[i], y[i], z[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_10(x[i], y[i], z[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_11(x[i], y[i], z[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_12(x[i], y[i], z[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_13(x[i], y[i], z[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_14(x[i], y[i], z[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_15(x[i], y[i], z[i]);
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_16(x[i], y[i], z[i]);
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_17(x[i], y[i], z[i]);
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_18(x[i], y[i], z[i]);
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_19(x[i], y[i], z[i]);
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_20(x[i], y[i], z[i]);
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_21(x[i], y[i], z[i]);
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_22(x[i], y[i], z[i]);
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_23(x[i], y[i], z[i]);
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_24(x[i], y[i], z[i]);
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_25(x[i], y[i], z[i]);
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_26(x[i], y[i], z[i]);
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_27(x[i], y[i], z[i]);
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_28(x[i], y[i], z[i]);
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_29(x[i], y[i], z[i]);
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_30(x[i], y[i], z[i]);
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_31(x[i], y[i], z[i]);
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_32(x[i], y[i], z[i]);
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_33(x[i], y[i], z[i]);
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = p_4_basis_value_3d_single_34(x[i], y[i], z[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{double helper_0 = 160*x;
double helper_1 = y*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 128*x;
double helper_6 = 128*y;
double helper_7 = 128*z;
val[0] = -helper_0*y - helper_0*z + 256*helper_1*x - 160*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 80*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 80*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 80*helper_4 + (128.0/3.0)*pow(x, 3) + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y + (128.0/3.0)*pow(z, 3) + (140.0/3.0)*z - 25.0/3.0;}
{double helper_0 = 160*x;
double helper_1 = y*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 128*x;
double helper_6 = 128*y;
double helper_7 = 128*z;
val[1] = -helper_0*y - helper_0*z + 256*helper_1*x - 160*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 80*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 80*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 80*helper_4 + (128.0/3.0)*pow(x, 3) + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y + (128.0/3.0)*pow(z, 3) + (140.0/3.0)*z - 25.0/3.0;}
{double helper_0 = 160*x;
double helper_1 = y*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 128*x;
double helper_6 = 128*y;
double helper_7 = 128*z;
val[2] = -helper_0*y - helper_0*z + 256*helper_1*x - 160*helper_1 + helper_2*helper_6 + helper_2*helper_7 - 80*helper_2 + helper_3*helper_5 + helper_3*helper_7 - 80*helper_3 + helper_4*helper_5 + helper_4*helper_6 - 80*helper_4 + (128.0/3.0)*pow(x, 3) + (140.0/3.0)*x + (128.0/3.0)*pow(y, 3) + (140.0/3.0)*y + (128.0/3.0)*pow(z, 3) + (140.0/3.0)*z - 25.0/3.0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = (128.0/3.0)*pow(x, 3) - 48*pow(x, 2) + (44.0/3.0)*x - 1;}
{val[1] = 0;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = (128.0/3.0)*pow(y, 3) - 48*pow(y, 2) + (44.0/3.0)*y - 1;}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = (128.0/3.0)*pow(z, 3) - 48*pow(z, 2) + (44.0/3.0)*z - 1;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
double helper_2 = pow(z, 2);
double helper_3 = 16*x;
double helper_4 = 24*helper_0;
val[0] = 288*helper_0 - 16*helper_1*helper_3 - 128*helper_1*z + 96*helper_1 - 16*helper_2*helper_3 - 128*helper_2*y + 96*helper_2 - 16*helper_4*y - 16*helper_4*z - 512.0/3.0*pow(x, 3) - 512*x*y*z + 384*x*y + 384*x*z - 416.0/3.0*x - 128.0/3.0*pow(y, 3) + 192*y*z - 208.0/3.0*y - 128.0/3.0*pow(z, 3) - 208.0/3.0*z + 16;}
{double helper_0 = 48*x;
val[1] = -16.0/3.0*x*(helper_0*y + helper_0*z + 24*pow(x, 2) - 36*x + 24*pow(y, 2) + 48*y*z - 36*y + 24*pow(z, 2) - 36*z + 13);}
{double helper_0 = 48*x;
val[2] = -16.0/3.0*x*(helper_0*y + helper_0*z + 24*pow(x, 2) - 36*x + 24*pow(y, 2) + 48*y*z - 36*y + 24*pow(z, 2) - 36*z + 13);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{double helper_0 = 72*x;
double helper_1 = y*z;
double helper_2 = 96*pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 32*x;
val[0] = -4*helper_0*y - 4*helper_0*z + 256*helper_1*x - 32*helper_1 + 4*helper_2*y + 4*helper_2*z - 4*helper_2 + 4*helper_3*helper_5 - 16*helper_3 + 4*helper_4*helper_5 - 16*helper_4 + 256*pow(x, 3) + 152*x + 28*y + 28*z - 12;}
{double helper_0 = 32*x;
val[1] = 4*x*(helper_0*y + helper_0*z + 32*pow(x, 2) - 36*x - 8*y - 8*z + 7);}
{double helper_0 = 32*x;
val[2] = 4*x*(helper_0*y + helper_0*z + 32*pow(x, 2) - 36*x - 8*y - 8*z + 7);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{double helper_0 = pow(x, 2);
double helper_1 = 8*helper_0;
val[0] = 224*helper_0 - 16*helper_1*y - 16*helper_1*z - 512.0/3.0*pow(x, 3) + 64*x*y + 64*x*z - 224.0/3.0*x - 16.0/3.0*y - 16.0/3.0*z + 16.0/3.0;}
{val[1] = -16.0/3.0*x*(8*pow(x, 2) - 6*x + 1);}
{val[2] = -16.0/3.0*x*(8*pow(x, 2) - 6*x + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = (16.0/3.0)*y*(24*pow(x, 2) - 12*x + 1);}
{val[1] = (16.0/3.0)*x*(8*pow(x, 2) - 6*x + 1);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{double helper_0 = 4*y;
val[0] = helper_0*(-helper_0 + 32*x*y - 8*x + 1);}
{double helper_0 = 4*x;
val[1] = helper_0*(-helper_0 + 32*x*y - 8*y + 1);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = (16.0/3.0)*y*(8*pow(y, 2) - 6*y + 1);}
{val[1] = (16.0/3.0)*x*(24*pow(y, 2) - 12*y + 1);}
{val[2] = 0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = -16.0/3.0*y*(8*pow(y, 2) - 6*y + 1);}
{double helper_0 = pow(y, 2);
double helper_1 = 8*helper_0;
val[1] = 224*helper_0 - 16*helper_1*x - 16*helper_1*z + 64*x*y - 16.0/3.0*x - 512.0/3.0*pow(y, 3) + 64*y*z - 224.0/3.0*y - 16.0/3.0*z + 16.0/3.0;}
{val[2] = -16.0/3.0*y*(8*pow(y, 2) - 6*y + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{double helper_0 = 32*y;
val[0] = 4*y*(helper_0*x + helper_0*z - 8*x + 32*pow(y, 2) - 36*y - 8*z + 7);}
{double helper_0 = 72*y;
double helper_1 = x*z;
double helper_2 = pow(x, 2);
double helper_3 = 96*pow(y, 2);
double helper_4 = pow(z, 2);
double helper_5 = 32*y;
val[1] = -4*helper_0*x - 4*helper_0*z + 256*helper_1*y - 32*helper_1 + 4*helper_2*helper_5 - 16*helper_2 + 4*helper_3*x + 4*helper_3*z - 4*helper_3 + 4*helper_4*helper_5 - 16*helper_4 + 28*x + 256*pow(y, 3) + 152*y + 28*z - 12;}
{double helper_0 = 32*y;
val[2] = 4*y*(helper_0*x + helper_0*z - 8*x + 32*pow(y, 2) - 36*y - 8*z + 7);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{double helper_0 = 48*x;
val[0] = -16.0/3.0*y*(helper_0*y + helper_0*z + 24*pow(x, 2) - 36*x + 24*pow(y, 2) + 48*y*z - 36*y + 24*pow(z, 2) - 36*z + 13);}
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
double helper_2 = pow(z, 2);
double helper_3 = 24*helper_1;
double helper_4 = 16*y;
val[1] = -16*helper_0*helper_4 - 128*helper_0*z + 96*helper_0 + 288*helper_1 - 16*helper_2*helper_4 - 128*helper_2*x + 96*helper_2 - 16*helper_3*x - 16*helper_3*z - 128.0/3.0*pow(x, 3) - 512*x*y*z + 384*x*y + 192*x*z - 208.0/3.0*x - 512.0/3.0*pow(y, 3) + 384*y*z - 416.0/3.0*y - 128.0/3.0*pow(z, 3) - 208.0/3.0*z + 16;}
{double helper_0 = 48*x;
val[2] = -16.0/3.0*y*(helper_0*y + helper_0*z + 24*pow(x, 2) - 36*x + 24*pow(y, 2) + 48*y*z - 36*y + 24*pow(z, 2) - 36*z + 13);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{double helper_0 = 48*x;
val[0] = -16.0/3.0*z*(helper_0*y + helper_0*z + 24*pow(x, 2) - 36*x + 24*pow(y, 2) + 48*y*z - 36*y + 24*pow(z, 2) - 36*z + 13);}
{double helper_0 = 48*x;
val[1] = -16.0/3.0*z*(helper_0*y + helper_0*z + 24*pow(x, 2) - 36*x + 24*pow(y, 2) + 48*y*z - 36*y + 24*pow(z, 2) - 36*z + 13);}
{double helper_0 = pow(x, 2);
double helper_1 = pow(y, 2);
double helper_2 = pow(z, 2);
double helper_3 = 24*helper_2;
double helper_4 = 16*z;
val[2] = -16*helper_0*helper_4 - 128*helper_0*y + 96*helper_0 - 16*helper_1*helper_4 - 128*helper_1*x + 96*helper_1 + 288*helper_2 - 16*helper_3*x - 16*helper_3*y - 128.0/3.0*pow(x, 3) - 512*x*y*z + 192*x*y + 384*x*z - 208.0/3.0*x - 128.0/3.0*pow(y, 3) + 384*y*z - 208.0/3.0*y - 512.0/3.0*pow(z, 3) - 416.0/3.0*z + 16;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{double helper_0 = 32*z;
val[0] = 4*z*(helper_0*x + helper_0*y - 8*x - 8*y + 32*pow(z, 2) - 36*z + 7);}
{double helper_0 = 32*z;
val[1] = 4*z*(helper_0*x + helper_0*y - 8*x - 8*y + 32*pow(z, 2) - 36*z + 7);}
{double helper_0 = x*y;
double helper_1 = 72*z;
double helper_2 = pow(x, 2);
double helper_3 = pow(y, 2);
double helper_4 = 96*pow(z, 2);
double helper_5 = 32*z;
val[2] = 256*helper_0*z - 32*helper_0 - 4*helper_1*x - 4*helper_1*y + 4*helper_2*helper_5 - 16*helper_2 + 4*helper_3*helper_5 - 16*helper_3 + 4*helper_4*x + 4*helper_4*y - 4*helper_4 + 28*x + 28*y + 256*pow(z, 3) + 152*z - 12;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = -16.0/3.0*z*(8*pow(z, 2) - 6*z + 1);}
{val[1] = -16.0/3.0*z*(8*pow(z, 2) - 6*z + 1);}
{double helper_0 = pow(z, 2);
double helper_1 = 8*helper_0;
val[2] = 224*helper_0 - 16*helper_1*x - 16*helper_1*y + 64*x*z - 16.0/3.0*x + 64*y*z - 16.0/3.0*y - 512.0/3.0*pow(z, 3) - 224.0/3.0*z + 16.0/3.0;}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = (16.0/3.0)*z*(24*pow(x, 2) - 12*x + 1);}
{val[1] = 0;}
{val[2] = (16.0/3.0)*x*(8*pow(x, 2) - 6*x + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{double helper_0 = 4*z;
val[0] = helper_0*(-helper_0 + 32*x*z - 8*x + 1);}
{val[1] = 0;}
{double helper_0 = 4*x;
val[2] = helper_0*(-helper_0 + 32*x*z - 8*z + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = (16.0/3.0)*z*(8*pow(z, 2) - 6*z + 1);}
{val[1] = 0;}
{val[2] = (16.0/3.0)*x*(24*pow(z, 2) - 12*z + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = (16.0/3.0)*z*(24*pow(y, 2) - 12*y + 1);}
{val[2] = (16.0/3.0)*y*(8*pow(y, 2) - 6*y + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{val[0] = 0;}
{double helper_0 = 4*z;
val[1] = helper_0*(-helper_0 + 32*y*z - 8*y + 1);}
{double helper_0 = 4*y;
val[2] = helper_0*(-helper_0 + 32*y*z - 8*z + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = (16.0/3.0)*z*(8*pow(z, 2) - 6*z + 1);}
{val[2] = (16.0/3.0)*y*(24*pow(z, 2) - 12*z + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{double helper_0 = x + y + z - 1;
double helper_1 = 4*x;
double helper_2 = helper_1 + 4*y + 4*z - 3;
val[0] = 32*y*(helper_0*helper_1 + helper_0*helper_2 + helper_2*x);}
{double helper_0 = x + y + z - 1;
double helper_1 = 4*y;
double helper_2 = helper_1 + 4*x + 4*z - 3;
val[1] = 32*x*(helper_0*helper_1 + helper_0*helper_2 + helper_2*y);}
{val[2] = 32*x*y*(8*x + 8*y + 8*z - 7);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{val[0] = -32*y*(4*y - 1)*(2*x + y + z - 1);}
{double helper_0 = 8*y;
val[1] = -32*x*(helper_0*x + helper_0*z - x + 12*pow(y, 2) - 10*y - z + 1);}
{val[2] = -32*x*y*(4*y - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{double helper_0 = 8*x;
val[0] = -32*y*(helper_0*y + helper_0*z + 12*pow(x, 2) - 10*x - y - z + 1);}
{val[1] = -32*x*(4*x - 1)*(x + 2*y + z - 1);}
{val[2] = -32*x*y*(4*x - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{double helper_0 = x + y + z - 1;
double helper_1 = 4*x;
double helper_2 = helper_1 + 4*y + 4*z - 3;
val[0] = 32*z*(helper_0*helper_1 + helper_0*helper_2 + helper_2*x);}
{val[1] = 32*x*z*(8*x + 8*y + 8*z - 7);}
{double helper_0 = x + y + z - 1;
double helper_1 = 4*z;
double helper_2 = helper_1 + 4*x + 4*y - 3;
val[2] = 32*x*(helper_0*helper_1 + helper_0*helper_2 + helper_2*z);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{val[0] = -32*z*(4*z - 1)*(2*x + y + z - 1);}
{val[1] = -32*x*z*(4*z - 1);}
{double helper_0 = 8*z;
val[2] = -32*x*(helper_0*x + helper_0*y - x - y + 12*pow(z, 2) - 10*z + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
{double helper_0 = 8*x;
val[0] = -32*z*(helper_0*y + helper_0*z + 12*pow(x, 2) - 10*x - y - z + 1);}
{val[1] = -32*x*z*(4*x - 1);}
{val[2] = -32*x*(4*x - 1)*(x + y + 2*z - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
{val[0] = 32*y*z*(8*x - 1);}
{val[1] = 32*x*z*(4*x - 1);}
{val[2] = 32*x*y*(4*x - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
{val[0] = 32*y*z*(4*z - 1);}
{val[1] = 32*x*z*(4*z - 1);}
{val[2] = 32*x*y*(8*z - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_30(double x, double y, double z, double *val) {
{val[0] = 32*y*z*(4*y - 1);}
{val[1] = 32*x*z*(8*y - 1);}
{val[2] = 32*x*y*(4*y - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_31(double x, double y, double z, double *val) {
{val[0] = -32*y*z*(4*y - 1);}
{double helper_0 = 8*y;
val[1] = -32*z*(helper_0*x + helper_0*z - x + 12*pow(y, 2) - 10*y - z + 1);}
{val[2] = -32*y*(4*y - 1)*(x + y + 2*z - 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_32(double x, double y, double z, double *val) {
{val[0] = -32*y*z*(4*z - 1);}
{val[1] = -32*z*(4*z - 1)*(x + 2*y + z - 1);}
{double helper_0 = 8*z;
val[2] = -32*y*(helper_0*x + helper_0*y - x - y + 12*pow(z, 2) - 10*z + 1);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_33(double x, double y, double z, double *val) {
{val[0] = 32*y*z*(8*x + 8*y + 8*z - 7);}
{double helper_0 = x + y + z - 1;
double helper_1 = 4*y;
double helper_2 = helper_1 + 4*x + 4*z - 3;
val[1] = 32*z*(helper_0*helper_1 + helper_0*helper_2 + helper_2*y);}
{double helper_0 = x + y + z - 1;
double helper_1 = 4*z;
double helper_2 = helper_1 + 4*x + 4*y - 3;
val[2] = 32*y*(helper_0*helper_1 + helper_0*helper_2 + helper_2*z);}
}

inline POLYFEM_BOTH void p_4_basis_grad_value_3d_single_34(double x, double y, double z, double *val) {
{val[0] = -256*y*z*(2*x + y + z - 1);}
{val[1] = -256*x*z*(x + 2*y + z - 1);}
{val[2] = -256*x*y*(x + y + 2*z - 1);}
}



POLYFEM_BOTH void p_4_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_20(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_21(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_22(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_23(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_24(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_25(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_26(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_27(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_28(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_29(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_30(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_31(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_32(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_33(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i) {
			p_4_basis_grad_value_3d_single_34(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}


POLYFEM_BOTH int p_basis_count_3d(const int p) {
switch(p) {
	case 0: return 1;
	case 1: return 4;
	case 2: return 10;
	case 3: return 20;
	case 4: return 35;
	default: return (p + 1) * (p + 2) * (p + 3) / 6;
}}

POLYFEM_BOTH void p_basis_value_3d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
if(bernstein) { b_basis_value_3d(p, local_index, x, y, z, val); return; }


switch(p){
	case 0: p_0_basis_value_3d(local_index, x, y, z, val); break;
	case 1: p_1_basis_value_3d(local_index, x, y, z, val); break;
	case 2: p_2_basis_value_3d(local_index, x, y, z, val); break;
	case 3: p_3_basis_value_3d(local_index, x, y, z, val); break;
	case 4: p_4_basis_value_3d(local_index, x, y, z, val); break;
	default:
#ifdef __CUDA_ARCH__
		assert(false && "generic Pn evaluation is host-only");
#else
		p_n_basis_value_3d(p, local_index, x, y, z, val);
#endif
}}

POLYFEM_BOTH void p_grad_basis_value_3d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
if(bernstein) { b_grad_basis_value_3d(p, local_index, x, y, z, grad_x, grad_y, grad_z); return; }


switch(p){
	case 0: p_0_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: p_1_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: p_2_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: p_3_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 4: p_4_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default:
#ifdef __CUDA_ARCH__
		assert(false && "generic Pn evaluation is host-only");
#else
		p_n_basis_grad_value_3d(p, local_index, x, y, z, grad_x, grad_y, grad_z);
#endif
}}


}}
