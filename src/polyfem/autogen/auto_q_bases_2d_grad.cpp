#include "auto_q_bases_2d_grad.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH void q_0_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 0;}
{val[1] = 0;}
}



POLYFEM_BOTH void q_0_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_0_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = 1.0*(y - 1);}
{val[1] = 1.0*(x - 1);}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = 1.0*(1 - y);}
{val[1] = -1.0*x;}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = 1.0*y;}
{val[1] = 1.0*x;}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = -1.0*y;}
{val[1] = 1.0*(1 - x);}
}



POLYFEM_BOTH void q_1_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = (4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = (4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 3.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = y*(4.0*x - 1.0)*(2.0*y - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 1.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = y*(4.0*x - 3.0)*(2.0*y - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = -4.0*(2*x - 1)*(y - 1)*(2.0*y - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 12.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -y*(16.0*x - 4.0)*(y - 1);}
{val[1] = -4.0*x*(2.0*x - 1.0)*(2*y - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = -4.0*y*(2*x - 1)*(2.0*y - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 4.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = -y*(16.0*x - 12.0)*(y - 1);}
{val[1] = -4.0*(x - 1)*(2.0*x - 1.0)*(2*y - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_2d_single_8(double x, double y, double *val) {
{val[0] = 16.0*y*(2*x - 1)*(y - 1);}
{val[1] = 16.0*x*(x - 1)*(2*y - 1);}
}



POLYFEM_BOTH void q_2_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_2d_single_8(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_0(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(3.0*helper_0*helper_1 + 1.5*helper_0*helper_2 + 1.0*helper_1*helper_2);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_1(double x, double y, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_2(double x, double y, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_3(double x, double y, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_4(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = -(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = -x*(x - 1)*(3.0*x - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_5(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = (y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 1.5*y - 1.0;
double helper_2 = 3.0*y - 1.0;
val[1] = x*(x - 1)*(3.0*x - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_6(double x, double y, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = y*(y - 1)*(3.0*y - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_7(double x, double y, double *val) {
{double helper_0 = 1.4999999999999998*x - 0.49999999999999989;
double helper_1 = 2.9999999999999996*x - 1.9999999999999996;
val[0] = -y*(y - 1)*(3.0*y - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*x + 6.7499999999999973*helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = -x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_8(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 4.4999999999999991;
val[0] = -y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = -x*(x - 1)*(3.0*x - 1.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_9(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 13.499999999999996*x - 8.9999999999999982;
val[0] = y*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996)*(helper_0*helper_1 + 13.499999999999996*helper_0*x + helper_1*x);}
{double helper_0 = 1.4999999999999998*y - 0.49999999999999989;
double helper_1 = 2.9999999999999996*y - 1.9999999999999996;
val[1] = x*(x - 1)*(3.0*x - 2.0)*(4.4999999999999991*helper_0*helper_1 + 13.499999999999995*helper_0*y + 6.7499999999999973*helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_10(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = y*(y - 1)*(3.0*y - 1.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 4.4999999999999991;
val[1] = (x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_11(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 1.5*x - 1.0;
double helper_2 = 3.0*x - 1.0;
val[0] = -y*(y - 1)*(3.0*y - 2.0)*(13.499999999999996*helper_0*helper_1 + 6.7499999999999982*helper_0*helper_2 + 4.4999999999999991*helper_1*helper_2);}
{double helper_0 = y - 1;
double helper_1 = 13.499999999999996*y - 8.9999999999999982;
val[1] = -(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(helper_0*helper_1 + 13.499999999999996*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_12(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = y*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = x*(x - 1)*(3.0*x - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_13(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 40.499999999999986;
val[0] = -y*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = -x*(x - 1)*(3.0*x - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_14(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = -y*(y - 1)*(3.0*y - 2.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 40.499999999999986;
val[1] = -x*(x - 1)*(3.0*x - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_2d_single_15(double x, double y, double *val) {
{double helper_0 = x - 1;
double helper_1 = 60.749999999999979*x - 20.249999999999993;
val[0] = y*(y - 1)*(3.0*y - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*x + helper_1*x);}
{double helper_0 = y - 1;
double helper_1 = 60.749999999999979*y - 20.249999999999993;
val[1] = x*(x - 1)*(3.0*x - 1.0)*(helper_0*helper_1 + 60.749999999999979*helper_0*y + helper_1*y);}
}



POLYFEM_BOTH void q_3_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_8(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_9(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_10(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_11(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_12(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_13(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_14(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_2d_single_15(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_0(double x, double y, double *val) {
{val[0] = -(y - 1)*(4.0*x + 2.0*y - 3.0);}
{val[1] = -(x - 1)*(2.0*x + 4.0*y - 3.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_1(double x, double y, double *val) {
{val[0] = (y - 1)*(-4.0*x + 2.0*y + 1.0);}
{val[1] = -x*(2.0*x - 4.0*y + 1.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_2(double x, double y, double *val) {
{val[0] = y*(4.0*x + 2.0*y - 3.0);}
{val[1] = x*(2.0*x + 4.0*y - 3.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_3(double x, double y, double *val) {
{val[0] = -y*(-4.0*x + 2.0*y + 1.0);}
{val[1] = (x - 1)*(2.0*x - 4.0*y + 1.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_4(double x, double y, double *val) {
{val[0] = 4*(2*x - 1)*(y - 1);}
{val[1] = 4*x*(x - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_5(double x, double y, double *val) {
{val[0] = -4*y*(y - 1);}
{val[1] = -4*x*(2*y - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_6(double x, double y, double *val) {
{val[0] = -4*y*(2*x - 1);}
{val[1] = -4*x*(x - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_2d_single_7(double x, double y, double *val) {
{val[0] = 4*y*(y - 1);}
{val[1] = 4*(x - 1)*(2*y - 1);}
}



POLYFEM_BOTH void q_m2_basis_grad_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
double gradient[2];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_0(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_1(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_2(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_3(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_4(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_5(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_6(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_2d_single_7(x[i], y[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
		}
		break;
	default: assert(false);
}
}

POLYFEM_BOTH void q_grad_basis_value_2d(const int q, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
switch(q){
	case 0: q_0_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: q_1_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: q_2_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: q_3_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case -2: q_m2_basis_grad_value_2d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default: assert(false);
}}

}}
