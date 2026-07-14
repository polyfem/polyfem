#include "auto_q_bases_1d_grad.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH void q_0_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = 0;}
}



POLYFEM_BOTH void q_0_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_0_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = -1.0;}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = 1.0;}
}



POLYFEM_BOTH void q_1_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = 4.0*x - 3.0;}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = 4.0 - 8.0*x;}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_1d_single_2(double x, double *val) {
{val[0] = 4.0*x - 1.0;}
}



POLYFEM_BOTH void q_2_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_1d_single_2(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = -13.5*pow(x, 2) + 18.0*x - 5.5;}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = 40.499999999999986*pow(x, 2) - 44.999999999999986*x + 8.9999999999999982;}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_1d_single_2(double x, double *val) {
{val[0] = -40.499999999999986*pow(x, 2) + 35.999999999999993*x - 4.4999999999999991;}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_1d_single_3(double x, double *val) {
{val[0] = 13.499999999999996*pow(x, 2) - 8.9999999999999964*x + 0.99999999999999956;}
}



POLYFEM_BOTH void q_3_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_1d_single_2(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_1d_single_3(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = 4.0*x - 3.0;}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = 4.0 - 8.0*x;}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_1d_single_2(double x, double *val) {
{val[0] = 4.0*x - 1.0;}
}



POLYFEM_BOTH void q_m2_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_1d_single_2(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_4_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = 42.666666666666664*pow(x, 3) - 80.0*pow(x, 2) + 46.666666666666657*x - 8.3333333333333339;}
}

inline POLYFEM_BOTH void q_4_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = -170.66666666666666*pow(x, 3) + 288.0*pow(x, 2) - 138.66666666666666*x + 16.0;}
}

inline POLYFEM_BOTH void q_4_basis_grad_value_1d_single_2(double x, double *val) {
{val[0] = 256.0*pow(x, 3) - 384.0*pow(x, 2) + 152.0*x - 12.0;}
}

inline POLYFEM_BOTH void q_4_basis_grad_value_1d_single_3(double x, double *val) {
{val[0] = -170.66666666666666*pow(x, 3) + 223.99999999999997*pow(x, 2) - 74.666666666666657*x + 5.333333333333333;}
}

inline POLYFEM_BOTH void q_4_basis_grad_value_1d_single_4(double x, double *val) {
{val[0] = 42.666666666666664*pow(x, 3) - 48.0*pow(x, 2) + 14.666666666666666*x - 1.0;}
}



POLYFEM_BOTH void q_4_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_4_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_4_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_4_basis_grad_value_1d_single_2(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_4_basis_grad_value_1d_single_3(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_4_basis_grad_value_1d_single_4(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_5_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = -130.20833333333337*pow(x, 4) + 312.5*pow(x, 3) - 265.625*pow(x, 2) + 93.75*x - 11.416666666666668;}
}

inline POLYFEM_BOTH void q_5_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = 651.04166666666652*pow(x, 4) - 1458.333333333333*pow(x, 3) + 1109.3749999999998*pow(x, 2) - 320.83333333333331*x + 25.0;}
}

inline POLYFEM_BOTH void q_5_basis_grad_value_1d_single_2(double x, double *val) {
{val[0] = -1302.0833333333337*pow(x, 4) + 2708.3333333333339*pow(x, 3) - 1843.7500000000005*pow(x, 2) + 445.83333333333348*x - 25.000000000000007;}
}

inline POLYFEM_BOTH void q_5_basis_grad_value_1d_single_3(double x, double *val) {
{val[0] = 1302.0833333333333*pow(x, 4) - 2500.0*pow(x, 3) + 1531.25*pow(x, 2) - 324.99999999999994*x + 16.666666666666664;}
}

inline POLYFEM_BOTH void q_5_basis_grad_value_1d_single_4(double x, double *val) {
{val[0] = -651.0416666666664*pow(x, 4) + 1145.833333333333*pow(x, 3) - 640.62499999999977*pow(x, 2) + 127.08333333333327*x - 6.2499999999999982;}
}

inline POLYFEM_BOTH void q_5_basis_grad_value_1d_single_5(double x, double *val) {
{val[0] = 130.20833333333337*pow(x, 4) - 208.3333333333334*pow(x, 3) + 109.37500000000003*pow(x, 2) - 20.833333333333339*x + 1.0000000000000002;}
}



POLYFEM_BOTH void q_5_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_5_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_5_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_5_basis_grad_value_1d_single_2(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_5_basis_grad_value_1d_single_3(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_5_basis_grad_value_1d_single_4(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_5_basis_grad_value_1d_single_5(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_0(double x, double *val) {
{val[0] = 388.80000000000001*pow(x, 5) - 1134.0*pow(x, 4) + 1260.0*pow(x, 3) - 661.5*pow(x, 2) + 162.39999999999998*x - 14.699999999999999;}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_1(double x, double *val) {
{val[0] = -2332.7999999999988*pow(x, 5) + 6479.9999999999982*pow(x, 4) - 6695.9999999999973*pow(x, 3) + 3131.9999999999986*pow(x, 2) - 626.39999999999975*x + 35.999999999999986;}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_2(double x, double *val) {
{val[0] = 5831.9999999999982*pow(x, 5) - 15389.999999999993*pow(x, 4) + 14795.999999999993*pow(x, 3) - 6223.4999999999982*pow(x, 2) + 1052.9999999999995*x - 44.999999999999986;}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_3(double x, double *val) {
{val[0] = -7775.9999999999991*pow(x, 5) + 19439.999999999996*pow(x, 4) - 17423.999999999993*pow(x, 3) + 6695.9999999999973*pow(x, 2) - 1015.9999999999997*x + 39.999999999999986;}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_4(double x, double *val) {
{val[0] = 5831.9999999999982*pow(x, 5) - 13769.999999999996*pow(x, 4) + 11555.999999999996*pow(x, 3) - 4144.4999999999982*pow(x, 2) + 593.99999999999989*x - 22.499999999999993;}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_5(double x, double *val) {
{val[0] = -2332.7999999999988*pow(x, 5) + 5183.9999999999982*pow(x, 4) - 4103.9999999999982*pow(x, 3) + 1403.9999999999995*pow(x, 2) - 194.39999999999989*x + 7.1999999999999957;}
}

inline POLYFEM_BOTH void q_6_basis_grad_value_1d_single_6(double x, double *val) {
{val[0] = 388.80000000000001*pow(x, 5) - 810.0*pow(x, 4) + 612.0*pow(x, 3) - 202.5*pow(x, 2) + 27.399999999999995*x - 0.99999999999999967;}
}



POLYFEM_BOTH void q_6_basis_grad_value_1d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
double gradient[1];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_0(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_1(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_2(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_3(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_4(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_5(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_6_basis_grad_value_1d_single_6(x[i], gradient);
			grad_x[i] = gradient[0];
		}
		break;
	default: assert(false);
}
}

POLYFEM_BOTH void q_grad_basis_value_1d(const int q, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
switch(q){
	case 0: q_0_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: q_1_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: q_2_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: q_3_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case -2: q_m2_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 4: q_4_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 5: q_5_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 6: q_6_basis_grad_value_1d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default: assert(false);
}}

}}
