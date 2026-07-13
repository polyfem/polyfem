#include "auto_q_bases_3d_grad.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH void q_0_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = 0;}
{val[1] = 0;}
{val[2] = 0;}
}



POLYFEM_BOTH void q_0_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_0_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = -1.0*(y - 1)*(z - 1);}
{val[1] = -1.0*(x - 1)*(z - 1);}
{val[2] = -1.0*(x - 1)*(y - 1);}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = 1.0*(y - 1)*(z - 1);}
{val[1] = 1.0*x*(z - 1);}
{val[2] = 1.0*x*(y - 1);}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = -1.0*y*(z - 1);}
{val[1] = -1.0*x*(z - 1);}
{val[2] = -1.0*x*y;}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = 1.0*y*(z - 1);}
{val[1] = 1.0*(x - 1)*(z - 1);}
{val[2] = 1.0*y*(x - 1);}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = 1.0*z*(y - 1);}
{val[1] = 1.0*z*(x - 1);}
{val[2] = 1.0*(x - 1)*(y - 1);}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = -1.0*z*(y - 1);}
{val[1] = -1.0*x*z;}
{val[2] = -1.0*x*(y - 1);}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = 1.0*y*z;}
{val[1] = 1.0*x*z;}
{val[2] = 1.0*x*y;}
}

inline POLYFEM_BOTH void q_1_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = -1.0*y*z;}
{val[1] = -1.0*z*(x - 1);}
{val[2] = -1.0*y*(x - 1);}
}



POLYFEM_BOTH void q_1_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_1_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = (4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = (x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = (4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 3.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = y*(4.0*x - 1.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = x*(2.0*x - 1.0)*(4.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = y*(4.0*x - 3.0)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = (x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 3.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = z*(4.0*x - 3.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = z*(x - 1)*(2.0*x - 1.0)*(4.0*y - 3.0)*(2.0*z - 1.0);}
{val[2] = (x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = z*(4.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = x*z*(2.0*x - 1.0)*(4.0*y - 3.0)*(2.0*z - 1.0);}
{val[2] = x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = y*z*(4.0*x - 1.0)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = x*z*(2.0*x - 1.0)*(4.0*y - 1.0)*(2.0*z - 1.0);}
{val[2] = x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = y*z*(4.0*x - 3.0)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = z*(x - 1)*(2.0*x - 1.0)*(4.0*y - 1.0)*(2.0*z - 1.0);}
{val[2] = y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(4.0*z - 1.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = -4.0*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 12.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(16.0*z - 12.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = -y*(16.0*x - 4.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*x*(2.0*x - 1.0)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -x*y*(2.0*x - 1.0)*(y - 1)*(16.0*z - 12.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = -4.0*y*(2*x - 1)*(2.0*y - 1.0)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -x*(x - 1)*(16.0*y - 4.0)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -x*y*(x - 1)*(2.0*y - 1.0)*(16.0*z - 12.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -y*(16.0*x - 12.0)*(y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[2] = -y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(16.0*z - 12.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = -z*(16.0*x - 12.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -z*(x - 1)*(2.0*x - 1.0)*(16.0*y - 12.0)*(z - 1);}
{val[2] = -4.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = -z*(16.0*x - 4.0)*(y - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -x*z*(2.0*x - 1.0)*(16.0*y - 12.0)*(z - 1);}
{val[2] = -4.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 4.0)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -x*z*(2.0*x - 1.0)*(16.0*y - 4.0)*(z - 1);}
{val[2] = -4.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 12.0)*(2.0*y - 1.0)*(z - 1);}
{val[1] = -z*(x - 1)*(2.0*x - 1.0)*(16.0*y - 4.0)*(z - 1);}
{val[2] = -4.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = -4.0*z*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = -x*z*(x - 1)*(16.0*y - 12.0)*(2.0*z - 1.0);}
{val[2] = -x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(16.0*z - 4.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 4.0)*(y - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*x*z*(2.0*x - 1.0)*(2*y - 1)*(2.0*z - 1.0);}
{val[2] = -x*y*(2.0*x - 1.0)*(y - 1)*(16.0*z - 4.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = -4.0*y*z*(2*x - 1)*(2.0*y - 1.0)*(2.0*z - 1.0);}
{val[1] = -x*z*(x - 1)*(16.0*y - 4.0)*(2.0*z - 1.0);}
{val[2] = -x*y*(x - 1)*(2.0*y - 1.0)*(16.0*z - 4.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = -y*z*(16.0*x - 12.0)*(y - 1)*(2.0*z - 1.0);}
{val[1] = -4.0*z*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(2.0*z - 1.0);}
{val[2] = -y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(16.0*z - 4.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
{val[0] = y*z*(64.0*x - 48.0)*(y - 1)*(z - 1);}
{val[1] = 16.0*z*(x - 1)*(2.0*x - 1.0)*(2*y - 1)*(z - 1);}
{val[2] = 16.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
{val[0] = y*z*(64.0*x - 16.0)*(y - 1)*(z - 1);}
{val[1] = 16.0*x*z*(2.0*x - 1.0)*(2*y - 1)*(z - 1);}
{val[2] = 16.0*x*y*(2.0*x - 1.0)*(y - 1)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
{val[0] = 16.0*z*(2*x - 1)*(y - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = x*z*(x - 1)*(64.0*y - 48.0)*(z - 1);}
{val[2] = 16.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
{val[0] = 16.0*y*z*(2*x - 1)*(2.0*y - 1.0)*(z - 1);}
{val[1] = x*z*(x - 1)*(64.0*y - 16.0)*(z - 1);}
{val[2] = 16.0*x*y*(x - 1)*(2.0*y - 1.0)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
{val[0] = 16.0*y*(2*x - 1)*(y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[1] = 16.0*x*(x - 1)*(2*y - 1)*(z - 1)*(2.0*z - 1.0);}
{val[2] = x*y*(x - 1)*(y - 1)*(64.0*z - 48.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
{val[0] = 16.0*y*z*(2*x - 1)*(y - 1)*(2.0*z - 1.0);}
{val[1] = 16.0*x*z*(x - 1)*(2*y - 1)*(2.0*z - 1.0);}
{val[2] = x*y*(x - 1)*(y - 1)*(64.0*z - 16.0);}
}

inline POLYFEM_BOTH void q_2_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
{val[0] = -64.0*y*z*(2*x - 1)*(y - 1)*(z - 1);}
{val[1] = -64.0*x*z*(x - 1)*(2*y - 1)*(z - 1);}
{val[2] = -64.0*x*y*(x - 1)*(y - 1)*(2*z - 1);}
}



POLYFEM_BOTH void q_2_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_20(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_21(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_22(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_23(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_24(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_25(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_2_basis_grad_value_3d_single_26(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_20(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_21(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_22(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_23(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_24(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_25(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_26(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_27(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_28(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_29(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_30(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_31(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_32(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_33(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_34(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_35(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_36(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_37(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_38(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_39(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_40(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_41(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_42(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_43(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_44(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_45(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_46(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_47(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_48(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_49(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_50(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_51(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_52(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_53(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_54(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_55(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_56(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_57(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_58(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_59(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_60(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_61(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_62(double x, double y, double z, double *val) {
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

inline POLYFEM_BOTH void q_3_basis_grad_value_3d_single_63(double x, double y, double z, double *val) {
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



POLYFEM_BOTH void q_3_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 20:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_20(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 21:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_21(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 22:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_22(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 23:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_23(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 24:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_24(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 25:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_25(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 26:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_26(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 27:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_27(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 28:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_28(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 29:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_29(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 30:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_30(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 31:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_31(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 32:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_32(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 33:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_33(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 34:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_34(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 35:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_35(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 36:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_36(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 37:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_37(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 38:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_38(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 39:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_39(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 40:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_40(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 41:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_41(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 42:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_42(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 43:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_43(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 44:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_44(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 45:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_45(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 46:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_46(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 47:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_47(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 48:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_48(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 49:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_49(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 50:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_50(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 51:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_51(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 52:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_52(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 53:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_53(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 54:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_54(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 55:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_55(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 56:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_56(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 57:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_57(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 58:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_58(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 59:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_59(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 60:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_60(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 61:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_61(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 62:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_62(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 63:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_3_basis_grad_value_3d_single_63(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_0(double x, double y, double z, double *val) {
{val[0] = (y - 1)*(z - 1)*(4.0*x + 2.0*y + 2.0*z - 3.0);}
{val[1] = (x - 1)*(z - 1)*(2.0*x + 4.0*y + 2.0*z - 3.0);}
{val[2] = (x - 1)*(y - 1)*(2.0*x + 2.0*y + 4.0*z - 3.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_1(double x, double y, double z, double *val) {
{val[0] = -(y - 1)*(z - 1)*(-4.0*x + 2.0*y + 2.0*z + 1.0);}
{val[1] = x*(z - 1)*(2.0*x - 4.0*y - 2.0*z + 1.0);}
{val[2] = x*(y - 1)*(2.0*x - 2.0*y - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_2(double x, double y, double z, double *val) {
{val[0] = -y*(z - 1)*(4.0*x + 2.0*y - 2.0*z - 3.0);}
{val[1] = -x*(z - 1)*(2.0*x + 4.0*y - 2.0*z - 3.0);}
{val[2] = -x*y*(2.0*x + 2.0*y - 4.0*z - 1.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_3(double x, double y, double z, double *val) {
{val[0] = -y*(z - 1)*(4.0*x - 2.0*y + 2.0*z - 1.0);}
{val[1] = -(x - 1)*(z - 1)*(2.0*x - 4.0*y + 2.0*z + 1.0);}
{val[2] = -y*(x - 1)*(2.0*x - 2.0*y + 4.0*z - 1.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_4(double x, double y, double z, double *val) {
{val[0] = -z*(y - 1)*(4.0*x + 2.0*y - 2.0*z - 1.0);}
{val[1] = -z*(x - 1)*(2.0*x + 4.0*y - 2.0*z - 1.0);}
{val[2] = -(x - 1)*(y - 1)*(2.0*x + 2.0*y - 4.0*z + 1.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_5(double x, double y, double z, double *val) {
{val[0] = -z*(y - 1)*(4.0*x - 2.0*y + 2.0*z - 3.0);}
{val[1] = -x*z*(2.0*x - 4.0*y + 2.0*z - 1.0);}
{val[2] = -x*(y - 1)*(2.0*x - 2.0*y + 4.0*z - 3.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_6(double x, double y, double z, double *val) {
{val[0] = y*z*(4.0*x + 2.0*y + 2.0*z - 5.0);}
{val[1] = x*z*(2.0*x + 4.0*y + 2.0*z - 5.0);}
{val[2] = x*y*(2.0*x + 2.0*y + 4.0*z - 5.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_7(double x, double y, double z, double *val) {
{val[0] = y*z*(4.0*x - 2.0*y - 2.0*z + 1.0);}
{val[1] = z*(x - 1)*(2.0*x - 4.0*y - 2.0*z + 3.0);}
{val[2] = y*(x - 1)*(2.0*x - 2.0*y - 4.0*z + 3.0);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_8(double x, double y, double z, double *val) {
{val[0] = -4*(2*x - 1)*(y - 1)*(z - 1);}
{val[1] = -4*x*(x - 1)*(z - 1);}
{val[2] = -4*x*(x - 1)*(y - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_9(double x, double y, double z, double *val) {
{val[0] = 4*y*(y - 1)*(z - 1);}
{val[1] = 4*x*(2*y - 1)*(z - 1);}
{val[2] = 4*x*y*(y - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_10(double x, double y, double z, double *val) {
{val[0] = 4*y*(2*x - 1)*(z - 1);}
{val[1] = 4*x*(x - 1)*(z - 1);}
{val[2] = 4*x*y*(x - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_11(double x, double y, double z, double *val) {
{val[0] = -4*y*(y - 1)*(z - 1);}
{val[1] = -4*(x - 1)*(2*y - 1)*(z - 1);}
{val[2] = -4*y*(x - 1)*(y - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_12(double x, double y, double z, double *val) {
{val[0] = -4*z*(y - 1)*(z - 1);}
{val[1] = -4*z*(x - 1)*(z - 1);}
{val[2] = -4*(x - 1)*(y - 1)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_13(double x, double y, double z, double *val) {
{val[0] = 4*z*(y - 1)*(z - 1);}
{val[1] = 4*x*z*(z - 1);}
{val[2] = 4*x*(y - 1)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_14(double x, double y, double z, double *val) {
{val[0] = -4*y*z*(z - 1);}
{val[1] = -4*x*z*(z - 1);}
{val[2] = -4*x*y*(2*z - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_15(double x, double y, double z, double *val) {
{val[0] = 4*y*z*(z - 1);}
{val[1] = 4*z*(x - 1)*(z - 1);}
{val[2] = 4*y*(x - 1)*(2*z - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_16(double x, double y, double z, double *val) {
{val[0] = 4*z*(2*x - 1)*(y - 1);}
{val[1] = 4*x*z*(x - 1);}
{val[2] = 4*x*(x - 1)*(y - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_17(double x, double y, double z, double *val) {
{val[0] = -4*y*z*(y - 1);}
{val[1] = -4*x*z*(2*y - 1);}
{val[2] = -4*x*y*(y - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_18(double x, double y, double z, double *val) {
{val[0] = -4*y*z*(2*x - 1);}
{val[1] = -4*x*z*(x - 1);}
{val[2] = -4*x*y*(x - 1);}
}

inline POLYFEM_BOTH void q_m2_basis_grad_value_3d_single_19(double x, double y, double z, double *val) {
{val[0] = 4*y*z*(y - 1);}
{val[1] = 4*z*(x - 1)*(2*y - 1);}
{val[2] = 4*y*(x - 1)*(y - 1);}
}



POLYFEM_BOTH void q_m2_basis_grad_value_3d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
assert(grad_x.size() == x.size());
assert(y.size() == x.size());
assert(grad_y.size() == x.size());
assert(z.size() == x.size());
assert(grad_z.size() == x.size());
double gradient[3];
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_0(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_1(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_2(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_3(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_4(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_5(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_6(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_7(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_8(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_9(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_10(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_11(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_12(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_13(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_14(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_15(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 16:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_16(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 17:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_17(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 18:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_18(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	case 19:
		for (std::size_t i = 0; i < x.size(); ++i) {
			q_m2_basis_grad_value_3d_single_19(x[i], y[i], z[i], gradient);
			grad_x[i] = gradient[0];
			grad_y[i] = gradient[1];
			grad_z[i] = gradient[2];
		}
		break;
	default: assert(false);
}
}

POLYFEM_BOTH void q_grad_basis_value_3d(const int q, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z){
switch(q){
	case 0: q_0_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 1: q_1_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 2: q_2_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case 3: q_3_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	case -2: q_m2_basis_grad_value_3d(local_index, x, y, z, grad_x, grad_y, grad_z); break;
	default: assert(false);
}}

}}
