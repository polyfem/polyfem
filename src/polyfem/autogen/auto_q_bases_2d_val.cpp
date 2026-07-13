#include "auto_q_bases_2d_val.hpp"

namespace polyfem {
namespace autogen {
inline POLYFEM_BOTH double q_0_basis_value_2d_single_0(double x, double y) {
double result;
result = 1;
return result;
}



POLYFEM_BOTH void q_0_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_0_basis_value_2d_single_0(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH double q_1_basis_value_2d_single_0(double x, double y) {
double result;
result = 1.0*(x - 1)*(y - 1);
return result;
}

inline POLYFEM_BOTH double q_1_basis_value_2d_single_1(double x, double y) {
double result;
result = -1.0*x*(y - 1);
return result;
}

inline POLYFEM_BOTH double q_1_basis_value_2d_single_2(double x, double y) {
double result;
result = 1.0*x*y;
return result;
}

inline POLYFEM_BOTH double q_1_basis_value_2d_single_3(double x, double y) {
double result;
result = -1.0*y*(x - 1);
return result;
}



POLYFEM_BOTH void q_1_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_1_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_1_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_1_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_1_basis_value_2d_single_3(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH double q_2_basis_value_2d_single_0(double x, double y) {
double result;
result = 1.0*(x - 1)*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_1(double x, double y) {
double result;
result = 1.0*x*(2.0*x - 1.0)*(y - 1)*(2.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_2(double x, double y) {
double result;
result = 1.0*x*y*(2.0*x - 1.0)*(2.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_3(double x, double y) {
double result;
result = 1.0*y*(x - 1)*(2.0*x - 1.0)*(2.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_4(double x, double y) {
double result;
result = -4.0*x*(x - 1)*(y - 1)*(2.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_5(double x, double y) {
double result;
result = -4.0*x*y*(2.0*x - 1.0)*(y - 1);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_6(double x, double y) {
double result;
result = -4.0*x*y*(x - 1)*(2.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_7(double x, double y) {
double result;
result = -4.0*y*(x - 1)*(2.0*x - 1.0)*(y - 1);
return result;
}

inline POLYFEM_BOTH double q_2_basis_value_2d_single_8(double x, double y) {
double result;
result = 16.0*x*y*(x - 1)*(y - 1);
return result;
}



POLYFEM_BOTH void q_2_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_7(x[i], y[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_2_basis_value_2d_single_8(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH double q_3_basis_value_2d_single_0(double x, double y) {
double result;
result = 1.0*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_1(double x, double y) {
double result;
result = -1.0*x*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_2(double x, double y) {
double result;
result = 1.0*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_3(double x, double y) {
double result;
result = -1.0*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_4(double x, double y) {
double result;
result = -4.4999999999999991*x*(x - 1)*(3.0*x - 2.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_5(double x, double y) {
double result;
result = 4.4999999999999991*x*(x - 1)*(3.0*x - 1.0)*(y - 1)*(1.5*y - 1.0)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_6(double x, double y) {
double result;
result = 4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 2.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_7(double x, double y) {
double result;
result = -4.4999999999999991*x*y*(1.4999999999999998*x - 0.49999999999999989)*(2.9999999999999996*x - 1.9999999999999996)*(y - 1)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_8(double x, double y) {
double result;
result = -4.4999999999999991*x*y*(x - 1)*(3.0*x - 1.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_9(double x, double y) {
double result;
result = 4.4999999999999991*x*y*(x - 1)*(3.0*x - 2.0)*(1.4999999999999998*y - 0.49999999999999989)*(2.9999999999999996*y - 1.9999999999999996);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_10(double x, double y) {
double result;
result = 4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_11(double x, double y) {
double result;
result = -4.4999999999999991*y*(x - 1)*(1.5*x - 1.0)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_12(double x, double y) {
double result;
result = 20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 2.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_13(double x, double y) {
double result;
result = -20.249999999999993*x*y*(x - 1)*(3.0*x - 2.0)*(y - 1)*(3.0*y - 1.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_14(double x, double y) {
double result;
result = -20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 2.0);
return result;
}

inline POLYFEM_BOTH double q_3_basis_value_2d_single_15(double x, double y) {
double result;
result = 20.249999999999993*x*y*(x - 1)*(3.0*x - 1.0)*(y - 1)*(3.0*y - 1.0);
return result;
}



POLYFEM_BOTH void q_3_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_7(x[i], y[i]);
		break;
	case 8:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_8(x[i], y[i]);
		break;
	case 9:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_9(x[i], y[i]);
		break;
	case 10:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_10(x[i], y[i]);
		break;
	case 11:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_11(x[i], y[i]);
		break;
	case 12:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_12(x[i], y[i]);
		break;
	case 13:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_13(x[i], y[i]);
		break;
	case 14:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_14(x[i], y[i]);
		break;
	case 15:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_3_basis_value_2d_single_15(x[i], y[i]);
		break;
	default: assert(false);
}
}
inline POLYFEM_BOTH double q_m2_basis_value_2d_single_0(double x, double y) {
double result;
result = -1.0*(x - 1)*(y - 1)*(2*x + 2*y - 1);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_1(double x, double y) {
double result;
result = 1.0*x*(y - 1)*(-2*x + 2*y + 1);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_2(double x, double y) {
double result;
result = x*y*(2.0*x + 2.0*y - 3.0);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_3(double x, double y) {
double result;
result = 1.0*y*(x - 1)*(2*x - 2*y + 1);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_4(double x, double y) {
double result;
result = 4*x*(x - 1)*(y - 1);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_5(double x, double y) {
double result;
result = -4*x*y*(y - 1);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_6(double x, double y) {
double result;
result = -4*x*y*(x - 1);
return result;
}

inline POLYFEM_BOTH double q_m2_basis_value_2d_single_7(double x, double y) {
double result;
result = 4*y*(x - 1)*(y - 1);
return result;
}



POLYFEM_BOTH void q_m2_basis_value_2d(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
assert(val.size() == x.size());
assert(y.size() == x.size());
switch(local_index){
	case 0:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_0(x[i], y[i]);
		break;
	case 1:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_1(x[i], y[i]);
		break;
	case 2:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_2(x[i], y[i]);
		break;
	case 3:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_3(x[i], y[i]);
		break;
	case 4:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_4(x[i], y[i]);
		break;
	case 5:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_5(x[i], y[i]);
		break;
	case 6:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_6(x[i], y[i]);
		break;
	case 7:
		for (std::size_t i = 0; i < x.size(); ++i)
			val[i] = q_m2_basis_value_2d_single_7(x[i], y[i]);
		break;
	default: assert(false);
}
}
POLYFEM_BOTH int q_basis_count_2d(const int q){
switch(q){
	case 0: return 1;
	case 1: return 4;
	case 2: return 9;
	case 3: return 16;
	case -2: return 8;
	default: assert(false); return 0;
}}

POLYFEM_BOTH void q_basis_value_2d(const int q, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val){
switch(q){
	case 0: q_0_basis_value_2d(local_index, x, y, z, val); break;
	case 1: q_1_basis_value_2d(local_index, x, y, z, val); break;
	case 2: q_2_basis_value_2d(local_index, x, y, z, val); break;
	case 3: q_3_basis_value_2d(local_index, x, y, z, val); break;
	case -2: q_m2_basis_value_2d(local_index, x, y, z, val); break;
	default: assert(false);
}}

}}
