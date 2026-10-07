#include "../P2TriVal.hpp"
namespace miso {
	RealVector<6> P2TriVal::CL_m0(const RealVector<6> &px, const RealVector<6> &py) {
		#pragma region Temporaries
		const Real _t0 = 3*px[0];
		const Real _t1 = -4*px[1];
		const Real _t2 = 3*py[0];
		const Real _t3 = -4*py[3];
		const Real _t4 = -4*px[3];
		const Real _t5 = -4*py[1];
		const Real _t6 = px[0] - px[2];
		const Real _t7 = 2*py[1];
		const Real _t8 = py[0] - 2*py[4];
		const Real _t9 = 2*py[3];
		const Real _t10 = -_t9;
		const Real _t11 = _t10 + py[5];
		const Real _t12 = py[0] - py[2];
		const Real _t13 = 2*px[1];
		const Real _t14 = px[0] - 2*px[4];
		const Real _t15 = 2*px[3];
		const Real _t16 = -_t15;
		const Real _t17 = _t16 + px[5];
		const Real _t18 = 4*py[4];
		const Real _t19 = py[0] - py[5];
		const Real _t20 = 4*px[4];
		const Real _t21 = px[0] - px[5];
		const Real _t22 = -_t7;
		const Real _t23 = _t22 + py[2];
		const Real _t24 = -_t13;
		const Real _t25 = _t24 + px[2];
		const Real _t26 = px[0] + 2*px[4];
		const Real _t27 = py[0] + 2*py[4];
		#pragma endregion
		#pragma region Expressions
		return {
			-(-_t0 - _t1 - px[2])*(-_t2 - _t3 - py[5]) + (-_t0 - _t4 - px[5])*(-_t2 - _t5 - py[2]),
			-_t12*(-_t13 - _t14 - _t17) + _t6*(-_t11 - _t7 - _t8),
			(_t1 + _t20 + _t21)*(_t5 + py[0] + 3*py[2]) - (_t1 + px[0] + 3*px[2])*(_t18 + _t19 + _t5),
			_t19*(-_t14 - _t15 - _t25) - _t21*(-_t23 - _t8 - _t9),
			(_t10 + _t23 + _t27)*(_t17 + _t24 + _t26) - (_t11 + _t22 + _t27)*(_t16 + _t25 + _t26),
			(_t12 + _t18 + _t3)*(_t4 + px[0] + 3*px[5]) - (_t20 + _t4 + _t6)*(_t3 + py[0] + 3*py[5]),
		};
		#pragma endregion
	}
}
