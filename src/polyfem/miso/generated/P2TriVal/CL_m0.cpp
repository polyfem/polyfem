#include "../P2TriVal.hpp"
namespace miso {
	RealVector<6> P2TriVal::CL_m0(const RealVector<6> &rx, const RealVector<6> &ry, const RealVector<6> &ux, const RealVector<6> &uy) {
		#pragma region Temporaries
		const Real _t0 = 3*rx[0] + 3*ux[0];
		const Real _t1 = -4*rx[1] - 4*ux[1];
		const Real _t2 = rx[2] + ux[2];
		const Real _t3 = 3*ry[0] + 3*uy[0];
		const Real _t4 = -4*ry[3] - 4*uy[3];
		const Real _t5 = ry[5] + uy[5];
		const Real _t6 = -4*rx[3] - 4*ux[3];
		const Real _t7 = rx[5] + ux[5];
		const Real _t8 = -4*ry[1] - 4*uy[1];
		const Real _t9 = ry[2] + uy[2];
		const Real _t10 = rx[0] + ux[0];
		const Real _t11 = _t10 - rx[2] - ux[2];
		const Real _t12 = 2*ry[1];
		const Real _t13 = 2*uy[1];
		const Real _t14 = ry[0] + uy[0];
		const Real _t15 = _t14 - 2*ry[4] - 2*uy[4];
		const Real _t16 = 2*ry[3];
		const Real _t17 = 2*uy[3];
		const Real _t18 = -_t16 - _t17;
		const Real _t19 = _t18 + _t5;
		const Real _t20 = _t14 - ry[2] - uy[2];
		const Real _t21 = 2*rx[1];
		const Real _t22 = 2*ux[1];
		const Real _t23 = _t10 - 2*rx[4] - 2*ux[4];
		const Real _t24 = 2*rx[3];
		const Real _t25 = 2*ux[3];
		const Real _t26 = -_t24 - _t25;
		const Real _t27 = _t26 + _t7;
		const Real _t28 = _t14 - ry[5] - uy[5];
		const Real _t29 = 4*ry[4] + 4*uy[4];
		const Real _t30 = _t10 - rx[5] - ux[5];
		const Real _t31 = 4*rx[4] + 4*ux[4];
		const Real _t32 = -_t12 - _t13;
		const Real _t33 = _t32 + _t9;
		const Real _t34 = -_t21 - _t22;
		const Real _t35 = _t2 + _t34;
		const Real _t36 = _t10 + 2*rx[4] + 2*ux[4];
		const Real _t37 = _t14 + 2*ry[4] + 2*uy[4];
		#pragma endregion
		#pragma region Expressions
		return {
			-(-_t0 - _t1 - _t2)*(-_t3 - _t4 - _t5) + (-_t0 - _t6 - _t7)*(-_t3 - _t8 - _t9),
			_t11*(-_t12 - _t13 - _t15 - _t19) - _t20*(-_t21 - _t22 - _t23 - _t27),
			(_t1 + _t30 + _t31)*(_t14 + _t8 + 3*ry[2] + 3*uy[2]) - (_t28 + _t29 + _t8)*(_t1 + _t10 + 3*rx[2] + 3*ux[2]),
			_t28*(-_t23 - _t24 - _t25 - _t35) - _t30*(-_t15 - _t16 - _t17 - _t33),
			(_t18 + _t33 + _t37)*(_t27 + _t34 + _t36) - (_t19 + _t32 + _t37)*(_t26 + _t35 + _t36),
			-(_t11 + _t31 + _t6)*(_t14 + _t4 + 3*ry[5] + 3*uy[5]) + (_t20 + _t29 + _t4)*(_t10 + _t6 + 3*rx[5] + 3*ux[5]),
		};
		#pragma endregion
	}
}
