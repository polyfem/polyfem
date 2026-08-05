#include "JacEval_P2Tet.hpp"
namespace miso {
	Real JacEval_P2Tet(const RealVector<10> &px, const RealVector<10> &py, const RealVector<10> &pz, const Real x0, const Real x1, const Real x2) {
		#pragma region Temporaries
		const Real _t0 = 4*px[1];
		const Real _t1 = 8*x2;
		const Real _t2 = 4*px[3];
		const Real _t3 = -_t2*x1;
		const Real _t4 = 4*x2;
		const Real _t5 = 4*x1;
		const Real _t6 = 4*x0;
		const Real _t7 = -3*px[0];
		const Real _t8 = 4*px[6];
		const Real _t9 = 4*px[0];
		const Real _t10 = _t9*x0;
		const Real _t11 = _t9*x1;
		const Real _t12 = _t9*x2;
		const Real _t13 = _t10 + _t11 + _t12 + _t7 - _t8*x0;
		const Real _t14 = -_t0*x0 - _t0*x1 + _t0 - _t1*px[1] + _t13 + _t3 + _t4*px[2] + _t5*px[4] + _t6*px[7] - px[2];
		const Real _t15 = -3*py[0];
		const Real _t16 = 4*py[3];
		const Real _t17 = _t6*py[0];
		const Real _t18 = _t5*py[0];
		const Real _t19 = _t4*py[0];
		const Real _t20 = 4*py[1];
		const Real _t21 = -_t20*x2;
		const Real _t22 = 8*x1;
		const Real _t23 = 4*py[6];
		const Real _t24 = -_t23*x0;
		const Real _t25 = _t15 - _t16*x0 - _t16*x2 + _t16 + _t17 + _t18 + _t19 + _t21 - _t22*py[3] + _t24 + _t4*py[4] + _t5*py[5] + _t6*py[8] - py[5];
		const Real _t26 = 4*pz[6];
		const Real _t27 = 8*x0;
		const Real _t28 = 4*pz[3];
		const Real _t29 = -_t28*x1;
		const Real _t30 = -3*pz[0];
		const Real _t31 = 4*pz[1];
		const Real _t32 = _t6*pz[0];
		const Real _t33 = _t5*pz[0];
		const Real _t34 = _t4*pz[0];
		const Real _t35 = _t30 - _t31*x2 + _t32 + _t33 + _t34;
		const Real _t36 = -_t26*x1 - _t26*x2 + _t26 - _t27*pz[6] + _t29 + _t35 + _t4*pz[7] + _t5*pz[8] + _t6*pz[9] - pz[9];
		const Real _t37 = -_t0*x2;
		const Real _t38 = _t13 - _t2*x0 - _t2*x2 + _t2 - _t22*px[3] + _t37 + _t4*px[4] + _t5*px[5] + _t6*px[8] - px[5];
		const Real _t39 = _t15 - _t16*x1 + _t17 + _t18 + _t19;
		const Real _t40 = _t21 - _t23*x1 - _t23*x2 + _t23 - _t27*py[6] + _t39 + _t4*py[7] + _t5*py[8] + _t6*py[9] - py[9];
		const Real _t41 = -_t26*x0;
		const Real _t42 = -_t1*pz[1] + _t29 + _t30 - _t31*x0 - _t31*x1 + _t31 + _t32 + _t33 + _t34 + _t4*pz[2] + _t41 + _t5*pz[4] + _t6*pz[7] - pz[2];
		const Real _t43 = _t10 + _t11 + _t12 - _t27*px[6] + _t3 + _t37 + _t4*px[7] + _t5*px[8] + _t6*px[9] + _t7 - _t8*x1 - _t8*x2 + _t8 - px[9];
		const Real _t44 = -_t1*py[1] - _t20*x0 - _t20*x1 + _t20 + _t24 + _t39 + _t4*py[2] + _t5*py[4] + _t6*py[7] - py[2];
		const Real _t45 = -_t22*pz[3] - _t28*x0 - _t28*x2 + _t28 + _t35 + _t4*pz[4] + _t41 + _t5*pz[5] + _t6*pz[8] - pz[5];
		#pragma endregion
		return -_t14*_t25*_t36 + _t14*_t40*_t45 + _t25*_t42*_t43 + _t36*_t38*_t44 - _t38*_t40*_t42 - _t43*_t44*_t45;
	}
}
