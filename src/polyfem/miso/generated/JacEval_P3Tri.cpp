#include "JacEval_P3Tri.hpp"
namespace miso {
	Real JacEval_P3Tri(const RealVector<10> &px, const RealVector<10> &py, const Real x0, const Real x1) {
		#pragma region Temporaries
		const Real _t0 = pown(x0, 2);
		const Real _t1 = 9*x1;
		const Real _t2 = 45*x1;
		const Real _t3 = pown(x1, 2);
		const Real _t4 = (4.5)*x0;
		const Real _t5 = (13.5)*_t0;
		const Real _t6 = (22.5)*x0;
		const Real _t7 = px[4]*x0;
		const Real _t8 = (40.5)*_t3;
		const Real _t9 = x0*x1;
		const Real _t10 = 27*_t9;
		const Real _t11 = (13.5)*px[0];
		const Real _t12 = 54*_t9;
		const Real _t13 = _t0*_t11 + _t10*px[0] + _t11*_t3 + _t12*px[5] - 18*px[0]*x0 - 18*px[0]*x1 + (5.5)*px[0];
		const Real _t14 = 9*x0;
		const Real _t15 = 27*_t3;
		const Real _t16 = (4.5)*x1;
		const Real _t17 = (13.5)*_t3;
		const Real _t18 = (22.5)*x1;
		const Real _t19 = (40.5)*_t0;
		const Real _t20 = _t10*py[0] + _t12*py[5] + _t17*py[0] + _t5*py[0] - 18*py[0]*x0 - 18*py[0]*x1 + (5.5)*py[0];
		#pragma endregion
		return -((13.5)*_t0*px[1] + 27*_t0*px[4] - 27*_t0*px[5] + (13.5)*_t0*px[8] - _t1*px[3] - _t10*px[2] - _t13 - _t2*px[1] + (40.5)*_t3*px[1] + (13.5)*_t3*px[3] - _t4*px[6] - _t4*px[8] - _t5*px[7] - _t6*px[1] - 22.5*_t7 - _t8*px[2] + 54*px[1]*x0*x1 + 9*px[1] + (4.5)*px[2]*x0 + 36*px[2]*x1 - 4.5*px[2] + px[3] + 27*px[4]*x0*x1 + 27*px[5]*x0 + 27*px[6]*x0*x1 + (4.5)*px[7]*x0)*((40.5)*_t0*py[4] + (13.5)*_t0*py[9] - _t10*py[7] - _t14*py[9] - _t15*py[5] - _t16*py[6] - _t16*py[8] - _t17*py[2] - _t18*py[1] - _t18*py[4] - _t19*py[7] - _t20 + 27*_t3*py[1] + (13.5)*_t3*py[4] + (13.5)*_t3*py[6] + 27*py[1]*x0*x1 + (4.5)*py[2]*x1 + 54*py[4]*x0*x1 - 45*py[4]*x0 + 9*py[4] + 27*py[5]*x1 + 36*py[7]*x0 + (4.5)*py[7]*x1 - 4.5*py[7] + 27*py[8]*x0*x1 + py[9]) + ((40.5)*_t0*px[4] + (13.5)*_t0*px[9] - _t10*px[7] - _t13 - _t14*px[9] - _t15*px[5] - _t16*px[6] - _t16*px[8] - _t17*px[2] - _t18*px[1] - _t18*px[4] - _t19*px[7] + 27*_t3*px[1] + (13.5)*_t3*px[4] + (13.5)*_t3*px[6] - 45*_t7 + 27*px[1]*x0*x1 + (4.5)*px[2]*x1 + 54*px[4]*x0*x1 + 9*px[4] + 27*px[5]*x1 + 36*px[7]*x0 + (4.5)*px[7]*x1 - 4.5*px[7] + 27*px[8]*x0*x1 + px[9])*((13.5)*_t0*py[1] + 27*_t0*py[4] - 27*_t0*py[5] + (13.5)*_t0*py[8] - _t1*py[3] - _t10*py[2] - _t2*py[1] - _t20 + (40.5)*_t3*py[1] + (13.5)*_t3*py[3] - _t4*py[6] - _t4*py[8] - _t5*py[7] - _t6*py[1] - _t6*py[4] - _t8*py[2] + 54*py[1]*x0*x1 + 9*py[1] + (4.5)*py[2]*x0 + 36*py[2]*x1 - 4.5*py[2] + py[3] + 27*py[4]*x0*x1 + 27*py[5]*x0 + 27*py[6]*x0*x1 + (4.5)*py[7]*x0);
	}
}
