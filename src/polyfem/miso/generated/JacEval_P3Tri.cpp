#include "JacEval_P3Tri.hpp"
namespace miso {
	Real JacEval_P3Tri(const RealVector<10> &rx, const RealVector<10> &ry, const RealVector<10> &ux, const RealVector<10> &uy, const Real x0, const Real x1) {
		#pragma region Temporaries
		const Real _t0 = pown(x0, 2);
		const Real _t1 = 9*x1;
		const Real _t2 = 27*_t0;
		const Real _t3 = 45*x1;
		const Real _t4 = pown(x1, 2);
		const Real _t5 = (4.5)*x0;
		const Real _t6 = (13.5)*_t0;
		const Real _t7 = (22.5)*x0;
		const Real _t8 = (40.5)*_t4;
		const Real _t9 = 27*x0*x1;
		const Real _t10 = (13.5)*rx[0];
		const Real _t11 = (13.5)*_t4;
		const Real _t12 = 54*x0*x1;
		const Real _t13 = _t0*_t10 + _t10*_t4 + _t11*ux[0] + _t12*rx[5] + _t12*ux[5] + _t6*ux[0] + _t9*rx[0] + _t9*ux[0] - 18*rx[0]*x0 - 18*rx[0]*x1 + (5.5)*rx[0] - 18*ux[0]*x0 - 18*ux[0]*x1 + (5.5)*ux[0];
		const Real _t14 = 9*x0;
		const Real _t15 = 27*_t4;
		const Real _t16 = 45*x0;
		const Real _t17 = (4.5)*x1;
		const Real _t18 = (22.5)*x1;
		const Real _t19 = (40.5)*_t0;
		const Real _t20 = _t11*ry[0] + _t11*uy[0] + _t12*ry[5] + _t12*uy[5] + _t6*ry[0] + _t6*uy[0] + _t9*ry[0] + _t9*uy[0] - 18*ry[0]*x0 - 18*ry[0]*x1 + (5.5)*ry[0] - 18*uy[0]*x0 - 18*uy[0]*x1 + (5.5)*uy[0];
		#pragma endregion
		return -((13.5)*_t0*rx[1] + 27*_t0*rx[4] + (13.5)*_t0*rx[8] + (13.5)*_t0*ux[1] + 27*_t0*ux[4] + (13.5)*_t0*ux[8] - _t1*rx[3] - _t1*ux[3] - _t13 - _t2*rx[5] - _t2*ux[5] - _t3*rx[1] - _t3*ux[1] + (40.5)*_t4*rx[1] + (13.5)*_t4*rx[3] + (40.5)*_t4*ux[1] + (13.5)*_t4*ux[3] - _t5*rx[6] - _t5*rx[8] - _t5*ux[6] - _t5*ux[8] - _t6*rx[7] - _t6*ux[7] - _t7*rx[1] - _t7*rx[4] - _t7*ux[1] - _t7*ux[4] - _t8*rx[2] - _t8*ux[2] - _t9*rx[2] - _t9*ux[2] + 54*rx[1]*x0*x1 + 9*rx[1] + (4.5)*rx[2]*x0 + 36*rx[2]*x1 - 4.5*rx[2] + rx[3] + 27*rx[4]*x0*x1 + 27*rx[5]*x0 + 27*rx[6]*x0*x1 + (4.5)*rx[7]*x0 + 54*ux[1]*x0*x1 + 9*ux[1] + (4.5)*ux[2]*x0 + 36*ux[2]*x1 - 4.5*ux[2] + ux[3] + 27*ux[4]*x0*x1 + 27*ux[5]*x0 + 27*ux[6]*x0*x1 + (4.5)*ux[7]*x0)*((40.5)*_t0*ry[4] + (13.5)*_t0*ry[9] + (40.5)*_t0*uy[4] + (13.5)*_t0*uy[9] - _t11*ry[2] - _t11*uy[2] - _t14*ry[9] - _t14*uy[9] - _t15*ry[5] - _t15*uy[5] - _t16*ry[4] - _t16*uy[4] - _t17*ry[6] - _t17*ry[8] - _t17*uy[6] - _t17*uy[8] - _t18*ry[1] - _t18*ry[4] - _t18*uy[1] - _t18*uy[4] - _t19*ry[7] - _t19*uy[7] - _t20 + 27*_t4*ry[1] + (13.5)*_t4*ry[4] + (13.5)*_t4*ry[6] + 27*_t4*uy[1] + (13.5)*_t4*uy[4] + (13.5)*_t4*uy[6] - _t9*ry[7] - _t9*uy[7] + 27*ry[1]*x0*x1 + (4.5)*ry[2]*x1 + 54*ry[4]*x0*x1 + 9*ry[4] + 27*ry[5]*x1 + 36*ry[7]*x0 + (4.5)*ry[7]*x1 - 4.5*ry[7] + 27*ry[8]*x0*x1 + ry[9] + 27*uy[1]*x0*x1 + (4.5)*uy[2]*x1 + 54*uy[4]*x0*x1 + 9*uy[4] + 27*uy[5]*x1 + 36*uy[7]*x0 + (4.5)*uy[7]*x1 - 4.5*uy[7] + 27*uy[8]*x0*x1 + uy[9]) + ((40.5)*_t0*rx[4] + (13.5)*_t0*rx[9] + (40.5)*_t0*ux[4] + (13.5)*_t0*ux[9] - _t11*rx[2] - _t11*ux[2] - _t13 - _t14*rx[9] - _t14*ux[9] - _t15*rx[5] - _t15*ux[5] - _t16*rx[4] - _t16*ux[4] - _t17*rx[6] - _t17*rx[8] - _t17*ux[6] - _t17*ux[8] - _t18*rx[1] - _t18*rx[4] - _t18*ux[1] - _t18*ux[4] - _t19*rx[7] - _t19*ux[7] + 27*_t4*rx[1] + (13.5)*_t4*rx[4] + (13.5)*_t4*rx[6] + 27*_t4*ux[1] + (13.5)*_t4*ux[4] + (13.5)*_t4*ux[6] - _t9*rx[7] - _t9*ux[7] + 27*rx[1]*x0*x1 + (4.5)*rx[2]*x1 + 54*rx[4]*x0*x1 + 9*rx[4] + 27*rx[5]*x1 + 36*rx[7]*x0 + (4.5)*rx[7]*x1 - 4.5*rx[7] + 27*rx[8]*x0*x1 + rx[9] + 27*ux[1]*x0*x1 + (4.5)*ux[2]*x1 + 54*ux[4]*x0*x1 + 9*ux[4] + 27*ux[5]*x1 + 36*ux[7]*x0 + (4.5)*ux[7]*x1 - 4.5*ux[7] + 27*ux[8]*x0*x1 + ux[9])*((13.5)*_t0*ry[1] + 27*_t0*ry[4] + (13.5)*_t0*ry[8] + (13.5)*_t0*uy[1] + 27*_t0*uy[4] + (13.5)*_t0*uy[8] - _t1*ry[3] - _t1*uy[3] - _t2*ry[5] - _t2*uy[5] - _t20 - _t3*ry[1] - _t3*uy[1] + (40.5)*_t4*ry[1] + (13.5)*_t4*ry[3] + (40.5)*_t4*uy[1] + (13.5)*_t4*uy[3] - _t5*ry[6] - _t5*ry[8] - _t5*uy[6] - _t5*uy[8] - _t6*ry[7] - _t6*uy[7] - _t7*ry[1] - _t7*ry[4] - _t7*uy[1] - _t7*uy[4] - _t8*ry[2] - _t8*uy[2] - _t9*ry[2] - _t9*uy[2] + 54*ry[1]*x0*x1 + 9*ry[1] + (4.5)*ry[2]*x0 + 36*ry[2]*x1 - 4.5*ry[2] + ry[3] + 27*ry[4]*x0*x1 + 27*ry[5]*x0 + 27*ry[6]*x0*x1 + (4.5)*ry[7]*x0 + 54*uy[1]*x0*x1 + 9*uy[1] + (4.5)*uy[2]*x0 + 36*uy[2]*x1 - 4.5*uy[2] + uy[3] + 27*uy[4]*x0*x1 + 27*uy[5]*x0 + 27*uy[6]*x0*x1 + (4.5)*uy[7]*x0);
	}
}
