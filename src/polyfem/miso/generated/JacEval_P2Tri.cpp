#include "JacEval_P2Tri.hpp"
namespace miso {
	Real JacEval_P2Tri(const RealVector<6> &rx, const RealVector<6> &ry, const RealVector<6> &ux, const RealVector<6> &uy, const Real x0, const Real x1) {
		#pragma region Temporaries
		const Real _t0 = 4*rx[1];
		const Real _t1 = 4*ux[1];
		const Real _t2 = 8*x1;
		const Real _t3 = 4*rx[3];
		const Real _t4 = 4*ux[3];
		const Real _t5 = 4*x1;
		const Real _t6 = 4*x0;
		const Real _t7 = 4*rx[0];
		const Real _t8 = _t5*ux[0] + _t6*ux[0] + _t7*x0 + _t7*x1 - 3*rx[0] - 3*ux[0];
		const Real _t9 = 4*ry[3];
		const Real _t10 = 4*uy[3];
		const Real _t11 = 8*x0;
		const Real _t12 = 4*ry[1];
		const Real _t13 = 4*uy[1];
		const Real _t14 = _t5*ry[0] + _t5*uy[0] + _t6*ry[0] + _t6*uy[0] - 3*ry[0] - 3*uy[0];
		#pragma endregion
		return -(-_t0*x0 + _t0 - _t1*x0 + _t1 - _t2*rx[1] - _t2*ux[1] - _t3*x0 - _t4*x0 + _t5*rx[2] + _t5*ux[2] + _t6*rx[4] + _t6*ux[4] + _t8 - rx[2] - ux[2])*(-_t10*x1 + _t10 - _t11*ry[3] - _t11*uy[3] - _t12*x1 - _t13*x1 + _t14 + _t5*ry[4] + _t5*uy[4] + _t6*ry[5] + _t6*uy[5] - _t9*x1 + _t9 - ry[5] - uy[5]) + (-_t0*x1 - _t1*x1 - _t11*rx[3] - _t11*ux[3] - _t3*x1 + _t3 - _t4*x1 + _t4 + _t5*rx[4] + _t5*ux[4] + _t6*rx[5] + _t6*ux[5] + _t8 - rx[5] - ux[5])*(-_t10*x0 - _t12*x0 + _t12 - _t13*x0 + _t13 + _t14 - _t2*ry[1] - _t2*uy[1] + _t5*ry[2] + _t5*uy[2] + _t6*ry[4] + _t6*uy[4] - _t9*x0 - ry[2] - uy[2]);
	}
}
