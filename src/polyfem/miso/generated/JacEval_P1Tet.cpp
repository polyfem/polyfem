#include "JacEval_P1Tet.hpp"
namespace miso {
	Real JacEval_P1Tet(const RealVector<4> &rx, const RealVector<4> &ry, const RealVector<4> &rz, const RealVector<4> &ux, const RealVector<4> &uy, const RealVector<4> &uz) {
		#pragma region Temporaries
		const Real _t0 = rx[0] + ux[0];
		const Real _t1 = -_t0 + rx[1] + ux[1];
		const Real _t2 = ry[0] + uy[0];
		const Real _t3 = -_t2 + ry[2] + uy[2];
		const Real _t4 = rz[0] + uz[0];
		const Real _t5 = -_t4 + rz[3] + uz[3];
		const Real _t6 = -_t0 + rx[2] + ux[2];
		const Real _t7 = -_t2 + ry[3] + uy[3];
		const Real _t8 = -_t4 + rz[1] + uz[1];
		const Real _t9 = -_t0 + rx[3] + ux[3];
		const Real _t10 = -_t2 + ry[1] + uy[1];
		const Real _t11 = -_t4 + rz[2] + uz[2];
		#pragma endregion
		return _t1*_t11*_t7 - _t1*_t3*_t5 - _t10*_t11*_t9 + _t10*_t5*_t6 + _t3*_t8*_t9 - _t6*_t7*_t8;
	}
}
