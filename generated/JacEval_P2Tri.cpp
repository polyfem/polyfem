#include "JacEval_P2Tri.hpp"
namespace miso {
	Real JacEval_P2Tri(const RealVector<6> &px, const RealVector<6> &py, const Real x0, const Real x1) {
		#pragma region Temporaries
		const Real _t0 = 4*px[1];
		const Real _t1 = 8*x1;
		const Real _t2 = 4*px[3];
		const Real _t3 = 4*x1;
		const Real _t4 = 4*x0;
		const Real _t5 = 4*px[0];
		const Real _t6 = _t5*x0 + _t5*x1 - 3*px[0];
		const Real _t7 = 4*py[3];
		const Real _t8 = 8*x0;
		const Real _t9 = 4*py[1];
		const Real _t10 = _t3*py[0] + _t4*py[0] - 3*py[0];
		#pragma endregion
		return -(_t10 + _t3*py[4] + _t4*py[5] - _t7*x1 + _t7 - _t8*py[3] - _t9*x1 - py[5])*(-_t0*x0 + _t0 - _t1*px[1] - _t2*x0 + _t3*px[2] + _t4*px[4] + _t6 - px[2]) + (-_t0*x1 - _t2*x1 + _t2 + _t3*px[4] + _t4*px[5] + _t6 - _t8*px[3] - px[5])*(-_t1*py[1] + _t10 + _t3*py[2] + _t4*py[4] - _t7*x0 - _t9*x0 + _t9 - py[2]);
	}
}
