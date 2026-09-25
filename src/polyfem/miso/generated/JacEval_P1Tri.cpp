#include "JacEval_P1Tri.hpp"
namespace miso {
	Real JacEval_P1Tri(const RealVector<3> &rx, const RealVector<3> &ry, const RealVector<3> &ux, const RealVector<3> &uy) {
		#pragma region Temporaries
		const Real _t0 = rx[0] + ux[0];
		const Real _t1 = ry[0] + uy[0];
		#pragma endregion
		return -(-_t0 + rx[1] + ux[1])*(-_t1 + ry[2] + uy[2]) + (-_t0 + rx[2] + ux[2])*(-_t1 + ry[1] + uy[1]);
	}
}
