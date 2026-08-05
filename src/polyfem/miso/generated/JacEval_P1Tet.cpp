#include "JacEval_P1Tet.hpp"
namespace miso {
	Real JacEval_P1Tet(const RealVector<4> &px, const RealVector<4> &py, const RealVector<4> &pz) {
		#pragma region Temporaries
		const Real _t0 = -px[0] + px[1];
		const Real _t1 = -py[0] + py[2];
		const Real _t2 = -pz[0] + pz[3];
		const Real _t3 = -px[0] + px[2];
		const Real _t4 = -py[0] + py[3];
		const Real _t5 = -pz[0] + pz[1];
		const Real _t6 = -px[0] + px[3];
		const Real _t7 = -py[0] + py[1];
		const Real _t8 = -pz[0] + pz[2];
		#pragma endregion
		return -_t0*_t1*_t2 + _t0*_t4*_t8 + _t1*_t5*_t6 + _t2*_t3*_t7 - _t3*_t4*_t5 - _t6*_t7*_t8;
	}
}
