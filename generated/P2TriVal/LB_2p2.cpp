#include "../P2TriVal.hpp"
namespace miso {
	RealVector<6> P2TriVal::LB_2p2(const RealVector<6> &_l) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_l[0];
		const Real _t1 = (0.5)*_l[2];
		const Real _t2 = (0.5)*_l[5];
		#pragma endregion
		#pragma region Expressions
		return {
			_l[0],
			2*_l[1] - _t0 - _t1,
			_l[2],
			2*_l[3] - _t0 - _t2,
			2*_l[4] - _t1 - _t2,
			_l[5],
		};
		#pragma endregion
	}
}
