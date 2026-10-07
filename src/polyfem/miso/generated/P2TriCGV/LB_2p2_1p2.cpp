#include "../P2TriCGV.hpp"
namespace miso {
	RealVector<18> P2TriCGV::LB_2p2_1p2(const RealVector<18> &_l) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_l[0];
		const Real _t1 = (0.5)*_l[2];
		const Real _t2 = (0.5)*_l[6];
		const Real _t3 = (0.25)*_l[0] - _l[1] + (0.25)*_l[2];
		const Real _t4 = (0.25)*_l[6] - _l[7] + (0.25)*_l[8];
		const Real _t5 = (0.5)*_l[8];
		const Real _t6 = (0.5)*_l[15];
		const Real _t7 = (0.25)*_l[15] - _l[16] + (0.25)*_l[17];
		const Real _t8 = (0.5)*_l[17];
		#pragma endregion
		#pragma region Expressions
		return {
			_l[0],
			2*_l[1] - _t0 - _t1,
			_l[2],
			2*_l[3] - _t0 - _t2,
			-_l[3] + 4*_l[4] - _l[5] + _t3 + _t4,
			2*_l[5] - _t1 - _t5,
			_l[6],
			2*_l[7] - _t2 - _t5,
			_l[8],
			2*_l[9] - _t0 - _t6,
			4*_l[10] - _l[11] - _l[9] + _t3 + _t7,
			2*_l[11] - _t1 - _t8,
			2*_l[12] - _t2 - _t6,
			-_l[12] + 4*_l[13] - _l[14] + _t4 + _t7,
			2*_l[14] - _t5 - _t8,
			_l[15],
			2*_l[16] - _t6 - _t8,
			_l[17],
		};
		#pragma endregion
	}
}
