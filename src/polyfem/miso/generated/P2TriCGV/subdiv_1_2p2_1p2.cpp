#include "../P2TriCGV.hpp"
namespace miso {
	std::array<RealVector<18>, 2> P2TriCGV::subdiv_1_2p2_1p2(const RealVector<18> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[1];
		const Real _t1 = (0.25)*_b[0] + (0.25)*_b[2] + _t0;
		const Real _t2 = (0.5)*_b[4];
		const Real _t3 = (0.25)*_b[3] + (0.25)*_b[5] + _t2;
		const Real _t4 = (0.5)*_b[7];
		const Real _t5 = (0.25)*_b[6] + (0.25)*_b[8] + _t4;
		const Real _t6 = (0.5)*_b[10];
		const Real _t7 = (0.25)*_b[11] + (0.25)*_b[9] + _t6;
		const Real _t8 = (0.5)*_b[13];
		const Real _t9 = (0.25)*_b[12] + (0.25)*_b[14] + _t8;
		const Real _t10 = (0.5)*_b[16];
		const Real _t11 = (0.25)*_b[15] + (0.25)*_b[17] + _t10;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				(0.5)*_b[0] + _t0,
				_t1,
				_b[3],
				(0.5)*_b[3] + _t2,
				_t3,
				_b[6],
				(0.5)*_b[6] + _t4,
				_t5,
				_b[9],
				(0.5)*_b[9] + _t6,
				_t7,
				_b[12],
				(0.5)*_b[12] + _t8,
				_t9,
				_b[15],
				(0.5)*_b[15] + _t10,
				_t11,
			},
			{
				_t1,
				(0.5)*_b[2] + _t0,
				_b[2],
				_t3,
				(0.5)*_b[5] + _t2,
				_b[5],
				_t5,
				(0.5)*_b[8] + _t4,
				_b[8],
				_t7,
				(0.5)*_b[11] + _t6,
				_b[11],
				_t9,
				(0.5)*_b[14] + _t8,
				_b[14],
				_t11,
				(0.5)*_b[17] + _t10,
				_b[17],
			},
		}};
		#pragma endregion
	}
}
