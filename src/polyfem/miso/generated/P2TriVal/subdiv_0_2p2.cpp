#include "../P2TriVal.hpp"
namespace miso {
	std::array<RealVector<6>, 4> P2TriVal::subdiv_0_2p2(const RealVector<6> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = (0.25)*_b[0];
		const Real _t3 = (0.25)*_b[2];
		const Real _t4 = _t1 + _t2 + _t3;
		const Real _t5 = (0.5)*_b[3];
		const Real _t6 = (0.25)*_b[1] + (0.25)*_b[3] + (0.25)*_b[4];
		const Real _t7 = _t2 + _t6;
		const Real _t8 = (0.25)*_b[5];
		const Real _t9 = _t2 + _t5 + _t8;
		const Real _t10 = _t6 + _t8;
		const Real _t11 = (0.5)*_b[4];
		const Real _t12 = _t11 + _t3 + _t8;
		const Real _t13 = (0.5)*_b[5];
		const Real _t14 = (0.5)*_b[2];
		const Real _t15 = _t3 + _t6;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0 + _t1,
				_t4,
				_t0 + _t5,
				_t7,
				_t9,
			},
			{
				_t9,
				_t10,
				_t12,
				_t13 + _t5,
				_t11 + _t13,
				_b[5],
			},
			{
				_t4,
				_t1 + _t14,
				_b[2],
				_t15,
				_t11 + _t14,
				_t12,
			},
			{
				_t12,
				_t10,
				_t9,
				_t15,
				_t7,
				_t4,
			},
		}};
		#pragma endregion
	}
}
