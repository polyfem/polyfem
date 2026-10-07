#include "../P1TriCGV.hpp"
namespace miso {
	std::array<RealVector<6>, 8> P1TriCGV::subdiv_0_2p1_1p1(const RealVector<6> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = _t0 + _t1;
		const Real _t3 = (0.5)*_b[2];
		const Real _t4 = _t0 + _t3;
		const Real _t5 = (0.25)*_b[0] + (0.25)*_b[1];
		const Real _t6 = (0.25)*_b[2] + (0.25)*_b[3];
		const Real _t7 = _t5 + _t6;
		const Real _t8 = (0.5)*_b[4];
		const Real _t9 = _t0 + _t8;
		const Real _t10 = (0.25)*_b[4] + (0.25)*_b[5];
		const Real _t11 = _t10 + _t5;
		const Real _t12 = (0.5)*_b[3];
		const Real _t13 = _t1 + _t12;
		const Real _t14 = (0.5)*_b[5];
		const Real _t15 = _t1 + _t14;
		const Real _t16 = _t3 + _t8;
		const Real _t17 = _t10 + _t6;
		const Real _t18 = _t14 + _t8;
		const Real _t19 = _t12 + _t14;
		const Real _t20 = _t12 + _t3;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t2,
				_t4,
				_t7,
				_t9,
				_t11,
			},
			{
				_t2,
				_b[1],
				_t7,
				_t13,
				_t11,
				_t15,
			},
			{
				_t9,
				_t11,
				_t16,
				_t17,
				_b[4],
				_t18,
			},
			{
				_t11,
				_t15,
				_t17,
				_t19,
				_t18,
				_b[5],
			},
			{
				_t4,
				_t7,
				_b[2],
				_t20,
				_t16,
				_t17,
			},
			{
				_t7,
				_t13,
				_t20,
				_b[3],
				_t17,
				_t19,
			},
			{
				_t16,
				_t17,
				_t9,
				_t11,
				_t4,
				_t7,
			},
			{
				_t17,
				_t19,
				_t11,
				_t15,
				_t7,
				_t13,
			},
		}};
		#pragma endregion
	}
}
