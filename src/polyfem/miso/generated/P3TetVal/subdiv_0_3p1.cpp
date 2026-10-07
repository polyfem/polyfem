#include "../P3TetVal.hpp"
namespace miso {
	std::array<RealVector<4>, 8> P3TetVal::subdiv_0_3p1(const RealVector<4> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = _t0 + _t1;
		const Real _t3 = (0.5)*_b[2];
		const Real _t4 = _t0 + _t3;
		const Real _t5 = (0.5)*_b[3];
		const Real _t6 = _t0 + _t5;
		const Real _t7 = _t1 + _t5;
		const Real _t8 = _t3 + _t5;
		const Real _t9 = _t1 + _t3;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t2,
				_t4,
				_t6,
			},
			{
				_t6,
				_t7,
				_t8,
				_b[3],
			},
			{
				_t4,
				_t9,
				_b[2],
				_t8,
			},
			{
				_t2,
				_b[1],
				_t9,
				_t7,
			},
			{
				_t6,
				_t7,
				_t2,
				_t4,
			},
			{
				_t6,
				_t7,
				_t8,
				_t4,
			},
			{
				_t4,
				_t9,
				_t7,
				_t2,
			},
			{
				_t4,
				_t9,
				_t7,
				_t8,
			},
		}};
		#pragma endregion
	}
}
