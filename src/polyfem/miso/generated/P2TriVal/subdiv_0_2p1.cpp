#include "../P2TriVal.hpp"
namespace miso {
	std::array<RealVector<3>, 4> P2TriVal::subdiv_0_2p1(const RealVector<3> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0];
		const Real _t1 = (0.5)*_b[1];
		const Real _t2 = _t0 + _t1;
		const Real _t3 = (0.5)*_b[2];
		const Real _t4 = _t0 + _t3;
		const Real _t5 = _t1 + _t3;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t2,
				_t4,
			},
			{
				_t4,
				_t5,
				_b[2],
			},
			{
				_t2,
				_b[1],
				_t5,
			},
			{
				_t5,
				_t4,
				_t2,
			},
		}};
		#pragma endregion
	}
}
