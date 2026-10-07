#include "../P3TriCGV.hpp"
namespace miso {
	std::array<RealVector<6>, 2> P3TriCGV::subdiv_1_2p1_1p1(const RealVector<6> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0] + (0.5)*_b[1];
		const Real _t1 = (0.5)*_b[2] + (0.5)*_b[3];
		const Real _t2 = (0.5)*_b[4] + (0.5)*_b[5];
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0,
				_b[2],
				_t1,
				_b[4],
				_t2,
			},
			{
				_t0,
				_b[1],
				_t1,
				_b[3],
				_t2,
				_b[5],
			},
		}};
		#pragma endregion
	}
}
