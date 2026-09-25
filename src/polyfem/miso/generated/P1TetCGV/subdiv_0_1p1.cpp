#include "../P1TetCGV.hpp"
namespace miso {
	std::array<RealVector<2>, 2> P1TetCGV::subdiv_0_1p1(const RealVector<2> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[0] + (0.5)*_b[1];
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				_t0,
			},
			{
				_t0,
				_b[1],
			},
		}};
		#pragma endregion
	}
}
