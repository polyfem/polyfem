#include "../P1TetCGV.hpp"
namespace miso {
	std::array<RealVector<4>, 2> P1TetCGV::subdiv_0_1p3(const RealVector<4> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[1];
		const Real _t1 = (0.125)*_b[0] + (0.375)*_b[1] + (0.375)*_b[2] + (0.125)*_b[3];
		const Real _t2 = (0.5)*_b[2];
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				(0.5)*_b[0] + _t0,
				(0.25)*_b[0] + (0.25)*_b[2] + _t0,
				_t1,
			},
			{
				_t1,
				(0.25)*_b[1] + (0.25)*_b[3] + _t2,
				(0.5)*_b[3] + _t2,
				_b[3],
			},
		}};
		#pragma endregion
	}
}
