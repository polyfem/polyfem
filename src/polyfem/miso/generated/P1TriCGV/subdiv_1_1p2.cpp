#include "../P1TriCGV.hpp"
namespace miso {
	std::array<RealVector<3>, 2> P1TriCGV::subdiv_1_1p2(const RealVector<3> &_b) {
		#pragma region Temporaries
		const Real _t0 = (0.5)*_b[1];
		const Real _t1 = (0.25)*_b[0] + (0.25)*_b[2] + _t0;
		#pragma endregion
		#pragma region Expressions
		return {{
			{
				_b[0],
				(0.5)*_b[0] + _t0,
				_t1,
			},
			{
				_t1,
				(0.5)*_b[2] + _t0,
				_b[2],
			},
		}};
		#pragma endregion
	}
}
