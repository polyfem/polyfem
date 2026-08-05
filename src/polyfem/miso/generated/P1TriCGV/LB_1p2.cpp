#include "../P1TriCGV.hpp"
namespace miso {
	RealVector<3> P1TriCGV::LB_1p2(const RealVector<3> &_l) {
		#pragma region Expressions
		return {
			_l[0],
			-0.5*_l[0] + 2*_l[1] - 0.5*_l[2],
			_l[2],
		};
		#pragma endregion
	}
}
