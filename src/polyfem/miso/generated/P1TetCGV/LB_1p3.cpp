#include "../P1TetCGV.hpp"
namespace miso {
	RealVector<4> P1TetCGV::LB_1p3(const RealVector<4> &_l) {
		#pragma region Expressions
		return {
			_l[0],
			-_R(0.8333333333333333, 0.8333333333333334)*_l[0] + 3*_l[1] - 1.5*_l[2] + (_R(0.3333333333333333, 0.33333333333333337))*_l[3],
			(_R(0.3333333333333333, 0.33333333333333337))*_l[0] - 1.5*_l[1] + 3*_l[2] - _R(0.8333333333333333, 0.8333333333333334)*_l[3],
			_l[3],
		};
		#pragma endregion
	}
}
