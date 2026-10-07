#include "JacEval_P1Tri.hpp"
namespace miso {
	Real JacEval_P1Tri(const RealVector<3> &px, const RealVector<3> &py) {
		return -(-px[0] + px[1])*(-py[0] + py[2]) + (-px[0] + px[2])*(-py[0] + py[1]);
	}
}
