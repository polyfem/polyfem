#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P2Tri(const RealVector<6> &rx, const RealVector<6> &ry, const RealVector<6> &ux, const RealVector<6> &uy, const Real x0, const Real x1);
}
