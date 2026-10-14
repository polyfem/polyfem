#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P4Tri(const RealVector<15> &rx, const RealVector<15> &ry, const RealVector<15> &ux, const RealVector<15> &uy, const Real x0, const Real x1);
}
