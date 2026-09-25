#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P3Tri(const RealVector<10> &rx, const RealVector<10> &ry, const RealVector<10> &ux, const RealVector<10> &uy, const Real x0, const Real x1);
}
