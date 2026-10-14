#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P3Tet(const RealVector<20> &rx, const RealVector<20> &ry, const RealVector<20> &rz, const RealVector<20> &ux, const RealVector<20> &uy, const RealVector<20> &uz, const Real x0, const Real x1, const Real x2);
}
