#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P2Tet(const RealVector<10> &rx, const RealVector<10> &ry, const RealVector<10> &rz, const RealVector<10> &ux, const RealVector<10> &uy, const RealVector<10> &uz, const Real x0, const Real x1, const Real x2);
}
