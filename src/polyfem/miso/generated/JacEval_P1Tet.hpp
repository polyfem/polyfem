#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P1Tet(const RealVector<4> &rx, const RealVector<4> &ry, const RealVector<4> &rz, const RealVector<4> &ux, const RealVector<4> &uy, const RealVector<4> &uz);
}
