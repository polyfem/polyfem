#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P3Tet(const RealVector<20> &px, const RealVector<20> &py, const RealVector<20> &pz, const Real x0, const Real x1, const Real x2);
}
