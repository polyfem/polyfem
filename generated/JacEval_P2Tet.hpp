#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P2Tet(const RealVector<10> &px, const RealVector<10> &py, const RealVector<10> &pz, const Real x0, const Real x1, const Real x2);
}
