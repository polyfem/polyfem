#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P2Tri(const RealVector<6> &px, const RealVector<6> &py, const Real x0, const Real x1);
}
