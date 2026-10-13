#pragma once
#include "core/RealVector.hpp"
namespace miso {
	Real JacEval_P4Tri(const RealVector<15> &px, const RealVector<15> &py, const Real x0, const Real x1);
}
