#pragma once

#include <polyfem/State.hpp>

namespace polyfem::mesh
{
	void remesh(State &state, const double t0, const double dt, const int t);
} // namespace polyfem::mesh
