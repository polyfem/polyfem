#pragma once

#include <polyfem/State.hpp>

namespace polyfem::mesh
{
	void remesh(State &state, const double time, const double dt);
} // namespace polyfem::mesh
