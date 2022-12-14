#pragma once

#include <polyfem/State.hpp>

namespace polyfem::mesh
{
	bool remesh(State &state, Eigen::MatrixXd &sol, const double time, const double dt);
} // namespace polyfem::mesh
