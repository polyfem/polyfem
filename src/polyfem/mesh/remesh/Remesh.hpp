#pragma once

#include <polyfem/State.hpp>

namespace polyfem::mesh
{
	void remesh(State &state, Eigen::MatrixXd &sol, const double time, const double dt);
} // namespace polyfem::mesh
