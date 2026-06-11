#pragma once

#include <polyfem/Common.hpp>

namespace polyfem
{
	namespace mesh
	{
		class Mesh;
	}
} // namespace polyfem

namespace polyfem::varform
{
	bool should_use_isoparametric(const mesh::Mesh &mesh, const json &args);
} // namespace polyfem::varform
