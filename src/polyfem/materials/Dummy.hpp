#pragma once

namespace polyfem::material
{
	/// Dummy type for kernel that don't require material.
	struct Dummy
	{
		using ExprType = Dummy;
		using ExprViewType = Dummy;
	};
} // namespace polyfem::material
