#pragma once

namespace polyfem::utils
{

	/// @brief Atomic add via CAS. Do target += value.
	/// Replace this with std::atomic_ref once we upgrade to C++20.
	void atomic_add(double &target, double value);

} // namespace polyfem::utils
