#include <polyfem/utils/AtomicAdd.hpp>

#include <cassert>
#include <cstring>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace polyfem::utils
{
	void atomic_add(double &target, double value)
	{
#if defined(_MSC_VER)
		static_assert(sizeof(double) == sizeof(__int64), "Expected double and __int64 to have the same size.");
		assert(reinterpret_cast<std::uintptr_t>(&target) % alignof(__int64) == 0);

		auto *target_bits = reinterpret_cast<volatile __int64 *>(&target);
		__int64 old_bits = _InterlockedCompareExchange64(target_bits, 0, 0);
		while (true)
		{
			double old_value;
			std::memcpy(&old_value, &old_bits, sizeof(old_value));

			double new_value = old_value + value;
			__int64 new_bits;
			std::memcpy(&new_bits, &new_value, sizeof(new_bits));

			__int64 observed = _InterlockedCompareExchange64(target_bits, new_bits, old_bits);
			if (observed == old_bits)
				break;

			old_bits = observed;
		}
#else
		double old_value;
		__atomic_load(&target, &old_value, __ATOMIC_RELAXED);
		while (true)
		{
			double new_value = old_value + value;
			if (__atomic_compare_exchange(
					&target,
					&old_value,
					&new_value,
					true,
					__ATOMIC_RELAXED,
					__ATOMIC_RELAXED))
			{
				break;
			}
		}
#endif
	}
} // namespace polyfem::utils
