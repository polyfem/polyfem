#pragma once

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <cassert>

namespace polyfem
{
	struct Range
	{
		int offset = -1;
		int num = -1;

		POLYFEM_BOTH operator bool() const
		{
			return (offset >= 0 && num >= 0);
		}
	};

	/// @brief Slice input span by range. Return empty span if range is empty.
	template <typename T>
	POLYFEM_BOTH Span<T> slice_by_range(Span<T> s, Range r)
	{
		if (!r)
		{
			return {};
		}
		assert(r.offset >= 0 && r.num >= 0);
		assert(r.offset + r.num <= s.size());
		return s.subspan(r.offset, r.num);
	}

} // namespace polyfem
