#pragma once

#ifdef POLYFEM_WITH_CUDA
#include <cuda/std/span>
#else
#include <polyfem/utils/tcb_span.hpp>
#endif

namespace polyfem
{
#ifdef POLYFEM_WITH_CUDA
	template <typename T>
	using Span = cuda::std::span<T>;
#else
	template <typename T>
	using Span = tcb::span<T>;
#endif
} // namespace polyfem
