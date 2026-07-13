#pragma once

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/Span.hpp>

#include <cuda/buffer>
#include <cuda/algorithm>
#include <optional>

namespace polyfem
{
	POLYFEM_BOTH constexpr int div_round_up(int n, int d)
	{
		return (n + d - 1) / d;
	}

	/// @brief Nullable device buffer.
	/// It's very annoying device_buffer does not have default ctor for empty buffer.
	template <typename T>
	using DeviceBuf = std::optional<cuda::device_buffer<T>>;

	template <typename T>
	cuda::device_buffer<T> copy_to_device_async(Span<const T> src, ExecutionPolicy policy)
	{
		auto dst = cuda::make_buffer<T>(*policy.stream, *policy.mr, src.size(), cuda::no_init);
		cuda::copy_bytes(*policy.stream, src, dst);
		return dst;
	}

	template <typename T>
	cuda::device_buffer<T> copy_to_device_async(const T &src, ExecutionPolicy policy)
	{
		auto dst = cuda::make_buffer<T>(*policy.stream, *policy.mr, 1, cuda::no_init);
		cuda::copy_bytes(*policy.stream, Span<const T>{&src, 1}, dst);
		return dst;
	}

	template <typename T>
	T copy_to_host(const T *src, ExecutionPolicy policy)
	{
		T dst;
		cuda::copy_bytes(*policy.stream, Span<const T>{src, 1}, Span<T>{&dst, 1});
		policy.stream->sync();
		return dst;
	}

} // namespace polyfem
