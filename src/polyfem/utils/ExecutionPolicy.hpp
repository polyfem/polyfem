#pragma once

#include <optional>
#include <string>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/memory_pool>
#include <cuda/stream>
#endif

namespace polyfem
{
	enum class ExecutionMode
	{
		CPU,
		Hybrid
	};

	ExecutionMode execution_mode_from_string(const std::string &mode);
	std::string execution_mode_to_string(ExecutionMode mode);

	struct ExecutionPolicy
	{
		ExecutionMode mode = ExecutionMode::CPU;

#ifdef POLYFEM_WITH_CUDA
		// Must be non-empty if execution mode == Hybrid.

		/// Cuda stream.
		std::optional<cuda::stream_ref> stream;
		/// Cuda device memory pool ref.
		std::optional<cuda::device_memory_pool_ref> mr;
#endif
	};

	/// Owns cuda execution resources, must lives longer than all cuda storage.
	struct ExecutionRuntime
	{
		ExecutionMode mode = ExecutionMode::CPU;

#ifdef POLYFEM_WITH_CUDA
		struct CudaData
		{
			cuda::device_ref device;
			cuda::stream stream;
			cuda::device_memory_pool mr;

			explicit CudaData(cuda::device_ref selected_device);
		};

		std::optional<CudaData> cuda_data;
#endif

		/// Construct default CPU runtime.
		ExecutionRuntime() = default;
		explicit ExecutionRuntime(ExecutionMode mode, int cuda_device = 0);

		ExecutionPolicy policy();
	};
} // namespace polyfem
