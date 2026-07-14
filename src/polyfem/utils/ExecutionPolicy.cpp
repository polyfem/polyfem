#include <polyfem/utils/ExecutionPolicy.hpp>

#include <stdexcept>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/devices>
#endif

namespace polyfem
{
	ExecutionMode execution_mode_from_string(const std::string &mode)
	{
		if (mode == "CPU" || mode == "cpu")
			return ExecutionMode::CPU;
		if (mode == "Hybrid" || mode == "hybrid")
			return ExecutionMode::Hybrid;
		throw std::runtime_error("Unknown execution mode: " + mode);
	}

	std::string execution_mode_to_string(const ExecutionMode mode)
	{
		switch (mode)
		{
		case ExecutionMode::CPU:
			return "CPU";
		case ExecutionMode::Hybrid:
			return "Hybrid";
		default:
			return "Unknown";
		}
	}

#ifdef POLYFEM_WITH_CUDA
	ExecutionRuntime::CudaData::CudaData(cuda::device_ref selected_device)
		: device(selected_device),
		  stream(device),
		  mr(device)
	{
	}
#endif

	ExecutionRuntime::ExecutionRuntime(const ExecutionMode selected_mode, const int cuda_device)
		: mode(selected_mode)
	{
		if (mode == ExecutionMode::CPU)
			return;

#ifndef POLYFEM_WITH_CUDA
		throw std::runtime_error("Hybrid execution requested, but PolyFEM was built without CUDA support.");
#else
		if (cuda_device < 0 || cuda_device >= static_cast<int>(cuda::devices.size()))
			throw std::runtime_error("Invalid CUDA device index " + std::to_string(cuda_device) + ".");

		cuda_data.emplace(cuda::devices[cuda_device]);
		cudaSetDevice(cuda_device);
#endif
	}

	ExecutionPolicy ExecutionRuntime::policy()
	{
		ExecutionPolicy policy;
		policy.mode = mode;

		if (mode == ExecutionMode::CPU)
			return policy;

#ifndef POLYFEM_WITH_CUDA
		throw std::runtime_error("Hybrid execution requested, but PolyFEM was built without CUDA support.");
#else
		if (!cuda_data)
			throw std::runtime_error("Hybrid execution runtime is not initialized.");

		policy.stream.emplace(cuda_data->stream);
		policy.mr.emplace(cuda_data->mr.as_ref());
		return policy;
#endif
	}
} // namespace polyfem
