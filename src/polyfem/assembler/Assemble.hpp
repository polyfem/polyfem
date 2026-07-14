#pragma once

#include <polyfem/assembler/AssembleOnHost.hpp>
#include <polyfem/utils/DualVector.hpp>

#if defined(POLYFEM_WITH_CUDA) && defined(__CUDACC__)
#include <polyfem/assembler/AssembleOnDevice.cuh>
#endif

namespace polyfem::assembler
{
	/// @brief Assemble scalar value.
	/// @tparam ScalarKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param data Assembly data for FESpace.
	/// @param geom_data Assembly data for geometry mapping.
	/// @param unknown Unknow variables to solve.
	template <typename ScalarKernel>
	double assemble_scalar(
		ScalarKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		double time = 0.0,
		ExecutionPolicy policy = {})
	{
#if defined(POLYFEM_WITH_CUDA) && defined(__CUDACC__)
		if constexpr (ScalarKernel::SUPPORT_DEVICE_EVAL)
		{
			if (policy.mode == ExecutionMode::Hybrid)
			{
				return assemble_scalar_on_device(
					kernel, data, cache, material_registry, unknown, time, policy);
			}
		}
#endif

		return assemble_scalar_on_host(
			kernel, data, geom_data, cache, material_registry, unknown, time);
	}

	/// @brief Assemble scalar value per element.
	/// @tparam ScalarKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param data Assembly data for FESpace.
	/// @param geom_data Assembly data for geometry mapping.
	/// @param unknown Unknow variables to solve.
	/// @param scalar_out Output vector of size element num. Result is accumulated on top of input value.
	template <typename ScalarKernel>
	void assemble_scalar_per_element(
		ScalarKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		DualVector &scalar_out,
		double time = 0.0,
		ExecutionPolicy policy = {})
	{
#if defined(POLYFEM_WITH_CUDA) && defined(__CUDACC__)
		if constexpr (ScalarKernel::SUPPORT_DEVICE_EVAL)
		{
			if (policy.mode == ExecutionMode::Hybrid)
			{
				assemble_scalar_per_element_on_device(
					kernel,
					data,
					cache,
					material_registry,
					unknown,
					scalar_out.device_view(policy),
					policy,
					time);
				return;
			}
		}
#endif

		assemble_scalar_per_element_on_host(
			kernel,
			data,
			geom_data,
			cache,
			material_registry,
			unknown,
			scalar_out.host_view(),
			time);
	}

	/// @brief Assemble vector.
	/// @tparam VectorKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param data Assembly data for FESpace.
	/// @param geom_data Assembly data for geometry mapping.
	/// @param unknown Unknow variables to solve.
	/// @param vector_out Output vector of size total dof num. Result is accumulated on top of input value.
	template <typename VectorKernel>
	void assemble_vector(
		VectorKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		DualVector &vector_out,
		double time = 0.0,
		double extra_scaling = 1.0,
		ExecutionPolicy policy = {})
	{
#if defined(POLYFEM_WITH_CUDA) && defined(__CUDACC__)
		if constexpr (VectorKernel::SUPPORT_DEVICE_EVAL)
		{
			if (policy.mode == ExecutionMode::Hybrid)
			{
				assemble_vector_on_device(
					kernel,
					data,
					cache,
					material_registry,
					unknown,
					vector_out.device_view(policy),
					policy,
					time,
					extra_scaling);
				return;
			}
		}
#endif

		assemble_vector_on_host(
			kernel,
			data,
			geom_data,
			cache,
			material_registry,
			unknown,
			vector_out.host_view(),
			time,
			extra_scaling);
	}

	/// @brief Assemble vector.
	/// @tparam MatrixKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param data Assembly data for FESpace.
	/// @param geom_data Assembly data for geometry mapping.
	/// @param unknown Unknow variables to solve.
	/// @param matrix_out Output matrix of size (total dof x total dof). Result is accumulated
	/// on top of input value. Matrix storage must be pre-allocated.
	template <typename MatrixKernel>
	void assemble_matrix(
		MatrixKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrix &matrix_out,
		bool project_to_psd = false,
		double time = 0.0,
		double extra_scaling = 1.0,
		bool is_mass = false,
		ExecutionPolicy policy = {})
	{
#if defined(POLYFEM_WITH_CUDA) && defined(__CUDACC__)
		if constexpr (MatrixKernel::SUPPORT_DEVICE_EVAL)
		{
			if (policy.mode == ExecutionMode::Hybrid && !project_to_psd)
			{
				assemble_matrix_on_device(
					kernel,
					data,
					cache,
					material_registry,
					unknown,
					matrix_out.device_static_view(policy),
					policy,
					time,
					extra_scaling);
				return;
			}
		}
#endif

		assemble_matrix_on_host(
			kernel,
			data,
			geom_data,
			cache,
			material_registry,
			unknown,
			matrix_out.static_view(),
			project_to_psd,
			time,
			extra_scaling,
			is_mass);
	}
} // namespace polyfem::assembler
