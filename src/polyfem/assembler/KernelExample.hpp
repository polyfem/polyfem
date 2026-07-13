#pragma once

#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/AssemblyData.hpp>
#include <polyfem/materials/Dummy.hpp>
#include <polyfem/utils/CudaBoth.hpp>

namespace polyfem::assembler
{

	// Below are examples of low level assemble kernel, you are required to index and fetch
	// assembly data manually. If the physics you are targeting can be can be represented
	// in the form of scalar energy, please checkout AutoDiffKernelExample.hpp, which provides
	// a simple high level API.

	struct ScalarKernel
	{
		/// Required material type. Use Dummy if you do not need one.
		using Material = material::Dummy;
		/// Dimension of the unknown you are solving. Ex. 3 for displacement in 3D.
		static constexpr int VALUE_DIM = 1;
		/// Dimension of the space you are integrating. Ex. 3 if you do volume integral.
		static constexpr int DIM = 1;
		/// True if this kernel support GPU evaluation.
		static constexpr bool SUPPORT_DEVICE_EVAL = true;

		/// Kernel is passed by value, you can store custom data here.
		int custom_data = 0;

		/// @brief Evaluate element local scalar at quadrature.
		/// @param elem_id Element id.
		/// @param quad_id Quadrature id.
		/// @param data Assembly data.
		/// @param cache Assembly cache. Always non-empty.
		/// @param material Material required.
		/// @param unknown Global unknown vector.
		/// @return Scalar value.
		POLYFEM_BOTH double eval_scalar(
			int elem_id,
			int quad_id,
			const AssemblyDataView &data,
			const ElementAssemblyCacheView &cache,
			const Material &material,
			Span<const double> unknown) const;
	};

	struct VectorKernel
	{
		/// Required material type. Use Dummy if you do not need one.
		using Material = material::Dummy;
		/// Dimension of the unknown you are solving. Ex. 3 for displacement in 3D.
		static constexpr int VALUE_DIM = 1;
		/// Dimension of the space you are integrating. Ex. 3 if you do volume integral.
		static constexpr int DIM = 1;
		/// True if this kernel support GPU evaluation.
		static constexpr bool SUPPORT_DEVICE_EVAL = true;

		/// Kernel is passed by value, you can store custom data here.
		int custom_data = 0;

		/// @brief Evaluate i-th component of element local vector at quadrature.
		/// @param elem_id Element id.
		/// @param quad_id Quadrature id.
		/// @param local_i Element local basis node i.
		/// @param data Assembly data.
		/// @param cache Assembly cache. Always non-empty.
		/// @param material Material required.
		/// @param unknown Global unknown vector.
		/// @return Vector of size value_dim.
		POLYFEM_BOTH Eigen::Vector<double, VALUE_DIM> eval_vector(
			int elem_id,
			int quad_id,
			int local_i,
			const AssemblyDataView &data,
			const ElementAssemblyCacheView &cache,
			const Material &material,
			Span<const double> unknown) const;
	};

	struct MatrixKernel
	{
		/// Required material type. Use Dummy if you do not need one.
		using Material = material::Dummy;
		/// Dimension of the unknown you are solving. Ex. 3 for displacement in 3D.
		static constexpr int VALUE_DIM = 1;
		/// Dimension of the space you are integrating. Ex. 3 if you do volume integral.
		static constexpr int DIM = 1;
		/// True if this kernel support GPU evaluation.
		static constexpr bool SUPPORT_DEVICE_EVAL = true;

		/// Kernel is passed by value, you can store custom data here.
		int custom_data = 0;

		/// @brief Evaluate ij component of element local matrix at quadrature.
		/// @param elem_id Element id.
		/// @param quad_id Quadrature id.
		/// @param local_i Element local basis node i.
		/// @param local_i Element local basis node j.
		/// @param data Assembly data.
		/// @param cache Assembly cache. Always non-empty.
		/// @param material Material required.
		/// @param unknown Global unknown vector.
		/// @return Matrix of size (value_dim x value_dim).
		POLYFEM_BOTH Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor> eval_matrix(
			int elem_id,
			int quad_id,
			int local_i,
			int local_j,
			const AssemblyDataView &data,
			const ElementAssemblyCacheView &cache,
			const Material &material,
			Span<const double> unknown) const;
	};

} // namespace polyfem::assembler
