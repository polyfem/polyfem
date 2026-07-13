#pragma once

#include <polyfem/materials/Dummy.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>

namespace polyfem::assembler
{
	/// To use energy kernel, wrap them in autodiff kernel. For example,
	///
	/// assemble_scalar<AutoDiffScalarKernel<EnergyKernel>>(...);
	/// assemble_vector<AutoDiffGradientVectorKernel<EnergyKernel>>(...);
	/// assemble_matrix<AutoDiffHessianMatrixKernel<EnergyKernel>>(...);
	struct EnergyKernel
	{
		/// Required material type. Use Dummy if you do not need one.
		using Material = material::Dummy;
		/// Dimension of the unknown you are solving. Ex. 3 for displacement in 3D.
		static constexpr int VALUE_DIM = 1;
		/// Dimension of the space you are integrating. Ex. 3 if you do volume integral.
		static constexpr int DIM = 1;
		/// True if this kernel support GPU evaluation.
		static constexpr bool SUPPORT_DEVICE_EVAL = true;
		/// True if energy needs unknown value to evaluate.
		static constexpr bool NEED_UNKNOWN_VALUE = true;
		/// True if energy needs unknown gradient to evaluate.
		static constexpr bool NEED_UNKNOWN_GRAD = true;

		/// Kernel is passed by value, you can store custom data here.
		int custom_data = 0;

		/// @brief Evaluate energy.
		/// @tparam Scalar double or autodiff types.
		/// @param u Unknown vector of size VALUE_DIM. Empty if NEED_UNKNOWN_VALUE is false.
		/// @param gradu Row-major du_i/dx_j matrix of size (VALUE_DIM x DIM). Empty if NEED_UNKNOWN_GRAD is false;
		/// @param material Material required.
		/// @return Scalar energy.
		template <typename Scalar>
		POLYFEM_BOTH Scalar eval_scalar(
			Span<const Scalar> u,
			Span<const Scalar> gradu,
			const Material &material) const;
	};

} // namespace polyfem::assembler
