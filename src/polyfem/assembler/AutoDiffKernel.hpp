#pragma once

#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/AssemblyData.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/AutoDiff.hpp>

#include <Eigen/Core>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/std/type_traits>
#else
#include <type_traits>
#endif

namespace polyfem::assembler
{
	namespace detail
	{
		struct NoStorage
		{
		};

		template <int value_dim>
		POLYFEM_BOTH Eigen::Vector<double, value_dim> local_node_unknown(
			int elem_id,
			int local_node_id,
			const AssemblyDataView &data,
			Span<const double> unknown)
		{
			using Vec = Eigen::Vector<double, value_dim>;

			auto &elem_desc = data.element_desc[elem_id];
			auto &mappings = data.dof_mapping_store;
			int mapping_id = elem_desc.dof_mapping_range.offset + local_node_id;

			auto node_ids = mappings.get_node_ids(mapping_id);
			auto node_weights = mappings.get_weights(mapping_id);

			Vec ret = Vec::Zero();
			for (int i = 0; i < node_ids.size(); ++i)
			{
				ret += Eigen::Map<const Vec>(unknown.data() + value_dim * node_ids[i]) * node_weights[i];
			}
			return ret;
		}

	} // namespace detail

#ifdef POLYFEM_WITH_CUDA
	template <bool cond, typename If, typename Then>
	using Conditional = cuda::std::conditional_t<cond, If, Then>;
#else
	template <bool cond, typename If, typename Then>
	using Conditional = std::conditional_t<cond, If, Then>;
#endif

	template <typename EnergyKernel>
	struct AutoDiffScalarKernel
	{
		using Material = typename EnergyKernel::Material;
		static constexpr int VALUE_DIM = EnergyKernel::VALUE_DIM;
		static constexpr int DIM = EnergyKernel::DIM;
		static constexpr bool SUPPORT_DEVICE_EVAL = EnergyKernel::SUPPORT_DEVICE_EVAL;

		EnergyKernel energy;

		POLYFEM_BOTH double eval_scalar(
			int elem_id,
			int quad_id,
			const AssemblyDataView &data,
			const ElementAssemblyCacheView &cache,
			const Material &material,
			Span<const double> unknown) const
		{

			using Vec1 = Eigen::Vector<double, VALUE_DIM>;
			using Vec2 = Eigen::Vector<double, DIM>;
			using Mat = Eigen::Matrix<double, VALUE_DIM, DIM, Eigen::RowMajor>;

			constexpr int NEED_UNKNOWN_VALUE = EnergyKernel::NEED_UNKNOWN_VALUE;
			constexpr int NEED_UNKNOWN_GRAD = EnergyKernel::NEED_UNKNOWN_GRAD;

			auto &elem_desc = data.element_desc[elem_id];
			int basis_num = elem_desc.basis_desc.basis_num;

			Vec1 u_value = Vec1::Zero();
			Mat gradu_value = Mat::Zero();
			for (int b = 0; b < basis_num; ++b)
			{
				if constexpr (NEED_UNKNOWN_VALUE)
				{
					double phi = cache.get_basis_value(b, quad_id);
					Vec1 local_node_unknown = detail::local_node_unknown<VALUE_DIM>(elem_id, b, data, unknown);
					u_value += phi * local_node_unknown;
				}
				if constexpr (NEED_UNKNOWN_GRAD)
				{
					Vec2 grad_phi = cache.get_basis_grad_phy<DIM>(b, quad_id);
					Vec1 local_node_unknown = detail::local_node_unknown<VALUE_DIM>(elem_id, b, data, unknown);
					gradu_value += local_node_unknown * grad_phi.transpose();
				}
			}

			Span<const double> u;
			if constexpr (NEED_UNKNOWN_VALUE)
			{
				u = Span<const double>(u_value.data(), u_value.size());
			}
			Span<const double> gradu;
			if constexpr (NEED_UNKNOWN_GRAD)
			{
				gradu = Span<const double>(gradu_value.data(), gradu_value.size());
			}

			return energy.template eval_scalar<double>(u, gradu, material);
		}
	};

	template <typename EnergyKernel>
	struct AutoDiffGradientVectorKernel
	{
		using Material = typename EnergyKernel::Material;
		static constexpr int VALUE_DIM = EnergyKernel::VALUE_DIM;
		static constexpr int DIM = EnergyKernel::DIM;
		static constexpr bool SUPPORT_DEVICE_EVAL = EnergyKernel::SUPPORT_DEVICE_EVAL;

		EnergyKernel energy;

		POLYFEM_BOTH Eigen::Vector<double, VALUE_DIM> eval_vector(
			int elem_id,
			int quad_id,
			int local_i,
			const AssemblyDataView &data,
			const ElementAssemblyCacheView &cache,
			const Material &material,
			Span<const double> unknown) const
		{
			using Vec1 = Eigen::Vector<double, VALUE_DIM>;
			using Vec2 = Eigen::Vector<double, DIM>;
			using Mat = Eigen::Matrix<double, VALUE_DIM, DIM, Eigen::RowMajor>;

			constexpr int NEED_UNKNOWN_VALUE = EnergyKernel::NEED_UNKNOWN_VALUE;
			constexpr int NEED_UNKNOWN_GRAD = EnergyKernel::NEED_UNKNOWN_GRAD;

			auto &elem_desc = data.element_desc[elem_id];
			int basis_num = elem_desc.basis_desc.basis_num;

			Vec1 u_value = Vec1::Zero();
			Mat gradu_value = Mat::Zero();
			for (int b = 0; b < basis_num; ++b)
			{
				if constexpr (NEED_UNKNOWN_VALUE)
				{
					double phi = cache.get_basis_value(b, quad_id);
					Vec1 local_node_unknown = detail::local_node_unknown<VALUE_DIM>(elem_id, b, data, unknown);
					u_value += phi * local_node_unknown;
				}
				if constexpr (NEED_UNKNOWN_GRAD)
				{
					Vec2 grad_phi = cache.get_basis_grad_phy<DIM>(b, quad_id);
					Vec1 local_node_unknown = detail::local_node_unknown<VALUE_DIM>(elem_id, b, data, unknown);
					gradu_value += local_node_unknown * grad_phi.transpose();
				}
			}

			using AD = autodiff::Double1<VALUE_DIM>;
			using ADGrad = typename AD::Grad;
			double phi_i = cache.get_basis_value(local_i, quad_id);
			Vec2 grad_phi_i = cache.get_basis_grad_phy<DIM>(local_i, quad_id);

			// Seed autodiff unknown vector u.
			// If NEED_UNKNOWN_VALUE, autodiff_u is Eigen::Matrix<AD, VALUE_DIM, 1>. Else empty dummy type.
			Conditional<NEED_UNKNOWN_VALUE, Eigen::Matrix<AD, VALUE_DIM, 1>, detail::NoStorage> autodiff_u;
			if constexpr (NEED_UNKNOWN_VALUE)
			{
				for (int vd = 0; vd < VALUE_DIM; ++vd)
				{
					ADGrad g = ADGrad::Zero();
					g(vd) = phi_i;
					autodiff_u(vd) = AD(u_value(vd), g);
				}
			}

			// Seed autodiff unknown gradient matrix gradu.
			// If NEED_UNKNOWN_GRAD, autodiff_u is Eigen::Matrix<AD, VALUE_DIM, DIM>. Else empty dummy type.
			Conditional<NEED_UNKNOWN_GRAD, Eigen::Matrix<AD, VALUE_DIM, DIM>, detail::NoStorage> autodiff_gradu;
			if constexpr (NEED_UNKNOWN_GRAD)
			{
				for (int vd = 0; vd < VALUE_DIM; ++vd)
				{
					for (int d = 0; d < DIM; ++d)
					{
						ADGrad g = ADGrad::Zero();
						g(vd) = grad_phi_i(d);
						autodiff_gradu(vd, d) = AD(gradu_value(vd, d), g);
					}
				}
			}

			Span<const AD> u;
			if constexpr (NEED_UNKNOWN_VALUE)
			{
				u = Span<const AD>(autodiff_u.data(), autodiff_u.size());
			}
			Span<const AD> gradu;
			if constexpr (NEED_UNKNOWN_GRAD)
			{
				gradu = Span<const AD>(autodiff_gradu.data(), autodiff_gradu.size());
			}

			AD energy_value = energy.template eval_scalar<AD>(u, gradu, material);

			return energy_value.get_grad();
		}
	};

	template <typename EnergyKernel>
	struct AutoDiffHessianMatrixKernel
	{
		using Material = typename EnergyKernel::Material;
		static constexpr int VALUE_DIM = EnergyKernel::VALUE_DIM;
		static constexpr int DIM = EnergyKernel::DIM;
		static constexpr bool SUPPORT_DEVICE_EVAL = EnergyKernel::SUPPORT_DEVICE_EVAL;

		using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

		EnergyKernel energy;

		POLYFEM_BOTH Mat eval_matrix(
			int elem_id,
			int quad_id,
			int local_i,
			int local_j,
			const AssemblyDataView &data,
			const ElementAssemblyCacheView &cache,
			const Material &material,
			Span<const double> unknown) const
		{
			using Vec1 = Eigen::Vector<double, VALUE_DIM>;
			using Vec2 = Eigen::Vector<double, DIM>;
			// gradu := du/dX. So the shape is (value dim x dim).
			using GradU = Eigen::Matrix<double, VALUE_DIM, DIM, Eigen::RowMajor>;

			constexpr int NEED_UNKNOWN_VALUE = EnergyKernel::NEED_UNKNOWN_VALUE;
			constexpr int NEED_UNKNOWN_GRAD = EnergyKernel::NEED_UNKNOWN_GRAD;

			auto &elem_desc = data.element_desc[elem_id];
			int basis_num = elem_desc.basis_desc.basis_num;

			Vec1 u_value = Vec1::Zero();
			GradU gradu_value = GradU::Zero();
			for (int b = 0; b < basis_num; ++b)
			{
				if constexpr (NEED_UNKNOWN_VALUE)
				{
					double phi = cache.get_basis_value(b, quad_id);
					Vec1 local_node_unknown = detail::local_node_unknown<VALUE_DIM>(elem_id, b, data, unknown);
					u_value += phi * local_node_unknown;
				}
				if constexpr (NEED_UNKNOWN_GRAD)
				{
					Vec2 grad_phi = cache.get_basis_grad_phy<DIM>(b, quad_id);
					Vec1 local_node_unknown = detail::local_node_unknown<VALUE_DIM>(elem_id, b, data, unknown);
					gradu_value += local_node_unknown * grad_phi.transpose();
				}
			}

			using AD = autodiff::Double2<VALUE_DIM>;
			using ADGrad = typename AD::Grad;
			using ADHess = typename AD::Hess;
			double phi_i = cache.get_basis_value(local_i, quad_id);
			double phi_j = cache.get_basis_value(local_j, quad_id);
			Vec2 grad_phi_i = cache.get_basis_grad_phy<DIM>(local_i, quad_id);
			Vec2 grad_phi_j = cache.get_basis_grad_phy<DIM>(local_j, quad_id);

			// Autodiff type is storage intensive. A single Hessian autodiff scalar (Double2) takes 128 bytes.
			// Register/L1 cache are precious resources on GPU, avoid storing unecessary AD type.

			// Seed autodiff unknown vector u.
			// If NEED_UNKNOWN_VALUE, autodiff_u is Eigen::Matrix<AD, VALUE_DIM, 1>. Else empty dummy type.
			Conditional<NEED_UNKNOWN_VALUE, Eigen::Matrix<AD, VALUE_DIM, 1>, detail::NoStorage> autodiff_u;
			if constexpr (NEED_UNKNOWN_VALUE)
			{
				for (int vd = 0; vd < VALUE_DIM; ++vd)
				{
					ADGrad g = ADGrad::Zero();
					g(vd) = phi_i;
					g(VALUE_DIM + vd) = phi_j;
					autodiff_u(vd) = AD(u_value(vd), g, AD::Hess::Zero());
				}
			}

			// Seed autodiff unknown gradient matrix gradu.
			// If NEED_UNKNOWN_GRAD, autodiff_u is Eigen::Matrix<AD, VALUE_DIM, DIM>. Else empty dummy type.
			Conditional<NEED_UNKNOWN_GRAD, Eigen::Matrix<AD, VALUE_DIM, DIM>, detail::NoStorage> autodiff_gradu;
			if constexpr (NEED_UNKNOWN_GRAD)
			{
				for (int vd = 0; vd < VALUE_DIM; ++vd)
				{
					for (int d = 0; d < DIM; ++d)
					{
						ADGrad g = ADGrad::Zero();
						g(vd) = grad_phi_i(d);
						g(VALUE_DIM + vd) = grad_phi_j(d);
						autodiff_gradu(vd, d) = AD(gradu_value(vd, d), g, ADHess::Zero());
					}
				}
			}

			Span<const AD> u;
			if constexpr (NEED_UNKNOWN_VALUE)
			{
				u = Span<const AD>(autodiff_u.data(), autodiff_u.size());
			}
			Span<const AD> gradu;
			if constexpr (NEED_UNKNOWN_GRAD)
			{
				gradu = Span<const AD>(autodiff_gradu.data(), autodiff_gradu.size());
			}

			AD energy_value = energy.template eval_scalar<AD>(u, gradu, material);
			return energy_value.get_hess();
		}
	};

} // namespace polyfem::assembler
