#pragma once

#include <polyfem/materials/MaterialExprRegistry.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/ComputeAssemblyCache.hpp>
#include <polyfem/assembler/AssemblyData.hpp>
#include <polyfem/utils/AtomicAdd.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Span.hpp>

#include <ipc/utils/eigen_ext.hpp>

#include <Eigen/Core>

#include <cassert>
#include <stdexcept>
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <spdlog/fmt/fmt.h>

namespace polyfem::assembler
{
	namespace detail
	{
		/// Get element assembly cache. Compute on the fly if missing.
		ElementAssemblyCacheView get_element_cache(
			const AssemblyDataView &data,
			const AssemblyDataView &geom_data,
			const AssemblyCacheView &cache,
			int element_id,
			bool is_mass,
			AssemblyTempStorage &temp,
			AssemblyCache &temp_cache);

		template <typename Material, int dim>
		Material eval_material(
			const material::MaterialExprRegistry &material_registry,
			const ElementAssemblyCacheView &cache,
			int element_id,
			int quad_id,
			double time)
		{
			if constexpr (std::is_same_v<Material, material::Dummy>)
			{
				return {};
			}
			else
			{
				auto material_expr = material_registry.template get<typename Material::ExprType>(element_id);
				if (material_expr == nullptr)
				{
					auto err_string =
						fmt::format("Material {} missing for element {}. Please check your volume selection id in material json.",
									typeid(material_expr).name(), element_id);
					throw std::runtime_error(err_string);
				}

				double x = cache.get_physical_x(quad_id);
				double y = 0.0;
				double z = 0.0;
				if constexpr (dim > 1)
				{
					y = cache.get_physical_y(quad_id);
				}
				if constexpr (dim > 2)
				{
					z = cache.get_physical_z(quad_id);
				}
				return material::eval_expr(*material_expr, x, y, z, time, element_id);
			}
		}

		// Scatter element local vector to global.
		void scatter_element_vector(
			const ElementDesc &element_desc,
			DofMappingStoreView mapping,
			Span<const double> local_vector,
			int basis_num,
			int value_dim,
			Span<double> global_vector);

		// Scatter element local matrix to global.
		void scatter_element_matrix(
			const ElementDesc &element_desc,
			DofMappingStoreView mapping,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &local_matrix,
			int basis_num,
			int value_dim,
			BSRMatrixMutableView global_matrix);

	} // namespace detail

	/// @brief Assemble scalar value on host.
	/// @tparam ScalarKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	template <typename ScalarKernel>
	double assemble_scalar_on_host(
		ScalarKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		double time = 0.0)
	{
		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		AssemblyDataView data_view = data.view();
		AssemblyDataView geom_data_view = geom_data.view();
		AssemblyCacheView cache_view = cache.view();
		int element_num = data_view.element_desc.size();

		// Per thread scalar (double) storage.
		auto thread_storage = utils::create_thread_storage<double>(0.0);

		utils::maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;

			double &local_scalar = utils::get_local_thread_storage(thread_storage, thread_id);
			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				ElementAssemblyCacheView elem_cache = detail::get_element_cache(
					data_view,
					geom_data_view,
					cache_view,
					elem_id,
					false,
					temp,
					temp_cache);

				int quad_num = elem_cache.quad_num();
				for (int quad_id = 0; quad_id < quad_num; ++quad_id)
				{
					Material material = detail::eval_material<Material, DIM>(material_registry, elem_cache, elem_id, quad_id, time);
					double val = kernel.eval_scalar(elem_id, quad_id, data_view, elem_cache, material, unknown);
					local_scalar += val * elem_cache.get_weighted_measure(quad_id);
				}
			}
		});

		// Sum per thread assembly result.
		double scalar_out = 0.0;
		for (auto &s : thread_storage)
		{
			scalar_out += s;
		}

		return scalar_out;
	}

	/// @brief Assemble per element scalar value on host.
	/// @tparam ScalarKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param scalar_out Output vector of size element num. Result is accumulated on top of input value.
	template <typename ScalarKernel>
	void assemble_scalar_per_element_on_host(
		ScalarKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> scalar_out,
		double time = 0.0)
	{
		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		AssemblyDataView data_view = data.view();
		AssemblyDataView geom_data_view = geom_data.view();
		AssemblyCacheView cache_view = cache.view();
		int element_num = data_view.element_desc.size();
		assert(scalar_out.size() == element_num);

		utils::maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;

			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				ElementAssemblyCacheView elem_cache = detail::get_element_cache(
					data_view,
					geom_data_view,
					cache_view,
					elem_id,
					false,
					temp,
					temp_cache);

				int quad_num = elem_cache.quad_num();
				double local_scalar = 0.0;
				for (int quad_id = 0; quad_id < quad_num; ++quad_id)
				{
					Material material = detail::eval_material<Material, DIM>(material_registry, elem_cache, elem_id, quad_id, time);
					double val = kernel.eval_scalar(elem_id, quad_id, data_view, elem_cache, material, unknown);
					local_scalar += val * elem_cache.get_weighted_measure(quad_id);
				}
				scalar_out[elem_id] += local_scalar;
			}
		});
	}

	/// @brief Assemble vector on host.
	/// @tparam VectorKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param vector_out Output vector of size total dof num. Result is accumulated on top of input value.
	template <typename VectorKernel>
	void assemble_vector_on_host(
		VectorKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vector_out,
		double time = 0.0,
		double extra_scaling = 1.0)
	{
		using Material = typename VectorKernel::Material;
		constexpr int VALUE_DIM = VectorKernel::VALUE_DIM;
		constexpr int DIM = VectorKernel::DIM;

		AssemblyDataView data_view = data.view();
		AssemblyDataView geom_data_view = geom_data.view();
		AssemblyCacheView cache_view = cache.view();
		int element_num = data_view.element_desc.size();

		// Per thread global vector out.
		auto thread_storage = utils::create_thread_storage(std::vector<double>(vector_out.size(), 0.0));

		utils::maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;
			std::vector<double> elem_vector; // element local vector out.

			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				const ElementDesc &elem_desc = data_view.element_desc[elem_id];
				int basis_num = elem_desc.basis_desc.basis_num;

				ElementAssemblyCacheView elem_cache = detail::get_element_cache(
					data_view,
					geom_data_view,
					cache_view,
					elem_id,
					false,
					temp,
					temp_cache);

				int quad_num = elem_cache.quad_num();
				elem_vector.resize(basis_num * VALUE_DIM, 0.0);
				for (int basis_id = 0; basis_id < basis_num; ++basis_id)
				{
					using Vec = Eigen::Vector<double, VALUE_DIM>;
					// i-th component of element local vector from basis node i.
					Vec vec_i = Vec::Zero();
					for (int quad_id = 0; quad_id < quad_num; ++quad_id)
					{
						Material material = detail::eval_material<Material, DIM>(material_registry, elem_cache, elem_id, quad_id, time);
						Vec kernel_out = kernel.eval_vector(
							elem_id,
							quad_id,
							basis_id,
							data_view,
							elem_cache,
							material,
							unknown);
						vec_i += kernel_out * extra_scaling * elem_cache.get_weighted_measure(quad_id);
					}

					for (int d = 0; d < VALUE_DIM; ++d)
					{
						elem_vector[basis_id * VALUE_DIM + d] = vec_i(d);
					}
				}

				// Scatter element local vector to thread local global vector.
				auto &global_vector = utils::get_local_thread_storage(thread_storage, thread_id);
				detail::scatter_element_vector(
					elem_desc,
					data_view.dof_mapping_store,
					elem_vector,
					basis_num,
					VALUE_DIM,
					global_vector);
			}
		});

		// Sum thread local global vector.
		for (auto &s : thread_storage)
		{
			for (int i = 0; i < vector_out.size(); ++i)
			{
				vector_out[i] += s[i];
			}
		}
	}

	/// @brief Assemble matrix on host.
	/// @tparam MatrixKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param matrix_out Output matrix of size (total dof x total dof). Result is accumulated
	/// on top of input value. Matrix storage must be pre-allocated.
	template <typename MatrixKernel>
	void assemble_matrix_on_host(
		MatrixKernel kernel,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrixMutableView matrix_out,
		bool project_to_psd = false,
		double time = 0.0,
		double extra_scaling = 1.0,
		bool is_mass = false)
	{
		using Material = typename MatrixKernel::Material;
		constexpr int VALUE_DIM = MatrixKernel::VALUE_DIM;
		constexpr int DIM = MatrixKernel::DIM;

		AssemblyDataView data_view = data.view();
		AssemblyDataView geom_data_view = geom_data.view();
		AssemblyCacheView cache_view = cache.view();
		int element_num = data_view.element_desc.size();

		utils::maybe_parallel_for(element_num, [&](int start, int end, int) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;
			// element local matrix.
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> elem_matrix;

			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				const ElementDesc &elem_desc = data_view.element_desc[elem_id];
				int basis_num = elem_desc.basis_desc.basis_num;
				int local_dof_num = basis_num * VALUE_DIM;
				elem_matrix.setZero(local_dof_num, local_dof_num);

				ElementAssemblyCacheView elem_cache = detail::get_element_cache(
					data_view,
					geom_data_view,
					cache_view,
					elem_id,
					is_mass,
					temp,
					temp_cache);

				// Loop over ij component of element local matrix M.
				// Mij represents contributions from element local basis node i and j.
				// Take advantage of the symmetry, compute upper part only.
				for (int bi = 0; bi < basis_num; ++bi)
				{
					for (int bj = bi; bj < basis_num; ++bj)
					{
						using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;
						// ij component of element local matrix M from basis node i and j.
						Mat mat_ij = Mat::Zero();

						int quad_num = elem_cache.quad_num();
						for (int quad_id = 0; quad_id < quad_num; ++quad_id)
						{
							Material material = detail::eval_material<Material, DIM>(material_registry, elem_cache, elem_id, quad_id, time);
							Mat kernel_out = kernel.eval_matrix(
								elem_id,
								quad_id,
								bi,
								bj,
								data_view,
								elem_cache,
								material,
								unknown);
							mat_ij += kernel_out * extra_scaling * elem_cache.get_weighted_measure(quad_id);
						}

						// Scatter Mij to to element local matrix M.
						elem_matrix.block<VALUE_DIM, VALUE_DIM>(bi * VALUE_DIM, bj * VALUE_DIM) = mat_ij;
						if (bj > bi)
						{
							elem_matrix.block<VALUE_DIM, VALUE_DIM>(bj * VALUE_DIM, bi * VALUE_DIM) = mat_ij.transpose();
						}
					}
				}

				if (project_to_psd)
				{
					elem_matrix = ipc::project_to_psd(elem_matrix);
				}

				detail::scatter_element_matrix(
					elem_desc,
					data_view.dof_mapping_store,
					elem_matrix,
					basis_num,
					VALUE_DIM,
					matrix_out);
			}
		});
	}
} // namespace polyfem::assembler
