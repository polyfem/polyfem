#pragma once

#include <polyfem/materials/MaterialExprRegistry.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/AssemblyData.hpp>
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/CudaUtils.hpp>

#include <cuda/buffer>
#include <cuda/algorithm>
#include <cuda/warp>
#include <cuda/std/utility>
#include <cuda/std/optional>
#include <cub/cub.cuh>
#include <Eigen/Core>

#include <cassert>
#include <cstddef>
#include <vector>

namespace polyfem::assembler
{
	namespace detail
	{
		struct DeviceAssemblyError
		{
			bool is_material_missing = false;
		};

		/// @brief Scatter ij block of element local matrix to global.
		/// @param local_mat Row-major (value dim x value dim) matrix.
		template <int value_dim>
		__device__ void scatter_mat_ij(
			int elem_id,
			int basis_i,
			int basis_j,
			AssemblyDataView data,
			Span<const double> local_mat,
			BSRMatrixMutableView global_mat)
		{
			assert(local_mat.size() == value_dim * value_dim);

			using Mat = Eigen::Matrix<double, value_dim, value_dim, Eigen::RowMajor>;
			auto mat = Eigen::Map<const Mat>(local_mat.data());

			auto &elem_desc = data.element_desc[elem_id];
			auto &mappings = data.dof_mapping_store;
			int row_mapping_id = elem_desc.dof_mapping_range.offset + basis_i;
			int col_mapping_id = elem_desc.dof_mapping_range.offset + basis_j;

			auto row_node_ids = mappings.get_node_ids(row_mapping_id);
			auto row_node_weights = mappings.get_weights(row_mapping_id);
			auto col_node_ids = mappings.get_node_ids(col_mapping_id);
			auto col_node_weights = mappings.get_weights(col_mapping_id);
			assert(row_node_ids.size() == row_node_weights.size());
			assert(col_node_ids.size() == col_node_weights.size());

			for (int i = 0; i < row_node_ids.size(); ++i)
			{
				for (int j = 0; j < col_node_ids.size(); ++j)
				{
					double weight = row_node_weights[i] * col_node_weights[j];
					for (int r = 0; r < value_dim; ++r)
					{
						for (int c = 0; c < value_dim; ++c)
						{
							double value = mat(r, c);
							if (value == 0.0)
							{
								continue;
							}

							int global_row = row_node_ids[i] * value_dim + r;
							int global_col = col_node_ids[j] * value_dim + c;
							double *dst = global_mat.get_entry(global_row, global_col);
							assert(dst);
							atomicAdd(dst, weight * value);
						}
					}
				}
			}
		}

		/// @brief Scatter i block of element local vector to global.
		/// @param local_vec Vector of size value_dim.
		template <int value_dim>
		__device__ void scatter_vec_i(
			int elem_id,
			int basis_i,
			AssemblyDataView data,
			Span<const double> local_vec,
			Span<double> global_vec)
		{
			assert(local_vec.size() == value_dim);

			auto &elem_desc = data.element_desc[elem_id];
			auto &mappings = data.dof_mapping_store;
			int mapping_id = elem_desc.dof_mapping_range.offset + basis_i;

			auto node_ids = mappings.get_node_ids(mapping_id);
			auto node_weights = mappings.get_weights(mapping_id);

			for (int i = 0; i < node_ids.size(); ++i)
			{
				for (int k = 0; k < value_dim; ++k)
				{
					if (local_vec[k] == 0.0)
					{
						continue;
					}
					int offset = node_ids[i] * value_dim + k;
					assert(offset >= 0 && offset < global_vec.size());
					double val = node_weights[i] * local_vec[k];
					atomicAdd(global_vec.data() + offset, val);
				}
			}
		}

		/// Evaluate material expression view.
		/// Return nullopt if material is missing.
		template <typename Material, int dim>
		__device__ cuda::std::optional<Material> eval_material_expr(
			int elem_id,
			int quad_id,
			double time,
			const AssemblyCacheView &cache,
			const material::MaterialStoreView<typename Material::ExprViewType> &store_view)
		{

			if constexpr (std::is_same_v<Material, material::Dummy>)
			{
				return material::Dummy{};
			}
			else
			{
				if (elem_id < 0 || elem_id >= store_view.expr_ids.size())
				{
					return cuda::std::nullopt;
				}

				int expr_id = store_view.expr_ids[elem_id];
				if (expr_id < 0 || expr_id >= store_view.expr.size())
				{
					return cuda::std::nullopt;
				}
				auto &material_expr = store_view.expr[expr_id];

				ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
				double x = elem_cache.get_physical_x(quad_id);
				double y = (dim >= 2) ? elem_cache.get_physical_y(quad_id) : 0.0;
				double z = (dim >= 3) ? elem_cache.get_physical_z(quad_id) : 0.0;
				return material::eval_expr(material_expr, x, y, z, time, elem_id);
			}
		}

		template <typename K, int block_size>
		__global__ void assemble_scalar_kernel(
			K kernel,
			AssemblyDataView data,
			AssemblyCacheView cache,
			int elem_num,
			double time,
			Span<const typename K::Material> material_evals,
			material::MaterialStoreView<typename K::Material::ExprViewType> material_views,
			Span<const double> unknown,
			double *scalar_out,
			DeviceAssemblyError *error)
		{
			using Material = typename K::Material;
			constexpr int DIM = K::DIM;

			// One element per thread.
			assert(blockDim.x == block_size);
			int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
			double scalar = 0.0; // local scalar value.
			bool material_missing = false;

			if (elem_id < elem_num)
			{
				auto &cache_desc = cache.desc[elem_id];
				ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
				int quad_num = cache_desc.weighted_measure_range.num;

				for (int quad_id = 0; quad_id < quad_num; ++quad_id)
				{
					// If pre-compute material is available, load from memory. Else evaluate on the fly.
					cuda::std::optional<Material> material;
					if (material_evals.empty())
					{
						material = eval_material_expr<Material, DIM>(elem_id, quad_id, time, cache, material_views);
					}
					else
					{
						// Material is per cached quadrature point, matching weighted_measure.
						material = material_evals[cache_desc.weighted_measure_range.offset + quad_id];
					}

					if (!material)
					{
						error->is_material_missing = true;
						material_missing = true;
						break;
					}

					double local_scalar = kernel.eval_scalar(elem_id, quad_id, data, elem_cache, *material, unknown);
					scalar += local_scalar * elem_cache.get_weighted_measure(quad_id);
				}
			}
			if (material_missing)
				scalar = 0.0;

			using BlockReduce = cub::BlockReduce<double, block_size>;
			__shared__ typename BlockReduce::TempStorage reduce_temp;
			double sum = BlockReduce(reduce_temp).Sum(scalar);
			if (threadIdx.x == 0)
			{
				atomicAdd(scalar_out, sum);
			}
		}

		template <typename K>
		__global__ void assemble_scalar_per_element_kernel(
			K kernel,
			AssemblyDataView data,
			AssemblyCacheView cache,
			double time,
			Span<const typename K::Material> material_evals,
			material::MaterialStoreView<typename K::Material::ExprViewType> material_views,
			Span<const double> unknown,
			Span<double> scalar_out,
			DeviceAssemblyError *error)
		{
			using Material = typename K::Material;
			constexpr int DIM = K::DIM;

			// One element per thread.
			int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (elem_id >= scalar_out.size())
			{
				return;
			}

			auto &cache_desc = cache.desc[elem_id];
			ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
			int quad_num = cache_desc.weighted_measure_range.num;
			double scalar = 0.0; // local scalar value.
			bool material_missing = false;

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// If pre-compute material is available, load from memory. Else evaluate on the fly.
				cuda::std::optional<Material> material;
				if (material_evals.empty())
				{
					material = eval_material_expr<Material, DIM>(elem_id, quad_id, time, cache, material_views);
				}
				else
				{
					// Material is per cached quadrature point, matching weighted_measure.
					material = material_evals[cache_desc.weighted_measure_range.offset + quad_id];
				}

				if (!material)
				{
					error->is_material_missing = true;
					material_missing = true;
					break;
				}

				double local_scalar = kernel.eval_scalar(elem_id, quad_id, data, elem_cache, *material, unknown);
				scalar += local_scalar * elem_cache.get_weighted_measure(quad_id);
			}

			if (!material_missing)
				scalar_out[elem_id] += scalar;
		}

		template <typename K>
		__global__ void assemble_vector_kernel(
			K kernel,
			AssemblyDataView data,
			AssemblyCacheView cache,
			double time,
			Span<const typename K::Material> material_evals,
			material::MaterialStoreView<typename K::Material::ExprViewType> material_views,
			Span<const double> unknown,
			Span<double> vec_out,
			double extra_scaling,
			DeviceAssemblyError *error)
		{
			constexpr int VALUE_DIM = K::VALUE_DIM;
			constexpr int DIM = K::DIM;

			using Material = typename K::Material;
			using Vec = Eigen::Vector<double, VALUE_DIM>;

			// Each thread compute i-th component of element local vector.
			auto tasks = data.vector_assembly_tasks;
			int task_num = tasks.size();
			int task_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (task_id >= task_num)
			{
				return;
			}

			DeviceVectorAssemblyTask task = tasks[task_id];
			int elem_id = task.elem_id;
			int basis_id = task.basis_i;
			auto &cache_desc = cache.desc[elem_id];
			ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
			int quad_num = cache_desc.weighted_measure_range.num;
			Vec grad_i = Vec::Zero(); // local vector.
			bool material_missing = false;

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// If pre-compute material is available, load from memory. Else evaluate on the fly.
				cuda::std::optional<Material> material;
				if (material_evals.empty())
				{
					material = eval_material_expr<Material, DIM>(elem_id, quad_id, time, cache, material_views);
				}
				else
				{
					// Material is per cached quadrature point, matching weighted_measure.
					material = material_evals[cache_desc.weighted_measure_range.offset + quad_id];
				}

				if (!material)
				{
					error->is_material_missing = true;
					material_missing = true;
					break;
				}

				Vec kernel_out = kernel.eval_vector(elem_id, quad_id, basis_id, data, elem_cache, *material, unknown);
				grad_i += kernel_out * extra_scaling * elem_cache.get_weighted_measure(quad_id);
			}

			if (!material_missing && !grad_i.isZero())
			{
				Span<const double> grad_i_span(grad_i.data(), grad_i.size());
				scatter_vec_i<VALUE_DIM>(elem_id, basis_id, data, grad_i_span, vec_out);
			}
		}

		template <typename K>
		__global__ void assemble_matrix_kernel(
			K kernel,
			AssemblyDataView data,
			AssemblyCacheView cache,
			double time,
			Span<const typename K::Material> material_evals,
			material::MaterialStoreView<typename K::Material::ExprViewType> material_views,
			Span<const double> unknown,
			BSRMatrixMutableView mat_out,
			double extra_scaling,
			DeviceAssemblyError *error)
		{
			constexpr int VALUE_DIM = K::VALUE_DIM;
			constexpr int DIM = K::DIM;

			using Material = typename K::Material;
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

			// Each thread compute ij block of element local matrix M.
			auto tasks = data.matrix_assembly_tasks;
			int task_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (task_id >= tasks.size())
			{
				return;
			}

			DeviceMatrixAssemblyTask task = tasks[task_id];
			int elem_id = task.elem_id;
			int bi = task.basis_i;
			int bj = task.basis_j;
			auto &cache_desc = cache.desc[elem_id];
			ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
			int quad_num = cache_desc.weighted_measure_range.num;
			// ij component of element local matrix M.
			// Represents contribution from element local basis node i and j.
			Mat mat_ij = Mat::Zero();
			bool material_missing = false;

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// If pre-compute material is available, load from memory. Else evaluate on the fly.
				cuda::std::optional<Material> material;
				if (material_evals.empty())
				{
					material = eval_material_expr<Material, DIM>(elem_id, quad_id, time, cache, material_views);
				}
				else
				{
					// Material is per cached quadrature point, matching weighted_measure.
					material = material_evals[cache_desc.weighted_measure_range.offset + quad_id];
				}

				if (!material)
				{
					error->is_material_missing = true;
					material_missing = true;
					break;
				}

				Mat kernel_out = kernel.eval_matrix(elem_id, quad_id, bi, bj, data, elem_cache, *material, unknown);
				mat_ij += kernel_out * extra_scaling * elem_cache.get_weighted_measure(quad_id);
			}

			if (!material_missing && !mat_ij.isZero())
			{
				Span<const double> hess_ij_span(mat_ij.data(), mat_ij.size());
				scatter_mat_ij<VALUE_DIM>(elem_id, bi, bj, data, hess_ij_span, mat_out);
				// We only eval upper half of M. Scatters to lower part too.
				if (bj > bi)
				{
					Mat hess_ji = mat_ij.transpose();
					Span<const double> hess_ji_span(hess_ji.data(), hess_ji.size());
					scatter_mat_ij<VALUE_DIM>(elem_id, bj, bi, data, hess_ji_span, mat_out);
				}
			}
		}

		template <typename Material, int dim>
		cuda::device_buffer<Material> precompute_materials_on_host(
			const AssemblyData &data,
			const AssemblyCache &cache,
			const material::MaterialExprRegistry &material_registry,
			double time,
			ExecutionPolicy policy)
		{
			auto cache_view = cache.view();

			int global_material_num = cache_view.weighted_measure.size();
			assert(global_material_num != 0);
			int elem_num = data.view().element_desc.size();
			assert(elem_num != 0);

			if constexpr (std::is_same_v<Material, material::Dummy>)
			{
				auto d_materials =
					cuda::make_buffer<Material>(*policy.stream, *policy.mr, 0, cuda::no_init);
				policy.stream->sync();
				return d_materials;
			}
			else
			{
				std::vector<Material> materials(global_material_num);
				utils::maybe_parallel_for(elem_num, [&cache_view, &material_registry, &materials, time](int elem_id) {
					auto material_expr = material_registry.get<typename Material::ExprType>(elem_id);
					if (material_expr == nullptr)
					{
						throw std::runtime_error("Material missing!");
					}

					auto &cache_desc = cache_view.desc[elem_id];
					ElementAssemblyCacheView elem_cache = cache_view.slice(elem_id);
					for (int q = 0; q < cache_desc.weighted_measure_range.num; ++q)
					{
						double x = elem_cache.get_physical_x(q);
						double y = (dim >= 2) ? elem_cache.get_physical_y(q) : 0.0;
						double z = (dim >= 3) ? elem_cache.get_physical_z(q) : 0.0;
						Material m = material::eval_expr(*material_expr, x, y, z, time, elem_id);

						materials[cache_desc.weighted_measure_range.offset + q] = std::move(m);
					}
				});

				auto d_materials = copy_to_device_async<Material>(materials, policy);
				policy.stream->sync();
				return d_materials;
			}
		}
	} // namespace detail

	/// @brief Assemble scalar value on host.
	/// @tparam ScalarKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	template <typename ScalarKernel>
	double assemble_scalar_on_device(
		ScalarKernel kernel,
		const AssemblyData &data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		double time,
		ExecutionPolicy policy)
	{
		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		auto &p = policy;

		auto d_data = data.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_unknown = copy_to_device_async(unknown, policy);
		auto d_scalar_out = cuda::make_buffer<double>(*p.stream, *p.mr, 1, 0.0);
		auto d_error = copy_to_device_async(detail::DeviceAssemblyError{}, policy);

		// If material is device compatible, get material store view and eval on the fly.
		// Else precompute them on host.
		DeviceBuf<Material> d_materials_evals;
		Span<const Material> d_materials_evals_span;
		material::MaterialStoreView<typename Material::ExprViewType> d_material_views;
		if constexpr (!std::is_same_v<Material, material::Dummy>)
		{
			if (material_registry.is_device_compatible<typename Material::ExprType>())
			{
				d_material_views = material_registry.get_all_device_expr_views<typename Material::ExprType>(policy);
			}
			else
			{
				d_materials_evals = detail::precompute_materials_on_host<Material, DIM>(data, cache, material_registry, time, policy);
				d_materials_evals_span = *d_materials_evals;
			}
		}

		int elem_num = data.view().element_desc.size();
		int grid_num = div_round_up(elem_num, 128);
		detail::assemble_scalar_kernel<ScalarKernel, 128><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_data,
			d_cache,
			elem_num,
			time,
			d_materials_evals_span,
			d_material_views,
			d_unknown,
			d_scalar_out.data(),
			d_error.data());
		p.stream->sync();

		double scalar_out = copy_to_host<double>(d_scalar_out.data(), policy);
		auto error = copy_to_host<detail::DeviceAssemblyError>(d_error.data(), policy);
		if (error.is_material_missing)
			throw std::runtime_error("Device assembly failed. Reason: material missing.");

		return scalar_out;
	}

	/// @brief Assemble per element scalar value on host.
	/// @tparam ScalarKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param scalar_out Output vector of size element num. Result is accumulated on top of input value.
	template <typename ScalarKernel>
	void assemble_scalar_per_element_on_device(
		ScalarKernel kernel,
		const AssemblyData &data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vec_out,
		ExecutionPolicy policy,
		double time)
	{
		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		auto &p = policy;

		auto d_data = data.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_unknown = copy_to_device_async(unknown, policy);
		auto d_error = copy_to_device_async(detail::DeviceAssemblyError{}, policy);

		// If material is device compatible, get material store view and eval on the fly.
		// Else precompute them on host.
		DeviceBuf<Material> d_materials_evals;
		Span<const Material> d_materials_evals_span;
		material::MaterialStoreView<typename Material::ExprViewType> d_material_views;
		if constexpr (!std::is_same_v<Material, material::Dummy>)
		{
			if (material_registry.is_device_compatible<typename Material::ExprType>())
			{
				d_material_views = material_registry.get_all_device_expr_views<typename Material::ExprType>(policy);
			}
			else
			{
				d_materials_evals = detail::precompute_materials_on_host<Material, DIM>(data, cache, material_registry, time, policy);
				d_materials_evals_span = *d_materials_evals;
			}
		}

		int elem_num = data.view().element_desc.size();
		assert(vec_out.size() == elem_num);
		int grid_num = div_round_up(elem_num, 128);
		detail::assemble_scalar_per_element_kernel<ScalarKernel><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_data,
			d_cache,
			time,
			d_materials_evals_span,
			d_material_views,
			d_unknown,
			vec_out,
			d_error.data());
		p.stream->sync();

		auto error = copy_to_host<detail::DeviceAssemblyError>(d_error.data(), policy);
		if (error.is_material_missing)
			throw std::runtime_error("Device assembly failed. Reason: material missing.");
	}

	/// @brief Assemble vector on host.
	/// @tparam VectorKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param vector_out Output vector of size total dof num. Result is accumulated on top of input value.
	template <typename VectorKernel>
	void assemble_vector_on_device(
		VectorKernel kernel,
		const AssemblyData &data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vec_out,
		ExecutionPolicy policy,
		double time = 0.0,
		double extra_scaling = 1.0)
	{
		using Material = typename VectorKernel::Material;
		constexpr int DIM = VectorKernel::DIM;

		auto &p = policy;

		auto d_data = data.device_view(p);
		int task_num = d_data.vector_assembly_tasks.size();
		auto d_cache = cache.device_view(p);
		auto d_unknown = copy_to_device_async(unknown, policy);
		auto d_error = copy_to_device_async(detail::DeviceAssemblyError{}, policy);

		// If material is device compatible, get material store view and eval on the fly.
		// Else precompute them on host.
		DeviceBuf<Material> d_materials_evals;
		Span<const Material> d_materials_evals_span;
		material::MaterialStoreView<typename Material::ExprViewType> d_material_views;
		if constexpr (!std::is_same_v<Material, material::Dummy>)
		{
			if (material_registry.is_device_compatible<typename Material::ExprType>())
			{
				d_material_views = material_registry.get_all_device_expr_views<typename Material::ExprType>(policy);
			}
			else
			{
				d_materials_evals = detail::precompute_materials_on_host<Material, DIM>(data, cache, material_registry, time, policy);
				d_materials_evals_span = *d_materials_evals;
			}
		}

		int grid_num = div_round_up(task_num, 128);
		detail::assemble_vector_kernel<VectorKernel><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_data,
			d_cache,
			time,
			d_materials_evals_span,
			d_material_views,
			d_unknown,
			vec_out,
			extra_scaling,
			d_error.data());
		p.stream->sync();

		auto error = copy_to_host<detail::DeviceAssemblyError>(d_error.data(), policy);
		if (error.is_material_missing)
			throw std::runtime_error("Device assembly failed. Reason: material missing.");
	}

	/// @brief Assemble matrix on host.
	/// @tparam MatrixKernel Kernel type. See KernelExample.hpp.
	/// @param kernel Kernel functor.
	/// @param matrix_out Output matrix of size (total dof x total dof). Result is accumulated
	/// on top of input value. Matrix storage must be pre-allocated.
	template <typename MatrixKernel>
	void assemble_matrix_on_device(
		MatrixKernel kernel,
		const AssemblyData &data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrixMutableView mat_out,
		ExecutionPolicy policy,
		double time = 0.0,
		double extra_scaling = 1.0)
	{
		using Material = typename MatrixKernel::Material;
		constexpr int DIM = MatrixKernel::DIM;

		auto &p = policy;

		auto d_data = data.device_view(p);
		int task_num = d_data.matrix_assembly_tasks.size();
		auto d_cache = cache.device_view(p);
		auto d_unknown = copy_to_device_async(unknown, policy);
		auto d_error = copy_to_device_async(detail::DeviceAssemblyError{}, policy);

		// If material is device compatible, get material store view and eval on the fly.
		// Else precompute them on host.
		DeviceBuf<Material> d_materials_evals;
		Span<const Material> d_materials_evals_span;
		material::MaterialStoreView<typename Material::ExprViewType> d_material_views;
		if constexpr (!std::is_same_v<Material, material::Dummy>)
		{
			if (material_registry.is_device_compatible<typename Material::ExprType>())
			{
				d_material_views = material_registry.get_all_device_expr_views<typename Material::ExprType>(policy);
			}
			else
			{
				d_materials_evals = detail::precompute_materials_on_host<Material, DIM>(data, cache, material_registry, time, policy);
				d_materials_evals_span = *d_materials_evals;
			}
		}

		int grid_num = div_round_up(task_num, 128);
		detail::assemble_matrix_kernel<MatrixKernel><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_data,
			d_cache,
			time,
			d_materials_evals_span,
			d_material_views,
			d_unknown,
			mat_out,
			extra_scaling,
			d_error.data());
		p.stream->sync();

		auto error = copy_to_host<detail::DeviceAssemblyError>(d_error.data(), policy);
		if (error.is_material_missing)
			throw std::runtime_error("Device assembly failed. Reason: material missing.");
	}

} // namespace polyfem::assembler
