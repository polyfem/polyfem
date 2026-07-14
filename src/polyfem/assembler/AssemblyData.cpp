#include <polyfem/assembler/AssemblyData.hpp>

#include <polyfem/basis/BasisStore.hpp>
#include <polyfem/basis/EvalBasis.hpp>
#include <polyfem/quadrature/QuadratureStore.hpp>
#include <polyfem/assembler/DofMappingStore.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <Eigen/Core>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem::assembler
{
	namespace
	{
		Eigen::MatrixXd span_components_to_points(
			const Span<const double> x,
			const Span<const double> y,
			const Span<const double> z,
			const int dim)
		{
			const int n = static_cast<int>(x.size());
			Eigen::MatrixXd pts(n, dim);
			for (int i = 0; i < n; ++i)
			{
				pts(i, 0) = x[i];
				if (dim > 1)
					pts(i, 1) = y[i];
				if (dim > 2)
					pts(i, 2) = z[i];
			}
			return pts;
		}

		void legacy_uv_matrix_to_spans(
			const Eigen::MatrixXd &uv,
			std::vector<double> &x,
			std::vector<double> &y,
			std::vector<double> &z)
		{
			x.resize(uv.rows());
			y.resize(uv.cols() > 1 ? uv.rows() : 0);
			z.resize(uv.cols() > 2 ? uv.rows() : 0);
			for (int i = 0; i < uv.rows(); ++i)
			{
				x[i] = uv(i, 0);
				if (uv.cols() > 1)
					y[i] = uv(i, 1);
				if (uv.cols() > 2)
					z[i] = uv(i, 2);
			}
		}

		struct LegacyBasisEvalData
		{
			basis::BasisDesc desc;
			std::vector<double> rational_weights;
			std::vector<basis::BasisEvalCallback> eval_callbacks;

			basis::BasisStoreView view() const
			{
				return basis::BasisStoreView{rational_weights, eval_callbacks};
			}
		};

		std::shared_ptr<const LegacyBasisEvalData> copy_legacy_basis_eval_data(
			const basis::BasisDesc &basis_desc,
			const basis::BasisStoreView basis_store)
		{
			auto data = std::make_shared<LegacyBasisEvalData>();
			data->desc = basis_desc;

			if (basis_desc.basis_family == basis::BasisFamily::Rational)
			{
				const Span<const double> weights = slice_by_range(
					basis_store.rational_weights, basis_desc.rational_weight_range);
				data->rational_weights.assign(weights.begin(), weights.end());
				data->desc.rational_weight_range = Range{0, int(data->rational_weights.size())};
			}
			else if (basis_desc.basis_family == basis::BasisFamily::Unknown)
			{
				assert(basis_desc.eval_callback_id >= 0);
				assert(basis_desc.eval_callback_id < int(basis_store.eval_callbacks.size()));
				data->eval_callbacks.push_back(basis_store.eval_callbacks[basis_desc.eval_callback_id]);
				data->desc.eval_callback_id = 0;
			}

			return data;
		}

		basis::Basis make_legacy_basis(
			int local_basis_id,
			const ElementDesc &desc,
			std::shared_ptr<const LegacyBasisEvalData> basis_eval_data,
			const DofMappingStoreView dof_mapping_store)
		{
			int dim = desc.basis_desc.dim;
			int mapping_id = desc.dof_mapping_range.offset + local_basis_id;
			std::vector<basis::Local2Global> mapping = dof_mapping_store.get_local_to_global(mapping_id, dim);
			assert(!mapping.empty());

			basis::Basis basis;
			basis.init(desc.basis_desc.order, mapping.front().index, local_basis_id, mapping.front().node);
			basis.global() = std::move(mapping);

			// Legacy callback takes (quad_num x dim) uv matrix input and return (quad num x 1) output.
			basis.set_basis([local_basis_id, basis_eval_data](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
				const basis::BasisDesc &basis_desc = basis_eval_data->desc;
				assert(uv.cols() == basis_desc.dim);
				std::vector<double> x, y, z;
				legacy_uv_matrix_to_spans(uv, x, y, z);
				val.resize(uv.rows(), 1);
				basis::basis_values_single(
					local_basis_id,
					basis_desc,
					basis_eval_data->view(),
					x,
					y,
					z,
					Span<double>(val.data(), val.size()));
			});

			// Legacy callback takes (quad_num x dim) uv matrix input and return (quad num x dim) output.
			basis.set_grad([local_basis_id, basis_eval_data](const Eigen::MatrixXd &uv, Eigen::MatrixXd &grad) {
				const basis::BasisDesc &basis_desc = basis_eval_data->desc;
				assert(uv.cols() == basis_desc.dim);
				std::vector<double> x, y, z;
				legacy_uv_matrix_to_spans(uv, x, y, z);
				std::vector<double> gx(uv.rows());
				std::vector<double> gy(basis_desc.dim > 1 ? uv.rows() : 0);
				std::vector<double> gz(basis_desc.dim > 2 ? uv.rows() : 0);
				basis::basis_gradients_single(
					local_basis_id,
					basis_desc,
					basis_eval_data->view(),
					x,
					y,
					z,
					gx,
					gy,
					gz);

				grad.resize(uv.rows(), basis_desc.dim);
				for (int i = 0; i < uv.rows(); ++i)
				{
					grad(i, 0) = gx[i];
					if (basis_desc.dim > 1)
						grad(i, 1) = gy[i];
					if (basis_desc.dim > 2)
						grad(i, 2) = gz[i];
				}
			});

			return basis;
		}

#ifdef POLYFEM_WITH_CUDA
		std::vector<DeviceVectorAssemblyTask> build_device_vector_assembly_tasks(
			const std::vector<ElementDesc> &element_desc)
		{
			int task_num = 0;
			for (const ElementDesc &elem_desc : element_desc)
			{
				task_num += elem_desc.basis_desc.basis_num;
			}

			std::vector<DeviceVectorAssemblyTask> tasks;
			tasks.reserve(task_num);
			for (int elem_id = 0; elem_id < int(element_desc.size()); ++elem_id)
			{
				const int basis_num = element_desc[elem_id].basis_desc.basis_num;
				for (int bi = 0; bi < basis_num; ++bi)
				{
					tasks.push_back({elem_id, bi});
				}
			}

			return tasks;
		}

		std::vector<DeviceMatrixAssemblyTask> build_device_matrix_assembly_tasks(
			const std::vector<ElementDesc> &element_desc)
		{
			int task_num = 0;
			for (const ElementDesc &elem_desc : element_desc)
			{
				const int basis_num = elem_desc.basis_desc.basis_num;
				task_num += basis_num * (basis_num + 1) / 2;
			}

			std::vector<DeviceMatrixAssemblyTask> tasks;
			tasks.reserve(task_num);
			for (int elem_id = 0; elem_id < int(element_desc.size()); ++elem_id)
			{
				const int basis_num = element_desc[elem_id].basis_desc.basis_num;
				for (int bi = 0; bi < basis_num; ++bi)
				{
					for (int bj = bi; bj < basis_num; ++bj)
					{
						tasks.push_back({elem_id, bi, bj});
					}
				}
			}

			return tasks;
		}
#endif

		basis::BasisEvalCallback make_legacy_eval_callback(
			const basis::ElementBases &legacy_element,
			const int dim)
		{
			return [legacy_element, dim](
					   const Span<const double> x,
					   const Span<const double> y,
					   const Span<const double> z,
					   Span<double> values,
					   Span<double> grad_x,
					   Span<double> grad_y,
					   Span<double> grad_z) {
				const int n_points = static_cast<int>(x.size());
				const int n_bases = static_cast<int>(legacy_element.bases.size());
				assert(values.size() == n_bases * n_points);
				assert(grad_x.size() == n_bases * n_points);
				assert(dim < 2 || grad_y.size() == n_bases * n_points);
				assert(dim < 3 || grad_z.size() == n_bases * n_points);

				const Eigen::MatrixXd pts = span_components_to_points(x, y, z, dim);
				std::vector<AssemblyValues> basis_values;
				std::vector<AssemblyValues> basis_grads;
				legacy_element.evaluate_bases(pts, basis_values);
				legacy_element.evaluate_grads(pts, basis_grads);
				assert(int(basis_values.size()) == n_bases);
				assert(int(basis_grads.size()) == n_bases);

				for (int local_basis_id = 0; local_basis_id < n_bases; ++local_basis_id)
				{
					assert(basis_values[local_basis_id].val.size() == n_points);
					assert(basis_grads[local_basis_id].grad.rows() == n_points);
					assert(basis_grads[local_basis_id].grad.cols() >= dim);

					const int offset = local_basis_id * n_points;
					for (int q = 0; q < n_points; ++q)
					{
						values[offset + q] = basis_values[local_basis_id].val(q);
						grad_x[offset + q] = basis_grads[local_basis_id].grad(q, 0);
						if (dim > 1)
							grad_y[offset + q] = basis_grads[local_basis_id].grad(q, 1);
						if (dim > 2)
							grad_z[offset + q] = basis_grads[local_basis_id].grad(q, 2);
					}
				}
			};
		}
	} // namespace

	AssemblyDataView AssemblyData::view() const
	{
		return AssemblyDataView{
			element_desc,
			quadrature_store.view(),
			mass_quadrature_store.view(),
			basis_store.view(),
			dof_mapping_store.view(),
			local_nodes_from_primitive};
	}

	AssemblyDataMutView AssemblyData::mutable_view()
	{
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		return AssemblyDataMutView{
			&element_desc,
			&quadrature_store,
			&mass_quadrature_store,
			&basis_store,
			&dof_mapping_store,
			&local_nodes_from_primitive};
	}

	std::shared_ptr<std::vector<basis::ElementBases>> AssemblyData::legacy_bases_ptr() const
	{

		auto legacy_bases = std::make_shared<std::vector<basis::ElementBases>>();
		legacy_bases->resize(element_desc.size());

		auto basis_store_view = basis_store.view();
		auto dof_mapping_store_view = dof_mapping_store.view();
		auto quadrature_store_view = quadrature_store.view();
		auto mass_quadrature_store_view = mass_quadrature_store.view();

		for (int e = 0; e < int(element_desc.size()); ++e)
		{
			const ElementDesc &desc = element_desc[e];
			basis::ElementBases &legacy = (*legacy_bases)[e];
			legacy.has_parameterization = desc.basis_desc.is_parametric;

			// Polygonal elements are populated in a second pass. Until then their
			// descriptors are value-initialized placeholders with no bases (and no
			// evaluation callback) and must remain empty in the legacy container.
			const int n_bases = desc.basis_desc.basis_num;
			if (n_bases == 0)
				continue;

			quadrature::Quadrature quadrature = quadrature_store_view.get_quadrature(desc.quadrature_desc);
			legacy.set_quadrature([quadrature](quadrature::Quadrature &out) { out = quadrature; });

			quadrature::Quadrature mass_quadrature = mass_quadrature_store_view.get_quadrature(desc.mass_quadrature_desc);
			legacy.set_mass_quadrature([mass_quadrature](quadrature::Quadrature &out) { out = mass_quadrature; });

			const int callback_id = desc.local_nodes_from_primitive_id;
			if (callback_id >= 0
				&& callback_id < int(local_nodes_from_primitive.size())
				&& local_nodes_from_primitive[callback_id])
			{
				legacy.set_local_node_from_primitive_func(local_nodes_from_primitive[callback_id]);
			}

			auto basis_eval_data = copy_legacy_basis_eval_data(desc.basis_desc, basis_store_view);
			legacy.bases.reserve(n_bases);
			for (int local_basis_id = 0; local_basis_id < n_bases; ++local_basis_id)
			{
				legacy.bases.push_back(make_legacy_basis(
					local_basis_id,
					desc,
					basis_eval_data,
					dof_mapping_store_view));
			}
		}

		return legacy_bases;
	}

	void AssemblyData::set_legacy_element(
		int element_id,
		const basis::ElementBases &legacy_element)
	{
		assert(element_id >= 0);
		assert(element_id < int(element_desc.size()));
		assert(!legacy_element.bases.empty());

#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		// Wrap legacy quadrature.
		quadrature::Quadrature quadrature;
		legacy_element.compute_quadrature(quadrature);
		quadrature::Quadrature mass_quadrature;
		legacy_element.compute_mass_quadrature(mass_quadrature);
		assert(quadrature.size() > 0);
		assert(mass_quadrature.size() > 0);
		assert(quadrature.points.cols() == mass_quadrature.points.cols());
		int dim = quadrature.points.cols();

		ElementDesc desc{};
		desc.quadrature_desc = quadrature_store.append(quadrature);
		desc.mass_quadrature_desc = mass_quadrature_store.append(mass_quadrature);

		// Warp legacy basis.
		auto &basis_desc = desc.basis_desc;
		basis_desc.element_kind = basis::ElementKind::Unknown;
		basis_desc.basis_family = basis::BasisFamily::Unknown;
		basis_desc.order = legacy_element.bases.front().order();
		basis_desc.orderq = basis_desc.order;
		basis_desc.dim = dim;
		basis_desc.basis_num = legacy_element.bases.size();
		basis_desc.eval_callback_id = basis_store.append_eval_callback(make_legacy_eval_callback(legacy_element, dim));
		basis_desc.is_parametric = legacy_element.has_parameterization;
		basis_desc.is_bernstein = false;

		// Warp legacy local2global.
		int first_mapping_id = 0;
		for (int local_basis_id = 0; local_basis_id < legacy_element.bases.size(); ++local_basis_id)
		{
			const std::vector<basis::Local2Global> &mapping = legacy_element.bases[local_basis_id].global();

			std::vector<int> node_ids;
			std::vector<double> weights;
			std::vector<double> node_positions;
			for (const auto &entry : mapping)
			{
				assert(entry.node.size() >= dim);
				node_ids.push_back(entry.index);
				weights.push_back(entry.val);
				for (int d = 0; d < dim; ++d)
					node_positions.push_back(entry.node(d));
			}

			int mapping_id = dof_mapping_store.append(node_ids, weights, node_positions);
			if (local_basis_id == 0)
				first_mapping_id = mapping_id;
		}
		desc.dof_mapping_range = Range{first_mapping_id, basis_desc.basis_num};

		// Wrap legacy local nodes for primitive callback.
		auto legacy_element_copy = std::make_shared<basis::ElementBases>(legacy_element);
		local_nodes_from_primitive.push_back([legacy_element_copy](const int primitive_id, const mesh::Mesh &mesh) {
			return legacy_element_copy->local_nodes_for_primitive(primitive_id, mesh);
		});
		desc.local_nodes_from_primitive_id = int(local_nodes_from_primitive.size()) - 1;

		element_desc[element_id] = desc;
	}

#ifdef POLYFEM_WITH_CUDA
	AssemblyDataView AssemblyData::device_view(ExecutionPolicy policy) const
	{
		if (need_host_device_sync_)
		{
			d_element_desc_ = copy_to_device_async<ElementDesc>(element_desc, policy);
			auto vecotr_tasks = build_device_vector_assembly_tasks(element_desc);
			d_vector_assembly_tasks_ = copy_to_device_async<DeviceVectorAssemblyTask>(vecotr_tasks, policy);
			auto matrix_tasks = build_device_matrix_assembly_tasks(element_desc);
			d_matrix_assembly_tasks_ = copy_to_device_async<DeviceMatrixAssemblyTask>(matrix_tasks, policy);
			policy.stream->sync();
			need_host_device_sync_ = false;
		}

		return AssemblyDataView{
			*d_element_desc_,
			quadrature_store.device_view(policy),
			mass_quadrature_store.device_view(policy),
			basis_store.device_view(policy),
			dof_mapping_store.device_view(policy),
			{} /* callback is not compatible with GPU */,
			*d_vector_assembly_tasks_,
			*d_matrix_assembly_tasks_};
	}

	void AssemblyData::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_element_desc_ = {};
		d_vector_assembly_tasks_ = {};
		d_matrix_assembly_tasks_ = {};
		quadrature_store.clear_device_storage();
		mass_quadrature_store.clear_device_storage();
		basis_store.clear_device_storage();
		dof_mapping_store.clear_device_storage();
	}
#endif

} // namespace polyfem::assembler
