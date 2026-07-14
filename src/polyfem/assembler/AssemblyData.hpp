#pragma once

#include <polyfem/basis/BasisStore.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/quadrature/QuadratureStore.hpp>
#include <polyfem/assembler/DofMappingStore.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>
#include <functional>
#include <memory>
#include <Eigen/Core>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem::assembler
{

	struct ElementDesc
	{
		quadrature::QuadratureDesc quadrature_desc;
		quadrature::QuadratureDesc mass_quadrature_desc;
		basis::BasisDesc basis_desc;
		Range dof_mapping_range;
		int local_nodes_from_primitive_id = -1;
	};

	// given primitive index (i.e. edge id), return element local node indexes.
	using LocalNodeFromPrimitiveFunc = std::function<Eigen::VectorXi(const int local_index, const mesh::Mesh &mesh)>;

#ifdef POLYFEM_WITH_CUDA
	/// Device task info per CUDA thread for vector assembly.
	struct DeviceVectorAssemblyTask
	{
		int elem_id;
		int basis_i;
	};

	/// Device task info per CUDA thread for matrix assembly.
	struct DeviceMatrixAssemblyTask
	{
		int elem_id;
		int basis_i;
		int basis_j;
	};
#endif

	struct AssemblyDataView
	{
		/// Per element descriptor.
		Span<const ElementDesc> element_desc;
		/// Stores quadrature points and weights.
		quadrature::QuadratureStoreView quadrature_store;
		/// Stores mass quadrature points and weights.
		quadrature::QuadratureStoreView mass_quadrature_store;
		/// Stores basis info.
		basis::BasisStoreView basis_store;
		/// Stores element local basis node to global mappings.
		DofMappingStoreView dof_mapping_store;
		Span<const LocalNodeFromPrimitiveFunc> local_nodes_from_primitive;

#ifdef POLYFEM_WITH_CUDA
		/// GPU per thread vector assembly tasks. Empty for host.
		Span<const DeviceVectorAssemblyTask> vector_assembly_tasks;
		/// GPU per thread matrix assembly tasks. Empty for host.
		Span<const DeviceMatrixAssemblyTask> matrix_assembly_tasks;
#endif
	};

	struct AssemblyDataMutView
	{
		std::vector<ElementDesc> *element_desc;
		quadrature::QuadratureStore *quadrature_store;
		quadrature::QuadratureStore *mass_quadrature_store;
		basis::BasisStore *basis_store;
		DofMappingStore *dof_mapping_store;
		std::vector<LocalNodeFromPrimitiveFunc> *local_nodes_from_primitive;
	};

	class AssemblyData
	{
	private:
		std::vector<ElementDesc> element_desc;
		quadrature::QuadratureStore quadrature_store;
		quadrature::QuadratureStore mass_quadrature_store;
		basis::BasisStore basis_store;
		DofMappingStore dof_mapping_store;
		std::vector<LocalNodeFromPrimitiveFunc> local_nodes_from_primitive;

#ifdef POLYFEM_WITH_CUDA
		mutable bool need_host_device_sync_ = true;
		mutable DeviceBuf<ElementDesc> d_element_desc_;
		mutable DeviceBuf<DeviceVectorAssemblyTask> d_vector_assembly_tasks_;
		mutable DeviceBuf<DeviceMatrixAssemblyTask> d_matrix_assembly_tasks_;
#endif

	public:
		AssemblyDataView view() const;
		AssemblyDataMutView mutable_view();

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device memory. Lazily sync data.
		AssemblyDataView device_view(ExecutionPolicy policy) const;

		/// Release device storage.
		void clear_device_storage();
#endif

		// ------------------------------------------------
		// Legacy compatibility layer.
		// ------------------------------------------------

		[[deprecated]] std::shared_ptr<std::vector<basis::ElementBases>> legacy_bases_ptr() const;

		[[deprecated]] void set_legacy_element(
			int element_id,
			const basis::ElementBases &legacy_element);
	};

} // namespace polyfem::assembler
