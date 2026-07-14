#pragma once

#include <polyfem/basis/Basis.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#endif

namespace polyfem::assembler
{

	struct DofMappingDesc
	{
		Range id_and_weight_range;
		Range node_position_range;
	};

	struct DofMappingStoreView
	{
		/// Per element mapping descriptor.
		Span<const DofMappingDesc> mapping_desc;
		/// Element local node to global node indexes.
		Span<const int> node_ids;
		/// Element local node to global node weights.
		Span<const double> weights;
		/// Element local node physical position.
		Span<const double> node_positions;

		POLYFEM_BOTH Span<const int> get_node_ids(int mapping_id) const
		{
			auto &desc = mapping_desc[mapping_id];
			return slice_by_range(node_ids, desc.id_and_weight_range);
		}
		POLYFEM_BOTH Span<const double> get_weights(int mapping_id) const
		{
			auto &desc = mapping_desc[mapping_id];
			return slice_by_range(weights, desc.id_and_weight_range);
		}
		POLYFEM_BOTH Span<const double> get_positions(int mapping_id) const
		{
			auto &desc = mapping_desc[mapping_id];
			return slice_by_range(node_positions, desc.node_position_range);
		}

		/// Get legacy local to global.
		std::vector<basis::Local2Global> get_local_to_global(int mapping_id, int dim) const;
	};

	class DofMappingStore
	{
	private:
		std::vector<DofMappingDesc> mapping_desc_;
		std::vector<int> node_ids_;
		std::vector<double> weights_;
		std::vector<double> node_positions_;

#ifdef POLYFEM_WITH_CUDA
		mutable bool need_host_device_sync_ = true;
		mutable DeviceBuf<DofMappingDesc> d_mapping_desc_;
		mutable DeviceBuf<int> d_node_ids_;
		mutable DeviceBuf<double> d_weights_;
		mutable DeviceBuf<double> d_node_positions_;
#endif

	public:
		DofMappingStoreView view() const;

		/// Return mapping id.
		int append(Span<const int> node_ids, Span<const double> weights, Span<const double> node_positions);

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device memory. Lazily sync data.
		DofMappingStoreView device_view(ExecutionPolicy policy) const;

		/// Release device storage.
		void clear_device_storage();
#endif
	};

} // namespace polyfem::assembler
