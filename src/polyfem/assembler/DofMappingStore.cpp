#include <polyfem/assembler/DofMappingStore.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::assembler
{

	DofMappingStoreView DofMappingStore::view() const
	{
		return DofMappingStoreView{mapping_desc_, node_ids_, weights_, node_positions_};
	}

	std::vector<basis::Local2Global> DofMappingStoreView::get_local_to_global(int mapping_id, int dim) const
	{
		assert(dim >= 1 && dim <= 3);
		auto ids = get_node_ids(mapping_id);
		auto weights = get_weights(mapping_id);
		auto positions = get_positions(mapping_id);
		assert(ids.size() == weights.size());
		assert(positions.size() == ids.size() * dim);

		std::vector<basis::Local2Global> result;
		result.reserve(ids.size());
		for (int i = 0; i < ids.size(); ++i)
		{
			RowVectorNd node(dim);
			for (int d = 0; d < dim; ++d)
				node(d) = positions[dim * i + d];
			result.emplace_back(ids[i], node, weights[i]);
		}
		return result;
	}

	int DofMappingStore::append(Span<const int> node_ids, Span<const double> weights, Span<const double> node_positions)
	{
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		DofMappingDesc desc;
		desc.id_and_weight_range = Range{static_cast<int>(this->node_ids_.size()),
										 static_cast<int>(node_ids.size())};
		desc.node_position_range = Range{static_cast<int>(this->node_positions_.size()),
										 static_cast<int>(node_positions.size())};

		this->node_ids_.insert(this->node_ids_.end(), node_ids.begin(), node_ids.end());
		this->weights_.insert(this->weights_.end(), weights.begin(), weights.end());
		this->node_positions_.insert(this->node_positions_.end(), node_positions.begin(), node_positions.end());
		this->mapping_desc_.push_back(desc);
		return this->mapping_desc_.size() - 1;
	}

#ifdef POLYFEM_WITH_CUDA
	DofMappingStoreView DofMappingStore::device_view(ExecutionPolicy policy) const
	{

		if (need_host_device_sync_)
		{
			d_mapping_desc_ = copy_to_device_async<DofMappingDesc>(mapping_desc_, policy);
			d_node_ids_ = copy_to_device_async<int>(node_ids_, policy);
			d_weights_ = copy_to_device_async<double>(weights_, policy);
			d_node_positions_ = copy_to_device_async<double>(node_positions_, policy);
			policy.stream->sync();
			need_host_device_sync_ = false;
		}
		return DofMappingStoreView{*d_mapping_desc_, *d_node_ids_, *d_weights_, *d_node_positions_};
	}

	void DofMappingStore::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_mapping_desc_ = {};
		d_node_ids_ = {};
		d_weights_ = {};
		d_node_positions_ = {};
	}
#endif

} // namespace polyfem::assembler
