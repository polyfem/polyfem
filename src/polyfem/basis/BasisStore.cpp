#include "polyfem/basis/ElementBases.hpp"
#include <polyfem/basis/BasisStore.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::basis
{
	Range BasisStore::append_rational_weights(Span<const double> weights)
	{
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		Range r{static_cast<int>(rational_weights_.size()), static_cast<int>(weights.size())};
		rational_weights_.insert(rational_weights_.end(), weights.begin(), weights.end());
		return r;
	}

	int BasisStore::append_eval_callback(BasisEvalCallback callback)
	{
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		int idx = eval_callbacks_.size();
		eval_callbacks_.push_back(std::move(callback));
		return idx;
	}

	BasisStoreView BasisStore::view() const { return BasisStoreView{rational_weights_, eval_callbacks_}; }

#ifdef POLYFEM_WITH_CUDA
	/// Return view on device memory. Lazily sync data.
	BasisStoreView BasisStore::device_view(ExecutionPolicy policy) const
	{

		if (need_host_device_sync_)
		{
			assert(policy.stream && policy.mr);
			d_rational_weights_ = copy_to_device_async<double>(rational_weights_, policy);
			policy.stream->sync();
			need_host_device_sync_ = false;
		}
		return BasisStoreView{*d_rational_weights_, {} /*callback is not device compatible*/};
	}

	/// Release device storage.
	void BasisStore::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_rational_weights_ = {};
	}
#endif

} // namespace polyfem::basis
