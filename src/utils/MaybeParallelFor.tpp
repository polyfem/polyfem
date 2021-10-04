#include <polyfem/MaybeParallelFor.hpp>

#if defined(POLYFEM_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>
#elif defined(POLYFEM_WITH_CPP_THREADS)
#include <polyfem/par_for.hpp>
#else
// Not using parallel for
#endif

namespace polyfem
{

	inline void maybe_parallel_for(int size, const std::function<void(int, int)> &partial_for)
	{
#if defined(POLYFEM_WITH_CPP_THREADS)
		par_for(size, partial_for);
#elif defined(POLYFEM_WITH_TBB)
		tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int> &r) {
			partial_for(r.begin(), r.end());
		});
#else
		partial_for(0, size); // actually the full for loop
#endif
	}

	template <typename LocalStorage>
	inline auto create_thread_storage(const LocalStorage &initial_local_storage)
	{
#if defined(POLYFEM_WITH_CPP_THREADS)
		return std::vector<LocalStorage>(polyfem::get_n_threads(), initial_local_storage);
#elif defined(POLYFEM_WITH_TBB)
		return tbb::enumerable_thread_specific<LocalStorage>(initial_local_storage);
#else
		return std::array<LocalStorage, 1>{{initial_local_storage}};
#endif
	}

	template <typename Storages>
	inline auto &get_local_thread_storage(Storages &storage)
	{
#if defined(POLYFEM_WITH_CPP_THREADS)
		assert(par_for_thread_id >= 0);
		return storage[par_for_thread_id];
#elif defined(POLYFEM_WITH_TBB)
		return storage.local();
#else
		assert(storage.size() == 0);
		return storage[0];
#endif
	}

} // namespace polyfem
