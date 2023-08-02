#include <polyfem/utils/MaybeParallelFor.hpp>

#if defined(POLYFEM_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>
#elif defined(POLYFEM_WITH_CPP_THREADS)
#include <polyfem/utils/par_for.hpp>
#include <execution>
#else
// Not using parallel for
#endif

namespace polyfem
{
	namespace utils
	{
		inline void maybe_parallel_for(int size, const std::function<void(int, int, int)> &partial_for)
		{
#if defined(POLYFEM_WITH_CPP_THREADS)
			par_for(size, partial_for);
#elif defined(POLYFEM_WITH_TBB)
			tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int> &r) {
				partial_for(r.begin(), r.end(), tbb::this_task_arena::current_thread_index());
			});
#else
			partial_for(0, size, /*thread_id=*/0); // actually the full for loop
#endif
		}

		inline void maybe_parallel_for(int size, const std::function<void(int)> &body)
		{
#if defined(POLYFEM_WITH_CPP_THREADS)
			for (int i = 0; i < size; ++i)
				body(i);
#elif defined(POLYFEM_WITH_TBB)
			tbb::parallel_for(0, size, body);
#else
			for (int i = 0; i < size; ++i)
				body(i);
#endif
		}

		template <typename LocalStorage>
		inline auto create_thread_storage(const LocalStorage &initial_local_storage)
		{
#if defined(POLYFEM_WITH_CPP_THREADS)
			return std::vector<LocalStorage>(get_n_threads(), initial_local_storage);
#elif defined(POLYFEM_WITH_TBB)
			return tbb::enumerable_thread_specific<LocalStorage>(initial_local_storage);
#else
			return std::array<LocalStorage, 1>{{initial_local_storage}};
#endif
		}

		template <typename Storages>
		inline auto &get_local_thread_storage(Storages &storage, int thread_id)
		{
#if defined(POLYFEM_WITH_CPP_THREADS)
			return storage[thread_id];
#elif defined(POLYFEM_WITH_TBB)
			return storage.local();
#else
			assert(thread_id == 0);
			assert(storage.size() == 1);
			return storage[0];
#endif
		}
	} // namespace utils
} // namespace polyfem
