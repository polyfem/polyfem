#pragma once

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
	// Perform a parallel (maybe) for loop.
	// The parallel for used depends on the compile definitions.
	// The overall for loop is from 0 up to `size` with an increment of 1.
	inline void maybe_parallel_for(int size, const std::function<void(int, int)> &partial_for);

	// Returns thread specific storage for further use in `maybe_parallel_for()`.
	// The return type depends on the threading library used.
	//     TBB         ⟹ `std::vector<LocalStorage>`
	//     C++ Threads ⟹ `tbb::enumerable_thread_specific<LocalStorage>`
	//     none        ⟹ `LocalStorage`
	template <typename LocalStorage>
	inline auto create_thread_storage(const LocalStorage &initial_local_storage);

	template <typename Storages>
	inline auto &get_local_thread_storage(Storages &storage);

} // namespace polyfem

#include <polyfem/MaybeParallelFor.tpp>
