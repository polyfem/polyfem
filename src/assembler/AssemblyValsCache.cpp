#include <polyfem/AssemblyValsCache.hpp>
#include <polyfem/par_for.hpp>

#ifdef POLYFEM_WITH_TBB
#include <tbb/parallel_for.h>
#endif

namespace polyfem
{
	void AssemblyValsCache::init(const bool is_volume, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases)
	{
		const int n_bases = bases.size();
		cache.resize(n_bases);

#if defined(POLYFEM_WITH_CPP_THREADS)
		polyfem::par_for(n_bases, [&](int start, int end, int t)
						 {
							 for (int e = start; e < end; ++e)
							 {
#elif defined(POLYFEM_WITH_TBB)
		tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
			for (int e = r.begin(); e != r.end(); ++e)
			{
#else
		for (int e = 0; e < n_bases; ++e)
		{
#endif
								 cache[e].compute(e, is_volume, bases[e], gbases[e]);
							 }
#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
						 });
#endif

		// for (int e = 0; e < n_bases; ++e)
		// {
		// 	cache[e].compute(e, is_volume, bases[e], gbases[e]);
		// }
	}

	void AssemblyValsCache::compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis, ElementAssemblyValues &vals) const
	{
		if (cache.empty())
			vals.compute(el_index, is_volume, basis, gbasis);
		else
			vals = cache[el_index];
	}

} // namespace polyfem