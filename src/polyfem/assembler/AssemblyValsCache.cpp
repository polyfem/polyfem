#include "AssemblyValsCache.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>

namespace polyfem
{
	using namespace basis;

	namespace assembler
	{
		void AssemblyValsCache::init(const bool is_volume, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases, const bool is_mass)
		{
			is_mass_ = is_mass;
			const int n_bases = bases.size();
			cache.resize(n_bases);

			utils::maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
				for (int e = start; e < end; ++e)
				{
					if (is_mass_)
					{
						auto &quadrature = cache[e].quadrature;
						bases[e].compute_mass_quadrature(quadrature);
						cache[e].compute(e, is_volume, quadrature.points, bases[e], gbases[e]);
					}
					else
						cache[e].compute(e, is_volume, bases[e], gbases[e]);
				}
			});
		}

		void AssemblyValsCache::compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis, ElementAssemblyValues &vals) const
		{
			if (cache.empty())
			{
				if (is_mass_)
				{
					auto &quadrature = vals.quadrature;
					basis.compute_mass_quadrature(quadrature);
					vals.compute(el_index, is_volume, quadrature.points, basis, gbasis);
				}
				else
					vals.compute(el_index, is_volume, basis, gbasis);
			}
			else
				vals = cache[el_index];
		}
	} // namespace assembler

} // namespace polyfem
