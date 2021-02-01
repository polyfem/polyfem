#include <polyfem/AssemblyValsCache.hpp>

namespace polyfem
{
    void AssemblyValsCache::init(const bool is_volume, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases)
    {
        cache.resize(bases.size());
        for (int e = 0; e < bases.size(); ++e)
        {
            cache[e].compute(e, is_volume, bases[e], gbases[e]);
        }
    }

    void AssemblyValsCache::compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis, ElementAssemblyValues &vals) const
    {
        if (cache.empty())
            vals.compute(el_index, is_volume, basis, gbasis);
        else
            vals = cache[el_index];
    }

} // namespace polyfem