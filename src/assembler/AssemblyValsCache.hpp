#pragma once

#include <polyfem/ElementAssemblyValues.hpp>

namespace polyfem
{
	class AssemblyValsCache
	{
	public:
		void init(const bool is_volume, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases);
		void compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis, ElementAssemblyValues &vals) const;

		void clear()
		{
			cache.clear();
		}

	private:
		std::vector<ElementAssemblyValues> cache;
	};

} // namespace polyfem
