#pragma once

#include <polyfem/ElementAssemblyValues.hpp>

namespace polyfem
{
	namespace assembler
	{
		class AssemblyValsCache
		{
		public:
			void init(const bool is_volume, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases);
			void compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis, ElementAssemblyValues &vals) const;

			void clear()
			{
				cache.clear();
			}

		private:
			std::vector<ElementAssemblyValues> cache;
		};
	} // namespace assembler
} // namespace polyfem
