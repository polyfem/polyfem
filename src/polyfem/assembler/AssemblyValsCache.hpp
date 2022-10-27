#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>

namespace polyfem
{
	namespace assembler
	{
		class AssemblyValsCache
		{
		public:
			void init(const bool is_volume, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const bool is_mass = false);
			void compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis, ElementAssemblyValues &vals) const;

			void clear()
			{
				cache.clear();
			}

			inline bool is_mass() const { return is_mass_; }

		private:
			std::vector<ElementAssemblyValues> cache;
			bool is_mass_;
		};
	} // namespace assembler
} // namespace polyfem
