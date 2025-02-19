#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>

namespace polyfem
{
	namespace assembler
	{
		/// Caches basis evaluation and geometric mapping at every element
		class AssemblyValsCache
		{
		public:
			/// computes the basis evaluation and geometric mapping
			/// for each of the given ElementBases in bases
			/// initializes cache member
			void init(const bool is_volume, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const bool is_mass = false);

			/// retrieves cached basis evaluation and geometric for the given element
			/// if it doesn't exist, computes and caches it (modifies cache member in the latter case)
			void compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis, ElementAssemblyValues &vals) const;

			void update(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis);

			void clear()
			{
				cache.clear();
			}

			inline bool is_initialized() const { return !cache.empty(); }

			inline bool is_mass() const { return is_mass_; }

		private:
			std::vector<ElementAssemblyValues> cache; ///< vector of basis values and geometric mapping with one entry per element
			bool is_mass_;
		};
	} // namespace assembler
} // namespace polyfem
