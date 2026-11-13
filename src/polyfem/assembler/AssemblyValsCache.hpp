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
			/// for each of the given ElementBases in bases initializes cache member.
			/// will clear the old cache if exists.
			void init(const bool is_volume, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const bool is_mass = false);

			/// initialize an empty cache.
			/// does not computes the basis evaluation and geometric mapping.
			/// will clear the old cache if exists.
			void init_empty(const bool is_mass = false);

			/// Copies cached values into vals, or computes into vals when the cache is empty.
			void compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis, ElementAssemblyValues &vals) const;

			/// Returns cached values, or computes into scratch when the cache is empty.
			/// The returned reference is valid while its cache entry or scratch remains unchanged.
			const ElementAssemblyValues &get_or_compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis, ElementAssemblyValues &scratch) const
			{
				if (is_initialized())
					return cached_values(el_index);
				compute(el_index, is_volume, basis, gbasis, scratch);
				return scratch;
			}

			void update(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis);

			void clear()
			{
				cache.clear();
			}

			inline bool is_initialized() const { return !cache.empty(); }

			const ElementAssemblyValues &cached_values(const int element) const
			{
				assert(element >= 0 && static_cast<size_t>(element) < cache.size());
				return cache[element];
			}

			inline bool is_mass() const { return is_mass_; }

		private:
			std::vector<ElementAssemblyValues> cache; ///< vector of basis values and geometric mapping with one entry per element
			bool is_mass_;
		};
	} // namespace assembler
} // namespace polyfem
