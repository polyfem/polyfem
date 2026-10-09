#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/Jacobian.hpp>

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

			/// retrieves cached basis evaluation and geometric for the given element
			/// if it doesn't exist, computes and caches it (modifies cache member in the latter case)
			void compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis, ElementAssemblyValues &vals) const;

			void update(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis);

			void clear()
			{
				cache.clear();
				control_points_.clear();
			}

			/// Control points of every element, which compute() hands to each
			/// element's values for the robust Jacobian in the energy. They
			/// belong to the bases the cache was initialized with, so init()
			/// and clear() drop them.
			void set_control_points(std::vector<utils::ElementControlPoints> control_points)
			{
				control_points_ = std::move(control_points);
			}

			inline bool is_initialized() const { return !cache.empty(); }

			inline bool is_mass() const { return is_mass_; }

		private:
			std::vector<ElementAssemblyValues> cache; ///< vector of basis values and geometric mapping with one entry per element
			bool is_mass_;
			std::vector<utils::ElementControlPoints> control_points_;
		};
	} // namespace assembler
} // namespace polyfem
