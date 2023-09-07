#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <cassert>

namespace polyfem
{
	namespace mesh
	{
		enum class BoundaryType
		{
			TRI_LINE,  ///< Boundary of a triangle in 2D
			QUAD_LINE, ///< Boundary of a quad in 2D
			TRI,
			QUAD,
			POLYGON,
			POLYHEDRON,
			INVALID
		};

		/// @brief Boundary primitive IDs for a single element.
		class LocalBoundary
		{
		public:
			/// @brief Construct a new Local Boundary object for a given element.
			/// @param global_element_id Element ID
			/// @param type Type of boundary for the element
			LocalBoundary(const int global_element_id, BoundaryType type);

			/// @brief Copy constructor
			/// @param other LocalBoundary to copy
			LocalBoundary(const LocalBoundary &other);

			/// @brief Mark a boundary primitive as a part of the global boundary.
			/// @param global_index Global index of the boundary primitive
			/// @param local_index Local index of the boundary primitive
			void add_boundary_primitive(const int global_index, const int local_index);

			/// @brief Number of boundary primitives for the element.
			/// @return Number of boundary primitives for the element.
			int size() const { return local_primitive_id_.size(); }

			/// @brief Check if the element has any boundary primitives.
			/// @return True if the element has no boundary primitives.
			bool empty() const { return size() <= 0; }

			/// @brief Get the element's ID.
			/// @return Element ID.
			int element_id() const { return global_element_id_; }

			/// @brief Get the type of boundary for the element.
			/// @return Type of boundary for the element.
			BoundaryType type() const { return type_; }

			/// @brief Get the i-th boundary primitive's local ID.
			/// @param index Index of the boundary primitive.
			/// @return Local ID of the boundary primitive.
			int local_primitive_id(const int index) const { return local_primitive_id_[index]; }

			/// @brief Get the i-th boundary primitive's global ID.
			/// @param index Index of the boundary primitive.
			/// @return Global ID of the boundary primitive.
			int global_primitive_id(const int index) const { return global_primitive_id_[index]; }

			/// @brief Get the i-th boundary primitive's local ID.
			/// @param index Index of the boundary primitive.
			/// @return Local ID of the boundary primitive.
			int operator[](const int index) const { return local_primitive_id(index); }

			/// @brief Remove all boundary primitives that are also in another LocalBoundary.
			/// @param other Other LocalBoundary to remove from this one.
			void remove_from(const LocalBoundary &other);

			/// @brief Remove a boundary primitive from the element.
			/// @param index Index of the boundary primitive to remove.
			void remove_tag_for_index(const int index);

			/// @brief Print the LocalBoundary to an output stream.
			/// @param os Output stream
			/// @param lb LocalBoundary to print
			/// @return Output stream
			friend std::ostream &operator<<(std::ostream &os, const LocalBoundary &lb);

		private:
			/// @brief Global IDs of the boundary primitives.
			std::vector<int> global_primitive_id_;
			/// @brief Local IDs of the boundary primitives.
			std::vector<int> local_primitive_id_;

			/// @brief Element ID.
			const int global_element_id_;
			/// @brief Type of boundary primitives for the element.
			const BoundaryType type_;
		};
	} // namespace mesh
} // namespace polyfem
