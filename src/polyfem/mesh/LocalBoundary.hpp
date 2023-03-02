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
			TRI_LINE,
			QUAD_LINE,
			TRI,
			QUAD,
			POLYGON,
			POLYHEDRON,
			INVALID
		};

		class LocalBoundary
		{
		public:
			LocalBoundary(const int global_id, BoundaryType type)
				: global_element_id_(global_id), type_(type)
			{
			}

			LocalBoundary(const LocalBoundary &other)
				: global_primitive_id_(other.global_primitive_id_), local_primitive_id_(other.local_primitive_id_),
				  global_element_id_(other.global_element_id_), type_(other.type_)
			{
			}

			void add_boundary_primitive(const int global_index, const int local_index)
			{
				global_primitive_id_.emplace_back(global_index);
				local_primitive_id_.emplace_back(local_index);
			}

			int size() const { return local_primitive_id_.size(); }
			bool empty() const { return size() <= 0; }

			int element_id() const { return global_element_id_; }

			BoundaryType type() const { return type_; }

			inline int operator[](const int index) const
			{
				assert(index < size());
				return local_primitive_id_[index];
			}

			inline int global_primitive_id(const int index) const { return global_primitive_id_[index]; }

			void remove_from(const LocalBoundary &other)
			{
				std::vector<int> to_remove;

				for (int j = 0; j < size(); ++j)
				{
					const int loc_id = (*this)[j];
					for (int i = 0; i < other.size(); ++i)
					{
						if (other[i] == loc_id)
						{
							to_remove.push_back(j);
							break;
						}
					}
				}

				for (int j : to_remove)
					remove_tag_for_index(j);
			}

			void remove_tag_for_index(const int index)
			{
				global_primitive_id_.erase(global_primitive_id_.begin() + index);
				local_primitive_id_.erase(local_primitive_id_.begin() + index);
			}

			friend std::ostream &operator<<(std::ostream &os, const LocalBoundary &lb)
			{
				for (int i = 0; i < lb.size(); ++i)
					os << lb[i] << " -> " << lb.global_primitive_id(i) << ", ";
				return os;
			}

		private:
			std::vector<int> global_primitive_id_;
			std::vector<int> local_primitive_id_;

			const int global_element_id_;
			const BoundaryType type_;
		};
	} // namespace mesh
} // namespace polyfem
