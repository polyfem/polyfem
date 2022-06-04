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
			TriLine,
			QuadLine,
			Tri,
			Quad,
			Polygon,
			Polyhedron,
			Invalid
		};

		class LocalBoundary
		{
		public:
			LocalBoundary(const int global_id, BoundaryType type)
				: global_element_id_(global_id), type_(type), counter_(0)
			{
			}

			LocalBoundary(const LocalBoundary &other)
				: global_primitive_id_(other.global_primitive_id_), local_primitive_id_(other.local_primitive_id_),
				  global_element_id_(other.global_element_id_), type_(other.type_), counter_(other.counter_)
			{
			}

			void add_boundary_primitive(const int global_index, const int local_index)
			{
				global_primitive_id_[counter_] = global_index;
				local_primitive_id_[counter_] = local_index;

				++counter_;
			}

			int size() const { return counter_; }
			bool empty() const { return counter_ <= 0; }

			int element_id() const { return global_element_id_; }

			BoundaryType type() const { return type_; }

			inline int operator[](const int index) const
			{
				assert(index < counter_);
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
				for (int i = index + 1; i < size(); ++i)
				{
					global_primitive_id_[i - 1] = global_primitive_id_[i];
					local_primitive_id_[i - 1] = local_primitive_id_[i];
				}

				--counter_;
			}

			friend std::ostream &operator<<(std::ostream &os, const LocalBoundary &lb)
			{
				for (int i = 0; i < lb.size(); ++i)
					os << lb[i] << " -> " << lb.global_primitive_id(i) << ", ";
				return os;
			}

		private:
			std::array<int, 6> global_primitive_id_ = {{-1, -1, -1, -1, -1, -1}};
			std::array<int, 6> local_primitive_id_ = {{-1, -1, -1, -1, -1, -1}};

			const int global_element_id_;
			const BoundaryType type_;

			int counter_;
		};
	} // namespace mesh
} // namespace polyfem
