#include "LocalBoundary.hpp"

namespace polyfem::mesh
{
	LocalBoundary::LocalBoundary(const int global_element_id, BoundaryType type)
		: global_element_id_(global_element_id), type_(type)
	{
	}

	LocalBoundary::LocalBoundary(const LocalBoundary &other)
		: global_primitive_id_(other.global_primitive_id_), local_primitive_id_(other.local_primitive_id_),
		  global_element_id_(other.global_element_id_), type_(other.type_)
	{
	}

	void LocalBoundary::add_boundary_primitive(const int global_index, const int local_index)
	{
		global_primitive_id_.emplace_back(global_index);
		local_primitive_id_.emplace_back(local_index);
	}

	void LocalBoundary::remove_from(const LocalBoundary &other)
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

	void LocalBoundary::remove_tag_for_index(const int index)
	{
		global_primitive_id_.erase(global_primitive_id_.begin() + index);
		local_primitive_id_.erase(local_primitive_id_.begin() + index);
	}

	std::ostream &operator<<(std::ostream &os, const LocalBoundary &lb)
	{
		for (int i = 0; i < lb.size(); ++i)
			os << lb[i] << " -> " << lb.global_primitive_id(i) << ", ";
		return os;
	}
} // namespace polyfem::mesh
