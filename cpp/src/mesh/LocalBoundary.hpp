#ifndef LOCAL_HPP
#define LOCAL_HPP

#include <array>
#include <cassert>

namespace poly_fem
{
	enum class BoundaryType
	{
		TriLine,
		QuadLine,
		Tri,
		Quad,
		Invalid
	};

	class LocalBoundary
	{
	public:
		LocalBoundary()
		: global_element_id_(-1), type_(BoundaryType::Invalid), counter_(0)
		{}

		LocalBoundary(const int global_id, BoundaryType type)
		: global_element_id_(global_id), type_(type), counter_(0)
		{ }

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

		inline int operator[](const int index) const { assert(index<counter_); return local_primitive_id_[index];}

	private:
		std::array<int, 6> global_primitive_id_= {{-1, -1, -1, -1, -1, -1}};
		std::array<int, 6> local_primitive_id_= {{-1, -1, -1, -1, -1, -1}};

		const int global_element_id_;
		const BoundaryType type_;

		int counter_;
	};
}

#endif //LOCAL_HPP
