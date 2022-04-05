#include <polyfem/Basis.hpp>

#include <iostream>

namespace polyfem
{
	Basis::Basis()
		: order_(-1)
	{
	}

	void Basis::init(const int order, const int global_index, const int local_index, const RowVectorNd &node)
	{
		order_ = order;
		global_.resize(1);
		global_.front().index = global_index;
		global_.front().val = 1;
		global_.front().node = node;

		local_index_ = local_index;
	}

} // namespace polyfem
