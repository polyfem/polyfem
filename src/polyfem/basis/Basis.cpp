#include <polyfem/basis/Basis.hpp>

namespace polyfem::basis
{
	Basis::Basis()
		: order_(-1)
	{
	}

	void Basis::init(const int order, const int global_index, const int local_index, const RowVectorNd &node)
	{
		order_ = order;
		global_ = {{Local2Global(global_index, node, 1)}};
		local_index_ = local_index;
	}

	std::ostream &operator<<(std::ostream &os, const Basis &obj)
	{
		os << obj.local_index_ << ":\n";
		for (auto l2g : obj.global_)
			os << "\tl2g: " << l2g.index << " (" << l2g.node << ") " << l2g.val << "\n";
		return os;
	}
} // namespace polyfem::basis
