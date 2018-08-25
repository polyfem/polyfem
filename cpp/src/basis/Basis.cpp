#include <polyfem/Basis.hpp>

#include <iostream>


namespace polyfem
{
	Basis::Basis()
	{ }


	void Basis::init(const int global_index, const int local_index, const RowVectorNd &node)
	{
		global_.resize(1);
		global_.front().index = global_index;
		global_.front().val = 1;
		global_.front().node = node;

		local_index_ = local_index;
	}

}
