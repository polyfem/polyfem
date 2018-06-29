#include <polyfem/Basis.hpp>

#include <iostream>


namespace poly_fem
{
	Basis::Basis()
	{ }


	void Basis::init(const int global_index, const int local_index, const Eigen::MatrixXd &node)
	{
		global_.resize(1);
		global_.front().index = global_index;
		global_.front().val = 1;
		global_.front().node = node;

		local_index_ = local_index;
	}

}
