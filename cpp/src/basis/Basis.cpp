#include "Basis.hpp"

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

	void Basis::basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		basis_(uv, val);
	}

	void Basis::grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		grad_(uv, val);
	}
}