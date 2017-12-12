#include "TriBasis.hpp"

#include <cassert>
#include <algorithm>

namespace poly_fem
{
	void TriBasis::basis(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
	{
		assert(discr_order == 1);

		switch(local_index)
		{
			case 0: val = 1-uv.col(0).array()-uv.col(1).array(); break;
			case 1: val = uv.col(0); break;
			case 2: val = uv.col(1); break;
			default: assert(false);
		}
	}

	void TriBasis::grad(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
	{
		assert(discr_order == 1);

		val.resize(uv.rows(),2);
		val.setConstant(0);

		switch(local_index)
		{
			case 0:
			{
				val.col(0).setConstant(-1);
				val.col(1).setConstant(-1);
				break;
			}
			case 1:
			{
				val.col(0).setConstant(1);
				break;
			}
			case 2:
			{
				val.col(1).setConstant(1);
				break;
			}
			default:
			assert(false);
		}
	}
}
