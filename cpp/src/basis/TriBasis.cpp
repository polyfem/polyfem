#include "TriBasis.hpp"

#include <cassert>
#include <algorithm>

namespace poly_fem
{
	void TriBasis::basis(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
	{
		switch(discr_order)
		{
			case 1:
			{
				switch(local_index)
				{
					case 0: val = 1-uv.col(0).array()-uv.col(1).array(); break;
					case 1: val = uv.col(0); break;
					case 2: val = uv.col(1); break;
					default: assert(false);
				}
				break;
			}

			case 2:
			{
				auto u=uv.col(0).array();
				auto v=uv.col(1).array();

				switch(local_index)
				{
					case 0: val = (1 - u - v) * (1 -2*u -2*v); break;
					case 1: val = u * (2*u - 1); break;
					case 2: val = v * (2*v - 1); break;

					case 3: val = 4*u * (1 - u - v); break;
					case 4: val = 4 * u * v; break;
					case 5: val = 4*v * (1 - u - v); break;
					default: assert(false);
				}
				break;
			}
			default: assert(false); break;
		}
	}

	void TriBasis::grad(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
	{

		val.resize(uv.rows(),2);

		switch(discr_order)
		{
			case 1:
			{
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

				break;
			}

			case 2:
			{
				auto u=uv.col(0).array();
				auto v=uv.col(1).array();

				switch(local_index)
				{
					case 0:
					{
						val.col(0) = -3 + 4 * u + 4 * v;
						val.col(1) = -3 + 4 * u + 4 * v;
						break;
					}
					case 1:
					{
						val.col(0) = 4 * u - 1;
						val.col(1).setZero();
						break;
					}
					case 2:
					{
						val.col(0).setZero();
						val.col(1) = 4 * v - 1;
						break;
					}

					case 3:
					{
						val.col(0) = 4 - 8 * u - 4 * v;
						val.col(1) = -4 * u;
						break;
					}
					case 4:
					{
						val.col(0) = 4 * v;
						val.col(1) = 4 * u;
						break;
					}
					case 5:
					{
						val.col(0) = -4 * v;
						val.col(1) = 4 - 4 * u - 8 * v;
						break;
					}
					default: assert(false);
				}

				break;
			}

			default: assert(false); break;
		}
	}
}
