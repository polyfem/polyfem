#include "HexBasis.hpp"

#include <cassert>

namespace poly_fem
{
	HexBasis::HexBasis(const int global_index, const Eigen::MatrixXd &coeff, const int disc_order)
	: Basis(global_index, coeff), disc_order_(disc_order)
	{ }


	void HexBasis::basis(const Eigen::MatrixXd &xne, const int index, Eigen::MatrixXd &val) const
	{
		auto x=xne.col(0).array();
		auto n=xne.col(1).array();
		auto e=xne.col(2).array();

		switch(disc_order_)
		{
			case 1:
			{
				switch(index)
				{
					case 0: val = (1-x)*(1-n)*(1-e); break;
					case 1: val = (  x)*(1-n)*(1-e); break;
					case 2: val = (  x)*(  n)*(1-e); break;
					case 3: val = (1-x)*(  n)*(1-e); break;
					case 4: val = (1-x)*(1-n)*(  e); break;
					case 5: val = (  x)*(1-n)*(  e); break;
					case 6: val = (  x)*(  n)*(  e); break;
					case 7: val = (1-x)*(  n)*(  e); break;
					default: assert(false);
				}

				break;
			}

			//No H2 implemented
			default: assert(false);
		}
	}

	void HexBasis::grad(const Eigen::MatrixXd &xne, const int index, Eigen::MatrixXd &val) const
	{
		val.resize(xne.rows(), 3);

		auto x=xne.col(0).array();
		auto n=xne.col(1).array();
		auto e=xne.col(2).array();

		switch(disc_order_)
		{
			case 1:
			{
				switch(index)
				{
					case 0:
					{
						//(1-x)*(1-n)*(1-e);
						val.col(0) = -      (1-n)*(1-e);
						val.col(1) = -(1-x)      *(1-e);
						val.col(2) = -(1-x)*(1-n);

						break;
					}
					case 1:
					{
						//(  x)*(1-n)*(1-e)
						val.col(0) =        (1-n)*(1-e);
						val.col(1) = -(  x)      *(1-e);
						val.col(2) = -(  x)*(1-n);

						break;
					}
					case 2:
					{
						//(  x)*(  n)*(1-e)
						val.col(0) =        (  n)*(1-e);
						val.col(1) =  (  x)      *(1-e);
						val.col(2) = -(  x)*(  n);

						break;
					}
					case 3:
					{
						//(1-x)*(  n)*(1-e);
						val.col(0) = -       (  n)*(1-e);
						val.col(1) =  (1-x)      *(1-e);
						val.col(2) = -(1-x)*(  n);

						break;
					}
					case 4:
					{
						//(1-x)*(1-n)*(  e);
						val.col(0) = -      (1-n)*(  e);
						val.col(1) = -(1-x)      *(  e);
						val.col(2) =  (1-x)*(1-n);

						break;
					}
					case 5:
					{
						//(  x)*(1-n)*(  e);
						val.col(0) =        (1-n)*(  e);
						val.col(1) = -(  x)      *(  e);
						val.col(2) =  (  x)*(1-n);

						break;
					}
					case 6:
					{
						//(  x)*(  n)*(  e);
						val.col(0) =       (  n)*(  e);
						val.col(1) = (  x)      *(  e);
						val.col(2) = (  x)*(  n);

						break;
					}
					case 7:
					{
						//(1-x)*(  n)*(  e);
						val.col(0) = -      (  n)*(  e);
						val.col(1) =  (1-x)      *(  e);
						val.col(2) =  (1-x)*(  n);

						break;
					}

					default: assert(false);
				}

				break;
			}

			default: assert(false);
		}
	}
}
