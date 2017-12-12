#include "HexBasis.hpp"

#include <cassert>

namespace poly_fem
{

	int HexBasis::build_bases(const Mesh &mesh, std::vector< std::vector<Basis> > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
	{
		assert(mesh.is_volume());
		assert(false);
		return 0;

		// int disc_order = 1;

		// bases.resize(mesh.els.rows());
		// int n_bases = int(mesh.pts.rows());

		// for(long i=0; i < mesh.els.rows(); ++i)
		// {
		// 	std::vector<Basis> &b=bases[i];
		// 	b.resize(8);

		// 	for(int j = 0; j < 8; ++j)
		// 	{
		// 		const int global_index = mesh.els(i,j);
		// 		b[j].init(global_index, j, mesh.pts.row(global_index));

		// 		b[j].set_basis([disc_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { HexBasis::basis(disc_order, j, uv, val); });
		// 		b[j].set_grad( [disc_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  HexBasis::grad(disc_order, j, uv, val); });
		// 	}
		// }

		// return n_bases;
	}

	void HexBasis::basis(const int disc_order, const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
	{
		auto x=xne.col(0).array();
		auto n=xne.col(1).array();
		auto e=xne.col(2).array();

		switch(disc_order)
		{
			case 1:
			{
				switch(local_index)
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

	void HexBasis::grad(const int disc_order, const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
	{
		val.resize(xne.rows(), 3);

		auto x=xne.col(0).array();
		auto n=xne.col(1).array();
		auto e=xne.col(2).array();

		switch(disc_order)
		{
			case 1:
			{
				switch(local_index)
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
