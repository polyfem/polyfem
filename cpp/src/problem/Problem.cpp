#include "Problem.hpp"

#include <iostream>

namespace poly_fem
{
	bool Problem::has_exact_sol() const
	{
		return problem_num_ < 3;
	}

	void Problem::rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		switch(problem_num_)
		{
			case 0: val = Eigen::MatrixXd::Zero(pts.rows(), 1); return;
			case 1: val = 2*Eigen::MatrixXd::Ones(pts.rows(), 1); return;

			case 2: {

				auto cx2 = (9*x-2) * (9*x-2);
				auto cy2 = (9*y-2) * (9*y-2);

				auto cx1 = (9*x+1) * (9*x+1);
				auto cx7 = (9*x-7) * (9*x-7);

				auto cy3 = (9*y-3) * (9*y-3);
				auto cx4 = (9*x-4) * (9*x-4);

				auto cy7 = (9*y-7) * (9*y-7);

				auto s1 = (-40.5 * x+9) * (-40.5 * x + 9);
				auto s2 = (-162./49. * x - 18./49.) * (-162./49. * x - 18./49.);
				auto s3 = (-40.5 * x + 31.5) * (-40.5 * x + 31.5);
				auto s4 = (-162. * x + 72) * (-162 * x + 72);

				auto s5 = (-40.5 * y + 9) * (-40.5 * y + 9);
				auto s6 = (-40.5 * y + 13.5) * (-40.5 * y + 13.5);
				auto s7 = (-162 * y + 126) * (-162 * y + 126);

				val = 243./4. * (-0.25 * cx2 - 0.25 * cy2).exp() -   0.75 * s1 * (-0.25 * cx2 - 0.25 *cy2).exp() +
				36693./19600. * (-1./49. * cx1 - 0.9 * y - 0.1).exp()  - 0.75 * s2 * (- 1./49 * cx1 - 0.9 * y - 0.1).exp() +
				40.5 * (-0.25 * cx7 - 0.25 * cy3).exp()   - 0.5 * s3 * (-0.25 * cx7 - 0.25 * cy3).exp() -
				324./5.  * (-cx4-cy7).exp() + 0.2 * s4 * (-cx4-cy7).exp() -
				0.75 * s5 * (-0.25 * cx2 - 0.25 *cy2).exp()  - 0.5 * s6 * (-0.25 * cx7 - 0.25 * cy3).exp() +
				0.2 * s7 * (-cx4-cy7).exp();
				val*=-1;

				return;
			}

			case 3: val = Eigen::MatrixXd::Zero(pts.rows(), 2); return;

			default: assert(false);
		}
	}

	void Problem::bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}


	void Problem::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		switch(problem_num_)
		{
			case 0: val = x; return;
			case 1: val = x * x; return;
			case 2:
			{
				auto cx2 = (9*x-2) * (9*x-2);
				auto cy2 = (9*y-2) * (9*y-2);

				auto cx1 = (9*x+1) * (9*x+1);
				auto cx7 = (9*x-7) * (9*x-7);

				auto cy3 = (9*y-3) * (9*y-3);
				auto cx4 = (9*x-4) * (9*x-4);

				auto cy7 = (9*y-7) * (9*y-7);

		// val = 0.75 * (-0.25 * cx2 - 0.25 * cy2).exp() +
		// 0.75 * (-1./49. * cx1 - 0.9 * y - 0.1).exp() +
		// 0.5 * (-0.25 * cx7 - 0.25 * cy3).exp() -
		// 0.2 * (-cx4-cy7).exp();

				val = (3./4.)*exp(-(1./4.)*cx2-(1./4.)*cy2)+(3./4.)*exp(-(1./49.)*cx1-(9./10.)*y-1./10.)+(1./2.)*exp(-(1./4.)*cx7-(1./4.)*cy3)-(1./5.)*exp(-cx4-cy7);

				return;
			}

			case 3:
			{
				val = Eigen::MatrixXd::Zero(pts.rows(), 2);

				for(long i = 0; i < x.size(); ++i)
				{
					if(fabs(x(i)-1)<1e-8)
						val(i, 0)=-0.1;
					else if(fabs(x(i))<1e-8)
						val(i, 0)=0.1;
					// else
						// assert(false);
				}

				return;
			}

			default: assert(false);
		}
	}

	void Problem::remove_neumann_nodes(const std::vector< ElementBases > &bases, const std::vector<int> &boundary_tag, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes)
	{
		if(problem_num_ < 3)
			return;

		//TODO use b tag for everything
		for(std::size_t j = 0; j < local_boundary.size(); ++j)
		{
			if(!local_boundary[j].is_boundary()) continue;
			// std::cout<<j<<" before "<<local_boundary[j].flags()<<std::endl;
			for(int i = 0; i < int(boundary_tag.size()); ++i)
			{
				const int tag = boundary_tag[i];
				// std::cout<<i<<" "<<tag<<std::endl;
				if(tag == 1 || tag == 3) continue;

				local_boundary[j].clear_edge_tag(i);
			}
			// std::cout<<j<<" after "<<local_boundary[j].flags()<<std::endl;
		}

		std::vector<int> old_b_nodes = boundary_nodes;
		boundary_nodes.clear();

		for(std::size_t i = 0; i < bases.size(); ++i)
		{
			const ElementBases &bs = bases[i];

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				if(std::find(old_b_nodes.begin(), old_b_nodes.end(), bs.bases[j].global().front().index) != old_b_nodes.end())
				{
					const auto &node = bs.bases[j].global().front().node;

					if(fabs(node(0)-1)<1e-8 || fabs(node(0))<1e-8)
						boundary_nodes.push_back(bs.bases[j].global().front().index);
				}
			}
		}
	}
}