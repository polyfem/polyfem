#include "Problem.hpp"

#include <iostream>

namespace poly_fem
{
	bool Problem::has_exact_sol() const
	{
		return problem_num_ != 3;
	}

	void Problem::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		switch(problem_num_)
		{
			case 0: val = Eigen::MatrixXd::Zero(pts.rows(), 1); return;
			// case 1: val = Eigen::MatrixXd::Zero(pts.rows(), 1); return;
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

			case 3: val = Eigen::MatrixXd::Zero(pts.rows(), mesh.is_volume()? 3:2); return;

			case 4: {
				assert(mesh.is_volume());
				auto &z = pts.col(2).array();
				val =  -4 * x * y * (1 - y) *(1 - y) * z * (1 - z) + 2 * (1 - x) * y * (1 - y) * (1 - y) * z * (1 - z) - 4 * (1 - x) * x * x * (1 - y) * z * (1 - z) + 2 * (1 - x) * x * x * y * z * (1 - z) - 2 * (1 - x) * x * y * (1 - y) * (1 - y);

				// val = -4 * x * y * (1 - y) * (1 - y) + 2 * (1 - x) * y * (1 - y) *(1 - y) - 4 * (1 - x) * x * x * (1 - y) + 2 * (1 - x) * x * x * y;
				return;
			}

			case 5:
			{
				assert(mesh.is_volume());
				auto &z = pts.col(2).array();

				val =
				(1181472075 * x * x + 1181472075 * y * y + 1181472075 * z * z - 525098700 * x - 525098700 * y - 525098700 * z + 87516450) / 960400. *
				exp(-81./4. * x * x + 9 * x - 3 - 81./4. * y * y + 9 * y - 81./4. * z * z + 9 * z) +

				(787648050 * x * x + 3150592200 * y * y - 1225230300 * x - 2800526400 * y + 1040473350) / 960400. *
				exp(-81./4. * x * x + 63./2. * x - 83./4. - 81./2. * y * y + 36 * y) +

				(7873200 * x * x + 1749600 * x - 1117314) / 960400. *
				exp(-81./49. * x * x - 18./49. * x - 54./245. - 9./10. * y - 9./10. * z) -

				26244./ 5. * (x * x + 4 * y * y - 8./9. * x - 16./3. * y + 317./162.) *
				exp(-81 * x * x - 162 * y * y + 72 * x + 216 * y - 90);


				return;
			}


			default: assert(false);
		}
	}

	void Problem::bc(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		if(problem_num_ == 3)
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.is_volume()? 3 : 2);

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
		else
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
			// case 1: val = x * y; return;
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


			case 4:
			{
				auto &z = pts.col(2).array();

				val = (1 - x)  * x * x * y * (1-y) *(1-y) * z * (1 - z);

				return;
			}

			case 5:
			{
				auto &z = pts.col(2).array();

				auto cx2 = (9*x-2) * (9*x-2);
				auto cy2 = (9*y-2) * (9*y-2);
				auto cz2 = (9*z-2) * (9*z-2);

				auto cx1 = (9*x+1) * (9*x+1);
				auto cx7 = (9*x-7) * (9*x-7);

				auto cy3 = (9*y-3) * (9*y-3);
				auto cx4 = (9*x-4) * (9*x-4);
				auto cy7 = (9*y-7) * (9*y-7);

				auto cz5 = (9*y-5) * (9*y-5);

				val =
				3./4. * exp( -1./4.*cx2 - 1./4.*cy2 - 1./4.*cz2) +
				3./4. * exp(-1./49. * cx1 - 9./10.*y - 1./10. -  9./10.*z - 1./10.) +
				1./2. * exp(-1./4. * cx7 - 1./4. * cy3 - 1./4. * cz5) -
				1./5. * exp(- cx4 - cy7 - cz5);

				return;
			}

			default: assert(false);
		}
	}

	void Problem::remove_neumann_nodes(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes)
	{
		if(problem_num_ != 3)
			return;

		std::vector< LocalBoundary > new_local_boundary;
		for(auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			LocalBoundary new_lb(lb.element_id(), lb.type());
			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_g_id = lb.global_primitive_id(i);
				const int tag = mesh.get_boundary_id(primitive_g_id);

				if(tag == 1 || tag == 3)
					new_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
			}

			if(!new_lb.empty())
				new_local_boundary.emplace_back(new_lb);
		}
		std::swap(local_boundary, new_local_boundary);

		boundary_nodes.clear();

		for(auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = bases[lb.element_id()];
			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = b.local_nodes_for_primitive(primitive_global_id, mesh);

				for(long n = 0; n < nodes.size(); ++n){
					auto &bs = b.bases[nodes(n)];
					for(size_t g = 0; g < bs.global().size(); ++g)
						boundary_nodes.push_back(bs.global()[g].index);
				}
			}
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.resize(std::distance(boundary_nodes.begin(), it));
	}
}
