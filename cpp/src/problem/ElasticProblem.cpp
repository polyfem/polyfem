#include "ElasticProblem.hpp"
#include "State.hpp"

#include <iostream>

namespace poly_fem
{
	ElasticProblem::ElasticProblem(const std::string &name)
	: Problem(name)
	{
		// boundary_ids_ = {1, 3, 5, 6};
		boundary_ids_ = {1, 2, 3, 4, 5, 6};
	}

	void ElasticProblem::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());
		val.col(1).setConstant(0.5);
		// val = Eigen::MatrixXd::Constant(pts.rows(), mesh.dimension(), 0.5);
	}

	void ElasticProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		// for(long i = 0; i < pts.rows(); ++i)
		// {
		// 	if(mesh.get_boundary_id(global_ids(i))== 1)
		// 		val(i, 0)=-0.25;
		// 	else if(mesh.get_boundary_id(global_ids(i))== 3)
		// 		val(i, 0)=0.25;
		// 	if(mesh.get_boundary_id(global_ids(i))== 5)
		// 		val(i, 1)=-0.025;
		// 	else if(mesh.get_boundary_id(global_ids(i))== 6)
		// 		val(i, 1)=0.025;
		// }

		for(long i = 0; i < pts.rows(); ++i)
		{
			if(mesh.get_boundary_id(global_ids(i)) > 0)
				val.row(i).setZero();
		}
	}






	ElasticProblemExact::ElasticProblemExact(const std::string &name)
	: Problem(name)
	{ }

	void ElasticProblemExact::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		double lambda = State::state().lambda;
		double mu = State::state().mu;

		val.resize(pts.rows(), mesh.dimension());
		if(pts.cols() == 2)
		{
			val.col(0) = 2./5.*mu + lambda*(1./5+1./5.*y) + 4./5.*mu*y;
			val.col(1) = 2*mu*(9./5.*x*x + 1./20.) + 2./5.*mu*x + lambda*(1./10.+1./5.*x);
		}
		else
		{
			auto &z = pts.col(2).array();

			val.col(0) = 2./5.*mu + lambda * (1./5. + 3./10.*y) + 9./10.*mu*y;
			val.col(1) = 2*mu* (9./5. * x * x + 1./20.) + 2./5. * mu * x + lambda * (1./10. + 3./10. * x + 2./5. * y * z) + 2 * mu * (1./5. * y * z + 1./20. * x - 3./10.*z);
			val.col(2) = 1./5. * mu * z * z + 2./5. * mu * y * y + 1./5. * lambda * y * y;
		}
	}

	void ElasticProblemExact::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}

	void ElasticProblemExact::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		val.resize(pts.rows(), pts.cols());

		if(pts.cols() == 2)
		{
			val.col(0) = (y*y*y + x*x + x*y)/10.;
			val.col(1) = (3*x*x*x*x + x*y*y + x)/10.;
		}
		else
		{
			auto &z = pts.col(2).array();

			val.col(0) = (x*y + x*x + y*y*y + 6*z)/10.;
			val.col(1) = (z*x - z*z*z + x*y*y + 3*x*x*x*x)/10.;
			val.col(2) = (x*y*z + z*z*y*y - 2*x)/10.;
		}
	}
}
