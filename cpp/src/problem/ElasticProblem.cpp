#include "ElasticProblem.hpp"
#include "State.hpp"

#include <iostream>

namespace poly_fem
{
	ElasticProblem::ElasticProblem()
	{
		problem_num_ = ProblemType::Elastic;

		boundary_ids_ = {1, 3, 5, 6};
	}

	void ElasticProblem::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());
	}

	void ElasticProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for(long i = 0; i < pts.rows(); ++i)
		{
			if(mesh.get_boundary_id(global_ids(i))== 1)
				val(i, 0)=-0.25;
			else if(mesh.get_boundary_id(global_ids(i))== 3)
				val(i, 0)=0.25;
			if(mesh.get_boundary_id(global_ids(i))== 5)
				val(i, 1)=-0.025;
			else if(mesh.get_boundary_id(global_ids(i))== 6)
				val(i, 1)=0.025;
		}
	}






	ElasticProblemExact::ElasticProblemExact()
	{
		problem_num_ = ProblemType::ElasticExact;
	}

	void ElasticProblemExact::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();
		
		double lambda = State::state().lambda;
		double mu = State::state().mu;

		val.resize(pts.rows(), mesh.dimension());
		val.col(0) = 2./5.*mu + lambda*(1./5+1./5.*y) + 4./5.*mu*y;
		val.col(1) = 2*mu*(9./5.*x*x + 1./20.) + 2./5.*mu*x + lambda*(1./10.+1./5.*x);
		
		// val.col(0).setConstant((2./5.)*mu+(1./5.)*lambda);
		// val.col(0).setZero();
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
		val.col(0) = (y*y*y + x*x + x*y)/10.;
		val.col(1) = (3*x*x*x*x + x*y*y + x)/10.;
		// val.col(0) = x*x/10.;
		// val.col(1) = y/10.;
	}
}
