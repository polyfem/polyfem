#include "ElasticProblem.hpp"

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
				val(i, 0)=-0.025;
			else if(mesh.get_boundary_id(global_ids(i))== 3)
				val(i, 0)=0.025;
			if(mesh.get_boundary_id(global_ids(i))== 5)
				val(i, 1)=-0.025;
			else if(mesh.get_boundary_id(global_ids(i))== 6)
				val(i, 1)=0.025;
		}
	}
}
