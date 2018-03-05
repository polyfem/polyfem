#include "LinearProblem.hpp"

#include <iostream>

namespace poly_fem
{
	LinearProblem::LinearProblem(const std::string &name)
	: Problem(name)
	{ }

	void LinearProblem::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);
	}

	void LinearProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}

	void LinearProblem::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = pts.col(0).array();
	}
}
