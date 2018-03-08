#include "QuadraticProblem.hpp"

#include <iostream>

namespace poly_fem
{
	QuadraticProblem::QuadraticProblem(const std::string &name)
	: Problem(name)
	{ }

	void QuadraticProblem::rhs(const std::string &formulation, const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = 10*Eigen::MatrixXd::Ones(pts.rows(), 1);
	}

	void QuadraticProblem::bc(const std::string &formulation, const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(formulation, pts, val);
	}


	void QuadraticProblem::exact(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		val = 5 * x * x;
	}
}
