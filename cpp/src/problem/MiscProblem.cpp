#include "MiscProblem.hpp"

#include <iostream>

namespace poly_fem
{
	LinearProblem::LinearProblem(const std::string &name)
	: Problem(name)
	{ }

	void LinearProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
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



	QuadraticProblem::QuadraticProblem(const std::string &name)
	: Problem(name)
	{ }

	void QuadraticProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = 10*Eigen::MatrixXd::Ones(pts.rows(), 1);
	}

	void QuadraticProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}


	void QuadraticProblem::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		val = 5 * x * x;
	}



	ZeroBCProblem::ZeroBCProblem(const std::string &name)
	: Problem(name)
	{ }

	void ZeroBCProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		assert(pts.cols() == 3);

		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();
		auto &z = pts.col(2).array();

		val =  -4 * x * y * (1 - y) *(1 - y) * z * (1 - z) + 2 * (1 - x) * y * (1 - y) * (1 - y) * z * (1 - z) - 4 * (1 - x) * x * x * (1 - y) * z * (1 - z) + 2 * (1 - x) * x * x * y * z * (1 - z) - 2 * (1 - x) * x * y * (1 - y) * (1 - y);

		// val = -4 * x * y * (1 - y) * (1 - y) + 2 * (1 - x) * y * (1 - y) *(1 - y) - 4 * (1 - x) * x * x * (1 - y) + 2 * (1 - x) * x * x * y;
	}

	void ZeroBCProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}


	void ZeroBCProblem::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();
		auto &z = pts.col(2).array();

		val = (1 - x)  * x * x * y * (1-y) *(1-y) * z * (1 - z);
	}
}
