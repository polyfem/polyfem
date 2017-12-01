#include "Quadratic.hpp"


namespace poly_fem
{
	void Quadratic::rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = 2*Eigen::MatrixXd::Ones(pts.rows(), 1);
	}

	void Quadratic::bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}

	void Quadratic::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		val = x * x;
	}
}