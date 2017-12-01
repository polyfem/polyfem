#include "Linear.hpp"


namespace poly_fem
{
	void Linear::rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);
	}

	void Linear::bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}

	void Linear::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = pts.col(0);
	}
}