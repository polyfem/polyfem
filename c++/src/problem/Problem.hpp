#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Problem
	{
	public:
		virtual ~Problem() {}
		virtual void rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const = 0;
		virtual void bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const = 0;

		virtual void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const { }
	};
}

#endif //PROBLEM_HPP

