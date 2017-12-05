#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Problem
	{
	public:
		void rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;
		void bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;

		inline void set_problem_num(const int num) { problem_num_ = num; }
		inline int problem_num() const { return problem_num_; }
	private:
		int problem_num_;
	};
}

#endif //PROBLEM_HPP

