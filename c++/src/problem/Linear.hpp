#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "Problem.hpp"

#include <Eigen/Dense>

namespace poly_fem
{
	class Linear : public Problem
	{
	public:
		virtual ~Linear() {}

		void rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	};
}

#endif //LINEAR_HPP
