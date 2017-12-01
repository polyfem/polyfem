#ifndef QUADRATIC_HPP
#define QUADRATIC_HPP

#include "Problem.hpp"

#include <Eigen/Dense>

namespace poly_fem
{
	class Quadratic : public Problem
	{
	public:
		virtual ~Quadratic() {}

		void rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	};
}

#endif //QUADRATIC_HPP
