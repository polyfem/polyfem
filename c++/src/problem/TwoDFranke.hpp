#ifndef TWO_D_FRANKE_HPP
#define TWO_D_FRANKE_HPP

#include "Problem.hpp"

#include <Eigen/Dense>

namespace poly_fem
{
	class TwoDFranke : public Problem
	{
	public:
		virtual ~TwoDFranke() {}

		void rhs(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	};
}

#endif //TWO_D_FRANKE_HPP
