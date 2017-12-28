#ifndef BIHARMONIC_HPP
#define BIHARMONIC_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Biharmonic
	{
	public:
		Biharmonic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs);

		void basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;
		void grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;
	private:
		void compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs);

		Eigen::MatrixXd centers_;
		Eigen::MatrixXd weights_;

	};
}

#endif
