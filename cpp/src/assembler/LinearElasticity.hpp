#ifndef LINEAR_ELASTICITY_HPP
#define LINEAR_ELASTICITY_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class LinearElasticity
	{
	public:
		void assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, const Eigen::MatrixXd &da, Eigen::MatrixXd &res) const;

		inline int &size() { return size_; }
		inline int size() const { return size_; }

		inline double &mu() { return mu_; }
		inline double mu() const { return mu_; }

		inline double &lambda() { return lambda_; }
		inline double lambda() const { return lambda_; }
	private:
		int size_ = 2;
		double mu_ = 1;
		double lambda_ = 1;
	};
}

#endif //LINEAR_ELASTICITY_HPP
