#ifndef LINEAR_ELASTICITY_HPP
#define LINEAR_ELASTICITY_HPP


#include "ElementAssemblyValues.hpp"
#include "ElementBases.hpp"

#include <Eigen/Dense>


namespace poly_fem
{
	class LinearElasticity
	{
	public:
		// res is R^{m x dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>

		assemble(const ElementAssemblyValues &vals, const AssemblyValues &values_i, const AssemblyValues &values_j, const Eigen::VectorXd &da) const;

		void compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;

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
