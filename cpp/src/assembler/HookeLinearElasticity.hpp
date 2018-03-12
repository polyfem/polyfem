#ifndef HOOKE_LINEAR_ELASTICITY_HPP
#define HOOKE_LINEAR_ELASTICITY_HPP

#include "Common.hpp"

#include "ElementAssemblyValues.hpp"
#include "ElementBases.hpp"

#include "AutodiffTypes.hpp"

#include <Eigen/Dense>
#include <array>

namespace poly_fem
{
	class HookeLinearElasticity
	{
	public:
		HookeLinearElasticity();

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;

		inline int size() const { return size_; }

		void set_size(const int size);
		void set_lambda_mu(const double lambda, const double mu);

		void set_stiffness_tensor(int i, int j, const double val);

		void set_parameters(const json &params);
	private:
		int size_ = 2;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 21, 1> stifness_tensor_;

		double stifness_tensor(int i, int j) const;

		template<int D>
		double stress(const std::array<double, D> &strain, const int j) const;
	};
}

#endif //HOOKE_LINEAR_ELASTICITY_HPP
