#ifndef HOOKE_LINEAR_ELASTICITY_HPP
#define HOOKE_LINEAR_ELASTICITY_HPP

#include <polyfem/Common.hpp>
#include <polyfem/ElasticityUtils.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>

#include <polyfem/AutodiffTypes.hpp>

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
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void compute_von_mises_stresses(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		inline int size() const { return size_; }

		void set_size(const int size);

		void set_parameters(const json &params);
	private:
		int size_ = 2;

		ElasticityTensor elasticity_tensor_;

		void assign_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
}

#endif //HOOKE_LINEAR_ELASTICITY_HPP
