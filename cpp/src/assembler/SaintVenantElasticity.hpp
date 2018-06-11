#pragma once

#include "Common.hpp"
#include "ElasticityUtils.hpp"

#include "ElementAssemblyValues.hpp"
#include "ElementBases.hpp"
#include "AutodiffTypes.hpp"
#include "Types.hpp"

#include <Eigen/Dense>
#include <array>

namespace poly_fem
{
	class SaintVenantElasticity
	{
	public:
		SaintVenantElasticity();

		Eigen::VectorXd	assemble(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
		Eigen::MatrixXd	assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
		double 			compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;


		inline int size() const { return size_; }
		void set_size(const int size);

		void set_stiffness_tensor(int i, int j, const double val);
		double stifness_tensor(int i, int j) const;

		void compute_von_mises_stresses(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		void set_parameters(const json &params);
	private:
		int size_ = 2;

		ElasticityTensor elasticity_tensor_;

		template <typename T, unsigned long N>
		T stress(const std::array<T, N> &strain, const int j) const;

		template<typename T>
		T compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

		void assign_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
}

