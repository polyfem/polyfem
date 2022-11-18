#pragma once

#include "AssemblerData.hpp"
#include "MatParams.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <array>

// local assembler for HookeLinearElasticity C : (F+F^T)/2, see linear elasticity
namespace polyfem::assembler
{
	class HookeLinearElasticity
	{
	public:
		HookeLinearElasticity();

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		inline int size() const { return size_; }
		void set_size(const int size);

		// sets the elasticty tensor
		void add_multimaterial(const int index, const json &params);

		const ElasticityTensor &elasticity_tensor() const { return elasticity_tensor_; }

	private:
		int size_ = -1;

		ElasticityTensor elasticity_tensor_;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
} // namespace polyfem::assembler
