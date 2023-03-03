#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// local assembler for HookeLinearElasticity C : (F+F^T)/2, see linear elasticity
namespace polyfem::assembler
{
	class HookeLinearElasticity : public LinearAssembler, ElasticityAssembler
	{
	public:
		using LinearAssembler::assemble;

		HookeLinearElasticity();

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;
		void set_size(const int size) override;

		// sets the elasticty tensor
		void add_multimaterial(const int index, const json &params) override;

		const ElasticityTensor &elasticity_tensor() const { return elasticity_tensor_; }

		std::string name() const override { return "Hooke"; }
		std::map<std::string, ParamFunc> parameters() const override;

	protected:
		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		ElasticityTensor elasticity_tensor_;
	};
} // namespace polyfem::assembler
