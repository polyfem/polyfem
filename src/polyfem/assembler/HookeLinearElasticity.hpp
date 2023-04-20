#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// local assembler for HookeLinearElasticity C : (F+F^T)/2, see linear elasticity
namespace polyfem::assembler
{
	class HookeLinearElasticity : public LinearAssembler, NLAssembler, ElasticityAssembler
	{
	public:
		using LinearAssembler::assemble;
		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		HookeLinearElasticity();

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		// compute elastic energy
		double compute_energy(const NonLinearAssemblerData &data) const override;
		// neccessary for mixing linear model with non-linear collision response
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
		// compute gradient of elastic energy, as assembler
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;

		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		void set_size(const int size) override;

		// sets the elasticty tensor
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		const ElasticityTensor &elasticity_tensor() const { return elasticity_tensor_; }

		virtual bool is_linear() const override { return true; }
		std::string name() const override { return "HookeLinearElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		ElasticityTensor elasticity_tensor_;

		// aux function that computes energy
		// double compute_energy is the same with T=double
		// assemble_gradient is the same with T=DScalar1 and return .getGradient()
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
	};
} // namespace polyfem::assembler
