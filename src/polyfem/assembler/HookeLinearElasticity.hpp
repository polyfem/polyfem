#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// local assembler for HookeLinearElasticity C : (F+F^T)/2, see linear elasticity
namespace polyfem::assembler
{
	class HookeLinearElasticity : public TensorLinearAssembler, NLAssembler
	{
	public:
		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_grad;
		using NLAssembler::assemble_hessian;
		using TensorLinearAssembler::assemble;

		HookeLinearElasticity();

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		// compute elastic energy
		double compute_energy(const NonLinearAssemblerData &data) const override;
		// neccessary for mixing linear model with non-linear collision response
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
		// compute gradient of elastic energy, as assembler
		Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const override;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const ElasticityTensorType &type, Eigen::MatrixXd &tensor) const;

		void set_size(const int size) override;

		// sets the elasticty tensor
		void add_multimaterial(const int index, const json &params);

		const ElasticityTensor &elasticity_tensor() const { return elasticity_tensor_; }

	private:
		ElasticityTensor elasticity_tensor_;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;

		// aux function that computes energy
		// double compute_energy is the same with T=double
		// assemble_grad is the same with T=DScalar1 and return .getGradient()
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
	};
} // namespace polyfem::assembler
