#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::assembler
{
	// Similar to HookeLinear but with non-linear stress strain: C:½(F+Fᵀ+FᵀF)
	class SaintVenantElasticity : public NLAssembler, ElasticityAssembler
	{
	public:
		SaintVenantElasticity();

		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		void set_size(const int size) override;

		void set_stiffness_tensor(int i, int j, const double val);
		double stifness_tensor(int i, int j) const;

		void add_multimaterial(const int index, const json &params, const Units &units) override;

		std::string name() const override { return "SaintVenant"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		ElasticityTensor elasticity_tensor_;

		template <typename T, unsigned long N>
		T stress(const std::array<T, N> &strain, const int j) const;

		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
	};
} // namespace polyfem::assembler
