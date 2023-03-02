#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::assembler
{
	// Similar to HookeLinear but with non-linear stress strain: C:½(F+Fᵀ+FᵀF)
	class SaintVenantElasticity : public NLAssembler
	{
	public:
		SaintVenantElasticity();

		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_grad;
		using NLAssembler::assemble_hessian;

		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size) override;

		void set_stiffness_tensor(int i, int j, const double val);
		double stifness_tensor(int i, int j) const;

		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const ElasticityTensorType &type, Eigen::MatrixXd &tensor) const;

		void add_multimaterial(const int index, const json &params);

	private:
		ElasticityTensor elasticity_tensor_;

		template <typename T, unsigned long N>
		T stress(const std::array<T, N> &strain, const int j) const;

		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
} // namespace polyfem::assembler
