#pragma once

#include <polyfem/assembler/NeoHookeanElasticity.hpp>

// NeoHookean with an added inversion barrier, ported from the original
// positive-jacobian branch (commit c59afca4, later reverted in 342cb860).
namespace polyfem::assembler
{
	class CorrectedNeoHookeanElasticity : public NeoHookeanElasticity
	{
	public:
		using ElasticityNLAssembler::assemble_energy;
		using ElasticityNLAssembler::assemble_gradient;
		using ElasticityNLAssembler::assemble_hessian;

		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		std::string name() const override { return "CorrectedNeoHookean"; }

	private:
		// Barrier correction added on top of the base NeoHookean energy/gradient/hessian.
		double barrier_energy(const NonLinearAssemblerData &data) const;
		Eigen::VectorXd barrier_gradient(const NonLinearAssemblerData &data) const;
		Eigen::MatrixXd barrier_hessian(const NonLinearAssemblerData &data) const;
	};
} // namespace polyfem::assembler
