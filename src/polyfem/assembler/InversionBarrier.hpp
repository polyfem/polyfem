#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>

// Standalone inversion barrier, split out of the former ModifiedNeoHookeanElasticity
// (ported from the original positive-jacobian branch, commit c59afca4, later reverted
// in 342cb860) so it can be combined with any elasticity model via SumModel/MaterialSum.
namespace polyfem::assembler
{
	class InversionBarrier : public ElasticityNLAssembler
	{
	public:
		InversionBarrier();

		using ElasticityNLAssembler::assemble_energy;
		using ElasticityNLAssembler::assemble_gradient;
		using ElasticityNLAssembler::assemble_hessian;

		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::string name() const override { return "InversionBarrier"; }
		bool allow_inversion() const override { return false; }
		std::map<std::string, ParamFunc> parameters() const override;

		void assign_stress_tensor(const OutputData &data,
								  const int all_size,
								  const ElasticityTensorType &type,
								  Eigen::MatrixXd &all,
								  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		LameParameters params_;
		// Activation threshold for the barrier: zero for J >= JBarrierThreshold, diverges to +inf as J -> 0.
		GenericMatParam JBarrierThreshold_;
	};
} // namespace polyfem::assembler
