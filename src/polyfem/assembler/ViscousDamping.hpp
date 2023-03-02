#pragma once

#include <polyfem/assembler/Assembler.hpp>

// non linear NeoHookean material model
namespace polyfem
{
	namespace assembler
	{
		class ViscousDamping : public NLAssembler
		{
		public:
			ViscousDamping() = default;

			using NLAssembler::assemble_energy;
			using NLAssembler::assemble_grad;
			using NLAssembler::assemble_hessian;

			// energy, gradient, and hessian used in newton method
			double compute_energy(const NonLinearAssemblerData &data) const override;
			Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
			Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const override;

			// sets material params
			void add_multimaterial(const int index, const json &params);
			void set_params(const double psi, const double phi)
			{
				psi_ = psi;
				phi_ = phi;
			}
			double get_psi() const { return psi_; }
			double get_phi() const { return phi_; }

			bool is_valid() const { return (psi_ > 0) && (phi_ > 0); }

		private:
			// material parameters controlling shear and bulk damping
			double psi_ = 0, phi_ = 0;

			void compute_stress_aux(const Eigen::MatrixXd &F, const Eigen::MatrixXd &dFdt, Eigen::MatrixXd &dRdF, Eigen::MatrixXd &dRdFdot) const;
			void compute_stress_grad_aux(const Eigen::MatrixXd &F, const Eigen::MatrixXd &dFdt, Eigen::MatrixXd &d2RdF2, Eigen::MatrixXd &d2RdFdFdot, Eigen::MatrixXd &d2RdFdot2) const;
		};
	} // namespace assembler
} // namespace polyfem
