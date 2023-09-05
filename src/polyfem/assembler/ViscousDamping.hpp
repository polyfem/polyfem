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
			using NLAssembler::assemble_energy;
			using NLAssembler::assemble_gradient;
			using NLAssembler::assemble_hessian;
			using NLAssembler::compute_energy;

			std::string name() const override { return "ViscousDamping"; }
			std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

			ViscousDamping() = default;

			// energy, gradient, and hessian used in newton method
			double compute_energy(const NonLinearAssemblerData &data) const override;
			virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
			virtual Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;

			// sets material params
			void add_multimaterial(const int index, const json &params, const Units &units) override;
			void set_params(const double psi, const double phi)
			{
				damping_params_[0] = psi;
				damping_params_[1] = phi;
			}

			void compute_stress_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;
			void compute_stress_prev_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &result) const override;
			static void compute_dstress_dpsi_dphi(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &dstress_dpsi, Eigen::MatrixXd &dstress_dphi);

			double get_psi() const { return damping_params_[0]; }
			double get_phi() const { return damping_params_[1]; }

			bool is_valid() const { return (damping_params_[0] > 0) && (damping_params_[1] > 0); }

			const DampingParameters &damping_params() const { return damping_params_; }

		protected:
			// material parameters controlling shear and bulk damping
			// double psi_ = 0, phi_ = 0;
			DampingParameters damping_params_;

			void compute_stress_aux(const Eigen::MatrixXd &F, const Eigen::MatrixXd &dFdt, Eigen::MatrixXd &dRdF, Eigen::MatrixXd &dRdFdot) const;
			void compute_stress_grad_aux(const Eigen::MatrixXd &F, const Eigen::MatrixXd &dFdt, Eigen::MatrixXd &d2RdF2, Eigen::MatrixXd &d2RdFdFdot, Eigen::MatrixXd &d2RdFdot2) const;
		};

		class ViscousDampingPrev: public ViscousDamping
		{
		public:
			using NLAssembler::assemble_energy;
			using NLAssembler::assemble_gradient;
			using NLAssembler::assemble_hessian;
			using ViscousDamping::compute_energy;

			//energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
			Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		};
	}
} // namespace polyfem
