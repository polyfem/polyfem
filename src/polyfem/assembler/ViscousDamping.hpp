#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <array>

//non linear NeoHookean material model
namespace polyfem
{
	namespace assembler
	{
		class ViscousDamping
		{
		public:
			ViscousDamping() { damping_params_[0] = 0; damping_params_[1] = 0; }
			virtual ~ViscousDamping() = default;

			//energy, gradient, and hessian used in newton method
			virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const;
			virtual Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const;
			// Eigen::VectorXd assemble_prev_grad(const NonLinearAssemblerData &data) const;
			// Eigen::MatrixXd assemble_stress_prev_grad(const NonLinearAssemblerData &data) const;

			double compute_energy(const NonLinearAssemblerData &data) const;

			void compute_stress_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const;
			void compute_stress_prev_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &result) const;
			static void compute_dstress_dpsi_dphi(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &dstress_dpsi, Eigen::MatrixXd &dstress_dphi);

			//sets material params
			void add_multimaterial(const int index, const json &params);
			void set_params(const double psi, const double phi) { damping_params_[0] = psi; damping_params_[1] = phi; }
			double get_psi() const { return damping_params_[0]; }
			double get_phi() const { return damping_params_[1]; }

			inline int size() const { return size_; }
			void set_size(const int size);

			bool is_valid() const { return (damping_params_[0] > 0) || (damping_params_[1] > 0); }

			const DampingParameters &damping_params() const { return damping_params_; }

		protected:
			// material parameters controlling shear and bulk damping
			// double psi_ = 0, phi_ = 0;
			DampingParameters damping_params_;

			int size_ = -1;

			void compute_stress_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& dFdt, Eigen::MatrixXd& dRdF, Eigen::MatrixXd& dRdFdot) const;
			void compute_stress_grad_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& dFdt, Eigen::MatrixXd& d2RdF2, Eigen::MatrixXd& d2RdFdFdot, Eigen::MatrixXd& d2RdFdot2) const;
		};

		class ViscousDampingPrev: public ViscousDamping
		{
		public:
			ViscousDampingPrev() = default;

			//energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
			Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const override;
		};
	}
} // namespace polyfem
