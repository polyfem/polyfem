#pragma once

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
			ViscousDamping() = default;

			//energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const;
			Eigen::VectorXd assemble_prev_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const;
			Eigen::MatrixXd assemble_stress_prev_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const;
			Eigen::VectorXd assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const;

			double compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const;

			void compute_stress_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const;
			void compute_stress_prev_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &result) const;
			void compute_dstress_dpsi_dphi(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &dstress_dpsi, Eigen::MatrixXd &dstress_dphi) const;

			//sets material params
			void add_multimaterial(const int index, const json &params);
			void set_params(const double psi, const double phi) { psi_ = psi; phi_ = phi; }
			double get_psi() const { return psi_; }
			double get_phi() const { return phi_; }

			inline int size() const { return size_; }
			void set_size(const int size);

			bool is_valid() const { return (psi_ > 0) && (phi_ > 0); }

		private:
			// material parameters controlling shear and bulk damping
			double psi_ = 0, phi_ = 0;

			int size_ = 2;

			void compute_stress_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& dFdt, Eigen::MatrixXd& dRdF, Eigen::MatrixXd& dRdFdot) const;
			void compute_stress_grad_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& dFdt, Eigen::MatrixXd& d2RdF2, Eigen::MatrixXd& d2RdFdFdot, Eigen::MatrixXd& d2RdFdot2) const;
		};
	}
} // namespace polyfem
