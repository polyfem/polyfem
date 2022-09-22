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
		class NeoHookeanElasticity
		{
		public:
			NeoHookeanElasticity();

			//energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
			Eigen::VectorXd assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

			double compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

			//rhs for fabbricated solution, compute with automatic sympy code
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
			compute_rhs(const AutodiffHessianPt &pt) const;

			inline int size() const { return size_; }
			void set_size(const int size);

			//von mises and stress tensor
			void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
			void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

			double compute_energy(const Eigen::MatrixXd &grad_disp, const double lambda, const double mu) const;
			void compute_energy_gradient(const Eigen::MatrixXd &grad_disp, const double lambda, const double mu, Eigen::MatrixXd &grad) const;
			void compute_stress_grad_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const;
			void compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const;

			//sets material params
			void add_multimaterial(const int index, const json &params);
			void set_params(const LameParameters &params) { params_ = params; }
			LameParameters &lame_params() { return params_; }
			const LameParameters &lame_params() const { return params_; }

		private:
			int size_ = -1;

			LameParameters params_;

			//utulity function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
			template <typename T>
			T compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
			template <int n_basis, int dim>
			void compute_energy_hessian_aux_fast(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da, Eigen::MatrixXd &H) const;
			template <int n_basis, int dim>
			void compute_energy_aux_gradient_fast(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da, Eigen::VectorXd &G_flattened) const;

			void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
		};
	} // namespace assembler
} // namespace polyfem
