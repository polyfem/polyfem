#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	class NeoHookeanElasticity : public NLAssembler, public ElasticityAssembler
	{
	public:
		NeoHookeanElasticity();

		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		// energy, gradient, and hessian used in newton method
		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		// rhs for fabbricated solution, compute with automatic sympy code
		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		void compute_stiffness_value(const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const override;

		void compute_stress_grad_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;
		void compute_stress_grad_multiply_stress(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;
		void compute_stress_grad_multiply_vect(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &vect, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;
		void compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const override;

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		void set_params(const LameParameters &params) { params_ = params; }
		const LameParameters &lame_params() const { return params_; }

		void update_lame_params(const Eigen::MatrixXd &lambdas, const Eigen::MatrixXd &mus) override
		{
			params_.lambda_mat_ = lambdas;
			params_.mu_mat_ = mus;
		}

		std::string name() const override { return "NeoHookean"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		LameParameters params_;

		// utility function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
		template <int n_basis, int dim>
		void compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const;
		template <int n_basis, int dim>
		void compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::VectorXd &G_flattened) const;
	};
} // namespace polyfem::assembler
