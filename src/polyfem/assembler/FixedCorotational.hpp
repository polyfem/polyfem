#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	class FixedCorotational : public NLAssembler, public ElasticityAssembler
	{
	public:
		FixedCorotational();

		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		// energy, gradient, and hessian used in newton method
		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		void compute_stress_grad_multiply_mat(const OptAssemblerData &data,
											  const Eigen::MatrixXd &mat,
											  Eigen::MatrixXd &stress,
											  Eigen::MatrixXd &result) const override;

		void compute_stress_grad_multiply_stress(const OptAssemblerData &data,
												 Eigen::MatrixXd &stress,
												 Eigen::MatrixXd &result) const override;

		void compute_dstress_dmu_dlambda(const OptAssemblerData &data,
										 Eigen::MatrixXd &dstress_dmu,
										 Eigen::MatrixXd &dstress_dlambda) const override;

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		void set_params(const LameParameters &params) { params_ = params; }
		const LameParameters &lame_params() const { return params_; }

		void update_lame_params(const Eigen::MatrixXd &lambdas, const Eigen::MatrixXd &mus) override
		{
			params_.lambda_mat_ = lambdas;
			params_.mu_mat_ = mus;
		}

		std::string name() const override { return "FixedCorotational"; }
		bool allow_inversion() const override { return true; }
		std::map<std::string, ParamFunc> parameters() const override;

		void assign_stress_tensor(const OutputData &data,
								  const int all_size,
								  const ElasticityTensorType &type,
								  Eigen::MatrixXd &all,
								  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		LameParameters params_;

		// utility function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <int dim>
		double compute_energy_aux(const NonLinearAssemblerData &data) const;
		template <int n_basis, int dim>
		void compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const;
		template <int n_basis, int dim>
		void compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::VectorXd &G_flattened) const;
	
		template <int dim>
		static double compute_energy_from_singular_values(const Eigen::Vector<double, dim> &sigmas, const double lambda, const double mu);
		template <int dim>
		static Eigen::Vector<double, dim> compute_stress_from_singular_values(const Eigen::Vector<double, dim> &sigmas, const double lambda, const double mu);
		template <int dim>
		static Eigen::Matrix<double, dim, dim> compute_stiffness_from_singular_values(const Eigen::Vector<double, dim> &sigmas, const double lambda, const double mu);

		template <int dim>
		static double compute_energy_from_def_grad(const Eigen::Matrix<double, dim, dim> &F, const double lambda, const double mu);
		template <int dim>
		static Eigen::Matrix<double, dim, dim> compute_stress_from_def_grad(const Eigen::Matrix<double, dim, dim> &F, const double lambda, const double mu);
		template <int dim>
		static Eigen::Matrix<double, dim*dim, dim*dim> compute_stiffness_from_def_grad(const Eigen::Matrix<double, dim, dim> &F, const double lambda, const double mu);
	};
} // namespace polyfem::assembler
