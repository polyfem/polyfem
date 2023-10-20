#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	template <typename T>
	using AutoDiffGradMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;
	template <typename T>
	using AutoDiffVect = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	class MooneyRivlin3ParamSymbolic : public NLAssembler, public ElasticityAssembler
	{
	public:
		MooneyRivlin3ParamSymbolic();

		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		const GenericMatParam &c1() const { return c1_; }
		const GenericMatParam &c2() const { return c2_; }
		const GenericMatParam &c3() const { return c3_; }
		const GenericMatParam &d1() const { return d1_; }

		// energy, gradient, and hessian used in newton method
		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		// sets material params
		void add_multimaterial(const int index, const json &params) override;

		std::string name() const override { return "MooneyRivlin3ParamSymbolic"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

		void compute_stress_grad_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;
		void compute_stress_grad_multiply_stress(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;
		void compute_stress_grad_multiply_vect(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &vect, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const override;

	private:
		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam c3_;
		GenericMatParam d1_;

		// utility function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
		template <int n_basis, int dim>
		void compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const;
		template <int n_basis, int dim>
		void compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::VectorXd &G_flattened) const;

		template <typename T>
		T elastic_energy(const RowVectorNd &p, const int el_id, const AutoDiffGradMat<T> &def_grad) const;
	};
} // namespace polyfem::assembler
