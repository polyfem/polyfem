#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

namespace polyfem::assembler
{
	class AMIPSEnergy : public NLAssembler, public ElasticityAssembler
	{
	public:
		AMIPSEnergy() {}

		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		// energy, gradient, and hessian used in newton method
		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override {}

		std::string name() const override { return "AMIPS"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		void assign_stress_tensor(const OutputData &data,
								  const int all_size,
								  const ElasticityTensorType &type,
								  Eigen::MatrixXd &all,
								  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

		bool allow_inversion() const override { return true; }

	private:
		// utility function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
		template <int n_basis, int dim>
		void compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::VectorXd &G_flattened) const;
		template <int n_basis, int dim>
		void compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const;
	};

	class AMIPSEnergyAutodiff : public GenericElastic<AMIPSEnergyAutodiff>
	{
	public:
		AMIPSEnergyAutodiff();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		std::string name() const override { return "AMIPSAutodiff"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			using std::pow;
			using MatrixNT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;

			MatrixNT F = MatrixNT::Zero(size(), size());
			for (int i = 0; i < size(); ++i)
			{
				for (int j = 0; j < size(); ++j)
				{
					for (int k = 0; k < size(); ++k)
					{
						F(i, j) += def_grad(i, k) * canonical_transformation_[el_id](k, j);
					}
				}
			}

			T J = polyfem::utils::determinant(F);
			if (J <= 0)
				J = T(std::nan(""));
			return (F.transpose() * F).trace() / pow(J, 2. / size());
		}

	private:
		std::vector<Eigen::MatrixXd> canonical_transformation_;
	};
} // namespace polyfem::assembler