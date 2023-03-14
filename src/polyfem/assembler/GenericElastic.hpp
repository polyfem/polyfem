#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/ElasticEnergyMacros.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	class GenericElastic : public NLAssembler, public ElasticityAssembler
	{
	public:
		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		GenericElastic();
		virtual ~GenericElastic() = default;

		// energy, gradient, and hessian used in newton method
		double compute_energy(const NonLinearAssemblerData &data) const override;
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;
		Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;

		// sets material params
		virtual void add_multimaterial(const int index, const json &params) override = 0;

		// This macro declares the virtual functions that compute the energy:
		// template <typename T>
		// virtual T elastic_energy(const RowVectorNd &p, const int el_id, const DefGradMatrix<T> &def_grad) const = 0;
		POLYFEM_DECLARE_VIRTUAL_ELASTIC_ENERGY

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		// utility function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const
		{
			typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
			typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

			AutoDiffVect local_disp;
			get_local_disp(data, size(), local_disp);

			AutoDiffGradMat def_grad(size(), size());

			T energy = T(0.0);

			const int n_pts = data.da.size();
			for (long p = 0; p < n_pts; ++p)
			{
				compute_disp_grad_at_quad(data, local_disp, p, size(), def_grad);

				// Id + grad d
				for (int d = 0; d < size(); ++d)
					def_grad(d, d) += T(1);

				const T val = elastic_energy(data.vals.val.row(p), data.vals.element_id, def_grad);

				energy += val * data.da(p);
			}
			return energy;
		}
	};
} // namespace polyfem::assembler