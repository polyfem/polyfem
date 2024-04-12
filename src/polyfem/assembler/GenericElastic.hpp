#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	template <typename T>
	using DefGradMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;

	template <typename Derived>
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

		void assign_stress_tensor(const OutputData &data,
								  const int all_size,
								  const ElasticityTensorType &type,
								  Eigen::MatrixXd &all,
								  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

		void compute_stress_grad_multiply_mat(const OptAssemblerData &data,
											  const Eigen::MatrixXd &mat,
											  Eigen::MatrixXd &stress,
											  Eigen::MatrixXd &result) const override;

		void compute_stress_grad_multiply_stress(const OptAssemblerData &data,
												 Eigen::MatrixXd &stress,
												 Eigen::MatrixXd &result) const override;

		void compute_stress_grad_multiply_vect(const OptAssemblerData &data,
											   const Eigen::MatrixXd &vect,
											   Eigen::MatrixXd &stress,
											   Eigen::MatrixXd &result) const override;

		/// @brief Returns this as a reference to derived class
		Derived &derived() { return static_cast<Derived &>(*this); }
		/// @brief Returns this as a const reference to derived class
		const Derived &derived() const { return static_cast<const Derived &>(*this); }

		// sets material params
		virtual void add_multimaterial(const int index, const json &params, const Units &units) override = 0;

		bool allow_inversion() const override { return true; }

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

				const T val = derived().elastic_energy(data.vals.val.row(p), data.t, data.vals.element_id, def_grad);

				energy += val * data.da(p);
			}
			return energy;
		}
	};
} // namespace polyfem::assembler