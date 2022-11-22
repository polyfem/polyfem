#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>

#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <functional>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	template <typename ElasticFormulation>
	class GenericElastic
	{
	public:
		GenericElastic();

		// energy, gradient, and hessian used in newton method
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const;
		Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const;

		double compute_energy(const NonLinearAssemblerData &data) const;

		// rhs for fabbricated solution, compute with automatic sympy code
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		inline int size() const { return size_; }
		void set_size(const int size);

		// von mises and stress tensor
		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		// sets material params
		void add_multimaterial(const int index, const json &params);

		const ElasticFormulation &formulation() const { return formulation_; }

	private:
		int size_ = -1;
		ElasticFormulation formulation_;

		// utulity function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const
		{
			typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
			typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

			AutoDiffVect local_disp;
			get_local_disp(data, size(), local_disp);

			AutoDiffGradMat disp_grad(size(), size());

			T energy = T(0.0);

			const int n_pts = data.da.size();
			for (long p = 0; p < n_pts; ++p)
			{
				compute_disp_grad_at_quad(data, local_disp, p, size(), disp_grad);

				const T val = formulation_.elastic_energy(size(), data.vals.val.row(p), data.vals.element_id, disp_grad);

				energy += val * data.da(p);
			}
			return energy;
		}

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
} // namespace polyfem::assembler