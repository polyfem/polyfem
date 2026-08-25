#pragma once

#include "Form.hpp"

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/MatrixCache.hpp>
#include <polyfem/utils/Types.hpp>

#include <memory>
#include <vector>

namespace polyfem::solver
{
	class MixedAssemblerForm : public Form
	{
	public:
		MixedAssemblerForm(
			const int n_phi_bases,
			const int n_psi_bases,
			const std::vector<basis::ElementBases> &phi_bases,
			const std::vector<basis::ElementBases> &psi_bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::MixedNLAssembler &assembler,
			assembler::AssemblyValsCache &phi_cache,
			assembler::AssemblyValsCache &psi_cache,
			const double t,
			const double dt,
			const bool is_volume);

		std::string name() const override { return "mixed-assembler"; }

		void set_row_weights(const double phi_weight, const double psi_weight);

		void update_quantities(const double t, const Eigen::VectorXd &x) override
		{
			t_ = t;
			x_prev_ = x;
		}

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		int phi_ndof() const;
		int psi_ndof() const;
		assembler::MixedNLAssembler::SolutionSplitter split_solution() const;

		const int n_phi_bases_;
		const int n_psi_bases_;
		const std::vector<basis::ElementBases> &phi_bases_;
		const std::vector<basis::ElementBases> &psi_bases_;
		const std::vector<basis::ElementBases> &geom_bases_;
		const assembler::MixedNLAssembler &assembler_;
		assembler::AssemblyValsCache &phi_cache_;
		assembler::AssemblyValsCache &psi_cache_;
		double t_;
		const double dt_;
		const bool is_volume_;
		Eigen::VectorXd x_prev_;
		double phi_row_weight_ = 1;
		double psi_row_weight_ = 1;
		mutable std::unique_ptr<utils::MatrixCache> mat_cache_;
	};
} // namespace polyfem::solver
