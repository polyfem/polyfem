#include "MixedAssemblerForm.hpp"

#include <polyfem/utils/Timer.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace polyfem::solver
{
	MixedAssemblerForm::MixedAssemblerForm(
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
		const bool is_volume)
		: n_phi_bases_(n_phi_bases),
		  n_psi_bases_(n_psi_bases),
		  phi_bases_(phi_bases),
		  psi_bases_(psi_bases),
		  geom_bases_(geom_bases),
		  assembler_(assembler),
		  phi_cache_(phi_cache),
		  psi_cache_(psi_cache),
		  t_(t),
		  dt_(dt),
		  is_volume_(is_volume)
	{
		assert(n_phi_bases_ > 0);
		assert(n_psi_bases_ > 0);
		assert(phi_bases_.size() == psi_bases_.size());
		assert(phi_bases_.size() == geom_bases_.size());
		mat_cache_ = std::make_unique<utils::SparseMatrixCache>();
	}

	void MixedAssemblerForm::set_row_weights(const double phi_weight, const double psi_weight)
	{
		assert(std::isfinite(phi_weight));
		assert(std::isfinite(psi_weight));
		phi_row_weight_ = phi_weight;
		psi_row_weight_ = psi_weight;
	}

	double MixedAssemblerForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return assembler_.assemble_energy(
			is_volume_,
			n_psi_bases_, n_phi_bases_,
			psi_bases_, phi_bases_, geom_bases_,
			psi_cache_, phi_cache_,
			t_, dt_, x, x_prev_,
			split_solution());
	}

	Eigen::VectorXd MixedAssemblerForm::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd out = assembler_.assemble_energy_per_element(
			is_volume_,
			n_psi_bases_, n_phi_bases_,
			psi_bases_, phi_bases_, geom_bases_,
			psi_cache_, phi_cache_,
			t_, dt_, x, x_prev_,
			split_solution());
		assert(std::abs(out.sum() - value_unweighted(x)) < std::max(1e-10 * std::abs(out.sum()), 1e-10));
		return out;
	}

	void MixedAssemblerForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_gradient(
			is_volume_,
			n_psi_bases_, n_phi_bases_,
			psi_bases_, phi_bases_, geom_bases_,
			psi_cache_, phi_cache_,
			t_, dt_, x, x_prev_,
			split_solution(),
			grad);
		gradv = grad;
		gradv.head(phi_ndof()) *= phi_row_weight_;
		gradv.tail(psi_ndof()) *= psi_row_weight_;
	}

	void MixedAssemblerForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("mixed assembler hessian");

		hessian.resize(x.size(), x.size());
		// The autodiff mixed Hessian can change its exact zero pattern between
		// Newton iterations, so the per-element sparse pattern cache is unsafe.
		mat_cache_ = std::make_unique<utils::SparseMatrixCache>();
		assembler_.assemble_hessian(
			is_volume_,
			n_psi_bases_, n_phi_bases_,
			project_to_psd_,
			psi_bases_, phi_bases_, geom_bases_,
			psi_cache_, phi_cache_,
			t_, dt_, x, x_prev_,
			split_solution(),
			*mat_cache_, hessian);

		for (int k = 0; k < hessian.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
			{
				it.valueRef() *= it.row() < phi_ndof() ? phi_row_weight_ : psi_row_weight_;
			}
		}
	}

	int MixedAssemblerForm::phi_ndof() const
	{
		return n_phi_bases_ * assembler_.size();
	}

	int MixedAssemblerForm::psi_ndof() const
	{
		return n_psi_bases_;
	}

	assembler::MixedNLAssembler::SolutionSplitter MixedAssemblerForm::split_solution() const
	{
		return [this](const Eigen::MatrixXd &x, Eigen::MatrixXd &x_phi, Eigen::MatrixXd &x_psi) {
			assert(x.rows() == phi_ndof() + psi_ndof());
			assert(x.cols() == 1);
			x_phi = x.topRows(phi_ndof());
			x_psi = x.middleRows(phi_ndof(), psi_ndof());
		};
	}
} // namespace polyfem::solver
