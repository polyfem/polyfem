#include "NavierStokesForm.hpp"

#include <polyfem/utils/Logger.hpp>

#include <cassert>
#include <vector>

namespace polyfem::solver
{
	NavierStokesForm::NavierStokesForm(
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		std::shared_ptr<assembler::Assembler> stokes_assembler,
		assembler::NavierStokesVelocity &navier_stokes_assembler,
		const assembler::AssemblyValsCache &ass_vals_cache,
		const double t,
		const bool is_volume)
		: n_bases_(n_bases),
		  bases_(bases),
		  geom_bases_(geom_bases),
		  stokes_assembler_(std::move(stokes_assembler)),
		  navier_stokes_assembler_(navier_stokes_assembler),
		  ass_vals_cache_(ass_vals_cache),
		  t_(t),
		  is_volume_(is_volume)
	{
		assemble_stokes(t_);
	}

	void NavierStokesForm::assemble_stokes(const double t)
	{
		stokes_assembler_->assemble(
			is_volume_, n_bases_, bases_, geom_bases_, ass_vals_cache_, t, stokes_stiffness_);
	}

	void NavierStokesForm::assemble_convection(
		const Eigen::VectorXd &x, const bool picard, StiffnessMatrix &jacobian) const
	{
		navier_stokes_assembler_.set_picard(picard);
		utils::SparseMatrixCache cache;
		navier_stokes_assembler_.assemble_hessian(
			is_volume_, n_bases_, /*project_to_psd=*/false,
			bases_, geom_bases_, ass_vals_cache_, t_, /*dt=*/0,
			x, Eigen::MatrixXd(), cache, jacobian);
	}

	double NavierStokesForm::value_unweighted(const Eigen::VectorXd &) const
	{
		log_and_throw_error("NavierStokesForm is a residual form and has no value()");
	}

	void NavierStokesForm::first_derivative_unweighted(
		const Eigen::VectorXd &x, Eigen::VectorXd &residual) const
	{
		StiffnessMatrix convection;
		assemble_convection(x, /*picard=*/true, convection);
		residual = (stokes_stiffness_ + convection) * x;
	}

	void NavierStokesForm::second_derivative_unweighted(
		const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const
	{
		StiffnessMatrix convection;
		// PolySolve's projected operator is the Oseen/Picard linearization.
		assemble_convection(x, /*picard=*/project_to_psd_, convection);
		jacobian = stokes_stiffness_ + convection;
	}

	void NavierStokesForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		t_ = t;
		assemble_stokes(t_);
	}

	MixedLinearForm::MixedLinearForm(
		const int n_velocity_bases,
		const int n_pressure_bases,
		const std::vector<basis::ElementBases> &velocity_bases,
		const std::vector<basis::ElementBases> &pressure_bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::MixedAssembler &assembler,
		const assembler::AssemblyValsCache &velocity_cache,
		const assembler::AssemblyValsCache &pressure_cache,
		const double t,
		const bool is_volume)
		: velocity_ndof_(n_velocity_bases * assembler.size()),
		  pressure_ndof_(n_pressure_bases)
	{
		StiffnessMatrix mixed;
		assembler.assemble(
			is_volume,
			n_pressure_bases, n_velocity_bases,
			pressure_bases, velocity_bases, geom_bases,
			pressure_cache, velocity_cache, t, mixed);

		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(2 * mixed.nonZeros());
		for (int k = 0; k < mixed.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(mixed, k); it; ++it)
			{
				entries.emplace_back(it.row(), velocity_ndof_ + it.col(), it.value());
				entries.emplace_back(velocity_ndof_ + it.col(), it.row(), it.value());
			}

		coupling_.resize(velocity_ndof_ + pressure_ndof_, velocity_ndof_ + pressure_ndof_);
		coupling_.setFromTriplets(entries.begin(), entries.end());
		coupling_.makeCompressed();
	}

	void MixedLinearForm::set_row_weights(
		const double velocity_weight, const double pressure_weight)
	{
		velocity_weight_ = velocity_weight;
		pressure_weight_ = pressure_weight;
	}

	double MixedLinearForm::value_unweighted(const Eigen::VectorXd &) const
	{
		log_and_throw_error("MixedLinearForm is a residual form and has no value()");
	}

	void MixedLinearForm::first_derivative_unweighted(
		const Eigen::VectorXd &x, Eigen::VectorXd &residual) const
	{
		residual = coupling_ * x;
		residual.head(velocity_ndof_) *= velocity_weight_;
		residual.tail(pressure_ndof_) *= pressure_weight_;
	}

	void MixedLinearForm::second_derivative_unweighted(
		const Eigen::VectorXd &, StiffnessMatrix &jacobian) const
	{
		jacobian = coupling_;
		for (int k = 0; k < jacobian.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(jacobian, k); it; ++it)
				it.valueRef() *= it.row() < velocity_ndof_ ? velocity_weight_ : pressure_weight_;
	}

	AveragePressureForm::AveragePressureForm(const int n_pressure_bases)
	{
		assert(n_pressure_bases > 0);
		const double value = 1.0 / n_pressure_bases;
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(2 * n_pressure_bases);
		for (int i = 0; i < n_pressure_bases; ++i)
		{
			entries.emplace_back(i, n_pressure_bases, value);
			entries.emplace_back(n_pressure_bases, i, value);
		}
		jacobian_.resize(n_pressure_bases + 1, n_pressure_bases + 1);
		jacobian_.setFromTriplets(entries.begin(), entries.end());
		jacobian_.makeCompressed();
	}

	double AveragePressureForm::value_unweighted(const Eigen::VectorXd &) const
	{
		log_and_throw_error("AveragePressureForm is a residual form and has no value()");
	}

	void AveragePressureForm::first_derivative_unweighted(
		const Eigen::VectorXd &x, Eigen::VectorXd &residual) const
	{
		residual = jacobian_ * x;
	}

	void AveragePressureForm::second_derivative_unweighted(
		const Eigen::VectorXd &, StiffnessMatrix &jacobian) const
	{
		jacobian = jacobian_;
	}
} // namespace polyfem::solver
