#pragma once

#include "Form.hpp"

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/NavierStokes.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/MatrixCache.hpp>

#include <memory>

namespace polyfem::solver
{
	/// Residual form for the viscous and convective velocity terms of Navier--Stokes.
	class NavierStokesForm : public Form
	{
	public:
		NavierStokesForm(
			int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			std::shared_ptr<assembler::Assembler> stokes_assembler,
			assembler::NavierStokesVelocity &navier_stokes_assembler,
			const assembler::AssemblyValsCache &ass_vals_cache,
			double t,
			bool is_volume);

		std::string name() const override { return "navier-stokes"; }

		void update_quantities(double t, const Eigen::VectorXd &x) override;

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override;

	private:
		void assemble_stokes(double t);
		void assemble_convection(const Eigen::VectorXd &x, bool picard, StiffnessMatrix &jacobian) const;

		int n_bases_;
		const std::vector<basis::ElementBases> &bases_;
		const std::vector<basis::ElementBases> &geom_bases_;
		std::shared_ptr<assembler::Assembler> stokes_assembler_;
		assembler::NavierStokesVelocity &navier_stokes_assembler_;
		const assembler::AssemblyValsCache &ass_vals_cache_;
		double t_;
		bool is_volume_;
		StiffnessMatrix stokes_stiffness_;
	};

	/// Constant saddle-point coupling between velocity and pressure.
	class MixedLinearForm : public Form
	{
	public:
		MixedLinearForm(
			int n_velocity_bases,
			int n_pressure_bases,
			const std::vector<basis::ElementBases> &velocity_bases,
			const std::vector<basis::ElementBases> &pressure_bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::MixedAssembler &assembler,
			const assembler::AssemblyValsCache &velocity_cache,
			const assembler::AssemblyValsCache &pressure_cache,
			double t,
			bool is_volume);

		std::string name() const override { return "mixed-linear"; }
		void set_row_weights(double velocity_weight, double pressure_weight);

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override;

	private:
		int velocity_ndof_;
		int pressure_ndof_;
		StiffnessMatrix coupling_;
		double velocity_weight_ = 1;
		double pressure_weight_ = 1;
	};

	/// Mean-zero pressure equation with a scalar Lagrange multiplier.
	class AveragePressureForm : public Form
	{
	public:
		explicit AveragePressureForm(int n_pressure_bases);

		std::string name() const override { return "average-pressure"; }

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override;

	private:
		StiffnessMatrix jacobian_;
	};
} // namespace polyfem::solver
