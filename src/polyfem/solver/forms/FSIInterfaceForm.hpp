#pragma once

#include "Form.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::solver
{
	/// Residual-only coupling of the physical fluid/solid interface and of the
	/// computational ALE mesh to the solid boundary. The physical multiplier
	/// applies equal and opposite interface forces. The mesh multiplier is
	/// deliberately one-way: its reaction acts on D_m, but not on D_s.
	class FSIInterfaceForm : public Form
	{
	public:
		FSIInterfaceForm(
			int total_size,
			int velocity_offset,
			int mesh_displacement_offset,
			int solid_displacement_offset,
			int fluid_multiplier_offset,
			int mesh_multiplier_offset,
			StiffnessMatrix fluid_velocity_trace,
			StiffnessMatrix fluid_solid_trace,
			StiffnessMatrix mesh_trace,
			StiffnessMatrix mesh_solid_trace,
			const time_integrator::ImplicitTimeIntegrator &fluid_integrator,
			const time_integrator::ImplicitTimeIntegrator &solid_integrator);

		std::string name() const override { return "fsi-interface"; }

		int fluid_multiplier_size() const { return fluid_velocity_trace_.rows(); }
		int mesh_multiplier_size() const { return mesh_trace_.rows(); }
		const StiffnessMatrix &fluid_multiplier_mass() const { return fluid_multiplier_mass_; }
		const StiffnessMatrix &mesh_multiplier_mass() const { return mesh_multiplier_mass_; }
		Eigen::VectorXd physical_constraint(
			const Eigen::VectorXd &velocity, const Eigen::VectorXd &solid_velocity) const;
		Eigen::VectorXd mesh_constraint(
			const Eigen::VectorXd &mesh_displacement,
			const Eigen::VectorXd &solid_displacement) const;

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override;

	private:
		static StiffnessMatrix make_multiplier_mass(const StiffnessMatrix &trace);

		const int total_size_;
		const int velocity_offset_;
		const int mesh_displacement_offset_;
		const int solid_displacement_offset_;
		const int fluid_multiplier_offset_;
		const int mesh_multiplier_offset_;
		const StiffnessMatrix fluid_velocity_trace_;
		const StiffnessMatrix fluid_solid_trace_;
		const StiffnessMatrix mesh_trace_;
		const StiffnessMatrix mesh_solid_trace_;
		const StiffnessMatrix fluid_multiplier_mass_;
		const StiffnessMatrix mesh_multiplier_mass_;
		const time_integrator::ImplicitTimeIntegrator &fluid_integrator_;
		const time_integrator::ImplicitTimeIntegrator &solid_integrator_;
	};
} // namespace polyfem::solver
