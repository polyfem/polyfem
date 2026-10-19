#pragma once

#include "Form.hpp"

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/NavierStokesFSI.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <array>
#include <memory>

namespace polyfem::solver
{
	/// Mean-zero pressure constraint on the current ALE domain. The form acts on
	/// the complete [velocity, pressure, mesh displacement, optional intervening
	/// blocks, multiplier] vector, but deliberately contributes no
	/// mesh-displacement residual.
	class NavierStokesFSIAveragePressureForm : public Form
	{
	public:
		NavierStokesFSIAveragePressureForm(
			int total_size,
			int n_velocity_bases,
			int n_pressure_bases,
			int n_mesh_displacement_bases,
			int multiplier_offset,
			int dim,
			const std::vector<basis::ElementBases> &pressure_bases,
			const std::vector<basis::ElementBases> &mesh_displacement_bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::AssemblyValsCache &pressure_cache,
			const assembler::AssemblyValsCache &mesh_displacement_cache,
			bool is_volume);

		std::string name() const override { return "navier-stokes-fsi-average-pressure"; }

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override;

	private:
		void compute_constraint(
			const Eigen::VectorXd &x,
			Eigen::VectorXd &weights,
			Eigen::MatrixXd &weight_derivative) const;

		const int total_size_;
		const int dim_;
		const int n_pressure_bases_;
		const int n_mesh_displacement_bases_;
		const int pressure_offset_;
		const int mesh_displacement_offset_;
		const int multiplier_offset_;
		const std::vector<basis::ElementBases> &pressure_bases_;
		const std::vector<basis::ElementBases> &mesh_displacement_bases_;
		const std::vector<basis::ElementBases> &geom_bases_;
		const assembler::AssemblyValsCache &pressure_cache_;
		const assembler::AssemblyValsCache &mesh_displacement_cache_;
		const bool is_volume_;
	};

	/// Global residual form for ALE Navier--Stokes on velocity, pressure, and
	/// mesh-displacement spaces. This form owns all element gather/scatter;
	/// MultiSpacesNLAssembler implementations remain strictly local.
	class NavierStokesFSIForm : public Form
	{
	public:
		using BodyForceEvaluator = assembler::NavierStokesFSIAssemblerData::BodyForceEvaluator;
		using VelocityTildeUpdater = std::function<void(
			double time,
			const Eigen::VectorXd &current_velocity,
			Eigen::VectorXd &velocity_tilde)>;

		NavierStokesFSIForm(
			int total_size,
			int n_velocity_bases,
			int n_pressure_bases,
			int n_mesh_displacement_bases,
			const std::vector<basis::ElementBases> &velocity_bases,
			const std::vector<basis::ElementBases> &pressure_bases,
			const std::vector<basis::ElementBases> &mesh_displacement_bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::AssemblyValsCache &velocity_cache,
			const assembler::AssemblyValsCache &pressure_cache,
			const assembler::AssemblyValsCache &mesh_displacement_cache,
			std::vector<std::shared_ptr<assembler::MultiSpacesNLAssembler>> assemblers,
			const time_integrator::ImplicitTimeIntegrator *velocity_time_integrator,
			const time_integrator::ImplicitTimeIntegrator *mesh_displacement_time_integrator,
			double t,
			double dt,
			bool is_volume,
			BodyForceEvaluator body_force_evaluator = {});

		std::string name() const override { return "navier-stokes-fsi"; }
		void update_quantities(double t, const Eigen::VectorXd &x) override;
		void set_velocity_tilde_updater(VelocityTildeUpdater updater) { velocity_tilde_updater_ = std::move(updater); }
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		int velocity_ndof() const { return global_sizes_[0]; }
		int pressure_ndof() const { return global_sizes_[1]; }
		int mesh_displacement_ndof() const { return global_sizes_[2]; }

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &residual) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const override;

	private:
		using SpaceValues = std::array<assembler::ElementAssemblyValues, 3>;
		using LocalCoefficients = std::array<Eigen::VectorXd, 3>;

		void compute_element_values(int element, SpaceValues &vals, QuadratureVector &da) const;
		Eigen::VectorXd gather(const Eigen::VectorXd &x, const assembler::ElementAssemblyValues &vals, int components, int global_offset) const;
		void scatter_local_residual(const SpaceValues &vals, const Eigen::VectorXd &local, Eigen::VectorXd &global) const;
		void scatter_local_block(
			const SpaceValues &vals,
			int row_space,
			int col_space,
			const Eigen::MatrixXd &local,
			std::vector<Eigen::Triplet<double>> &entries) const;
		assembler::NavierStokesFSIAssemblerData make_data(
			const SpaceValues &vals,
			const LocalCoefficients &x,
			const LocalCoefficients &x_prev,
			const QuadratureVector &da,
			const Eigen::VectorXd &velocity_tilde,
			const Eigen::VectorXd &mesh_velocity) const;
		bool has_valid_ale_mapping(const Eigen::VectorXd &x) const;

		const int total_size_;
		const int dim_;
		const std::array<int, 3> n_bases_;
		const std::array<int, 3> components_;
		const std::array<int, 3> global_offsets_;
		const std::array<int, 3> global_sizes_;
		const std::array<std::reference_wrapper<const std::vector<basis::ElementBases>>, 3> bases_;
		const std::vector<basis::ElementBases> &geom_bases_;
		const std::array<std::reference_wrapper<const assembler::AssemblyValsCache>, 3> caches_;
		const std::vector<std::shared_ptr<assembler::MultiSpacesNLAssembler>> assemblers_;
		const time_integrator::ImplicitTimeIntegrator *velocity_time_integrator_;
		const time_integrator::ImplicitTimeIntegrator *mesh_displacement_time_integrator_;
		double t_;
		const double dt_;
		const bool is_volume_;
		const BodyForceEvaluator body_force_evaluator_;
		VelocityTildeUpdater velocity_tilde_updater_;
		Eigen::VectorXd x_prev_;
	};
} // namespace polyfem::solver
