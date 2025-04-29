#pragma once

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <ipc/collision_mesh.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <unordered_map>

#include <polyfem/solver/forms/ElasticForm.hpp>

namespace polyfem::time_integrator
{
	class ImplicitTimeIntegrator;
} // namespace polyfem::time_integrator

namespace polyfem::utils
{
	class PeriodicBoundary;
}

namespace polyfem::assembler
{
	class ViscousDamping;
	class MacroStrainValue;
	class PressureAssembler;
} // namespace polyfem::assembler

namespace polyfem::solver
{
	class NLProblem;
	class Form;
	class ContactForm;
	class PeriodicContactForm;
	class MacroStrainALForm;
	class FrictionForm;
	class BodyForm;
	class AugmentedLagrangianForm;
	class MacroStrainLagrangianForm;
	class MacroStrainALForm;
	class InertiaForm;
	class ElasticForm;
	class PressureForm;

	/// class to store time stepping data
	class SolveData
	{
	public:
		/// @brief Initialize the forms and return a vector of pointers to them.
		/// @note Requires rhs_assembler (and time_integrator) to be initialized.
		std::vector<std::shared_ptr<Form>> init_forms(
			// General
			const Units &units,
			const int dim,
			const double t,

			// Elastic form
			const int n_bases,
			std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::Assembler &assembler,
			assembler::AssemblyValsCache &ass_vals_cache,
			const assembler::AssemblyValsCache &mass_ass_vals_cache,
			const double jacobian_threshold,
			const solver::ElementInversionCheck check_inversion,

			// Body form
			const int n_pressure_bases,
			const std::vector<int> &boundary_nodes,
			const std::vector<mesh::LocalBoundary> &local_boundary,
			const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
			const int n_boundary_samples,
			const Eigen::MatrixXd &rhs,
			const Eigen::MatrixXd &sol,
			const assembler::Density &density,

			// Pressure form
			const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
			const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
			const std::shared_ptr<assembler::PressureAssembler> pressure_assembler,

			// Inertia form
			const bool ignore_inertia,
			const StiffnessMatrix &mass,
			const std::shared_ptr<assembler::ViscousDamping> damping_assembler,

			// Lagged regularization form
			const double lagged_regularization_weight,
			const int lagged_regularization_iterations,

			// Augemented lagrangian form
			// const std::vector<int> &boundary_nodes,
			// const std::vector<mesh::LocalBoundary> &local_boundary,
			// const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
			// const int n_boundary_samples,
			// const StiffnessMatrix &mass,
			const size_t obstacle_ndof,

			// Contact form
			const bool contact_enabled,
			const ipc::CollisionMesh &collision_mesh,
			const double dhat,
			const double avg_mass,
			const bool use_convergent_contact_formulation,
			const json &barrier_stiffness,
			const ipc::BroadPhaseMethod broad_phase,
			const double ccd_tolerance,
			const long ccd_max_iterations,
			const bool enable_shape_derivatives,
			
			// Smooth contact form
			const json &contact_params,

			// Homogenization
			const assembler::MacroStrainValue &macro_strain_constraint,

			// Periodic contact
			const bool periodic_contact,
			const Eigen::VectorXi &tiled_to_single,
			const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc,

			// Friction form
			const double friction_coefficient,
			const double epsv,
			const int friction_iterations,

			// Rayleigh damping form
			const json &rayleigh_damping);

		/// @brief update the barrier stiffness for the forms
		/// @param x current solution
		void update_barrier_stiffness(const Eigen::VectorXd &x);

		/// @brief updates the dt inside the different forms
		void update_dt();

		std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> named_forms() const;

	public:
		std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
		std::shared_ptr<assembler::PressureAssembler> pressure_assembler;
		std::shared_ptr<solver::NLProblem> nl_problem;

		std::vector<std::shared_ptr<solver::AugmentedLagrangianForm>> al_form;
		std::shared_ptr<solver::MacroStrainLagrangianForm> strain_al_lagr_form;
		std::shared_ptr<solver::BodyForm> body_form;
		std::shared_ptr<solver::ContactForm> contact_form;
		std::shared_ptr<solver::ElasticForm> damping_form;
		std::shared_ptr<solver::ElasticForm> elastic_form;
		std::shared_ptr<solver::FrictionForm> friction_form;
		std::shared_ptr<solver::InertiaForm> inertia_form;
		std::shared_ptr<solver::PressureForm> pressure_form;

		std::shared_ptr<solver::PeriodicContactForm> periodic_contact_form;

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::solver
