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
			const Eigen::VectorXi &in_node_to_node,

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

			// Constraint forms
			const size_t obstacle_ndof,
			const std::vector<std::string> &hard_constraint_files,
			const std::vector<json> &soft_constraint_files,

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

		void set_AL_initial_weight(const double weight)
		{
			AL_initial_weight_ = weight;
		}
		void set_initial_barrier_stiffness_multiplier(const double multipler)
		{
			initial_barrier_stiffness_multipler_ = multipler;
		}
		void set_avg_edge_length(const double length)
		{
			avg_edge_length_ = length;
		}
		void set_min_edge_length(const double length)
		{
			min_edge_length_ = length;
		}
		bool update_al_weight(const Eigen::VectorXd &x, bool AL_adaptive);
		/// @brief updates the dt inside the different forms
		void update_dt();

		double avg_mass_ = 0;

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
	private:
		json barrier_stiffness_;
		double dt_ = 0;
		double AL_initial_weight_= 1.0;
		double initial_barrier_stiffness_multipler_=1.0;
		double avg_edge_length_=0.0;
		double min_edge_length_=0.0;

	};
} // namespace polyfem::solver
