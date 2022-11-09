#pragma once

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <ipc/collision_mesh.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <unordered_map>

namespace polyfem::time_integrator
{
	class ImplicitTimeIntegrator;
} // namespace polyfem::time_integrator

namespace polyfem::solver
{
	class NLProblem;
	class Form;
	class ContactForm;
	class FrictionForm;
	class BodyForm;
	class ALForm;
	class InertiaForm;
	class ElasticForm;

	/// class to store time stepping data
	class SolveData
	{
	public:
		/// @brief Initialize the forms and return a vector of pointers to them.
		/// @note Requires rhs_assembler and time_integrator to be initialized.
		std::vector<std::shared_ptr<Form>> init_forms(
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::AssemblerUtils &assembler,
			const assembler::AssemblyValsCache &ass_vals_cache,
			const std::string &formulation,
			const int dim,
			const int n_pressure_bases,
			const std::vector<int> &boundary_nodes,
			const std::vector<mesh::LocalBoundary> &local_boundary,
			const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
			const int n_boundary_samples,
			const Eigen::MatrixXd &rhs,
			const double t,
			const Eigen::MatrixXd &sol,
			const json &args,
			const StiffnessMatrix &mass,
			const polyfem::mesh::Obstacle &obstacle,
			const ipc::CollisionMesh &collision_mesh,
			const Eigen::MatrixXd &boundary_nodes_pos,
			const double avg_mass);

		/// @brief update the barrier stiffness for the forms
		/// @param x current solution
		void updated_barrier_stiffness(const Eigen::VectorXd &x);

		/// @brief updates the dt inside the different forms
		void update_dt();

		std::unordered_map<std::string, std::shared_ptr<solver::Form>> named_forms() const;

	public:
		std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
		std::shared_ptr<solver::NLProblem> nl_problem;

		std::shared_ptr<solver::ALForm> al_form;
		std::shared_ptr<solver::BodyForm> body_form;
		std::shared_ptr<solver::ContactForm> contact_form;
		std::shared_ptr<solver::ElasticForm> damping_form;
		std::shared_ptr<solver::ElasticForm> elastic_form;
		std::shared_ptr<solver::FrictionForm> friction_form;
		std::shared_ptr<solver::InertiaForm> inertia_form;

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::solver
