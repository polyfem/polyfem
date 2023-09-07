#include "SolveData.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/solver/forms/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/BCPenaltyForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	using namespace polyfem::time_integrator;

	std::vector<std::shared_ptr<Form>> SolveData::init_forms(
		// General
		const Units &units,
		const int dim,
		const double t,

		// Elastic form
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::Assembler &assembler,
		const assembler::AssemblyValsCache &ass_vals_cache,
		const assembler::AssemblyValsCache &mass_ass_vals_cache,

		// Body form
		const int n_pressure_bases,
		const std::vector<int> &boundary_nodes,
		const std::vector<mesh::LocalBoundary> &local_boundary,
		const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
		const int n_boundary_samples,
		const Eigen::MatrixXd &rhs,
		const Eigen::MatrixXd &sol,
		const assembler::Density &density,

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
		const polyfem::mesh::Obstacle &obstacle,

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

		// Friction form
		const double friction_coefficient,
		const double epsv,
		const int friction_iterations,

		// Rayleigh damping form
		const json &rayleigh_damping)
	{
		const bool is_time_dependent = time_integrator != nullptr;
		assert(!is_time_dependent || time_integrator != nullptr);
		const double dt = is_time_dependent ? time_integrator->dt() : 0.0;
		const int ndof = n_bases * dim;
		// if (is_formulation_mixed) // mixed not supported
		// 	ndof_ += n_pressure_bases; // pressure is a scalar
		const bool is_volume = dim == 3;

		std::vector<std::shared_ptr<Form>> forms;

		elastic_form = std::make_shared<ElasticForm>(
			n_bases, bases, geom_bases, assembler, ass_vals_cache,
			dt, is_volume);
		forms.push_back(elastic_form);

		if (rhs_assembler != nullptr)
		{
			body_form = std::make_shared<BodyForm>(
				ndof, n_pressure_bases, boundary_nodes, local_boundary,
				local_neumann_boundary, n_boundary_samples, rhs, *rhs_assembler,
				density, /*apply_DBC=*/true, /*is_formulation_mixed=*/false,
				is_time_dependent);
			body_form->update_quantities(t, sol);
			forms.push_back(body_form);
		}

		inertia_form = nullptr;
		damping_form = nullptr;
		if (is_time_dependent)
		{
			if (!ignore_inertia)
			{
				assert(time_integrator != nullptr);
				inertia_form = std::make_shared<InertiaForm>(mass, *time_integrator);
				forms.push_back(inertia_form);
			}

			if (damping_assembler != nullptr)
			{
				damping_form = std::make_shared<ElasticForm>(
					n_bases, bases, geom_bases, *damping_assembler, ass_vals_cache, dt, is_volume);
				forms.push_back(damping_form);
			}
		}
		else
		{
			if (lagged_regularization_weight > 0)
			{
				forms.push_back(std::make_shared<LaggedRegForm>(lagged_regularization_iterations));
				forms.back()->set_weight(lagged_regularization_weight);
			}
		}

		if (rhs_assembler != nullptr)
		{
			// assembler::Mass mass_mat_assembler;
			// mass_mat_assembler.set_size(dim);
			StiffnessMatrix mass_tmp = mass;
			// mass_mat_assembler.assemble(dim == 3, n_bases, bases, geom_bases, mass_ass_vals_cache, mass_tmp, true);
			// assert(mass_tmp.rows() == mass.rows() && mass_tmp.cols() == mass.cols());

			al_lagr_form = std::make_shared<BCLagrangianForm>(
				ndof, boundary_nodes, local_boundary, local_neumann_boundary,
				n_boundary_samples, mass_tmp, *rhs_assembler, obstacle, is_time_dependent, t);
			forms.push_back(al_lagr_form);

			al_pen_form = std::make_shared<BCPenaltyForm>(
				ndof, boundary_nodes, local_boundary, local_neumann_boundary,
				n_boundary_samples, mass_tmp, *rhs_assembler, obstacle, is_time_dependent, t);
			forms.push_back(al_pen_form);
		}

		contact_form = nullptr;
		friction_form = nullptr;
		if (contact_enabled)
		{
			const bool use_adaptive_barrier_stiffness = !barrier_stiffness.is_number();

			contact_form = std::make_shared<ContactForm>(
				collision_mesh, dhat, avg_mass, use_convergent_contact_formulation,
				use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase, ccd_tolerance * units.characteristic_length(),
				ccd_max_iterations);

			if (use_adaptive_barrier_stiffness)
			{
				contact_form->set_barrier_stiffness(1);
				// logger().debug("Using adaptive barrier stiffness");
			}
			else
			{
				assert(barrier_stiffness.get<double>() > 0);
				contact_form->set_barrier_stiffness(barrier_stiffness);
				// logger().debug("Using fixed barrier stiffness of {}", contact_form->barrier_stiffness());
			}

			forms.push_back(contact_form);

			// ----------------------------------------------------------------

			if (friction_coefficient != 0)
			{
				friction_form = std::make_shared<FrictionForm>(
					collision_mesh, time_integrator, epsv, friction_coefficient, dhat,
					broad_phase, *contact_form, friction_iterations);
				friction_form->init_lagging(sol);
				forms.push_back(friction_form);
			}
		}

		const std::vector<json> rayleigh_damping_jsons = utils::json_as_array(rayleigh_damping);
		if (is_time_dependent)
		{
			// Map from form name to form so RayleighDampingForm::create can get the correct form to damp
			const std::unordered_map<std::string, std::shared_ptr<Form>> possible_forms_to_damp = {
				{"elasticity", elastic_form},
				{"contact", contact_form},
			};

			for (const json &params : rayleigh_damping_jsons)
			{
				forms.push_back(RayleighDampingForm::create(
					params, possible_forms_to_damp,
					*time_integrator));
			}
		}
		else if (rayleigh_damping_jsons.size() > 0)
		{
			log_and_throw_error("Rayleigh damping is only supported for time-dependent problems");
		}

		update_dt();

		return forms;
	}

	void SolveData::update_barrier_stiffness(const Eigen::VectorXd &x)
	{
		if (contact_form == nullptr || !contact_form->use_adaptive_barrier_stiffness())
			return;

		Eigen::VectorXd grad_energy = Eigen::VectorXd::Zero(x.size());
		const std::array<std::shared_ptr<Form>, 3> energy_forms{
			{elastic_form, inertia_form, body_form}};
		for (const std::shared_ptr<Form> &form : energy_forms)
		{
			if (form == nullptr || !form->enabled())
				continue;

			Eigen::VectorXd grad_form;
			form->first_derivative(x, grad_form);
			grad_energy += grad_form;
		}

		contact_form->update_barrier_stiffness(x, grad_energy);
	}

	void SolveData::update_dt()
	{
		if (time_integrator == nullptr) // if is not time dependent
			return;

		const std::array<std::shared_ptr<Form>, 5> energy_forms{
			{elastic_form, body_form, damping_form, contact_form, friction_form}};
		for (const std::shared_ptr<Form> &form : energy_forms)
		{
			if (form == nullptr)
				continue;
			form->set_weight(time_integrator->acceleration_scaling());
		}
	}

	std::unordered_map<std::string, std::shared_ptr<solver::Form>> SolveData::named_forms() const
	{
		return {
			{"contact", contact_form},
			{"body", body_form},
			{"augmented_lagrangian_lagr", al_lagr_form},
			{"augmented_lagrangian_penalty", al_pen_form},
			{"damping", damping_form},
			{"friction", friction_form},
			{"inertia", inertia_form},
			{"elastic", elastic_form},
		};
	}
} // namespace polyfem::solver
