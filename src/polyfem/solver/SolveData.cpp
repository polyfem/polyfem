#include "SolveData.hpp"

#include <barrier>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/MatrixLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/MacroStrainLagrangianForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/PressureForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>
#include <polyfem/solver/forms/QuadraticPenaltyForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/MacroStrain.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/assembler/PeriodicBoundary.hpp>

#include <h5pp/h5pp.h>


namespace polyfem::solver
{
	using namespace polyfem::time_integrator;

	std::vector<std::shared_ptr<Form>> SolveData::init_forms(
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
		const ElementInversionCheck check_inversion,

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
		const json &rayleigh_damping)
	{
		this->barrier_stiffness_ = barrier_stiffness;

		const bool is_time_dependent = time_integrator != nullptr;
		assert(!is_time_dependent || time_integrator != nullptr);
		const double dt = is_time_dependent ? time_integrator->dt() : 0.0;
		const int ndof = n_bases * dim;
		// if (is_formulation_mixed) // mixed not supported
		// 	ndof_ += n_pressure_bases; // pressure is a scalar
		const bool is_volume = dim == 3;

		std::vector<std::shared_ptr<Form>> forms;
		al_form.clear();

		elastic_form = std::make_shared<ElasticForm>(
			n_bases, bases, geom_bases, assembler, ass_vals_cache,
			t, dt, is_volume, jacobian_threshold, check_inversion);
		forms.push_back(elastic_form);

		if (rhs_assembler != nullptr)
		{
			body_form = std::make_shared<BodyForm>(
				ndof, n_pressure_bases, boundary_nodes, local_boundary,
				local_neumann_boundary, n_boundary_samples, rhs, *rhs_assembler,
				density, /*is_formulation_mixed=*/false,
				is_time_dependent);
			body_form->update_quantities(t, sol);
			forms.push_back(body_form);
		}

		if (pressure_assembler != nullptr)
		{
			pressure_form = std::make_shared<PressureForm>(
				ndof,
				local_pressure_boundary,
				local_pressure_cavity,
				boundary_nodes,
				n_boundary_samples, *pressure_assembler,
				is_time_dependent);
			pressure_form->update_quantities(t, sol);
			forms.push_back(pressure_form);
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
					n_bases, bases, geom_bases, *damping_assembler, ass_vals_cache, t, dt, is_volume);
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

			if (!boundary_nodes.empty())
				al_form.push_back(std::make_shared<BCLagrangianForm>(
					ndof, boundary_nodes, local_boundary, local_neumann_boundary,
					n_boundary_samples, mass_tmp, *rhs_assembler, obstacle_ndof, is_time_dependent, t, periodic_bc));
			// forms.push_back(al_form.back());
		}

		for (const auto &path : hard_constraint_files)
		{
			logger().debug("Setting up hard constraints for {}", path);
			h5pp::File file(path, h5pp::FileAccess::READONLY);
			std::vector<int> local2global;
			if (!file.findDatasets("local2global").empty())
				local2global = file.readDataset<std::vector<int>>("local2global");

			if (local2global.empty())
			{
				local2global.resize(in_node_to_node.size());

				for (int i = 0; i < local2global.size(); ++i)
					local2global[i] = in_node_to_node[i];
			}
			else
			{
				for (auto &v : local2global)
					v = in_node_to_node[v];
			}

			Eigen::MatrixXd b = file.readDataset<Eigen::MatrixXd>("b");

			if (!file.findDatasets("A").empty())
			{
				Eigen::MatrixXd A = file.readDataset<Eigen::MatrixXd>("A");

				if (file.findDatasets("A_proj").empty())
					al_form.push_back(std::make_shared<MatrixLagrangianForm>(
						ndof, dim, A, b, local2global));
				else
				{
					Eigen::MatrixXd A_proj = file.readDataset<Eigen::MatrixXd>("A_proj");
					if (file.findDatasets("b_proj").empty())
						log_and_throw_error("Missing b_proj in hard constraint file");

					Eigen::MatrixXd b_proj = file.readDataset<Eigen::MatrixXd>("b_proj");
					al_form.push_back(std::make_shared<MatrixLagrangianForm>(
						ndof, dim, A, b, local2global, A_proj, b_proj));
				}
			}
			else
			{
				std::vector<double> values = file.readDataset<std::vector<double>>("A_triplets/values");
				std::vector<int> rows = file.readDataset<std::vector<int>>("A_triplets/rows");
				std::vector<int> cols = file.readDataset<std::vector<int>>("A_triplets/cols");

				if (file.findGroups("A_proj_triplets").empty())
					al_form.push_back(std::make_shared<MatrixLagrangianForm>(
						ndof, dim, rows, cols, values, b, local2global));
				else
				{
					if (file.findDatasets("b_proj").empty())
						log_and_throw_error("Missing b_proj in hard constraint file");
					if (file.findDatasets("rows", "/A_proj_triplets").empty())
						log_and_throw_error("Missing A_proj_triplets/rows in hard constraint file");
					if (file.findDatasets("cols", "/A_proj_triplets").empty())
						log_and_throw_error("Missing A_proj_triplets/cols in hard constraint file");
					if (file.findDatasets("values", "/A_proj_triplets").empty())
						log_and_throw_error("Missing A_proj_triplets/values in hard constraint file");

					std::vector<double> values_proj = file.readDataset<std::vector<double>>("A_proj_triplets/values");
					std::vector<int> rows_proj = file.readDataset<std::vector<int>>("A_proj_triplets/rows");
					std::vector<int> cols_proj = file.readDataset<std::vector<int>>("A_proj_triplets/cols");
					Eigen::MatrixXd b_proj = file.readDataset<Eigen::MatrixXd>("b_proj");

					al_form.push_back(std::make_shared<MatrixLagrangianForm>(
						ndof, dim, rows, cols, values, b, local2global, rows_proj, cols_proj, values_proj, b_proj));
				}
			}
			// forms.push_back(al_form.back());
		}

		for (const auto &j : soft_constraint_files)
		{
			const std::string &path = j["data"];
			double weight = j["weight"];

			logger().debug("Setting up soft constraints for {}", path);
			h5pp::File file(path, h5pp::FileAccess::READONLY);
			std::vector<int> local2global;
			if (!file.findDatasets("local2global").empty())
				local2global = file.readDataset<std::vector<int>>("local2global");

			if (local2global.empty())
			{
				local2global.resize(in_node_to_node.size());

				for (int i = 0; i < local2global.size(); ++i)
					local2global[i] = in_node_to_node[i];
			}
			else
			{
				for (auto &v : local2global)
					v = in_node_to_node[v];
			}

			Eigen::MatrixXd b = file.readDataset<Eigen::MatrixXd>("b");

			if (!file.findDatasets("A").empty())
			{
				Eigen::MatrixXd A = file.readDataset<Eigen::MatrixXd>("A");

				forms.push_back(std::make_shared<QuadraticPenaltyForm>(
					ndof, dim, A, b, weight, local2global));
			}
			else
			{
				std::vector<double> values = file.readDataset<std::vector<double>>("A_triplets/values");
				std::vector<int> rows = file.readDataset<std::vector<int>>("A_triplets/rows");
				std::vector<int> cols = file.readDataset<std::vector<int>>("A_triplets/cols");

				forms.push_back(std::make_shared<QuadraticPenaltyForm>(
					ndof, dim, rows, cols, values, b, weight, local2global));
			}
		}

		if (macro_strain_constraint.is_active())
		{
			// don't push these two into forms because they take a different input x
			strain_al_lagr_form = std::make_shared<MacroStrainLagrangianForm>(macro_strain_constraint);
		}

		contact_form = nullptr;
		periodic_contact_form = nullptr;
		friction_form = nullptr;
		if (contact_enabled)
		{
			const bool use_adaptive_barrier_stiffness = !barrier_stiffness.is_number();

			if (periodic_contact)
			{
				periodic_contact_form = std::make_shared<PeriodicContactForm>(
					collision_mesh, tiled_to_single, dhat, avg_mass, use_convergent_contact_formulation,
					use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase, ccd_tolerance,
					ccd_max_iterations);

				if (use_adaptive_barrier_stiffness)
				{
					periodic_contact_form->set_barrier_stiffness(1);
					// logger().debug("Using adaptive barrier stiffness");
				}
				else
				{
					assert(barrier_stiffness.is_number());
					assert(barrier_stiffness.get<double>() > 0);
					periodic_contact_form->set_barrier_stiffness(barrier_stiffness);
					// logger().debug("Using fixed barrier stiffness of {}", contact_form->barrier_stiffness());
				}
			}
			else
			{
				contact_form = std::make_shared<ContactForm>(
					collision_mesh, dhat, avg_mass, use_convergent_contact_formulation,
					use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase, ccd_tolerance * units.characteristic_length(),
					ccd_max_iterations);

				if (use_adaptive_barrier_stiffness)
				{
					contact_form->set_barrier_stiffness(1);
					logger().debug("Using adaptive barrier stiffness");
				}
				else
				{
					assert(barrier_stiffness.is_number());
					assert(barrier_stiffness.get<double>() > 0);
					contact_form->set_barrier_stiffness(barrier_stiffness);
					logger().debug("Scaling barrier stiffness by {}", contact_form->barrier_stiffness());
				}

				if (contact_form)
					forms.push_back(contact_form);

				// ----------------------------------------------------------------
			}

			if (friction_coefficient != 0)
			{
				friction_form = std::make_shared<FrictionForm>(
					collision_mesh, time_integrator, epsv, friction_coefficient,
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
			log_and_throw_adjoint_error("Rayleigh damping is only supported for time-dependent problems");
		}

		update_dt();

		this->dt_ = dt;

		return forms;
	}

	void SolveData::update_barrier_stiffness(const Eigen::VectorXd &x)
	{	//todo: fix this to make it work with fixed barrier stiffness
		if (contact_form == nullptr)
			return;

		StiffnessMatrix hessian_form;
		double max_stiffness = 0;
		double grad_energy = 0.0;
		const std::array<std::shared_ptr<Form>, 4> energy_forms{
					{elastic_form, inertia_form, body_form, pressure_form}};

		//Grabs the gradient of the energy to scale the barrier stiffness
		for (const std::shared_ptr<Form> &form : energy_forms)
		{
			if (form == nullptr || !form->enabled())
				continue;

			double weight = form->weight();
			Eigen::VectorXd grad_form = Eigen::VectorXd::Zero(x.size());
			form->first_derivative(x, grad_form);
			grad_energy += grad_form.colwise().maxCoeff()(0)/weight;
		}
		//Grabs the approximate stiffness of the material via the max coeff of the elastic hessian
		elastic_form->second_derivative(x, hessian_form);
		const double ef_weight = elastic_form->weight();

		for (int k = 0; k < hessian_form.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(hessian_form, k); it; ++it)
			{
				max_stiffness= std::max(max_stiffness, std::abs(it.value()));
			}
		}
		max_stiffness/= ef_weight;



		// grad_energy/(approx gradient of barrier function) provides a scaling factor based on changes in the energy relative to barrier stiffness
		const double current_barrier_stiffness = contact_form->barrier_stiffness();
		double ini_barrier_stiffness = 1.0;
		if (barrier_stiffness_.is_number()){
			ini_barrier_stiffness = barrier_stiffness_.get<double>();
		}
		const double dhat = contact_form->dhat();
		double contact_barrier_grad =  17.5408*dhat; //solving for d for d(barrier_function)/dd(barrier_function) gives constant relative to dhat
		double barrier_stiffness = grad_energy/contact_barrier_grad * ini_barrier_stiffness;
		if (barrier_stiffness <  1)
			barrier_stiffness = 1.0;
		contact_form->set_barrier_stiffness(barrier_stiffness);
		logger().debug("Barrier Stiffness set to {}", contact_form->barrier_stiffness());

	}
	void SolveData::update_dt()
	{
		if (time_integrator == nullptr) // if is not time dependent
			return;

		const std::array<std::shared_ptr<Form>, 6> energy_forms{
			{elastic_form, body_form, pressure_form, damping_form, contact_form, friction_form}};
		for (const std::shared_ptr<Form> &form : energy_forms)
		{
			if (form == nullptr)
				continue;
			form->set_weight(time_integrator->acceleration_scaling());
		}
	}

	std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> SolveData::named_forms() const
	{
		std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> res{
			{"elastic", elastic_form},
			{"inertia", inertia_form},
			{"body", body_form},
			{"contact", contact_form},
			{"friction", friction_form},
			{"damping", damping_form},
			{"pressure", pressure_form},
			{"strain_augmented_lagrangian_lagr", strain_al_lagr_form},
			{"periodic_contact", periodic_contact_form},
		};

		for (const auto &form : al_form)
			res.push_back({"augmented_lagrangian", form});

		return res;
	}
} // namespace polyfem::solver
