#include "SolveData.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/MatrixLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/PeriodicLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/MacroStrainLagrangianForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/solver/forms/PressureForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
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
		const bool use_area_weighting,
		const bool use_improved_max_operator,
		const bool use_physical_barrier,
		const json &barrier_stiffness,
		const ipc::BroadPhaseMethod broad_phase,
		const double ccd_tolerance,
		const long ccd_max_iterations,
		const bool enable_shape_derivatives,

		// Smooth Contact Form
		const bool use_gcp_formulation,
		const double alpha_t,
		const double alpha_n,
		const bool use_adaptive_dhat,
		const double min_distance_ratio,
		
		// Normal Adhesion Form
		const bool adhesion_enabled,
		const double dhat_p,
		const double dhat_a,
		const double Y,

		// Tangential Adhesion Form
		const double tangential_adhesion_coefficient,
		const double epsa,
		const int tangential_adhesion_iterations,

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
		this->avg_mass_ = avg_mass;

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
					n_boundary_samples, mass_tmp, *rhs_assembler, obstacle_ndof, is_time_dependent, t));
			// forms.push_back(al_form.back());
		}

		if (periodic_bc != nullptr)
		{
			al_form.push_back(std::make_shared<PeriodicLagrangianForm>(ndof, periodic_bc));
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

			Eigen::MatrixXd bin = file.readDataset<Eigen::MatrixXd>("b");

			StiffnessMatrix A, A_proj;
			Eigen::MatrixXd b, b_proj;

			if (!file.findDatasets("A").empty())
			{
				Eigen::MatrixXd Ain = file.readDataset<Eigen::MatrixXd>("A");
				utils::scatter_matrix(ndof, dim, Ain, bin, local2global, A, b);

				if (!file.findDatasets("A_proj").empty())
				{
					Eigen::MatrixXd A_proj_in = file.readDataset<Eigen::MatrixXd>("A_proj");
					if (file.findDatasets("b_proj").empty())
						log_and_throw_error("Missing b_proj in hard constraint file");

					Eigen::MatrixXd b_proj_in = file.readDataset<Eigen::MatrixXd>("b_proj");
					utils::scatter_matrix_col(ndof, dim, A_proj_in, b_proj_in, local2global, A_proj, b_proj);
				}
			}
			else
			{
				std::vector<double> values = file.readDataset<std::vector<double>>("A_triplets/values");
				std::vector<int> rows = file.readDataset<std::vector<int>>("A_triplets/rows");
				std::vector<int> cols = file.readDataset<std::vector<int>>("A_triplets/cols");
				std::vector<long> shape = file.readDataset<std::vector<long>>("A_triplets/shape");
				utils::scatter_matrix(ndof, dim, shape, rows, cols, values, bin, local2global, A, b);

				if (!file.findGroups("A_proj_triplets").empty())
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
					Eigen::MatrixXd b_projin = file.readDataset<Eigen::MatrixXd>("b_proj");
					std::vector<long> shape_proj = file.readDataset<std::vector<long>>("A_proj_triplets/shape");

					utils::scatter_matrix_col(ndof, dim, shape_proj, rows_proj, cols_proj, values_proj, b_projin, local2global, A_proj, b_proj);
				}
			}

			al_form.push_back(std::make_shared<MatrixLagrangianForm>(A, b, A_proj, b_proj));
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

			Eigen::MatrixXd bin = file.readDataset<Eigen::MatrixXd>("b");

			StiffnessMatrix A;
			Eigen::MatrixXd b;

			if (!file.findDatasets("A").empty())
			{
				Eigen::MatrixXd Ain = file.readDataset<Eigen::MatrixXd>("A");
				utils::scatter_matrix(ndof, dim, Ain, bin, local2global, A, b);
			}
			else
			{
				std::vector<double> values = file.readDataset<std::vector<double>>("A_triplets/values");
				std::vector<int> rows = file.readDataset<std::vector<int>>("A_triplets/rows");
				std::vector<int> cols = file.readDataset<std::vector<int>>("A_triplets/cols");
				std::vector<long> shape = file.readDataset<std::vector<long>>("A_triplets/shape");

				utils::scatter_matrix(ndof, dim, shape, rows, cols, values, bin, local2global, A, b);
			}

			forms.push_back(std::make_shared<QuadraticPenaltyForm>(A, b, weight));
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
					collision_mesh, tiled_to_single, dhat, avg_mass, use_area_weighting, use_improved_max_operator, use_physical_barrier,
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

				// periodic_contact_form is not pushed into forms since it takes different input vectors.
			}
			else
			{
				if (use_gcp_formulation)
				{
					contact_form = std::make_shared<SmoothContactForm>(
						collision_mesh, dhat, avg_mass, alpha_t, alpha_n, use_adaptive_dhat, min_distance_ratio,
						use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase, 
						ccd_tolerance * units.characteristic_length(), ccd_max_iterations);
				}
				else
				{
					contact_form = std::make_shared<BarrierContactForm>(
						collision_mesh, dhat, avg_mass, use_area_weighting, use_improved_max_operator, use_physical_barrier,
						use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase, ccd_tolerance * units.characteristic_length(),
						ccd_max_iterations);
				}

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
					logger().debug("Setting barrier stiffness to {}", contact_form->barrier_stiffness());
				}

				forms.push_back(contact_form);
			}

			if (friction_coefficient != 0)
			{
				friction_form = std::make_shared<FrictionForm>(
					collision_mesh, time_integrator, epsv, friction_coefficient,
					broad_phase, *contact_form, friction_iterations);
				friction_form->init_lagging(sol);
				forms.push_back(friction_form);
			}

			if (adhesion_enabled)
			{
				normal_adhesion_form = std::make_shared<NormalAdhesionForm>(
					collision_mesh, dhat_p, dhat_a, Y, is_time_dependent, enable_shape_derivatives,
					broad_phase, ccd_tolerance * units.characteristic_length(), ccd_max_iterations
				);
				forms.push_back(normal_adhesion_form);

				if (tangential_adhesion_coefficient != 0)
				{
					tangential_adhesion_form = std::make_shared<TangentialAdhesionForm>(
						collision_mesh, time_integrator, epsa, tangential_adhesion_coefficient,
						broad_phase, *normal_adhesion_form, tangential_adhesion_iterations
					);
					forms.push_back(tangential_adhesion_form);
				}
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
	{
		if (contact_form == nullptr)
			return;

		double barrier_stiffness;
		double AL_grad_energy = 0.0;
		for (const auto &f : al_form)
		{
			AL_grad_energy =  f->lagrangian_weight();
		}

		if (contact_form->use_adaptive_barrier_stiffness())
		{
			logger().debug("You have set to scale the adaptive barrier stiffness by {}", initial_barrier_stiffness_multipler_);


			double bs_multiplier = contact_form->get_bs_multiplier();
			const double dhat = contact_form->dhat();
			double prev_dist = contact_form->get_prev_distance();


			logger().debug("Prev dist is {}", prev_dist);
			if (prev_dist != -1 && prev_dist != INFINITY && prev_dist < .5*dhat*dhat)
			{
				bs_multiplier *= 2;
				contact_form->set_bs_multiplier(bs_multiplier);
			}
			if (prev_dist != -1 && prev_dist != INFINITY && prev_dist > .9*dhat*dhat)
			{
				bs_multiplier /= 2;
				contact_form->set_bs_multiplier(bs_multiplier);
			}

			barrier_stiffness = 10*AL_grad_energy*initial_barrier_stiffness_multipler_ * bs_multiplier;
			if (barrier_stiffness < 10*AL_grad_energy || prev_dist == INFINITY )
				barrier_stiffness = 10*AL_grad_energy;
		}
		else
		{
			barrier_stiffness = barrier_stiffness_;
			if (barrier_stiffness<AL_grad_energy)
				logger().warn("Your barrier stiffness is lower than your Gradient Energy {}. Likely to result in numerical instabilities!", AL_grad_energy);
		}
		contact_form->set_barrier_stiffness(barrier_stiffness);
		logger().debug("Barrier Stiffness set to {}", contact_form->barrier_stiffness());

	}

	void SolveData::update_al_weight(const Eigen::VectorXd &x)
	{
		double weight;

		StiffnessMatrix hessian_form;
		double max_stiffness = 0;
		const double scaling = time_integrator->acceleration_scaling();

		const std::array<std::shared_ptr<Form>, 7> energy_forms{
					{elastic_form, inertia_form, body_form, pressure_form, friction_form, normal_adhesion_form, tangential_adhesion_form}};

		for (const std::shared_ptr<Form> &form : energy_forms)
		{
			if (form != nullptr){
				double max_of_form = 0;
				form->second_derivative(x, hessian_form);

				for (int k = 0; k < hessian_form.outerSize(); ++k)
				{
					for (StiffnessMatrix::InnerIterator it(hessian_form, k); it; ++it)
					{
						max_stiffness = std::max(max_stiffness, std::abs(it.value()));
					}
				}
				if (max_stiffness < max_of_form)
				{
					max_stiffness = max_of_form;
				}
			}
		}


		int dbc_size = 0;
		for (const auto &f : al_form)
		{

		dbc_size += f->get_dbc_size();
		}

		//Scales AL to max hessian * 1000; Al_initial_weight acts as a multiplier for users to make this more or less aggressive; scaling accounts for acceleration scaling
		weight = 10*max_stiffness/scaling*AL_initial_weight_;


			/* still playing with this
			double error = 0.0;
			int dbc_size = 0;
			for (const auto &f : al_form)
			{
			error += f->get_dbcerror();
			dbc_size += f->get_dbc_size();
			}
			 if (error > 0)
				weight = 1000*max_stiffness/scaling*AL_initial_weight_*error;
			else if (dbc_size > 0)
				weight =  max_stiffness/scaling*AL_initial_weight_* dbc_size;
			else
				weight =  max_stiffness/scaling*AL_initial_weight_;
			*/

		for (const auto &f : al_form)
		{
				f->set_al_weight(weight);
		}

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
