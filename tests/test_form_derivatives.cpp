////////////////////////////////////////////////////////////////////////////////

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/assembler/FixedCorotational.hpp>

#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/PressureForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/InversionBarrierForm.hpp>
#include <polyfem/solver/forms/L2ProjectionForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>
#include <polyfem/solver/forms/adjoint_forms/AMIPSForm.hpp>

#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <finitediff.hpp>

#include <polyfem/State.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::solver;
using namespace polyfem::time_integrator;
using namespace polyfem::assembler;

namespace
{
	std::shared_ptr<State> get_state(int dim, const std::string &material_type = "NeoHookean")
	{
		const std::string path = POLYFEM_DATA_DIR;

		json material;
		if (material_type == "NeoHookean")
		{
			material = R"(
			{
				"type": "NeoHookean",
				"E": 20000,
				"nu": 0.3,
				"rho": 1000,
				"phi": 1,
				"psi": 1
			}
			)"_json;
		}
		else if (material_type == "MooneyRivlin3ParamSymbolic")
		{
			material = R"(
			{
				"type": "MooneyRivlin3ParamSymbolic",
				"c1": 1e5,
				"c2": 1e3,
				"c3": 1e3,
				"d1": 1e5,
				"rho": 1000
			}
			)"_json;
		}
		else
			assert(false);

		json in_args = R"(
		{
			"time": {
				"dt": 0.001,
				"tend": 1.0
			},

			"output": {
				"log": {
					"level": "warning"
				}
			}

		})"_json;
		in_args["materials"] = material;
		if (dim == 2)
		{
			in_args["geometry"] = R"([{
				"enabled": true,
				"surface_selection": 7
			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";
			in_args["boundary_conditions"] = R"({
				"dirichlet_boundary": [{
					"id": "all",
					"value": [0, 0]
				}],
				"rhs": [10, 10]
			})"_json;
		}
		else
		{
			in_args["geometry"] = R"([{
				"transformation": {
					"scale": [0.1, 1, 1]
				},
				"surface_selection": [
					{
						"id": 1,
						"axis": "z",
						"position": 0.8,
						"relative": true
					},
					{
						"id": 2,
						"axis": "-z",
						"position": 0.2,
						"relative": true
					},
					{
						"id": 3,
						"box": [[0, 0, 0.2], [1, 1, 0.8]],
						"relative": true
					}
				],
				"n_refs": 1
			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-6.msh";
			in_args["boundary_conditions"] = R"({
				"neumann_boundary": [{
					"id": 1,
					"value": [1000, 1000, 1000]
				}],
				"pressure_boundary": [{
					"id": 1,
					"value": -2000
				},
				{
					"id": 2,
					"value": -2000
				},
				{
					"id": 3,
					"value": -2000
				}],
				"rhs": [0, 0, 0]
			})"_json;
		}

		auto state = std::make_shared<State>();
		state->init(in_args, true);
		state->set_max_threads(1);

		state->load_mesh();

		state->build_basis();
		state->assemble_rhs();
		state->assemble_mass_mat();

		return state;
	}
} // namespace

template <typename Form>
void test_form(Form &form, const State &state, double step = 1e-8, double tol = 1e-4)
{
	static const int n_rand = 10;

	Eigen::VectorXd x = Eigen::VectorXd::Zero(state.n_bases * state.mesh->dimension());

	form.init(x);
	form.init_lagging(x);

	for (int rand = 0; rand < n_rand; ++rand)
	{
		// Test gradient with finite differences
		{
			Eigen::VectorXd grad;
			form.first_derivative(x, grad);

			Eigen::VectorXd fgrad;
			fd::finite_gradient(
				x, [&form](const Eigen::VectorXd &x) -> double { return form.value(x); }, fgrad,
				fd::AccuracyOrder::SECOND, step);

			if (!fd::compare_gradient(grad, fgrad))
			{
				std::cout << "Gradient mismatch" << std::endl;
				std::cout << "Gradient: " << grad.transpose() << std::endl;
				std::cout << "Finite gradient: " << fgrad.transpose() << std::endl;
			}

			CHECK(fd::compare_gradient(grad, fgrad, tol));
		}

		// Test hessian with finite differences
		{
			StiffnessMatrix hess;
			form.second_derivative(x, hess);

			Eigen::MatrixXd fhess;
			fd::finite_jacobian(
				x,
				[&form](const Eigen::VectorXd &x) -> Eigen::VectorXd {
					Eigen::VectorXd grad;
					form.first_derivative(x, grad);
					return grad;
				},
				fhess,
				fd::AccuracyOrder::SECOND, step);

			if (!fd::compare_hessian(Eigen::MatrixXd(hess), fhess))
			{
				std::cout << "Hessian mismatch" << std::endl;
				std::cout << "Hessian: " << hess << std::endl;
				std::cout << "Finite hessian: " << fhess << std::endl;
			}

			CHECK(fd::compare_hessian(Eigen::MatrixXd(hess), fhess, tol));
		}

		x.setRandom();
		x /= 100;
	}
}

TEST_CASE("body form derivatives 3d", "[form][form_derivatives][body_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	const auto rhs_assembler_ptr = state_ptr->build_rhs_assembler();
	const int ndof = state_ptr->n_bases * state_ptr->mesh->dimension();

	BodyForm form(ndof, state_ptr->n_pressure_bases,
				  state_ptr->boundary_nodes,
				  state_ptr->local_boundary,
				  state_ptr->local_neumann_boundary,
				  state_ptr->n_boundary_samples(),
				  state_ptr->rhs,
				  *rhs_assembler_ptr,
				  state_ptr->mass_matrix_assembler->density(),
				  false, state_ptr->problem->is_time_dependent());
	Eigen::VectorXd x_prev;
	x_prev.setRandom(state_ptr->n_bases * 3);
	x_prev /= 100.;
	form.update_quantities(state_ptr->args["time"]["t0"].get<double>() + 5 * state_ptr->args["time"]["dt"].get<double>(), x_prev);

	test_form(form, *state_ptr);
}

TEST_CASE("body form derivatives", "[form][form_derivatives][body_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	const auto rhs_assembler_ptr = state_ptr->build_rhs_assembler();
	const int ndof = state_ptr->n_bases * state_ptr->mesh->dimension();

	BodyForm form(ndof, state_ptr->n_pressure_bases,
				  state_ptr->boundary_nodes,
				  state_ptr->local_boundary,
				  state_ptr->local_neumann_boundary,
				  state_ptr->n_boundary_samples(),
				  state_ptr->rhs,
				  *rhs_assembler_ptr,
				  state_ptr->mass_matrix_assembler->density(),
				  false, state_ptr->problem->is_time_dependent());

	Eigen::VectorXd x_prev;
	x_prev.setRandom(state_ptr->n_bases * dim);
	x_prev /= 100.;
	form.update_quantities(state_ptr->args["time"]["t0"].get<double>() + 5 * state_ptr->args["time"]["dt"].get<double>(), x_prev);

	test_form(form, *state_ptr);
}

TEST_CASE("barrier contact form derivatives", "[form][form_derivatives][contact_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);

	const double dhat = 1e-3;
	const bool use_adaptive_barrier_stiffness = true; // GENERATE(true, false);
	const bool use_convergent_formulation = GENERATE(true, false);
	const double barrier_stiffness = 1e7;
	const bool is_time_dependent = GENERATE(true, false);
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);
	const double dt = 1e-3;

	BarrierContactForm form(
		state_ptr->collision_mesh, dhat, state_ptr->avg_mass,
		use_convergent_formulation, use_adaptive_barrier_stiffness,
		is_time_dependent, false, broad_phase_method, ccd_tolerance,
		ccd_max_iterations);

	test_form(form, *state_ptr);
}

TEST_CASE("smooth contact form derivatives", "[form][form_derivatives][contact_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);

	const double dhat = 1e-3;
	const double alpha = 0.2;
	const double r = 1;
	const double barrier_stiffness = 1e7;
	const bool is_time_dependent = true;
	const bool use_adaptive_barrier_stiffness = false;
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);
	const double a = 0;
	const json contact_args = json::object({ {"a", a}, {"alpha_t", alpha}, {"alpha_n", 0.1}, {"beta_t", 0}, {"beta_n", 0}, {"dhat", dhat}, {"use_adaptive_dhat", false}, {"min_distance_ratio", 0.5} });

	SmoothContactForm form(
		state_ptr->collision_mesh, contact_args, state_ptr->avg_mass,
		use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method, 
		ccd_tolerance, ccd_max_iterations);

	test_form(form, *state_ptr);
}

TEST_CASE("elastic form derivatives", "[form][form_derivatives][elastic_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim, GENERATE("NeoHookean", "MooneyRivlin3ParamSymbolic"));
	ElasticForm form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		*state_ptr->assembler,
		state_ptr->ass_vals_cache,
		0,
		state_ptr->args["time"]["dt"],
		state_ptr->mesh->is_volume());
	test_form(form, *state_ptr, 1e-7);
}

TEST_CASE("pressure form derivatives", "[form][form_derivatives][pressure_form]")
{
	const int dim = GENERATE(3);
	const bool is_time_dependent = GENERATE(true);
	const auto state_ptr = get_state(dim);
	state_ptr->elasticity_pressure_assembler = state_ptr->build_pressure_assembler();
	PressureForm form(
		state_ptr->n_bases,
		state_ptr->local_pressure_boundary,
		state_ptr->local_pressure_cavity,
		state_ptr->boundary_nodes,
		state_ptr->n_boundary_samples(),
		*state_ptr->elasticity_pressure_assembler,
		is_time_dependent);
	test_form(form, *state_ptr);
}

TEST_CASE("friction form derivatives", "[form][form_derivatives][friction_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	const bool use_convergent_formulation = GENERATE(true, false);
	const double epsv = 1e-3;
	const double mu = GENERATE(0.0, 0.01, 0.1, 1.0);
	const double dhat = 1e-3;
	const double barrier_stiffness = 1e7;
	const bool is_time_dependent = GENERATE(true, false);
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const double dt = 1e-3;

	const bool use_adaptive_barrier_stiffness = true; // GENERATE(true, false);
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);

	const BarrierContactForm contact_form(
		state_ptr->collision_mesh, dhat, state_ptr->avg_mass, use_convergent_formulation,
		use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method,
		ccd_tolerance, ccd_max_iterations);

	FrictionForm form(
		state_ptr->collision_mesh, nullptr, epsv, mu, broad_phase_method, contact_form,
		/*n_lagging_iters=*/-1);

	test_form(form, *state_ptr);
}

TEST_CASE("damping form derivatives", "[form][form_derivatives][damping_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	const double dt = 1e-2;
	std::shared_ptr<assembler::ViscousDamping> damping_assembler = std::make_shared<assembler::ViscousDamping>();
	state_ptr->set_materials(*damping_assembler);

	ElasticForm form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		*damping_assembler,
		state_ptr->ass_vals_cache,
		0,
		dt,
		state_ptr->mesh->is_volume());
	form.update_quantities(0, Eigen::VectorXd::Ones(state_ptr->n_bases * dim));
	test_form(form, *state_ptr);
}

TEST_CASE("inertia form derivatives", "[form][form_derivatives][inertia_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);

	const double dt = 1e-3;
	ImplicitEuler time_integrator;
	time_integrator.init(
		Eigen::VectorXd::Zero(state_ptr->n_bases * dim),
		Eigen::VectorXd::Zero(state_ptr->n_bases * dim),
		Eigen::VectorXd::Zero(state_ptr->n_bases * dim),
		dt);

	InertiaForm form(state_ptr->mass, time_integrator);

	test_form(form, *state_ptr);
}

TEST_CASE("lagged regularization form derivatives", "[form][form_derivatives][lagged_reg_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);

	const double weight = 1e3;
	LaggedRegForm form(/*n_lagging_iters=*/-1);
	form.set_weight(weight);

	test_form(form, *state_ptr);
}

TEST_CASE("Rayleigh damping form derivatives", "[form][form_derivatives][rayleigh_damping_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	ElasticForm elastic_form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		*state_ptr->assembler,
		state_ptr->ass_vals_cache,
		0,
		state_ptr->args["time"]["dt"],
		state_ptr->mesh->is_volume());

	const double dt = 1e-3;
	ImplicitEuler time_integrator;
	time_integrator.init(
		Eigen::VectorXd::Zero(state_ptr->n_bases * dim),
		Eigen::VectorXd::Zero(state_ptr->n_bases * dim),
		Eigen::VectorXd::Zero(state_ptr->n_bases * dim),
		dt);

	RayleighDampingForm form(
		elastic_form, time_integrator, true, 0.1, 1);

	test_form(form, *state_ptr);
}

TEST_CASE("BC lagrangian form derivatives", "[form][form_derivatives][bc_lagr_form]")
{
	static const int n_rand = 10;

	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	const int ndof = state_ptr->n_bases * dim;
	const auto rhs_assembler_ptr = state_ptr->build_rhs_assembler();

	assembler::Mass mass_mat_assembler;
	mass_mat_assembler.set_size(dim);
	StiffnessMatrix mass_tmp;
	state_ptr->mass_matrix_assembler->assemble(dim == 3,
											   state_ptr->n_bases,
											   state_ptr->bases,
											   state_ptr->geom_bases(),
											   state_ptr->mass_ass_vals_cache,
											   0,
											   mass_tmp,
											   true);

	BCLagrangianForm form(
		ndof,
		state_ptr->boundary_nodes,
		state_ptr->local_boundary,
		state_ptr->local_neumann_boundary,
		state_ptr->n_boundary_samples(),
		mass_tmp,
		*rhs_assembler_ptr,
		state_ptr->obstacle.ndof(),
		state_ptr->problem->is_time_dependent(),
		0);

	Eigen::VectorXd x = Eigen::VectorXd::Zero(ndof);
	double k_al = 0;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(1e6, 1e8);

	for (int i = 0; i < n_rand; ++i)
	{
		form.update_lagrangian(x, k_al);
		test_form(form, *state_ptr);

		x.setRandom();
		x /= 100;

		k_al = dis(gen);
	}
}

TEST_CASE("BC penalty form derivatives", "[form][form_derivatives][bc_penalty_form]")
{
	static const int n_rand = 10;

	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	const int ndof = state_ptr->n_bases * dim;
	const auto rhs_assembler_ptr = state_ptr->build_rhs_assembler();

	assembler::Mass mass_mat_assembler;
	mass_mat_assembler.set_size(dim);
	StiffnessMatrix mass_tmp;
	state_ptr->mass_matrix_assembler->assemble(dim == 3,
											   state_ptr->n_bases,
											   state_ptr->bases,
											   state_ptr->geom_bases(),
											   state_ptr->mass_ass_vals_cache,
											   0,
											   mass_tmp,
											   true);

	BCLagrangianForm form(
		ndof,
		state_ptr->boundary_nodes,
		state_ptr->local_boundary,
		state_ptr->local_neumann_boundary,
		state_ptr->n_boundary_samples(),
		mass_tmp,
		*rhs_assembler_ptr,
		state_ptr->obstacle.ndof(),
		state_ptr->problem->is_time_dependent(),
		0);

	test_form(form, *state_ptr);
}

TEST_CASE("Inversion barrier form derivatives", "[form][form_derivatives][inversion_barrier]")
{

	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);

	const double vhat = 1e-3;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	state_ptr->build_mesh_matrices(V, F);

	InversionBarrierForm form(V, F, state_ptr->mesh->dimension(), vhat);

	test_form(form, *state_ptr);
}

TEST_CASE("L2 projection form derivatives", "[form][form_derivatives][L2]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);

	REQUIRE(state_ptr->mass.size() > 0);
	L2ProjectionForm form(state_ptr->mass, state_ptr->mass, Eigen::VectorXd::Ones(state_ptr->mass.cols()));

	test_form(form, *state_ptr);
}

TEST_CASE("AMIPS form derivatives", "[form][form_derivatives][amips_form]")
{
	const int dim = GENERATE(2, 3);
	std::shared_ptr<State> state;
	{
		const std::string path = POLYFEM_DATA_DIR;
		json in_args = R"(
 		{
 			"materials": {
 				"type": "AMIPS",
 				"rho": 1000
 			},
 			"time": {
 				"dt": 0.001,
 				"tend": 1.0
 			},
 			"output": {
 				"log": {
 					"level": "warning"
 				}
 			}
 		})"_json;
		if (dim == 2)
		{
			in_args["geometry"] = R"([{
 				"enabled": true,
 				"surface_selection": 7
 			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";
			in_args["boundary_conditions"] = R"({
 				"dirichlet_boundary": [{
 					"id": "all",
 					"value": [0, 0]
 				}],
 				"rhs": [10, 10]
 			})"_json;
		}
		else
		{
			in_args["geometry"] = R"([{
 				"transformation": {
 					"scale": [0.1, 1, 1]
 				},
 				"surface_selection": [
 					{
 						"id": 1,
 						"axis": "z",
 						"position": 0.8,
 						"relative": true
 					},
 					{
 						"id": 2,
 						"axis": "-z",
 						"position": 0.2,
 						"relative": true
 					},
 					{
 						"id": 3,
 						"box": [[0, 0, 0.2], [1, 1, 0.8]],
 						"relative": true
 					}
 				],
 				"n_refs": 1
 			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-6.msh";
			in_args["boundary_conditions"] = R"({
 				"neumann_boundary": [{
 					"id": 1,
 					"value": [1000, 1000, 1000]
 				}],
 				"pressure_boundary": [{
 					"id": 1,
 					"value": -2000
 				},
 				{
 					"id": 2,
 					"value": -2000
 				},
 				{
 					"id": 3,
 					"value": -2000
 				}],
 				"rhs": [0, 0, 0]
 			})"_json;
		}

		state = std::make_shared<State>();
		state->init(in_args, true);
		state->set_max_threads(1);

		state->load_mesh();

		state->build_basis();
		state->assemble_rhs();
		state->assemble_mass_mat();
	}
	ElasticForm form(
		state->n_bases,
		state->bases,
		state->geom_bases(),
		*state->assembler,
		state->ass_vals_cache,
		0,
		state->args["time"]["dt"],
		state->mesh->is_volume());
	test_form(form, *state);
}

TEST_CASE("Fixed corotational form derivatives", "[form][form_derivatives][elastic_form]")
{
	const int dim = GENERATE(2, 3);
	const auto state_ptr = get_state(dim);
	std::shared_ptr<assembler::FixedCorotational> assembler = std::make_shared<assembler::FixedCorotational>();
	state_ptr->set_materials(*assembler);

	ElasticForm form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		*assembler,
		state_ptr->ass_vals_cache,
		0,
		1,
		state_ptr->mesh->is_volume());
	form.update_quantities(0, Eigen::VectorXd::Ones(state_ptr->n_bases * dim));
	test_form(form, *state_ptr, 1e-7, 1e-4);
}
