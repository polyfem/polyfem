////////////////////////////////////////////////////////////////////////////////
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>

#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <finitediff.hpp>

#include <polyfem/State.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::solver;
using namespace polyfem::time_integrator;
using namespace polyfem::assembler;

namespace
{
	std::shared_ptr<State> get_state()
	{
		const std::string path = POLYFEM_DATA_DIR;
		json in_args = R"(
		{
			"materials": {
                "type": "NeoHookean",
                "E": 20000,
                "nu": 0.3,
                "rho": 1000,
				"phi": 1,
				"psi": 1
            },

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh",
				"surface_selection": 7
			}],

			"time": {
				"dt": 0.001,
				"tend": 1.0
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": [0, 0]
				}],
				"rhs": [10, 10]
			},

			"output": {
				"log": {
					"level": "warning"
				}
			}

		})"_json;
		in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";

		auto state = std::make_shared<State>();
		state->init(in_args, true);
		state->set_max_threads(1);

		state->load_mesh();

		state->build_basis();
		state->assemble_rhs();
		state->assemble_stiffness_mat();

		return state;
	}
} // namespace

template <typename Form>
void test_form(Form &form, const State &state)
{
	static const int n_rand = 10;

	Eigen::VectorXd x = Eigen::VectorXd::Zero(state.n_bases * 2);

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
				x, [&form](const Eigen::VectorXd &x) -> double { return form.value(x); }, fgrad);

			if (!fd::compare_gradient(grad, fgrad))
			{
				std::cout << "Gradient mismatch" << std::endl;
				std::cout << "Gradient: " << grad.transpose() << std::endl;
				std::cout << "Finite gradient: " << fgrad.transpose() << std::endl;
			}

			CHECK(fd::compare_gradient(grad, fgrad));
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
				fhess);

			if (!fd::compare_hessian(hess, fhess))
			{
				std::cout << "Hessian mismatch" << std::endl;
				std::cout << "Hessian: " << hess << std::endl;
				std::cout << "Finite hessian: " << fhess << std::endl;
			}

			CHECK(fd::compare_hessian(hess, fhess));
		}

		x.setRandom();
		x /= 100;
	}
}

TEST_CASE("body form derivatives", "[form][form_derivatives][body_form]")
{
	const auto state_ptr = get_state();
	const auto rhs_assembler_ptr = state_ptr->build_rhs_assembler();
	const bool apply_DBC = false; // GENERATE(true, false);
	const int ndof = state_ptr->n_bases * state_ptr->mesh->dimension();

	BodyForm form(ndof, state_ptr->n_pressure_bases,
				  state_ptr->boundary_nodes,
				  state_ptr->local_boundary,
				  state_ptr->local_neumann_boundary, state_ptr->n_boundary_samples(),
				  state_ptr->rhs,
				  *rhs_assembler_ptr,
				  state_ptr->assembler.density(),
				  apply_DBC, false, state_ptr->problem->is_time_dependent());
	form.update_quantities(state_ptr->args["time"]["t0"].get<double>(), Eigen::VectorXd());

	CAPTURE(apply_DBC);
	test_form(form, *state_ptr);
}

TEST_CASE("contact form derivatives", "[form][form_derivatives][contact_form]")
{
	const auto state_ptr = get_state();

	const double dhat = 1e-3;
	const bool use_adaptive_barrier_stiffness = true; // GENERATE(true, false);
	const bool use_convergent_formulation = GENERATE(true, false);
	const double barrier_stiffness = 1e7;
	const bool is_time_dependent = GENERATE(true, false);
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);
	const double dt = 1e-3;

	ContactForm form(
		state_ptr->collision_mesh, dhat, state_ptr->avg_mass,
		use_convergent_formulation, use_adaptive_barrier_stiffness,
		is_time_dependent, broad_phase_method, ccd_tolerance,
		ccd_max_iterations);

	test_form(form, *state_ptr);
}

TEST_CASE("elastic form derivatives", "[form][form_derivatives][elastic_form]")
{
	const auto state_ptr = get_state();
	ElasticForm form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		state_ptr->assembler,
		state_ptr->ass_vals_cache,
		state_ptr->formulation(),
		state_ptr->args["time"]["dt"],
		state_ptr->mesh->is_volume());
	test_form(form, *state_ptr);
}

TEST_CASE("friction form derivatives", "[form][form_derivatives][friction_form]")
{
	const auto state_ptr = get_state();
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

	const ContactForm contact_form(
		state_ptr->collision_mesh, dhat, state_ptr->avg_mass, use_convergent_formulation,
		use_adaptive_barrier_stiffness, is_time_dependent, broad_phase_method,
		ccd_tolerance, ccd_max_iterations);

	FrictionForm form(
		state_ptr->collision_mesh, epsv, mu, dhat, broad_phase_method, dt,
		contact_form, /*n_lagging_iters=*/-1);

	test_form(form, *state_ptr);
}

TEST_CASE("damping form derivatives", "[form][form_derivatives][damping_form]")
{
	const auto state_ptr = get_state();
	const double dt = 1e-2;

	ElasticForm form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		state_ptr->assembler,
		state_ptr->ass_vals_cache,
		"Damping",
		dt,
		state_ptr->mesh->is_volume());
	form.update_quantities(0, Eigen::VectorXd::Ones(state_ptr->n_bases * 2));
	test_form(form, *state_ptr);
}

TEST_CASE("inertia form derivatives", "[form][form_derivatives][inertia_form]")
{
	const auto state_ptr = get_state();

	const double dt = 1e-3;
	ImplicitEuler time_integrator;
	time_integrator.init(
		Eigen::VectorXd::Zero(state_ptr->n_bases * 2),
		Eigen::VectorXd::Zero(state_ptr->n_bases * 2),
		Eigen::VectorXd::Zero(state_ptr->n_bases * 2),
		dt);

	InertiaForm form(state_ptr->mass, time_integrator);

	test_form(form, *state_ptr);
}

TEST_CASE("lagged regularization form derivatives", "[form][form_derivatives][lagged_reg_form]")
{
	const auto state_ptr = get_state();

	const double weight = 1e3;
	LaggedRegForm form(/*n_lagging_iters=*/-1);
	form.set_weight(weight);

	test_form(form, *state_ptr);
}

TEST_CASE("Rayleigh damping form derivatives", "[form][form_derivatives][rayleigh_damping_form]")
{
	const auto state_ptr = get_state();
	ElasticForm elastic_form(
		state_ptr->n_bases,
		state_ptr->bases,
		state_ptr->geom_bases(),
		state_ptr->assembler,
		state_ptr->ass_vals_cache,
		state_ptr->formulation(),
		state_ptr->args["time"]["dt"],
		state_ptr->mesh->is_volume());

	const double dt = 1e-3;
	ImplicitEuler time_integrator;
	time_integrator.init(
		Eigen::VectorXd::Zero(state_ptr->n_bases * 2),
		Eigen::VectorXd::Zero(state_ptr->n_bases * 2),
		Eigen::VectorXd::Zero(state_ptr->n_bases * 2),
		dt);

	RayleighDampingForm form(
		elastic_form, time_integrator, true, 0.1, 1);

	test_form(form, *state_ptr);
}