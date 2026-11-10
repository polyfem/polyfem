////////////////////////////////////////////////////////////////////////////////

#include <polyfem/State.hpp>
#include "VarFormTestAccess.hpp"

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/assembler/FixedCorotational.hpp>

#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/GCPContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/PressureForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/ESPContactForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/InversionBarrierForm.hpp>
#include <polyfem/solver/forms/L2ProjectionForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>
#include <polyfem/optimization/forms/AMIPSForm.hpp>

#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <finitediff.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::solver;
using namespace polyfem::time_integrator;
using namespace polyfem::assembler;

namespace
{
	struct ElasticFormFixture
	{
		State state;
		std::shared_ptr<varform::ElasticVarForm> formulation;
		test::ElasticFormDebugData data;
		ipc::CollisionMesh collision_mesh;

		explicit ElasticFormFixture(const json &args)
			: formulation(prepare_form(state, args)),
			  data(test::VarFormTestAccess::elastic_form_data(*formulation))
		{
			test::VarFormTestAccess::build_collision_mesh(*formulation, collision_mesh);
		}

	private:
		static std::shared_ptr<varform::ElasticVarForm> prepare_form(State &state, const json &args)
		{
			state.init(args, true);
			state.set_max_threads(1);
			state.load_mesh();
			auto form = std::dynamic_pointer_cast<varform::ElasticVarForm>(state.variational_formulation);
			if (!form)
				throw std::runtime_error("Form derivative fixture requires an elastic VarForm.");
			test::VarFormTestAccess::prepare(*form);
			return form;
		}
	};

	std::shared_ptr<ElasticFormFixture> get_fixture(int dim, const std::string &material_type = "NeoHookean", const bool cavity = false)
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

		if (cavity)
		{
			assert(dim == 2);
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/annulus/annulus.msh";
			in_args["geometry"][0]["surface_selection"] = {{{"id", 7}, {"center", {0, 0}}, {"radius", 0.51}}};
			in_args["boundary_conditions"].erase("dirichlet_boundary");
			in_args["boundary_conditions"]["pressure_cavity"] = {{{"id", 7}, {"value", -1000}}};
		}

		return std::make_shared<ElasticFormFixture>(in_args);
	}
} // namespace

template <typename Form>
void test_form(Form &form, const ElasticFormFixture &fixture, double step = 1e-8, double tol = 1e-4, const int n_rand = 10)
{
	Eigen::VectorXd x = Eigen::VectorXd::Zero(fixture.data.space.ndof());

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
				logger().debug("Gradient mismatch");
				logger().debug("Gradient: {}", grad.transpose());
				logger().debug("Finite gradient: {}", fgrad.transpose());
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
				logger().debug("Hessian mismatch");
				logger().debug("Hessian: {}", hess);
				logger().debug("Finite hessian: {}", fhess);
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
	const auto fixture = get_fixture(dim);
	const auto rhs_assembler_ptr = fixture->data.rhs_assembler;
	const int ndof = fixture->data.space.n_bases * fixture->data.mesh.dimension();

	BodyForm form(ndof, /*n_pressure_bases=*/0,
				  fixture->data.boundary.boundary_nodes,
				  fixture->data.boundary.local_boundary,
				  fixture->data.boundary.local_neumann_boundary,
				  fixture->data.boundary_samples,
				  fixture->data.rhs,
				  *rhs_assembler_ptr,
				  fixture->data.mass_assembler.density(),
				  false, fixture->data.problem.is_time_dependent());
	Eigen::VectorXd x_prev;
	x_prev.setRandom(ndof);
	x_prev /= 100.;
	form.update_quantities(fixture->state.args["time"]["t0"].get<double>() + 5 * fixture->state.args["time"]["dt"].get<double>(), x_prev);

	test_form(form, *fixture);
}

TEST_CASE("body form derivatives", "[form][form_derivatives][body_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	const auto rhs_assembler_ptr = fixture->data.rhs_assembler;
	const int ndof = fixture->data.space.n_bases * fixture->data.mesh.dimension();

	BodyForm form(ndof, /*n_pressure_bases=*/0,
				  fixture->data.boundary.boundary_nodes,
				  fixture->data.boundary.local_boundary,
				  fixture->data.boundary.local_neumann_boundary,
				  fixture->data.boundary_samples,
				  fixture->data.rhs,
				  *rhs_assembler_ptr,
				  fixture->data.mass_assembler.density(),
				  false, fixture->data.problem.is_time_dependent());

	Eigen::VectorXd x_prev;
	x_prev.setRandom(fixture->data.space.n_bases * dim);
	x_prev /= 100.;
	form.update_quantities(fixture->state.args["time"]["t0"].get<double>() + 5 * fixture->state.args["time"]["dt"].get<double>(), x_prev);

	test_form(form, *fixture);
}

TEST_CASE("barrier contact form derivatives", "[form][form_derivatives][contact_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

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
		fixture->collision_mesh, dhat, fixture->data.average_mass,
		use_convergent_formulation, use_convergent_formulation, use_convergent_formulation,
		use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method,
		ccd_tolerance, ccd_max_iterations, /*dhat_epsilon_scale=*/1e-7);
	form.set_barrier_stiffness(barrier_stiffness);

	test_form(form, *fixture);
}

TEST_CASE("ESP contact form derivatives", "[form][form_derivatives][contact_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

	const double dhat = 1e-3;
	const double barrier_stiffness = 1e7;
	const bool is_time_dependent = GENERATE(true, false);
	const bool use_adaptive_barrier_stiffness = false;
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);
	// Order 1 is vertex quadrature; a higher order exercises the fixed rule on
	// the reference element, which is the part that differs from IPC.
	const int quadrature_order = GENERATE(1, 5);
	const bool area_weights = GENERATE(true, false);
	const json esp_params = json::object(
		{{"quadrature_order", quadrature_order},
		 {"dbar_factor", 1.0},
		 {"normalize_weights", false},
		 {"area_weights", area_weights}});

	ESPContactForm form(
		fixture->collision_mesh, dhat, fixture->data.average_mass, esp_params,
		/*skip_obstacles=*/false, /*barrier=*/nullptr, /*use_adaptive_dhat=*/false,
		use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method,
		ccd_tolerance, ccd_max_iterations, /*dhat_epsilon_scale=*/1e-7);
	form.set_barrier_stiffness(barrier_stiffness);

	test_form(form, *fixture);
}

TEST_CASE("smooth contact form derivatives", "[form][form_derivatives][contact_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

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
	const json contact_args = json::object({{"a", a}, {"alpha_t", alpha}, {"alpha_n", 0.1}, {"beta_t", 0}, {"beta_n", 0}, {"dhat", dhat}, {"use_adaptive_dhat", false}, {"min_distance_ratio", 0.5}});

	GCPContactForm form(
		fixture->collision_mesh, dhat, fixture->data.average_mass, contact_args["alpha_t"], contact_args["alpha_n"], contact_args["use_adaptive_dhat"], contact_args["min_distance_ratio"],
		use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method,
		ccd_tolerance, ccd_max_iterations, /*dhat_epsilon_scale=*/1e-7);

	test_form(form, *fixture);
}

TEST_CASE("elastic form derivatives", "[form][form_derivatives][elastic_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim, GENERATE("NeoHookean", "MooneyRivlin3ParamSymbolic"));
	ElasticForm form(
		fixture->data.space.n_bases,
		fixture->data.bases,
		fixture->data.space.geometry_basis_list(),
		fixture->data.assembler,
		fixture->data.assembly_cache,
		0,
		fixture->state.args["time"]["dt"],
		fixture->data.mesh.is_volume());
	test_form(form, *fixture, 1e-7);
}

TEST_CASE("pressure form derivatives", "[form][form_derivatives][pressure_form]")
{
	const int dim = GENERATE(2, 3);
	const bool is_time_dependent = GENERATE(true);
	const auto fixture = get_fixture(dim);
	const auto pressure_assembler = test::VarFormTestAccess::build_pressure_assembler(*fixture->formulation);
	PressureForm form(
		fixture->data.space.n_bases,
		fixture->data.boundary.local_pressure_boundary,
		fixture->data.boundary.local_pressure_cavity,
		fixture->data.boundary.boundary_nodes,
		fixture->data.boundary_samples,
		*pressure_assembler,
		is_time_dependent);
	test_form(form, *fixture);
}

TEST_CASE("cavity pressure form derivatives", "[form][form_derivatives][pressure_form]")
{
	const auto fixture = get_fixture(2, "NeoHookean", true);
	REQUIRE(fixture->data.boundary.local_pressure_boundary.empty());
	REQUIRE(fixture->data.boundary.local_pressure_cavity.size() == 1);
	REQUIRE_FALSE(fixture->data.boundary.local_pressure_cavity.at(7).empty());
	const auto pressure_assembler = test::VarFormTestAccess::build_pressure_assembler(*fixture->formulation);
	PressureForm form(
		fixture->data.space.n_bases,
		fixture->data.boundary.local_pressure_boundary,
		fixture->data.boundary.local_pressure_cavity,
		fixture->data.boundary.boundary_nodes,
		fixture->data.boundary_samples,
		*pressure_assembler,
		true);
	const Eigen::VectorXd x = Eigen::VectorXd::Zero(fixture->data.space.ndof());
	const Eigen::VectorXd direction = Eigen::VectorXd::Random(x.size()).normalized();
	const double step = 1e-5;
	REQUIRE(std::isfinite(form.value(x)));
	Eigen::VectorXd grad, grad_plus, grad_minus;
	form.first_derivative(x, grad);
	REQUIRE(grad.norm() > 0);
	const double finite_directional_gradient = (form.value(x + step * direction) - form.value(x - step * direction)) / (2 * step);
	CHECK(std::abs(grad.dot(direction) - finite_directional_gradient) < 1e-4);

	StiffnessMatrix hess;
	form.second_derivative(x, hess);
	form.first_derivative(x + step * direction, grad_plus);
	form.first_derivative(x - step * direction, grad_minus);
	const Eigen::VectorXd hess_direction = hess * direction;
	CHECK((hess_direction - (grad_plus - grad_minus) / (2 * step)).norm() < 1e-4 * std::max(1.0, hess_direction.norm()));
}

namespace
{
	void test_friction_form_derivatives(
		const int dim,
		const bool use_convergent_formulation)
	{
		const auto fixture = get_fixture(dim);
		const double epsv = 1e-3;
		const double dhat = 1e-3;
		const double barrier_stiffness = 1e7;
		const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;

		const bool use_adaptive_barrier_stiffness = true; // GENERATE(true, false);
		const double ccd_tolerance = 1e-6;
		const int ccd_max_iterations = static_cast<int>(1e6);

		for (const double mu : {0.0, 0.01, 0.1, 1.0})
			for (const bool is_time_dependent : {true, false})
			{
				CAPTURE(dim, use_convergent_formulation, mu, is_time_dependent);

				BarrierContactForm contact_form(
					fixture->collision_mesh, dhat, fixture->data.average_mass,
					use_convergent_formulation, use_convergent_formulation,
					use_convergent_formulation, use_adaptive_barrier_stiffness,
					is_time_dependent, false, broad_phase_method,
					ccd_tolerance, ccd_max_iterations, /*dhat_epsilon_scale=*/1e-7);
				contact_form.set_barrier_stiffness(barrier_stiffness);

				FrictionForm form(
					fixture->collision_mesh, nullptr, epsv, mu,
					broad_phase_method, contact_form,
					/*n_lagging_iters=*/-1);

				test_form(form, *fixture, 1e-8, 1e-4, 3);
			}
	}
} // namespace

TEST_CASE("smooth contact friction form derivatives", "[form][form_derivatives][friction_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	const double epsv = 1e-3;
	const double mu = GENERATE(0.0, 0.01, 0.1, 1.0);
	const double dhat = 1e-3;
	const bool is_time_dependent = GENERATE(true, false);
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const bool use_adaptive_barrier_stiffness = false;
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);
	const json contact_args = json::object({{"alpha_t", 0.2}, {"alpha_n", 0.1}, {"dhat", dhat}, {"use_adaptive_dhat", false}, {"min_distance_ratio", 0.5}});

	const GCPContactForm contact_form(
		fixture->collision_mesh, dhat, fixture->data.average_mass,
		contact_args["alpha_t"], contact_args["alpha_n"], contact_args["use_adaptive_dhat"], contact_args["min_distance_ratio"],
		use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method,
		ccd_tolerance, ccd_max_iterations, /*dhat_epsilon_scale=*/1e-7);

	FrictionForm form(
		fixture->collision_mesh, nullptr, epsv, mu, broad_phase_method, contact_form,
		/*n_lagging_iters=*/-1);

	test_form(form, *fixture);
}

TEST_CASE("high-order contact friction form derivatives", "[form][form_derivatives][friction_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	const double epsv = 1e-3;
	const double mu = GENERATE(0.0, 0.01, 0.1, 1.0);
	const double dhat = 1e-3;
	const bool is_time_dependent = GENERATE(true, false);
	const ipc::BroadPhaseMethod broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
	const bool use_adaptive_barrier_stiffness = false;
	const double ccd_tolerance = 1e-6;
	const int ccd_max_iterations = static_cast<int>(1e6);
	const json esp_params = json::object({{"quadrature_order", 5}, {"dbar_factor", 1.0}, {"normalize_weights", false}, {"area_weights", true}});

	const ESPContactForm contact_form(
		fixture->collision_mesh, dhat, fixture->data.average_mass, esp_params, /*skip_obstacles=*/false, /*barrier=*/nullptr,
		/*use_adaptive_dhat=*/false, use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method,
		ccd_tolerance, ccd_max_iterations, /*dhat_epsilon_scale=*/1e-7);

	FrictionForm form(
		fixture->collision_mesh, nullptr, epsv, mu, broad_phase_method, contact_form,
		/*n_lagging_iters=*/-1);

	test_form(form, *fixture);
}

TEST_CASE("friction form derivatives 2d convergent", "[form][form_derivatives][friction_form]")
{
	test_friction_form_derivatives(2, true);
}

TEST_CASE("friction form derivatives 2d nonconvergent", "[form][form_derivatives][friction_form]")
{
	test_friction_form_derivatives(2, false);
}

TEST_CASE("friction form derivatives 3d convergent", "[form][form_derivatives][friction_form]")
{
	test_friction_form_derivatives(3, true);
}

TEST_CASE("friction form derivatives 3d nonconvergent", "[form][form_derivatives][friction_form]")
{
	test_friction_form_derivatives(3, false);
}

TEST_CASE("damping form derivatives", "[form][form_derivatives][damping_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	const double dt = 1e-2;
	std::shared_ptr<assembler::ViscousDamping> damping_assembler = std::make_shared<assembler::ViscousDamping>();
	test::VarFormTestAccess::set_materials(*fixture->formulation, *damping_assembler);

	ElasticForm form(
		fixture->data.space.n_bases,
		fixture->data.bases,
		fixture->data.space.geometry_basis_list(),
		*damping_assembler,
		fixture->data.assembly_cache,
		0,
		dt,
		fixture->data.mesh.is_volume());
	form.update_quantities(0, Eigen::VectorXd::Ones(fixture->data.space.n_bases * dim));
	test_form(form, *fixture);
}

TEST_CASE("inertia form derivatives", "[form][form_derivatives][inertia_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

	const double dt = 1e-3;
	ImplicitEuler time_integrator;
	time_integrator.init(
		Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim),
		Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim),
		Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim),
		dt);

	InertiaForm form(fixture->data.mass, time_integrator);
	form.update_quantities(0, Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim));

	test_form(form, *fixture);
}

TEST_CASE("lagged regularization form derivatives", "[form][form_derivatives][lagged_reg_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

	const double weight = 1e3;
	LaggedRegForm form(/*n_lagging_iters=*/-1);
	form.set_weight(weight);

	test_form(form, *fixture);
}

TEST_CASE("Rayleigh damping form derivatives", "[form][form_derivatives][rayleigh_damping_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	ElasticForm elastic_form(
		fixture->data.space.n_bases,
		fixture->data.bases,
		fixture->data.space.geometry_basis_list(),
		fixture->data.assembler,
		fixture->data.assembly_cache,
		0,
		fixture->state.args["time"]["dt"],
		fixture->data.mesh.is_volume());

	const double dt = 1e-3;
	ImplicitEuler time_integrator;
	time_integrator.init(
		Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim),
		Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim),
		Eigen::VectorXd::Zero(fixture->data.space.n_bases * dim),
		dt);

	RayleighDampingForm form(
		elastic_form, time_integrator, true, 0.1, 1);

	test_form(form, *fixture);
}

TEST_CASE("BC lagrangian form derivatives", "[form][form_derivatives][bc_lagr_form]")
{
	static const int n_rand = 10;

	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	const int ndof = fixture->data.space.n_bases * dim;
	const auto rhs_assembler_ptr = fixture->data.rhs_assembler;

	assembler::Mass mass_mat_assembler;
	mass_mat_assembler.set_size(dim);
	StiffnessMatrix mass_tmp;
	fixture->data.mass_assembler.assemble(dim == 3,
										  fixture->data.space.n_bases,
										  fixture->data.bases,
										  fixture->data.space.geometry_basis_list(),
										  fixture->data.mass_assembly_cache,
										  0,
										  mass_tmp,
										  true);

	BCLagrangianForm form(
		ndof,
		fixture->data.boundary.boundary_nodes,
		fixture->data.boundary.local_boundary,
		fixture->data.boundary.local_neumann_boundary,
		fixture->data.boundary_samples,
		mass_tmp,
		*rhs_assembler_ptr,
		fixture->formulation->output_space().obstacle->ndof(),
		fixture->data.problem.is_time_dependent(),
		0);

	Eigen::VectorXd x = Eigen::VectorXd::Zero(ndof);
	double k_al = 0;

	std::mt19937 gen(42);
	std::uniform_real_distribution<> dis(1e6, 1e8);

	for (int i = 0; i < n_rand; ++i)
	{
		form.update_lagrangian(x, k_al);
		test_form(form, *fixture);

		x.setRandom();
		x /= 100;

		k_al = dis(gen);
	}
}

TEST_CASE("BC penalty form derivatives", "[form][form_derivatives][bc_penalty_form]")
{
	static const int n_rand = 10;

	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	const int ndof = fixture->data.space.n_bases * dim;
	const auto rhs_assembler_ptr = fixture->data.rhs_assembler;

	assembler::Mass mass_mat_assembler;
	mass_mat_assembler.set_size(dim);
	StiffnessMatrix mass_tmp;
	fixture->data.mass_assembler.assemble(dim == 3,
										  fixture->data.space.n_bases,
										  fixture->data.bases,
										  fixture->data.space.geometry_basis_list(),
										  fixture->data.mass_assembly_cache,
										  0,
										  mass_tmp,
										  true);

	BCLagrangianForm form(
		ndof,
		fixture->data.boundary.boundary_nodes,
		fixture->data.boundary.local_boundary,
		fixture->data.boundary.local_neumann_boundary,
		fixture->data.boundary_samples,
		mass_tmp,
		*rhs_assembler_ptr,
		fixture->formulation->output_space().obstacle->ndof(),
		fixture->data.problem.is_time_dependent(),
		0);

	test_form(form, *fixture);
}

TEST_CASE("Inversion barrier form derivatives", "[form][form_derivatives][inversion_barrier]")
{

	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

	const double vhat = 1e-3;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	test::VarFormTestAccess::build_mesh_matrices(*fixture->formulation, V, F);

	InversionBarrierForm form(V, F, fixture->data.mesh.dimension(), vhat);

	test_form(form, *fixture);
}

TEST_CASE("L2 projection form derivatives", "[form][form_derivatives][L2]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);

	REQUIRE(fixture->data.mass.size() > 0);
	L2ProjectionForm form(fixture->data.mass, fixture->data.mass, Eigen::VectorXd::Ones(fixture->data.mass.cols()));

	test_form(form, *fixture);
}

TEST_CASE("AMIPS form derivatives", "[form][form_derivatives][amips_form]")
{
	const int dim = GENERATE(2, 3);
	std::shared_ptr<ElasticFormFixture> fixture;
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

		fixture = std::make_shared<ElasticFormFixture>(in_args);
	}
	ElasticForm form(
		fixture->data.space.n_bases,
		fixture->data.bases,
		fixture->data.space.geometry_basis_list(),
		fixture->data.assembler,
		fixture->data.assembly_cache,
		0,
		fixture->state.args["time"]["dt"],
		fixture->data.mesh.is_volume());
	test_form(form, *fixture);
}

TEST_CASE("Fixed corotational form derivatives", "[form][form_derivatives][elastic_form]")
{
	const int dim = GENERATE(2, 3);
	const auto fixture = get_fixture(dim);
	std::shared_ptr<assembler::FixedCorotational> assembler = std::make_shared<assembler::FixedCorotational>();
	test::VarFormTestAccess::set_materials(*fixture->formulation, *assembler);

	ElasticForm form(
		fixture->data.space.n_bases,
		fixture->data.bases,
		fixture->data.space.geometry_basis_list(),
		*assembler,
		fixture->data.assembly_cache,
		0,
		1,
		fixture->data.mesh.is_volume());
	form.update_quantities(0, Eigen::VectorXd::Ones(fixture->data.space.n_bases * dim));
	// at x = 0 the singular values of F are all equal and finite differences with
	// step 1e-7 are dominated by round-off (~1e-4 on entries that are exactly 0)
	test_form(form, *fixture, 1e-6, 1e-4);
}
