#include <polyfem/State.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/forms/SmoothingForms.hpp>
#include <polyfem/optimization/forms/TargetForms.hpp>
#include <polyfem/optimization/parametrization/Parametrizations.hpp>
#include <polyfem/optimization/parametrization/NodeCompositeParametrizations.hpp>
#include <polyfem/optimization/force_derivatives/BarrierContactForceDerivative.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/potentials/barrier_potential.hpp>
#include <ipc/collision_mesh.hpp>

#include <finitediff.hpp>
#include <catch2/catch_all.hpp>

#include <cmath>
#include <fstream>
#include <cstdlib>

// Tests here are *unstable*.
//
// There are two levels of randomness:
// 1. We generate radom perturbation (velocity) using Catch2 seed.
// 2. Polyfem itself is non-deterministic. Even test with the same
//    rng seed might fail randomly. Based on my test, non-deterministic
//    behavior only happens when multi-threading is enabled.
//    So I am guessing the culprit is non-associative floating point reduction.
//
// If you encouter test failure here don't panic. Try:
// 1. First fix the perturbation by specifying --rng-seed
// 2. If still non-deterministic, set max thread to 1.
// 3. Then decide if this is an actual regression.

using namespace polyfem;
using namespace solver;

namespace
{
	std::string append_root_path(const std::string &path)
	{
		return POLYFEM_DIFF_DIR + std::string("/input/") + path;
	}

	/// @brief Load config json and patch root_path.
	bool load_json(const std::string &json_file, json &out)
	{
		std::ifstream file(json_file);

		if (!file.is_open())
		{
			return false;
		}
		file >> out;

		out["root_path"] = json_file;

		return true;
	}

	void sample_field(const State &state, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> field, Eigen::MatrixXd &discrete_field, const int order = 1)
	{
		Eigen::MatrixXd tmp;
		tmp.setZero(1, state.mesh->dimension());
		tmp = field(tmp);
		const int actual_dim = tmp.cols();

		if (order >= 1)
		{
			Eigen::MatrixXd V;
			state.get_vertices(V);

			discrete_field = utils::flatten(field(V));
		}
		else if (order == 0)
		{
			Eigen::MatrixXd centers;
			if (state.mesh->is_volume())
				state.mesh->cell_barycenters(centers);
			else
				state.mesh->face_barycenters(centers);

			discrete_field = utils::flatten(field(centers));
		}
	}

	Eigen::MatrixXd unflatten(const Eigen::VectorXd &x, int dim)
	{
		if (x.size() == 0)
			return Eigen::MatrixXd(0, dim);

		assert(x.size() % dim == 0);
		Eigen::MatrixXd X(x.size() / dim, dim);
		for (int i = 0; i < x.size(); ++i)
		{
			X(i / dim, i % dim) = x(i);
		}
		return X;
	}

	void verify_adjoint(AdjointNLProblem &problem, const Eigen::VectorXd &x, const Eigen::MatrixXd &theta, const double dt, const double tol)
	{
		problem.solution_changed(x);
		problem.save_to_file(0, x);
		double functional_val = problem.value(x);

		Eigen::VectorXd one_form;
		problem.gradient(x, one_form);
		double derivative = (one_form.array() * theta.array()).sum();

		problem.solution_changed(x + theta * dt);
		double next_functional_val = problem.value(x + theta * dt);

		problem.solution_changed(x - theta * dt);
		double former_functional_val = problem.value(x - theta * dt);

		double finite_difference = (next_functional_val - former_functional_val) / dt / 2;
		double back_finite_difference = (functional_val - former_functional_val) / dt;
		double front_finite_difference = (next_functional_val - functional_val) / dt;
		logger().trace("f(x) {:.16f} f(x-dt) {:.16f} f(x+dt) {:.16f}", functional_val, former_functional_val, next_functional_val);
		logger().trace("forward fd {:.16f} backward fd {:.16f}", front_finite_difference, back_finite_difference);
		logger().trace("derivative: {:.12f} fd: {:.12f}", derivative, finite_difference);
		logger().trace("relative error: {:.12f}", abs((finite_difference - derivative) / derivative));
		REQUIRE(derivative == Catch::Approx(finite_difference).epsilon(tol));
	}

	/// @brief Verify ‖∇f‖ against a centered finite difference along ∇f/‖∇f‖.
	///
	/// This catches errors in the *direction* of the gradient itself (the
	/// "worst case" direction) — random theta in `verify_adjoint` can miss bugs
	/// that only manifest along the gradient direction.
	void verify_adjoint_along_gradient(AdjointNLProblem &problem, const Eigen::VectorXd &x, const double dt, const double tol)
	{
		problem.solution_changed(x);
		double functional_val = problem.value(x);

		Eigen::VectorXd one_form;
		problem.gradient(x, one_form);
		const double grad_norm = one_form.norm();
		REQUIRE(grad_norm > 0);

		Eigen::VectorXd theta = one_form / grad_norm;

		problem.solution_changed(x + theta * dt);
		double next_functional_val = problem.value(x + theta * dt);
		problem.solution_changed(x - theta * dt);
		double former_functional_val = problem.value(x - theta * dt);

		const double fd_centered = (next_functional_val - former_functional_val) / dt / 2;

		logger().trace("[directional] ||grad|| = {:.6e}, fd_centered = {:.6e}, rel_err = {:.3e}",
		              grad_norm, fd_centered, std::abs(grad_norm - fd_centered) / std::abs(fd_centered));

		REQUIRE(grad_norm == Catch::Approx(fd_centered).epsilon(tol));
	}

	class TestContext
	{
	public:
		json args;
		std::vector<std::shared_ptr<State>> states;
		std::vector<std::shared_ptr<DiffCache>> diff_caches;
		int ndof;
		std::vector<int> var_sizes;
		std::shared_ptr<AdjointForm> form;
		VariableToSimulationGroup var2sim;
		// Use unique_ptr here because it has no default ctor by design.
		std::unique_ptr<AdjointNLProblem> problem;

		TestContext(const std::string &json_name)
		{
			load_json(append_root_path(json_name), args);
			args = AdjointOptUtils::apply_opt_json_spec(args, false);

			// Build states.
			std::string root = POLYFEM_DIFF_DIR + std::string("/input/");
			states = from_json::build_states(root, args["states"], -1);

			// Build diff_caches.
			diff_caches.resize(states.size());
			for (auto &cache : diff_caches)
			{
				cache = std::make_shared<DiffCache>();
			}

			// Build dof and variable sizes.
			ndof = 0;
			for (const auto &arg : args["parameters"])
			{
				int size = AdjointOptUtils::compute_variable_size(arg, states);
				ndof += size;
				var_sizes.push_back(size);
			}

			// Build variable to simulation.
			var2sim =
				from_json::build_variable_to_simulation_group(args["variable_to_simulation"], states, diff_caches, var_sizes);

			// Build forms.
			form = from_json::build_form(
				args["functionals"], var2sim, states, diff_caches);

			// Build adjoint NL problem.
			problem = std::make_unique<AdjointNLProblem>(form, var2sim, states, diff_caches, args);
		}
	};

	/// @brief Run optimization test variant 1.
	///
	/// 1. Build test context.
	/// 2. Compute initial guess via inverse_evaluation.
	/// 3. Gen random velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	/// @param[in] rand_min Minimum random value.
	/// @param[in] rand_max Maximum random value.
	void run_test1(
		const std::string &json_name,
		float dt,
		float tol,
		double rand_min,
		double rand_max)
	{
		TestContext ctx{json_name};

		// Compute initial solution.
		Eigen::VectorXd x =
			AdjointOptUtils::inverse_evaluation(ctx.args["parameters"], ctx.ndof, ctx.var_sizes, ctx.var2sim);

		// Gen random velocity using reproducible Catch2 seed.
		std::srand(Catch::rngSeed());
		Eigen::MatrixXd velocity =
			((Eigen::MatrixXd::Random(x.size(), 1).array() + 1.0) * 0.5) * (rand_max - rand_min) + rand_min;

		verify_adjoint(*ctx.problem, x, velocity, dt, tol);
	}

	/// @brief Run optimization test variant 2.
	///
	/// Assume only a single state exists.
	/// 1. Build test context.
	/// 2. Use vertex position as initial guess.
	/// 3. Gen random velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	/// @param[in] rand_min Minimum random value.
	/// @param[in] rand_max Maximum random value.
	void run_test2(
		const std::string &json_name,
		float dt,
		float tol,
		double rand_min,
		double rand_max)
	{
		TestContext ctx{json_name};

		// Compute initial solution.
		Eigen::MatrixXd V;
		ctx.states[0]->get_vertices(V);
		Eigen::VectorXd x = utils::flatten(V);

		// Gen random velocity using reproducible Catch2 seed.
		std::srand(Catch::rngSeed());
		Eigen::MatrixXd velocity =
			((Eigen::MatrixXd::Random(x.size(), 1).array() + 1.0) * 0.5) * (rand_max - rand_min) + rand_min;

		verify_adjoint(*ctx.problem, x, velocity, dt, tol);
	}

	/// @brief Run optimization test variant 3.
	///
	/// Assume only a single state exists.
	/// 1. Build test context.
	/// 2. Use vertex position as initial guess.
	/// 3. Set gradient as velocity.
	/// 4. Check convergence through verify_adjoint.
	///
	/// @param[in] json Filename of the input json config.
	/// @param[in] dt Time step.
	/// @param[in] tol Convergence check tolerance.
	void run_test3(const std::string &json_name, float dt, float tol)
	{
		TestContext ctx{json_name};

		// Compute initial solution.
		Eigen::MatrixXd V;
		ctx.states[0]->get_vertices(V);
		Eigen::VectorXd x = utils::flatten(V);

		ctx.problem->solution_changed(x);
		Eigen::VectorXd one_form;
		ctx.problem->gradient(x, one_form);

		verify_adjoint(*ctx.problem, x, one_form.normalized(), dt, tol);
	}

} // namespace

TEST_CASE("laplacian", "[test_adjoint]")
{
	TestContext ctx{"laplacian-opt.json"};

	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();
	ctx.var2sim.update(x);

	auto velocity = [](const Eigen::MatrixXd &position) {
		auto vel = position;
		for (int i = 0; i < vel.size(); i++)
		{
			vel(i) = vel(i) * std::cos(vel(i));
		}
		return vel;
	};
	Eigen::MatrixXd velocity_discrete;
	sample_field(*(ctx.states[0]), velocity, velocity_discrete);

	verify_adjoint(*ctx.problem, x, velocity_discrete, 1e-7, 1e-8);
}

TEST_CASE("linear_elasticity-surface-3d", "[test_adjoint]")
{
	run_test2("linear_elasticity-surface-3d-opt.json", 1e-7, 1e-5, -1.0, 1.0);
}

TEST_CASE("linear_elasticity-surface", "[test_adjoint]")
{
	run_test2("linear_elasticity-surface-opt.json", 1e-6, 1e-5, -1.0, 1.0);
}

TEST_CASE("topology-compliance", "[test_adjoint]")
{
	run_test1("topology-compliance-opt.json", 1e-2, 1e-4, 0.0, 1.0);
}

#if defined(NDEBUG) && !defined(WIN32)
std::string tagsdiff = "[test_adjoint]";
#else
std::string tagsdiff = "[.][test_adjoint]";
#endif

TEST_CASE("neohookean-stress-3d", tagsdiff)
{
	run_test2("neohookean-stress-3d-opt.json", 1e-7, 1e-4, -1.0, 1.0);
}

TEST_CASE("shape-neumann-nodes", "[test_adjoint]")
{
	run_test1("shape-neumann-nodes-opt.json", 1e-7, 1e-2, 0.0, 1.0);
}

TEST_CASE("shape-pressure-nodes-2d", "[test_adjoint]")
{
	run_test1("shape-pressure-nodes-2d-opt.json", 1e-7, 1e-2, 0.0, 1.0);
}

TEST_CASE("static-control-pressure-nodes-3d", "[.][test_adjoint]")
{
	run_test2("static-control-pressure-nodes-3d-opt.json", 1e-3, 1e-3, 0.0, 1.0);
}

TEST_CASE("control-pressure-walker-2d", "[test_adjoint]")
{
	run_test1("walker-opt.json", 1e-6, 1e-1, 0.0, 1.0);
}

TEST_CASE("shape-walker-2d", "[test_adjoint]")
{
	run_test1("walker-shape-opt.json", 1e-7, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-contact-force-norm", "[test_adjoint]")
{
	run_test1("shape-contact-force-norm-opt.json", 1e-7, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-contact-force-norm-adhesion", "[test_adjoint]")
{
	run_test1("shape-contact-force-norm-opt-adhesion.json", 1e-7, 1e-1, 0.0, 1.0);
}

TEST_CASE("shape-contact-force-norm-3d", "[test_adjoint]")
{
	run_test1("shape-contact-force-norm-3d-opt.json", 1e-6, 1e-3, 0.0, 1.0);
}

TEST_CASE("shape-contact", "[test_adjoint]")
{
	run_test3("shape-contact-opt.json", 1e-7, 1e-5);
}

TEST_CASE("shape-contact-adhesion", "[test_adjoint]")
{
	run_test3("shape-contact-adhesion-opt.json", 1e-7, 1e-5);
}

TEST_CASE("node-trajectory", "[test_adjoint]")
{
	// Prepare test manually because we need random target form.
	// The opt json is mostly a dummy json.

	json opt_args;
	load_json(append_root_path("node-trajectory-opt.json"), opt_args);
	opt_args = AdjointOptUtils::apply_opt_json_spec(opt_args, false);

	// One state only.
	std::string root = POLYFEM_DIFF_DIR + std::string("/input/");
	auto states =
		from_json::build_states(root, opt_args["states"], -1);
	std::vector<std::shared_ptr<DiffCache>> diff_caches = {std::make_shared<DiffCache>()};

	auto elastic_var2sim =
		std::make_shared<ElasticVariableToSimulation>(states, diff_caches, CompositeParametrization{});
	VariableToSimulationGroup var2sim_group;
	var2sim_group.data.push_back(elastic_var2sim);

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd targets =
		Eigen::MatrixXd::Random(states[0]->n_bases, states[0]->mesh->dimension());
	targets = (targets.array() + 1.0f) * 0.5f * 10.0f;

	// All actice.
	std::vector<int> actives(targets.rows());
	for (int i = 0; i < targets.rows(); ++i)
	{
		actives[i] = i;
	}

	auto form =
		std::make_shared<NodeTargetForm>(states[0], diff_caches[0], var2sim_group, actives, targets);
	AdjointNLProblem problem{form, var2sim_group, states, diff_caches, opt_args};
	Eigen::VectorXd x = var2sim_group.data[0]->inverse_eval();
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(x.size(), 1);

	verify_adjoint(problem, x, velocity, 1e-5, 1e-4);
}

TEST_CASE("damping-transient", "[test_adjoint]")
{
	run_test1("damping-transient-opt.json", 1e-3, 1e-4, 1.0, 1.0);
}

TEST_CASE("material-transient", "[test_adjoint]")
{
	run_test1("material-transient-opt.json", 1e-5, 1e-4, 1e3, 1e3);
}

TEST_CASE("shape-transient-friction", "[test_adjoint]")
{
	run_test2("shape-transient-friction-opt.json", 1e-6, 1e-5, 0.0, 1.0);
}

TEST_CASE("shape-transient-friction-sdf", "[test_adjoint]")
{
	run_test2("shape-transient-friction-sdf-opt.json", 1e-7, 1e-4, 0.0, 1.0);
}

// 3D transient stress_norm shape adjoint with IPC contact.
// Two-body scene (rotated plank as a meshed obstacle + small sphere falling
// onto it under gravity) integrated over a few time steps. The two variants
// only differ in `friction_coefficient` and are intended to localize a bug
// in the transient adjoint when friction is disabled.
// See INVESTIGATION.md (in the project workspace) for full diagnosis.
//
// We test with a directional FD along the gradient direction (not random
// theta), because the bug shows up most dramatically along the gradient
// direction itself — a random theta tends to average it out.
//
// Tagged [.][test_adjoint] (hidden) because each variant takes ~30 s.
namespace
{
	void run_directional_grad_test(const std::string &json_name, double dt, double tol)
	{
		TestContext ctx{json_name};
		// Use inverse_evaluation (variant 1) so composite parametrizations
		// like "boundary" can size x correctly.
		Eigen::VectorXd x =
			AdjointOptUtils::inverse_evaluation(ctx.args["parameters"], ctx.ndof, ctx.var_sizes, ctx.var2sim);
		verify_adjoint_along_gradient(*ctx.problem, x, dt, tol);
	}
}

TEST_CASE("shape-transient-stress-3d-friction", "[.][test_adjoint]")
{
	run_directional_grad_test("shape-transient-stress-3d-friction-opt.json", 1e-5, 1e-3);
}

// Same scene as above but with friction_coefficient = 0. Used to exhibit
// a ~10^5 mismatch (often with wrong sign) between analytic gradient and FD
// in the EE-mollified shape_derivative path of IPC; fixed upstream in
// ipc-toolkit (squared mollifier + corrected edge_edge_mollifier_gradient_wrt_x).
TEST_CASE("shape-transient-stress-3d-frictionless", "[.][test_adjoint]")
{
	run_directional_grad_test("shape-transient-stress-3d-frictionless-opt.json", 1e-5, 1e-3);
}

// Faster repro of the frictionless transient contact adjoint regression
// (12 BDF steps — the bisection minimum that triggered the now-fixed IPC
// EE-mollifier bug). verify_adjoint_along_gradient performs exactly 3 forward
// solves (value(x), value(x+eps*d), value(x-eps*d)); the gradient call shares
// the cached forward at x. ~25 s per solve, so ~75 s total.
TEST_CASE("shape-transient-stress-3d-frictionless-fast", "[.][test_adjoint]")
{
	run_directional_grad_test("shape-transient-stress-3d-frictionless-fast-opt.json", 1e-5, 1e-3);
}

// SIMPLEST possible transient objective: J = (integral of y over sphere
// volume) at final time step = unnormalized barycenter height. J is linear
// in u_N, so ∂J/∂u_N is just a constant integral-of-basis vector — this
// isolates the EE-mollifier adjoint bug (now fixed upstream) from any
// stress-norm-specific quirks. kappa = 1e7 keeps per-step matrix magnitudes
// within a numerically benign range.
TEST_CASE("shape-transient-position-3d-frictionless-fast", "[.][test_adjoint]")
{
	run_directional_grad_test(
		"shape-transient-position-3d-frictionless-fast-opt.json", 1e-5, 1e-3);
}

// Same scene + objective + 12 steps, but the optimization variable is the
// INITIAL CONDITION (initial position + velocity) instead of rest mesh shape.
// Shape and IC share the same forward solve and the same adjoint trajectory
// p_i, nu_i produced by solve_transient_adjoint. They differ only in the
// final assembly:
//   shape:  one_form = Sum_i beta*dt * (force_shape_derivative)^T p_i  + explicit
//   IC:     one_form = [-nu_0; -p_0]                                  (no contact, no kappa)
// If IC matches FD, then p_0 and nu_0 are correct, and the bug is specifically
// in the per-step force_shape_derivative assembly path (or in p_i for i>0 only,
// since IC only reads p_0/nu_0).
// If IC also fails, then the BDF backward sweep produces a globally wrong
// p_i / nu_i (the bug sits deeper in StateDiff.cpp).
TEST_CASE("frictionless-fast-ic", "[.][test_adjoint]")
{
	// IC uses empty "parameters", so use var2sim's own inverse_eval (the
	// AdjointOptUtils helper would return a size-0 x and segfault).
	TestContext ctx{"shape-transient-stress-3d-frictionless-fast-ic-opt.json"};
	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();
	verify_adjoint_along_gradient(*ctx.problem, x, 1e-5, 1e-3);
}

// Cross-check the analytic adjoint gradient against a tangent (forward
// sensitivity) reference. Three estimates of dJ/dX along d := grad/||grad||:
//   adjoint  = grad_adjoint . d                         (what polyfem reports)
//   fd_J     = (J(X+eps*d) - J(X-eps*d)) / (2 eps)      (FD of objective)
//   tangent  = (dJ/du_N) . xi_N,  xi_N = (u_N(+) - u_N(-)) / (2 eps)
// "tangent" uses an FD of the forward state trajectory at frame N, composed
// with the analytic per-step adjoint RHS (= dJ/du_N for integral_type=final).
// It involves NO transient adjoint solve, so it is an independent check.
// Used during investigation of an EE-mollifier shape_derivative bug (now
// fixed upstream): with the bug present, tangent ~= fd_J but adjoint was
// off by ~10^5 — pinpointing the BDF backward sweep / per-step contact
// shape_derivative as the bug locus.
TEST_CASE("verify-adjoint-via-tangent-fd", "[.][test_adjoint]")
{
	TestContext ctx{"shape-transient-stress-3d-frictionless-fast-opt.json"};
	Eigen::VectorXd x = AdjointOptUtils::inverse_evaluation(
		ctx.args["parameters"], ctx.ndof, ctx.var_sizes, ctx.var2sim);

	auto &problem = *ctx.problem;
	auto &state = *ctx.states[0];
	auto &dc = *ctx.diff_caches[0];
	const int N = state.args["time"]["time_steps"];

	// Forward + adjoint at x.
	problem.solution_changed(x);
	const double J0 = problem.value(x);
	Eigen::VectorXd grad_adjoint;
	problem.gradient(x, grad_adjoint);

	// ctx.form is a SumCompositeForm wrapper (functionals: [...]) whose
	// default compute_adjoint_rhs returns zero. Build the inner TransientForm
	// directly to get the actual adjoint RHS trajectory.
	auto inner_form = from_json::build_form(
		ctx.args["functionals"][0], ctx.var2sim, ctx.states, ctx.diff_caches);
	const Eigen::MatrixXd adj_rhs =
		inner_form->compute_adjoint_rhs(x, state, dc);
	// For TransientForm with integral_type=final, only col N is nonzero.
	const Eigen::VectorXd dJ_duN = adj_rhs.col(N);

	const double grad_norm = grad_adjoint.norm();
	REQUIRE(grad_norm > 0);
	const Eigen::VectorXd d = grad_adjoint / grad_norm;

	const double adjoint_d = grad_norm; // = grad_adjoint . d

	// Decompose: total = explicit (dJ/dX at fixed u) + via_u (dJ/dX through u).
	// The via_u piece is what the BDF backward sweep + assembled per-step
	// force_shape_derivatives produce; that is the suspected bug location.
	Eigen::VectorXd explicit_grad;
	inner_form->compute_partial_gradient(x, explicit_grad);
	const double explicit_d = explicit_grad.dot(d);
	const double via_u_reported_d = adjoint_d - explicit_d;

	std::printf("[tangent cross-check] J0=%.6e  adjoint(total)=%.6e\n", J0, adjoint_d);
	std::printf("  explicit (compute_partial_gradient)·d = %.6e\n", explicit_d);
	std::printf("  via_u_reported = adjoint - explicit    = %.6e\n", via_u_reported_d);
	std::printf("  ||dJ/du_N|| = %.3e\n", dJ_duN.norm());

	double tangent_best = 0.0, fd_J_best = 0.0;
	for (double eps : {1e-4, 1e-5, 1e-6, 1e-7})
	{
		problem.solution_changed(x + eps * d);
		const double J_plus = problem.value(x + eps * d);
		const Eigen::VectorXd u_N_plus = dc.u(N);

		problem.solution_changed(x - eps * d);
		const double J_minus = problem.value(x - eps * d);
		const Eigen::VectorXd u_N_minus = dc.u(N);

		const double fd_J = (J_plus - J_minus) / (2.0 * eps);
		const Eigen::VectorXd xi_N = (u_N_plus - u_N_minus) / (2.0 * eps);
		const double tangent = dJ_duN.dot(xi_N);

		const double fd_via_u = fd_J - explicit_d; // via_u inferred from FD
		std::printf(
			"  eps=%.0e  fd_J=%.6e  tangent=%.6e  fd_via_u=%.6e  "
			"tangent/fd_via_u=%.4f  via_u_reported/fd_via_u=%.3e\n",
			eps, fd_J, tangent, fd_via_u, tangent / fd_via_u,
			via_u_reported_d / fd_via_u);
		if (eps == 1e-6) { tangent_best = tangent; fd_J_best = fd_J - explicit_d; }
	}
	std::fflush(stdout);

	// tangent (independent FD-of-state + analytic dJ/du_N) MUST match the
	// via-u piece of dJ/dX. This passes — confirming forward dynamics and
	// the objective gradient are correct.
	REQUIRE(tangent_best == Catch::Approx(fd_J_best).epsilon(1e-2));

	// The via-u piece as REPORTED by the adjoint (= adjoint_total - explicit)
	// must equal the same fd_via_u. This FAILS by ~10^5 in the buggy
	// frictionless transient configuration — pinning the bug to the BDF
	// backward sweep + per-step force_shape_derivative assembly.
	REQUIRE(via_u_reported_d == Catch::Approx(fd_J_best).epsilon(1e-2));
}

// Isolate the suspect partial: IPC's barrier_potential.shape_derivative at
// the final forward step. Hold the displacement u_N fixed; perturb the
// collision-mesh rest positions by ±eps*d; recompute kappa * F_contact at
// each perturbation; compare p . (centered FD) against the analytic
// p . (kappa * shape_derivative * d). All vectors live in the collision-DOF
// frame to bypass to_full_dof and basis_nodes_to_gbasis_nodes.
//
// Interpretation:
//   match (rel_err < 1e-3) -> IPC shape_derivative is correct, so the bug
//     in the transient frictionless adjoint must be in p_i produced by
//     solve_transient_adjoint (BDF assembly), not in this partial.
//   mismatch -> IPC shape_derivative is wrong; substituting FD for this
//     term should fix the total gradient.
TEST_CASE("barrier-contact-shape-derivative-fd", "[.][test_adjoint]")
{
	TestContext ctx{"shape-transient-stress-3d-frictionless-fast-opt.json"};
	Eigen::VectorXd x = AdjointOptUtils::inverse_evaluation(
		ctx.args["parameters"], ctx.ndof, ctx.var_sizes, ctx.var2sim);

	// Drive forward solve to populate diff_cache (u_i, collision_set(i)).
	ctx.problem->solution_changed(x);
	(void)ctx.problem->value(x);

	auto &state = *ctx.states[0];
	auto &dc = *ctx.diff_caches[0];
	const auto *form_ptr = dynamic_cast<const BarrierContactForm *>(
		state.solve_data.contact_form.get());
	REQUIRE(form_ptr != nullptr);
	const auto &form = *form_ptr;

	const int N = state.args["time"]["time_steps"];
	const Eigen::VectorXd u_full = dc.u(N);
	const ipc::NormalCollisions &col_set_0 = dc.collision_set(N);

	const ipc::CollisionMesh &mesh = form.collision_mesh();
	const ipc::BarrierPotential &barrier_potential = form.barrier_potential();
	const double kappa = form.barrier_stiffness();

	const Eigen::MatrixXd displaced_0 = form.compute_displaced_surface(u_full);
	const Eigen::MatrixXd rest_0 = mesh.rest_positions();
	const Eigen::MatrixXd u_col_mat = displaced_0 - rest_0; // collision-frame displacement

	const int n_col_dof = displaced_0.size();
	const int dim = mesh.dim();
	REQUIRE(col_set_0.size() > 0);

	std::srand(20260605);
	Eigen::VectorXd p_col = Eigen::VectorXd::Random(n_col_dof);
	Eigen::VectorXd d_col = Eigen::VectorXd::Random(n_col_dof);
	Eigen::MatrixXd d_mat = utils::unflatten(d_col, dim);

	// -------- Analytic: p . kappa * shape_derivative * d --------
	Eigen::SparseMatrix<double> SD =
		barrier_potential.shape_derivative(col_set_0, mesh, displaced_0);
	const double analytic = kappa * p_col.dot(SD * d_col);

	// -------- Centered FD: p . kappa * F_contact(u, X +/- eps*d) --------
	auto eval_pF = [&](double sign, double eps, size_t *n_out = nullptr) -> double {
		Eigen::MatrixXd rest_pert = rest_0 + sign * eps * d_mat;
		ipc::CollisionMesh mesh_pert(rest_pert, mesh.edges(), mesh.faces());
		if (mesh.are_area_jacobians_initialized())
			mesh_pert.init_area_jacobians();
		Eigen::MatrixXd disp_pert = rest_pert + u_col_mat;

		ipc::NormalCollisions set_pert;
		set_pert.set_use_area_weighting(col_set_0.use_area_weighting());
		set_pert.set_use_improved_max_approximator(
			col_set_0.use_improved_max_approximator());
		set_pert.build(mesh_pert, disp_pert, form.dhat());

		const Eigen::VectorXd F_unit =
			barrier_potential.gradient(set_pert, mesh_pert, disp_pert);
		if (n_out) *n_out = set_pert.size();
		return p_col.dot(kappa * F_unit);
	};

	std::printf("[shape_deriv FD check] step=N=%d kappa=%.3e n_col_0=%zu analytic=%.6e\n",
		N, kappa, (size_t)col_set_0.size(), analytic);

	double fd_best = 0.0;
	for (double eps : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7})
	{
		size_t n_p = 0, n_m = 0;
		const double pair_p = eval_pF(+1.0, eps, &n_p);
		const double pair_m = eval_pF(-1.0, eps, &n_m);
		const double fd = (pair_p - pair_m) / (2.0 * eps);
		const double rel_err =
			std::abs(analytic - fd) / std::max(std::abs(analytic), 1e-30);
		std::printf("  eps=%.0e fd=%.6e rel_err=%.3e n_col_p=%zu n_col_m=%zu\n",
			eps, fd, rel_err, n_p, n_m);
		std::fflush(stdout);
		if (eps == 1e-5)
			fd_best = fd;
	}

	REQUIRE(analytic == Catch::Approx(fd_best).epsilon(1e-3));
}

TEST_CASE("3d-shape-mesh-target", "[.][test_adjoint]")
{
	run_test1("3d-shape-mesh-target-opt.json", 1e-7, 1e-5, 0.0, 1.0);
}

TEST_CASE("initial-contact-min-dist", "[test_adjoint]")
{
	run_test1("initial-contact-min-dist-opt.json", 1e-5, 1e-5, -1.0, 1.0);
}

TEST_CASE("friction-contact", "[test_adjoint]")
{
	TestContext ctx{"friction-contact-opt.json"};

	Eigen::VectorXd velocity = Eigen::VectorXd::Ones(1);
	Eigen::VectorXd x = 0.2f * Eigen::VectorXd::Ones(1);

	verify_adjoint(*ctx.problem, x, velocity, 1e-4, 1e-6);
}

TEST_CASE("barycenter", "[test_adjoint]")
{
	TestContext ctx{"barycenter-opt.json"};

	Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(ctx.states[0]->ndof() * 2, 1);
	for (int i = 0; i < ctx.states[0]->n_bases; i++)
	{
		velocity(ctx.states[0]->ndof() + i * 2 + 0) = -2.0f;
		velocity(ctx.states[0]->ndof() + i * 2 + 1) = -1.0f;
	}

	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();

	verify_adjoint(*ctx.problem, x, velocity, 1e-6, 1e-5);
}

TEST_CASE("shape-contact-smooth", "[test_adjoint]")
{
	TestContext ctx{"shape-contact-opt.json"};

	// Because state configs are shared, tailor json args.
	for (auto &state : ctx.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["use_convergent_formulation"] = false;
		state->args["contact"]["alpha_t"] = 0.95;
	}

	Eigen::MatrixXd V;
	ctx.states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	ctx.problem->solution_changed(x);
	Eigen::VectorXd one_form;
	ctx.problem->gradient(x, one_form);

	verify_adjoint(*ctx.problem, x, one_form.normalized(), 1e-6, 1e-5);
}

TEST_CASE("initial-contact-smooth", "[test_adjoint]")
{
	TestContext ctx{"initial-contact-smooth-opt.json"};

	// Because state configs are shared, tailor json args.
	for (auto &state : ctx.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["use_convergent_formulation"] = false;
		state->args["contact"]["alpha_t"] = 0.95;
		state->args["contact"]["friction_coefficient"] = 0;
	}

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(ctx.ndof, 1);
	Eigen::VectorXd x = ctx.var2sim.data[0]->inverse_eval();

	verify_adjoint(*ctx.problem, x, velocity, 1e-6, 1e-4);
}

TEST_CASE("shape-transient-smooth", "[test_adjoint]")
{
	TestContext ctx{"shape-transient-friction-opt.json"};

	// Because states are shared, tailor json args.
	for (auto &state : ctx.states)
	{
		state->args["contact"]["use_gcp_formulation"] = true;
		state->args["contact"]["alpha_t"] = 0.95;
		state->args["contact"]["friction_coefficient"] = 0;
		state->args["solver"]["nonlinear"]["grad_norm_tol"] = 1e-8;
	}

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(ctx.ndof, 1);
	velocity.normalize();

	Eigen::MatrixXd V;
	ctx.states[0]->get_vertices(V);
	Eigen::VectorXd x = utils::flatten(V);

	verify_adjoint(*ctx.problem, x, velocity, 1e-6, 1e-4);
}

TEST_CASE("shape-pressure-nodes-3d", "[.][test_adjoint]")
{
	run_test1("shape-pressure-nodes-3d-opt.json", 1e-7, 1e-3, 0.0, 1.0);
}

TEST_CASE("control-pressure-nodes-3d", "[.][test_adjoint]")
{
	run_test1("control-pressure-nodes-3d-opt.json", 1e-8, 1e-3, 0.0, 1.0);
}

TEST_CASE("dirichlet-nodes-3d", "[.][test_adjoint]")
{
	TestContext ctx{"dirichlet-nodes-3d-opt.json"};

	Eigen::VectorXd x(12);
	// clang-format off
	x << 0.0f,  0.0f, 0.0f,
	     0.0f,  0.0f, 0.0f,
	     0.0f, -0.2f, 0.0f,
	     0.0f, -0.2f, 0.0f;
	// clang-format on

	std::srand(Catch::rngSeed());
	Eigen::MatrixXd velocity = Eigen::MatrixXd::Random(x.size(), 1);

	verify_adjoint(*ctx.problem, x, velocity, 1e-7, 1e-3);
}
