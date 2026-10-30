////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/solver/BarrierContactNonsmoothTerm.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polysolve/nonlinear/descent_strategies/NonsmoothNewton.hpp>

#include <ipc/ipc.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

// polysolve's NonsmoothNewton with the barrier contact form as the nonsmooth term (BarrierContactNonsmoothTerm).
// The solving tests need CHOLMOD: the strategy requires Eigen::CholmodSupernodalLLT (an LLT that reports indefinite
// matrices). Builds without it (POLYSOLVE_WITH_CHOLMOD=OFF) only run the option tests.
#ifdef POLYSOLVE_WITH_CHOLMOD
namespace
{
	/// 5 stacked cubes falling onto a plane (contact from the first steps), shortened to `steps` time steps
	json five_cubes(const int steps, const std::string &solver, const std::string &out_json)
	{
		const std::string path = std::string(POLYFEM_DATA_DIR) + "/contact/examples/3D/unit-tests/5-cubes.json";
		std::ifstream f(path);
		REQUIRE(f.is_open());
		json args;
		f >> args;
		const double dt = args["time"]["tend"].get<double>() / args["time"]["time_steps"].get<int>();
		args["time"] = json{{"dt", dt}, {"time_steps", steps}};
		args["root_path"] = path;
		args["output"] = json::object();
		args["output"]["advanced"]["save_time_sequence"] = false;
		if (!out_json.empty())
			args["output"]["json"] = out_json;
		args["/solver/max_threads"_json_pointer] = 1;
		args["/solver/linear/solver"_json_pointer] = "Eigen::CholmodSupernodalLLT";
		// the augmented Lagrangian subproblems inherit the solver (none in this scene: the plane does not move)
		args["/solver/nonlinear/solver"_json_pointer] = solver;
		args["/output/log/level"_json_pointer] = "error";
		return args;
	}

	/// five_cubes with the plane moving up into the stack: once the plane touches the bottom cube, the obstacle
	/// displacement of a step cannot be applied directly and the step starts with augmented Lagrangian subproblems
	json five_cubes_pushed(const int steps, const std::string &solver, const std::string &out_json)
	{
		json args = five_cubes(steps, solver, out_json);
		args["/boundary_conditions/obstacle_displacements"_json_pointer] = json::array({json{{"id", 0}, {"value", json::array({0, "2 * t", 0})}}});
		return args;
	}

	/// stock transient formulation that checks the collision mesh for intersections after every time step
	class CheckedTransientVarForm : public varform::NonlinearElasticTransientVarForm
	{
	public:
		using NonlinearElasticTransientVarForm::NonlinearElasticTransientVarForm;

		bool intersecting = false;
		int frames = 0;

	protected:
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, bool init_lagging = true) override
		{
			NonlinearElasticTransientVarForm::solve_tensor_nonlinear(step, sol, init_lagging);
			const Eigen::MatrixXd V = collision_mesh_.displace_vertices(utils::unflatten(sol, mesh_->dimension()));
			intersecting = intersecting || ipc::has_intersections(collision_mesh_, V);
			++frames;
		}

	public:
		int collision_vertices() const { return int(collision_mesh_.full_num_vertices()); }
		int fe_nodes() const { return space_.n_bases; }
	};

	Eigen::MatrixXd run(const json &args, std::shared_ptr<CheckedTransientVarForm> *checked = nullptr)
	{
		// resources of State::init: root_path, then the root of the common parameter file
		const io::FileSystemIO cwd_resources(std::filesystem::current_path());
		json resource_args = args;
		std::unique_ptr<const io::ResourceIO> resources = cwd_resources.with_root(
			utils::is_param_valid(resource_args, "root_path") ? resource_args["root_path"].get<std::string>() : std::string());
		resource_args.erase("root_path");
		if (auto common_resources = utils::apply_common_params(resource_args, *resources))
			resources = std::move(common_resources);

		State state;
		state.init(args, /*strict_validation=*/true);
		if (checked)
		{
			// same construction as State::init, with the checking subclass (the formulation is the single material of
			// the five cubes scene)
			Units units;
			units.init(state.args["units"]);
			*checked = std::make_shared<CheckedTransientVarForm>(*resources);
			state.args["contact"]["_dhat_was_explicit"] = args.contains("/contact/dhat"_json_pointer);
			(*checked)->init("NeoHookean", units, state.args, state.args["output"]["directory"].get<std::string>());
			state.args["contact"].erase("_dhat_was_explicit");
			state.variational_formulation = *checked;
		}
		state.load_mesh(false);
		Eigen::MatrixXd sol;
		state.solve(sol);
		state.variational_formulation->save_json(sol);
		return sol;
	}

	/// the counters of one reduced solve: polysolve's NonsmoothNewton block, PolyFEM's contact term block
	const json &block(const json &e, const char *name)
	{
		static const json empty = json::object();
		return e.contains("info") && e["info"].contains(name) ? e["info"][name] : empty;
	}

	/// a flag of the contact term on all / any of the reduced solves recorded in the output JSON
	bool all_rc(const json &out, const std::string &key)
	{
		bool all = true;
		for (const auto &e : out["solver_info"])
			if (e.value("type", "") == "rc" && block(e, "contact_term").contains(key))
				all = all && block(e, "contact_term")[key].get<bool>();
		return all;
	}

	bool any_rc(const json &out, const std::string &key)
	{
		bool any = false;
		for (const auto &e : out["solver_info"])
			if (e.value("type", "") == "rc" && block(e, "contact_term").contains(key))
				any = any || block(e, "contact_term")[key].get<bool>();
		return any;
	}

	/// sum of a NonsmoothNewton counter (or of the total "iterations") over the reduced solves ("rc") or the augmented
	/// Lagrangian subproblems ("al") in the output JSON
	long sum_stat(const json &out, const std::string &key, const std::string &type = "rc")
	{
		long s = 0;
		for (const auto &e : out["solver_info"])
		{
			if (e.value("type", "") != type)
				continue;
			const json &b = key == "iterations" ? e["info"] : block(e, "NonsmoothNewton");
			if (b.contains(key))
				s += b[key].get<long>();
		}
		return s;
	}
} // namespace
#endif

TEST_CASE("nonsmooth_newton_contact_options", "[nonsmooth_newton]")
{
	using polysolve::nonlinear::NonsmoothNewton;
	const auto d = NonsmoothNewton::Options::from_json(json::object());
	CHECK(d.max_inner == 10);
	CHECK(d.relative_stop == Catch::Approx(1e-3));
	CHECK(d.ftb_tau == Catch::Approx(0.2));
	CHECK(d.guard);
	CHECK(d.dedupe);
	CHECK(d.fast_validity);

	const auto o = NonsmoothNewton::Options::from_json(json{{"NonsmoothNewton", {{"max_inner", 20}, {"guard", {{"c", 3.0}}}, {"debug", {{"dedupe", false}}}}}});
	CHECK(o.max_inner == 20);
	CHECK(o.guard_c == Catch::Approx(3.0));
	CHECK(o.guard_abs == Catch::Approx(10.0));
	CHECK(!o.dedupe);

	auto bad = [](const json &p) { return NonsmoothNewton::Options::from_json(json{{"NonsmoothNewton", p}}); };
	CHECK_THROWS(bad(json{{"max_inner", 0}}));
	CHECK_THROWS(bad(json{{"fraction_to_boundary", 1.0}}));
	// rho_min <= 0 would accept a true energy increase
	CHECK_THROWS(bad(json{{"rho_min", -1.0}}));
	CHECK_THROWS(bad(json{{"rho_min", 0.0}}));
	CHECK_THROWS(bad(json{{"rho_min", 1.0}}));
	CHECK_THROWS(bad(json{{"rho_min", std::numeric_limits<double>::quiet_NaN()}}));
	CHECK_THROWS(bad(json{{"max_columns", 0}}));
	CHECK_THROWS(bad(json{{"guard", {{"c", 0.0}}}}));
	CHECK_THROWS(bad(json{{"guard", {{"check", -1.0}}}}));

	// problem-side switches
	const auto t = solver::BarrierContactNonsmoothTerm::Options::from_json(json{{"NonsmoothNewton", {{"debug", {{"local_blocks", false}, {"candidate_permutation", 3}}}}}});
	CHECK(!t.local_blocks);
	CHECK(t.certificate);
	CHECK(t.candidate_permutation == 3);

	CHECK(solver::selects_nonsmooth_newton(json{{"solver", "NonsmoothNewton"}}));
	CHECK(solver::selects_nonsmooth_newton(json{{"solver", json::array({json{{"type", "NonsmoothNewton"}}, json{{"type", "Newton"}}})}}));
	CHECK(!solver::selects_nonsmooth_newton(json{{"solver", "Newton"}}));
}

#ifdef POLYSOLVE_WITH_CHOLMOD
// The full simulations run only in release builds, like the other simulation tests (verify_run, test_output).
#ifdef NDEBUG
#define NONSMOOTH_NEWTON_SIM_TAGS "[nonsmooth_newton]"
#else
#define NONSMOOTH_NEWTON_SIM_TAGS "[.][nonsmooth_newton]"
#endif

TEST_CASE("nonsmooth_newton_contact_without_method_is_native_newton", NONSMOOTH_NEWTON_SIM_TAGS)
{
	// without the strategy (debug/method = false) the chain is polysolve's Newton chain; the installed contact term may
	// only change how max_step_size is computed (certified narrow phase), never its result
	const int steps = 10;
	const Eigen::MatrixXd native = run(five_cubes(steps, "Newton", ""));
	json args = five_cubes(steps, "NonsmoothNewton", "");
	args["/solver/nonlinear/NonsmoothNewton/debug/method"_json_pointer] = false;
	const Eigen::MatrixXd own = run(args);
	REQUIRE(native.size() == own.size());
	CHECK(native == own);
}

TEST_CASE("nonsmooth_newton_contact_stop_before_step", NONSMOOTH_NEWTON_SIM_TAGS)
{
	// the stopping criteria of Solver::minimize act before the strategy's own step: with x_delta_tol above every step
	// norm both solvers stop at their first direction and leave the iterate unchanged
	const int steps = 2;
	const Eigen::MatrixXd reference = run(five_cubes(steps, "Newton", ""));
	json a = five_cubes(steps, "Newton", "");
	json b = five_cubes(steps, "NonsmoothNewton", "");
	a["/solver/nonlinear/x_delta_tol"_json_pointer] = 1e10;
	b["/solver/nonlinear/x_delta_tol"_json_pointer] = 1e10;
	const Eigen::MatrixXd native = run(a);
	const Eigen::MatrixXd own = run(b);
	CHECK(native != reference); // the option changes the native result
	CHECK(native == own);
}

TEST_CASE("nonsmooth_newton_contact_five_cubes", NONSMOOTH_NEWTON_SIM_TAGS)
{
	// the stack hits the plane at step 8. The trajectory is not compared with the native solver: at the impact the
	// incremental potential is nonconvex and the two solvers may converge to different (valid) minimizers.
	const int steps = 12;
	const std::string out = (std::filesystem::temp_directory_path() / "nonsmooth_newton_contact_five_cubes.json").string();
	std::shared_ptr<CheckedTransientVarForm> checked;
	const Eigen::MatrixXd sol = run(five_cubes(steps, "NonsmoothNewton", out), &checked);
	REQUIRE(sol.allFinite());
	CHECK(checked->frames == steps);
	CHECK(!checked->intersecting);

	json o;
	{
		std::ifstream f(out);
		REQUIRE(f.is_open());
		f >> o;
	}
	// the protected path is taken on contact iterates, and its failures fall back to the native line search; once the
	// true energy no longer verifies the path (handoff "any", the default) the Newton chain finishes the solve
	CHECK(sum_stat(o, "active_iterations") > 0);
	CHECK(sum_stat(o, "path_iterations") > 0);
	CHECK(sum_stat(o, "inner_steps") >= sum_stat(o, "path_iterations"));
	CHECK(sum_stat(o, "handoff_noise_floor") + sum_stat(o, "handoff_grad_norm") > 0);
	for (const auto &e : o["solver_info"])
	{
		const json &s = block(e, "NonsmoothNewton");
		// Newton chain iterations only after a handoff or a failed direction
		if (e.value("type", "") == "rc" && s.value("fallback_iterations", 0L) > 0)
			CHECK(s.value("handoff_noise_floor", 0L) + s.value("handoff_grad_norm", 0L) + s.value("failed_directions", 0L) > 0);
	}
	CHECK(all_rc(o, "local_contact_blocks"));
}

TEST_CASE("nonsmooth_newton_contact_collision_proxy", NONSMOOTH_NEWTON_SIM_TAGS)
{
	// a sampled collision proxy (max_order tessellation, lattice order 4) has more collision vertices than FE nodes and
	// interpolated displacements: the per-collision blocks are not applicable and the global contact assembly is used
	const int steps = 9; // first contact at step 8
	const std::string out = (std::filesystem::temp_directory_path() / "nonsmooth_newton_contact_collision_proxy.json").string();
	json args = five_cubes(steps, "NonsmoothNewton", out);
	args["/contact/collision_mesh"_json_pointer] = json{{"enabled", true}, {"tessellation_type", "max_order"}, {"sampling_order", 4}};
	std::shared_ptr<CheckedTransientVarForm> checked;
	const Eigen::MatrixXd sol = run(args, &checked);
	REQUIRE(sol.allFinite());
	INFO("collision vertices " << checked->collision_vertices() << ", FE nodes " << checked->fe_nodes());
	CHECK(checked->collision_vertices() > checked->fe_nodes());
	CHECK(!checked->intersecting);

	json o;
	{
		std::ifstream f(out);
		REQUIRE(f.is_open());
		f >> o;
	}
	CHECK(sum_stat(o, "active_iterations") > 0);
	CHECK(!any_rc(o, "local_contact_blocks"));
}

TEST_CASE("nonsmooth_newton_contact_al_subproblems", NONSMOOTH_NEWTON_SIM_TAGS)
{
	// the plane pushes the stack up: the steps after the first contact start with augmented Lagrangian subproblems, where
	// the contact term acts in the full size when augmented_lagrangian/nonlinear selects NonsmoothNewton (by default it
	// inherits solver/nonlinear)
	const int steps = 8;
	const auto read = [](const std::string &path) {
		json o;
		std::ifstream f(path);
		REQUIRE(f.is_open());
		f >> o;
		return o;
	};
	const auto count = [](const json &out, const std::string &type, const char *name) {
		int n = 0;
		for (const auto &e : out["solver_info"])
			n += e.value("type", "") == type && block(e, name).size() > 0;
		return n;
	};

	const std::string out = (std::filesystem::temp_directory_path() / "nonsmooth_newton_contact_al_subproblems.json").string();
	std::shared_ptr<CheckedTransientVarForm> checked;
	const Eigen::MatrixXd sol = run(five_cubes_pushed(steps, "NonsmoothNewton", out), &checked);
	REQUIRE(sol.allFinite());
	CHECK(checked->frames == steps);
	CHECK(!checked->intersecting);
	const json o = read(out);
	REQUIRE(count(o, "al", "NonsmoothNewton") > 0);
	CHECK(count(o, "al", "contact_term") == count(o, "al", "NonsmoothNewton"));
	CHECK(sum_stat(o, "active_iterations", "al") > 0);
	CHECK(sum_stat(o, "path_iterations", "al") > 0);

	// native Newton in the augmented Lagrangian subproblems: no contact term there
	const std::string out_native = (std::filesystem::temp_directory_path() / "nonsmooth_newton_contact_al_native.json").string();
	json args = five_cubes_pushed(steps, "NonsmoothNewton", out_native);
	args["/solver/augmented_lagrangian/nonlinear/solver"_json_pointer] = "Newton";
	std::shared_ptr<CheckedTransientVarForm> checked_native;
	const Eigen::MatrixXd sol_native = run(args, &checked_native);
	REQUIRE(sol_native.allFinite());
	CHECK(!checked_native->intersecting);
	const json o_native = read(out_native);
	CHECK(count(o_native, "al", "contact_term") == 0);
	CHECK(count(o_native, "rc", "contact_term") > 0);
}

TEST_CASE("nonsmooth_newton_contact_rejects_unsupported", "[nonsmooth_newton]")
{
	using Catch::Matchers::ContainsSubstring;
	SECTION("indefinite-blind linear solver")
	{
		json args = five_cubes(2, "NonsmoothNewton", "");
		args["/solver/linear/solver"_json_pointer] = "Eigen::SimplicialLDLT";
		CHECK_THROWS_WITH(run(args), ContainsSubstring("CholmodSupernodalLLT"));
	}
	SECTION("static problem")
	{
		json args = five_cubes(2, "NonsmoothNewton", "");
		args.erase("time");
		CHECK_THROWS_WITH(run(args), ContainsSubstring("transient"));
	}
	SECTION("homogenization")
	{
		json args = five_cubes(2, "NonsmoothNewton", "");
		// the varform route takes time-dependent homogenization only as quasistatic
		args["/time/quasistatic"_json_pointer] = true;
		args["/constraints/macro_displacement_gradient/value"_json_pointer] = json::array({json::array({0, 0, 0}), json::array({0, 0, 0}), json::array({0, 0, 0})});
		CHECK_THROWS_WITH(run(args), ContainsSubstring("macro strain"));
	}
}
#endif
