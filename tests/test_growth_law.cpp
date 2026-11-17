// V2: solved growth on a homogeneous single-material cube (Stage 2 exit test).
//
// A unit cube (2x2x2 cells, Kuhn split into tets) with rollers on x=0, y=0,
// z=0, prescribed u_x on x=1 and u_y on y=1 (equibiaxial in-plane stretch
// lambda), traction-free top. Wall normal n0 = z, fiber in-plane along x, the
// fitted MaterialSum. The solution is homogeneous:
//
//     F = diag(lambda, lambda, b),  Fg = diag(sqrt(theta), sqrt(theta), 1),
//     Fe = diag(a, a, b),  a = lambda / sqrt(theta),  dpsi/db (Fe) = 0,
//
// so f_I depends on the elastic area stretch a^2 alone (the calibration
// path), and the growth law gives theta = lambda^2 / vartheta*, with
// f_I(vartheta*) = phi, wherever lambda^2 > vartheta*, and theta = theta_n
// otherwise. The reference below is an independent scalar implementation of
// the fitted energy on that path (not the assembler).

#include <polyfem/State.hpp>
#include <polyfem/varforms/GrowthElasticVarForm.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>

using namespace polyfem;

namespace
{
	// ---- fitted material, energy at Fe = diag(a, a, b), fiber in-plane ----
	constexpr double E_ = 2600.0, nu_ = 0.26, k1_ = 2400.0, k2_ = 0.01, kappa_ = 0.24, kchi_ = 100.0;
	constexpr double kvol_ = 1805.5555555555557;

	double psi(const double a, const double b)
	{
		const double mu = E_ / (2.0 * (1.0 + nu_));
		const double I1 = 2.0 * a * a + b * b, J = a * a * b, I4 = a * a;
		const double E4 = kappa_ * I1 + (1.0 - 3.0 * kappa_) * I4 - 1.0;
		const double nh = 0.5 * mu * (std::pow(J, -2.0 / 3.0) * I1 - 3.0);
		const double fib = k1_ / (2.0 * k2_) * (std::exp(k2_ * E4 * E4) - 1.0) / (1.0 + std::exp(-kchi_ * E4));
		const double vol = 0.5 * kvol_ * (0.5 * (J * J - 1.0) - std::log(J));
		return nh + fib + vol;
	}

	double bisect(const std::function<double(double)> &f, double lo, double hi)
	{
		double flo = f(lo);
		REQUIRE(flo * f(hi) < 0);
		for (int i = 0; i < 200 && hi - lo > 1e-15; ++i)
		{
			const double mid = 0.5 * (lo + hi);
			const double fm = f(mid);
			if ((fm < 0) == (flo < 0))
			{
				lo = mid;
				flo = fm;
			}
			else
				hi = mid;
		}
		return 0.5 * (lo + hi);
	}

	double dpsi_db(const double a, const double b, const double h = 1e-6)
	{
		return (psi(a, b + h) - psi(a, b - h)) / (2 * h);
	}

	/// f_I on the equibiaxial traction-free path at elastic area stretch v = a^2
	double f_I(const double v)
	{
		const double a = std::sqrt(v);
		const double b = bisect([&](double bb) { return dpsi_db(a, bb); }, 0.3, 1.8);
		const double h = 1e-6;
		const double dpsi_da = (psi(a + h, b) - psi(a - h, b)) / (2 * h);
		return 0.5 * a * dpsi_da - psi(a, b);
	}

	/// elastic area stretch at which f_I = phi
	double activation_stretch(const double phi)
	{
		return bisect([&](double v) { return f_I(v) - phi; }, 0.5, 3.0);
	}

	// ---- mesh: n x n x n unit-cube cells, 6 Kuhn tets each, positive orientation
	std::string write_cube_mesh(const std::filesystem::path &dir, const int n)
	{
		std::vector<Eigen::Vector3d> V;
		auto id = [n](int i, int j, int k) { return (i * (n + 1) + j) * (n + 1) + k; };
		for (int i = 0; i <= n; ++i)
			for (int j = 0; j <= n; ++j)
				for (int k = 0; k <= n; ++k)
					V.emplace_back(double(i) / n, double(j) / n, double(k) / n);

		const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
		std::vector<Eigen::Vector4i> T;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				for (int k = 0; k < n; ++k)
					for (const auto &p : perms)
					{
						int c[3] = {i, j, k};
						Eigen::Vector4i t;
						t(0) = id(c[0], c[1], c[2]);
						for (int s = 0; s < 3; ++s)
						{
							c[p[s]] += 1;
							t(s + 1) = id(c[0], c[1], c[2]);
						}
						const double vol = (V[t(1)] - V[t(0)]).cross(V[t(2)] - V[t(0)]).dot(V[t(3)] - V[t(0)]);
						if (vol < 0)
							std::swap(t(2), t(3));
						T.push_back(t);
					}

		const std::string path = (dir / "cube.mesh").string();
		std::ofstream out(path);
		out << "MeshVersionFormatted 1\nDimension 3\nVertices\n" << V.size() << "\n";
		for (const auto &v : V)
			out << v.x() << " " << v.y() << " " << v.z() << " 0\n";
		out << "Tetrahedra\n" << T.size() << "\n";
		for (const auto &t : T)
			out << t(0) + 1 << " " << t(1) + 1 << " " << t(2) + 1 << " " << t(3) + 1 << " 1\n";
		out << "End\n";
		return path;
	}

	/// u_x on x=1 and u_y on y=1 follow (lambda - 1) * ramp(t)
	json cube_args(const std::string &mesh, const double lambda, const double phi,
				   const int n_steps, const double t_end, const std::string &ramp)
	{
		const std::string u = "(" + std::to_string(lambda - 1.0) + ")*(" + ramp + ")";
		json args = R"({
			"materials": {
				"type": "GrowthElasticity", "rho": 1000.0,
				"displacement_space_id": 0, "growth_space_id": 1,
				"normal_direction": [0, 0, 1],
				"elastic_material": {"type": "MaterialSum", "models": [
					{"type": "IsochoricNeoHookean", "E": 2600.0, "nu": 0.26},
					{"type": "HGODispersion", "k1": 2400.0, "k2": 0.01, "kappa": 0.24, "k_chi": 100,
					 "fiber_direction": [1, 0, 0]},
					{"type": "VolumePenalty", "k": 1805.5555555555557}]}
			},
			"space": {"discr_order": [{"fe_space": 0, "order": 1}, {"fe_space": 1, "order": 1}]},
			"solver": {"linear": {"solver": "Eigen::SimplicialLDLT"},
					   "nonlinear": {"grad_norm_tol": 1e-10}},
			"output": {"log": {"level": "warning"}}
		})"_json;
		args["geometry"] = json::array({{{"mesh", mesh},
										 {"surface_selection", json::array({
																   {{"id", 1}, {"axis", "-x"}, {"position", 1e-6}},
																   {{"id", 2}, {"axis", "-y"}, {"position", 1e-6}},
																   {{"id", 3}, {"axis", "-z"}, {"position", 1e-6}},
																   {{"id", 4}, {"axis", "x"}, {"position", 1 - 1e-6}},
																   {{"id", 5}, {"axis", "y"}, {"position", 1 - 1e-6}},
															   })}}});
		args["boundary_conditions"]["dirichlet_boundary"] = json::array({
			{{"id", 1}, {"value", {0, 0, 0}}, {"dimension", {true, false, false}}, {"fe_space", 0}},
			{{"id", 2}, {"value", {0, 0, 0}}, {"dimension", {false, true, false}}, {"fe_space", 0}},
			{{"id", 3}, {"value", {0, 0, 0}}, {"dimension", {false, false, true}}, {"fe_space", 0}},
			{{"id", 4}, {"value", {u, "0", "0"}}, {"dimension", {true, false, false}}, {"fe_space", 0}},
			{{"id", 5}, {"value", {"0", u, "0"}}, {"dimension", {false, true, false}}, {"fe_space", 0}},
		});
		args["materials"]["growth_law"] = {{"threshold", phi}, {"n_steps", n_steps}, {"t_end", t_end}};
		return args;
	}

	/// theta at every growth node after the solve
	Eigen::VectorXd solve_growth(const json &args, const int n_vertices)
	{
		State state;
		state.init(args, /*strict_validation=*/true);
		state.set_max_threads(1);
		state.load_mesh();
		Eigen::MatrixXd sol;
		state.solve(sol);
		REQUIRE(sol.rows() == 4 * n_vertices);
		return sol.col(0).tail(n_vertices);
	}

	std::filesystem::path scratch_dir()
	{
		const auto dir = std::filesystem::temp_directory_path() / "polyfem_growth_law_test";
		std::filesystem::create_directories(dir);
		return dir;
	}
} // namespace

TEST_CASE("growth_law_off_keeps_prescribed_mode", "[growth][growth_law]")
{
	// Without a threshold the law is off and the Stage-1 prescribed path runs
	// unchanged (jse injects the growth_law block with threshold "none"). The
	// homogeneous V2 state is prescribed here, so the solve must hold it.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const auto dir = scratch_dir();
	const std::string mesh = write_cube_mesh(dir, n);
	const double theta_v2 = std::sqrt(1.5) * std::sqrt(1.5) / activation_stretch(469.0);

	const std::string field = (dir / "theta_prescribed.txt").string();
	{
		std::ofstream out(field);
		out.precision(17);
		for (int a = 0; a < nv; ++a)
			out << theta_v2 << "\n";
	}
	json args = cube_args(mesh, std::sqrt(1.5), 469.0, 1, 1.0, "t");
	args["materials"].erase("growth_law");
	args["boundary_conditions"]["growth_nodal_field"] = field;
	// the prescribed path is a static solve at t = 1
	for (auto &bc : args["boundary_conditions"]["dirichlet_boundary"])
		for (auto &v : bc["value"])
			if (v.is_string())
			{
				std::string s = v;
				for (size_t p = s.find("(t)"); p != std::string::npos; p = s.find("(t)"))
					s.replace(p, 3, "(1)");
				v = s;
			}

	const Eigen::VectorXd theta = solve_growth(args, nv);
	for (int a = 0; a < nv; ++a)
		CHECK(theta(a) == theta_v2);

	SECTION("settings without a threshold are refused, not ignored")
	{
		json bad = cube_args(mesh, std::sqrt(1.5), 469.0, 3, 1.0, "t");
		bad["materials"]["growth_law"].erase("threshold");
		State state;
		CHECK_THROWS(state.init(bad, /*strict_validation=*/true));
	}
}

TEST_CASE("growth_law_reference_path", "[growth][growth_law]")
{
	// The reference reproduces the calibration-path numbers of the formulation
	// (Q1): f_I = 469 Pa at vartheta^e = 1.2, and 0 at the stress-free state.
	CHECK(f_I(1.0) == Catch::Approx(0.0).margin(1e-5));
	CHECK(f_I(1.2) == Catch::Approx(469.0).margin(0.5));
}

TEST_CASE("growth_law_V2_cube", "[growth][growth_law]")
{
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const std::string mesh = write_cube_mesh(scratch_dir(), n);
	const double phi = 469.0;
	const double vstar = activation_stretch(phi);

	SECTION("above threshold: theta = lambda^2 / vartheta*, uniform")
	{
		const double lambda = std::sqrt(1.5);
		const Eigen::VectorXd theta = solve_growth(cube_args(mesh, lambda, phi, 1, 1.0, "t"), nv);
		const double expected = lambda * lambda / vstar;
		for (int a = 0; a < nv; ++a)
			CHECK(theta(a) == Catch::Approx(expected).epsilon(1e-6));
	}

	SECTION("below threshold: no growth")
	{
		const double lambda = std::sqrt(1.1);
		REQUIRE(lambda * lambda < vstar);
		const Eigen::VectorXd theta = solve_growth(cube_args(mesh, lambda, phi, 1, 1.0, "t"), nv);
		for (int a = 0; a < nv; ++a)
			CHECK(theta(a) == 1.0);
	}

	SECTION("negative threshold: Pi_inc bounded below, growth stopped by the elastic energy")
	{
		// Route B admits phi < 0 (the shear supply); growth must still stop at a
		// finite theta where f_I = phi, i.e. at in-plane elastic compression.
		const double phi_neg = -300.0;
		const double vneg = activation_stretch(phi_neg);
		REQUIRE(vneg < 1.0);
		const double lambda = std::sqrt(1.2);
		const Eigen::VectorXd theta = solve_growth(cube_args(mesh, lambda, phi_neg, 1, 1.0, "t"), nv);
		for (int a = 0; a < nv; ++a)
			CHECK(theta(a) == Catch::Approx(lambda * lambda / vneg).epsilon(1e-6));
	}

	SECTION("irreversibility over load and unload")
	{
		// stretch up to lambda at t = 1 and back to the reference at t = 2;
		// theta must keep its peak value through the unloading increments.
		const double lambda = std::sqrt(1.5);
		const Eigen::VectorXd theta = solve_growth(cube_args(mesh, lambda, phi, 4, 2.0, "min(t, 2 - t)"), nv);
		const double peak = lambda * lambda / vstar;
		for (int a = 0; a < nv; ++a)
			CHECK(theta(a) == Catch::Approx(peak).epsilon(1e-6));
	}
}

TEST_CASE("growth_law_increment_output", "[growth][growth_law]")
{
	// Ramp the V2 stretch over four increments: growth activates in the first
	// increment whose end-of-increment area stretch exceeds vartheta*, and
	// save_increments writes one VTU per converged increment plus a PVD.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const auto dir = scratch_dir();
	const std::string mesh = write_cube_mesh(dir, n);
	const double phi = 469.0, vstar = activation_stretch(phi);
	const double lambda = std::sqrt(1.5);
	const int n_steps = 4;

	int expected_step = 0;
	for (int k = 1; k <= n_steps && expected_step == 0; ++k)
	{
		const double lk = 1.0 + (lambda - 1.0) * double(k) / n_steps;
		if (lk * lk > vstar * (1 + 1e-6))
			expected_step = k;
	}
	REQUIRE(expected_step > 1); // the ramp must start below threshold

	for (const bool save : {true, false})
	{
		const auto out_dir = dir / (save ? "increments_on" : "increments_off");
		std::filesystem::remove_all(out_dir);
		std::filesystem::create_directories(out_dir);
		json args = cube_args(mesh, lambda, phi, n_steps, 1.0, "t");
		args["materials"]["growth_law"]["save_increments"] = save;
		args["output"]["directory"] = out_dir.string();

		State state;
		state.init(args, /*strict_validation=*/true);
		state.set_max_threads(1);
		state.load_mesh();
		Eigen::MatrixXd sol;
		state.solve(sol);
		REQUIRE(sol.rows() == 4 * nv);

		const auto vf = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(state.variational_formulation);
		REQUIRE(vf);
		const Eigen::VectorXi &act = vf->growth_activation_step();
		REQUIRE(act.size() == nv);
		for (int a = 0; a < nv; ++a)
			CHECK(act(a) == expected_step);

		for (int k = 1; k <= n_steps; ++k)
			CHECK(std::filesystem::exists(out_dir / ("growth_step_" + std::to_string(k) + ".vtu")) == save);
		CHECK(std::filesystem::exists(out_dir / "growth_steps.pvd") == save);
		if (save)
		{
			std::ifstream pvd(out_dir / "growth_steps.pvd");
			const std::string text((std::istreambuf_iterator<char>(pvd)), std::istreambuf_iterator<char>());
			int n_sets = 0;
			for (size_t p = text.find("<DataSet"); p != std::string::npos; p = text.find("<DataSet", p + 1))
				++n_sets;
			CHECK(n_sets == n_steps);
		}
	}
}
