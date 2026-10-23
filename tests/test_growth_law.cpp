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
#include <polyfem/assembler/GrowthElasticity.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/io/MatrixIO.hpp>
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

namespace
{
	/// psi_e at a general Ce (n0 = z, fiber x): the same fitted energy as psi(a, b)
	double psi_C(const Eigen::Matrix3d &Ce)
	{
		const double mu = E_ / (2.0 * (1.0 + nu_));
		const double I1 = Ce.trace(), J = std::sqrt(Ce.determinant()), I4 = Ce(0, 0);
		const double E4 = kappa_ * I1 + (1.0 - 3.0 * kappa_) * I4 - 1.0;
		const double nh = 0.5 * mu * (std::pow(J, -2.0 / 3.0) * I1 - 3.0);
		const double fib = k1_ / (2.0 * k2_) * (std::exp(k2_ * E4 * E4) - 1.0) / (1.0 + std::exp(-kchi_ * E4));
		const double vol = 0.5 * kvol_ * (0.5 * (J * J - 1.0) - std::log(J));
		return nh + fib + vol;
	}

	/// f_I(Ce) = -d/dtheta [theta psi(Fgi^T Ce Fgi)] at theta = 1, Fgi = diag(theta^-1/2, theta^-1/2, 1)
	double f_I_C(const Eigen::Matrix3d &Ce)
	{
		const auto g = [&](const double th) {
			const Eigen::Vector3d d(1 / std::sqrt(th), 1 / std::sqrt(th), 1.0);
			return th * psi_C(d.asDiagonal() * Ce * d.asDiagonal());
		};
		const double h = 1e-6;
		return -(g(1 + h) - g(1 - h)) / (2 * h);
	}

	json material_params()
	{
		return R"({
			"type": "GrowthElasticity", "rho": 1000.0,
			"displacement_space_id": 0, "growth_space_id": 1,
			"normal_direction": [0, 0, 1],
			"elastic_material": {"type": "MaterialSum", "models": [
				{"type": "IsochoricNeoHookean", "E": 2600.0, "nu": 0.26},
				{"type": "HGODispersion", "k1": 2400.0, "k2": 0.01, "kappa": 0.24, "k_chi": 100,
				 "fiber_direction": [1, 0, 0]},
				{"type": "VolumePenalty", "k": 1805.5555555555557}]}
		})"_json;
	}

	std::string write_threshold_vtk(const std::filesystem::path &dir, const int n_cells, const double phi, const double phi_par)
	{
		const std::string path = (dir / "threshold_lagged.vtk").string();
		std::ofstream out(path);
		out.precision(17);
		out << "# vtk DataFile Version 3.0\nthreshold\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 0 double\n"
			<< "CELLS 0 0\nCELL_TYPES 0\nCELL_DATA " << n_cells << "\n";
		for (const auto &[name, v] : {std::pair<std::string, double>{"PHI", phi}, {"PHI_PAR", phi_par}})
		{
			out << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
			for (int i = 0; i < n_cells; ++i)
				out << v << "\n";
		}
		return path;
	}
} // namespace

TEST_CASE("growth_driving_force_of_Ce", "[growth][growth_law]")
{
	// The Ce-based evaluator used for the lagged shear supply against an
	// independent scalar implementation of the fitted energy.
	assembler::GrowthElasticity ge;
	ge.set_size(3);
	Units units;
	ge.add_multimaterial(0, material_params(), units, "");
	const RowVectorNd pos = RowVectorNd::Zero(3);

	SECTION("diagonal Ce: f_I = a/2 dpsi/da - psi")
	{
		for (const auto &[a, b] : {std::pair<double, double>{1.0, 1.0}, {1.1, 0.9}, {std::sqrt(1.2), 0.85}, {0.95, 1.05}})
		{
			const double h = 1e-6;
			const double ref = 0.5 * a * (psi(a + h, b) - psi(a - h, b)) / (2 * h) - psi(a, b);
			const Eigen::Vector3d d(a * a, a * a, b * b);
			CHECK(ge.driving_force(0, pos, 0, d.asDiagonal()) == Catch::Approx(ref).margin(1e-3));
		}
	}

	SECTION("sheared Ce: general reference")
	{
		Eigen::Matrix3d Ce;
		Ce << 1.21, 0.05, 0.30,
			0.05, 1.15, -0.20,
			0.30, -0.20, 0.80;
		CHECK(ge.driving_force(0, pos, 0, Ce) == Catch::Approx(f_I_C(Ce)).margin(1e-3));
		// removing the transverse shear at the layer thickness changes f_I
		const Eigen::Vector3d n(0, 0, 1);
		const Eigen::Matrix3d nn = n * n.transpose(), P = Eigen::Matrix3d::Identity() - nn;
		const Eigen::Matrix3d Ce_ns = P * Ce * P + (1.0 / n.dot(Ce.inverse() * n)) * nn;
		CHECK(ge.driving_force(0, pos, 0, Ce_ns) == Catch::Approx(f_I_C(Ce_ns)).margin(1e-3));
		CHECK(std::abs(ge.driving_force(0, pos, 0, Ce) - ge.driving_force(0, pos, 0, Ce_ns)) > 1.0);
	}
}

TEST_CASE("growth_law_lagged_shear_supply", "[growth][growth_law]")
{
	// On the shear-free V2 cube the lagged shear supply is zero, so lagged mode
	// with PHI_PAR = phi reproduces the frozen solution and activation increment.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const auto dir = scratch_dir();
	const std::string mesh = write_cube_mesh(dir, n);
	const double phi = 469.0, vstar = activation_stretch(phi);
	const double lambda = std::sqrt(1.5);
	const std::string vtk = write_threshold_vtk(dir, 6 * n * n * n, -999.0, phi);

	json args = cube_args(mesh, lambda, phi, 4, 1.0, "t");
	args["materials"]["growth_law"]["threshold"] = {{"type", "per_element_file"}, {"path", vtk}, {"field", "PHI"}};
	args["materials"]["growth_law"]["shear_supply"] = "lagged";
	args["materials"]["growth_law"]["theta_crit"] = 1.2;

	State state;
	state.init(args, /*strict_validation=*/true);
	state.set_max_threads(1);
	state.load_mesh();
	Eigen::MatrixXd sol;
	state.solve(sol);
	REQUIRE(sol.rows() == 4 * nv);

	const auto vf = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(state.variational_formulation);
	REQUIRE(vf);
	// PHI (-999) must be ignored in lagged mode: only PHI_PAR + f_gamma enters
	CHECK(vf->growth_shear_supply().cwiseAbs().maxCoeff() < 1e-6);
	CHECK(vf->growth_threshold().minCoeff() == Catch::Approx(phi).margin(1e-6));
	const Eigen::VectorXd theta = sol.col(0).tail(nv);
	for (int a = 0; a < nv; ++a)
		CHECK(theta(a) == Catch::Approx(lambda * lambda / vstar).epsilon(1e-6));

	SECTION("predictor and per-pass updates agree on the shear-free cube")
	{
		for (const std::string mode : {"predictor", "pass"})
		{
			json a2 = args;
			a2["materials"]["growth_law"]["shear_update"] = mode;
			State s2;
			s2.init(a2, /*strict_validation=*/true);
			s2.set_max_threads(1);
			s2.load_mesh();
			Eigen::MatrixXd sol2;
			s2.solve(sol2);
			const auto vf2 = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(s2.variational_formulation);
			REQUIRE(vf2);
			CHECK(vf2->growth_shear_supply().cwiseAbs().maxCoeff() < 1e-6);
			for (int a = 0; a < nv; ++a)
				CHECK(sol2(sol2.rows() - nv + a) == Catch::Approx(lambda * lambda / vstar).epsilon(1e-6));
		}
	}

	SECTION("unknown shear_update is refused")
	{
		json bad = args;
		bad["materials"]["growth_law"]["shear_update"] = "sometimes";
		State s2;
		CHECK_THROWS(s2.init(bad, /*strict_validation=*/true));
	}

	SECTION("lagged needs theta_crit")
	{
		json bad = args;
		bad["materials"]["growth_law"].erase("theta_crit");
		State s2;
		CHECK_THROWS(s2.init(bad, /*strict_validation=*/true));
	}
}

TEST_CASE("growth_law_threshold_at_state", "[growth][growth_law]")
{
	// Variant (b) at the state: with PHI_PAR re-evaluated on the solved state and
	// iterated to its fixed point (parallel_threshold lagged, shear_update pass),
	// the cube stops exactly at vartheta^e = theta_crit, theta = lambda^2 / theta_crit,
	// for any theta_crit (here not the calibration value). A second solve with the
	// threshold frozen at that converged state (parallel_threshold state) must
	// reproduce it: the state threshold is the fixed point.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const auto dir = scratch_dir();
	const std::string mesh = write_cube_mesh(dir, n);
	const double lambda = std::sqrt(1.5), crit = 1.25;

	json args = cube_args(mesh, lambda, 469.0, 1, 1.0, "t");
	args["materials"]["growth_law"]["shear_supply"] = "lagged";
	args["materials"]["growth_law"]["parallel_threshold"] = "lagged";
	args["materials"]["growth_law"]["shear_update"] = "pass";
	args["materials"]["growth_law"]["shear_update_tol"] = 1e-4;
	args["materials"]["growth_law"]["theta_crit"] = crit;

	State state;
	state.init(args, /*strict_validation=*/true);
	state.set_max_threads(1);
	state.load_mesh();
	Eigen::MatrixXd sol;
	state.solve(sol);
	REQUIRE(sol.rows() == 4 * nv);
	const auto vf = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(state.variational_formulation);
	REQUIRE(vf);
	for (int a = 0; a < nv; ++a)
		CHECK(sol(3 * nv + a) == Catch::Approx(lambda * lambda / crit).epsilon(1e-6));
	REQUIRE(vf->growth_attainment().size() > 0);
	CHECK(vf->growth_attainment().minCoeff() == Catch::Approx(1.0).epsilon(1e-6));
	CHECK(vf->growth_attainment().maxCoeff() == Catch::Approx(1.0).epsilon(1e-6));
	CHECK(vf->growth_lag_residual().cwiseAbs().maxCoeff() < 1e-3);
	const double phi_par = vf->growth_threshold_parallel()(0);
	CHECK(phi_par > 0);
	// route-B split: no shear, so S = 0 and D = v_n PHI_PAR (theta - 1) per element
	REQUIRE(vf->growth_dissipation().size() > 0);
	CHECK(vf->growth_shear_supply_work().cwiseAbs().maxCoeff() < 1e-6);
	CHECK(vf->growth_dissipation().minCoeff() == Catch::Approx(phi_par * (lambda * lambda / crit - 1.0)).epsilon(1e-5));
	CHECK(vf->growth_dissipation().maxCoeff() == Catch::Approx(phi_par * (lambda * lambda / crit - 1.0)).epsilon(1e-5));

	SECTION("threshold frozen at that state reproduces it")
	{
		const std::string state_file = (dir / "threshold_state.hdf5").string();
		std::filesystem::remove(state_file);
		const Eigen::MatrixXd u = sol.topRows(3 * nv);
		REQUIRE(io::write_matrix(state_file, "u", u));

		json b = cube_args(mesh, lambda, 469.0, 1, 1.0, "t");
		b["materials"]["growth_law"]["parallel_threshold"] = "state";
		b["materials"]["growth_law"]["threshold_state"] = state_file;
		b["materials"]["growth_law"]["theta_crit"] = crit;
		State s2;
		s2.init(b, /*strict_validation=*/true);
		s2.set_max_threads(1);
		s2.load_mesh();
		Eigen::MatrixXd sol2;
		s2.solve(sol2);
		const auto vf2 = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(s2.variational_formulation);
		REQUIRE(vf2);
		CHECK(vf2->growth_threshold_parallel()(0) == Catch::Approx(phi_par).epsilon(1e-8));
		for (int a = 0; a < nv; ++a)
			CHECK(sol2(3 * nv + a) == Catch::Approx(lambda * lambda / crit).epsilon(1e-6));
	}

	SECTION("lagged parallel threshold needs a lagged shear supply")
	{
		json bad = args;
		bad["materials"]["growth_law"]["shear_supply"] = "frozen";
		State s3;
		CHECK_THROWS(s3.init(bad, /*strict_validation=*/true));
	}
}
