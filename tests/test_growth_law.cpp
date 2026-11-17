// Solved growth (growth_law) on a homogeneous single-material cube.
//
// A unit cube (2x2x2 cells, Kuhn split into tets) with rollers on x=0, y=0,
// z=0, prescribed u_x on x=1 and u_y on y=1 (equibiaxial in-plane stretch
// lambda), traction-free top. Wall normal n0 = z, fiber in-plane along x, the
// fitted MaterialSum. The solution is homogeneous:
//
//     F = diag(lambda, lambda, b),  Fg = diag(sqrt(theta), sqrt(theta), 1),
//     Fe = diag(a, a, b),  a = lambda / sqrt(theta),  dpsi/db (Fe) = 0,
//
// so f_I depends on the elastic area stretch a^2 alone (the equibiaxial,
// traction-free path), and the growth law gives theta = lambda^2 / vartheta*, with
// f_I(vartheta*) = phi, wherever lambda^2 > vartheta*, and theta = theta_n
// otherwise. The reference below is an independent scalar implementation of
// the fitted energy on that path (not the assembler).

#include <polyfem/State.hpp>
#include <polyfem/varforms/GrowthElasticVarForm.hpp>
#include <polyfem/assembler/GrowthElasticity.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OutputData.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

		// Gmsh MSH 4.1 ASCII: one volume entity (physical tag 1) holding every
		// node and tetrahedron
		const std::string path = (dir / "cube.msh").string();
		std::ofstream out(path);
		out.precision(17);
		out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
			<< "$Entities\n0 0 0 1\n1 0 0 0 1 1 1 1 1 0\n$EndEntities\n";
		out << "$Nodes\n1 " << V.size() << " 1 " << V.size() << "\n3 1 0 " << V.size() << "\n";
		for (size_t a = 0; a < V.size(); ++a)
			out << a + 1 << "\n";
		for (const auto &v : V)
			out << v.x() << " " << v.y() << " " << v.z() << "\n";
		out << "$EndNodes\n$Elements\n1 " << T.size() << " 1 " << T.size() << "\n3 1 4 " << T.size() << "\n";
		for (size_t e = 0; e < T.size(); ++e)
			out << e + 1 << " " << T[e](0) + 1 << " " << T[e](1) + 1 << " " << T[e](2) + 1 << " " << T[e](3) + 1 << "\n";
		out << "$EndElements\n";
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
	// Without a threshold the law is off and the prescribed-growth path runs
	// unchanged (jse injects the growth_law block with threshold "none"). The
	// homogeneous grown cube state is prescribed here, so the solve must hold it.
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
	// The scalar reference on the equibiaxial path: f_I = 469 Pa at an elastic
	// area stretch of 1.2 for the fitted material, and 0 at the stress-free state.
	CHECK(f_I(1.0) == Catch::Approx(0.0).margin(1e-5));
	CHECK(f_I(1.2) == Catch::Approx(469.0).margin(0.5));
}

TEST_CASE("growth_law_cube_analytic", "[growth][growth_law]")
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
		// A threshold built from the state admits phi < 0 (the shear supply); growth must still stop at a
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
	// Ramp the cube stretch over four increments: growth activates in the first
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

	/// legacy VTK with one per-element CELL_DATA field (no geometry: the reader binds by element id)
	std::string write_cell_field_vtk(const std::filesystem::path &path, const std::string &name, const Eigen::VectorXd &values)
	{
		std::ofstream out(path);
		out.precision(17);
		out << "# vtk DataFile Version 3.0\nthreshold\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 0 double\n"
			<< "CELLS 0 0\nCELL_TYPES 0\nCELL_DATA " << values.size() << "\n"
			<< "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
		for (int i = 0; i < values.size(); ++i)
			out << values(i) << "\n";
		return path.string();
	}

	struct GrowthRun
	{
		Eigen::MatrixXd sol;
		std::shared_ptr<varform::GrowthElasticVarForm> vf;
		std::shared_ptr<State> state; // keeps vf alive
	};

	GrowthRun run_growth(const json &args)
	{
		GrowthRun r;
		r.state = std::make_shared<State>();
		r.state->init(args, /*strict_validation=*/true);
		r.state->set_max_threads(1);
		r.state->load_mesh();
		r.state->solve(r.sol);
		r.vf = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(r.state->variational_formulation);
		REQUIRE(r.vf);
		return r;
	}
} // namespace

TEST_CASE("growth_driving_force_of_Ce", "[growth][growth_law]")
{
	// The Ce-based evaluator used for the lagged shear supply against an
	// independent scalar implementation of the fitted energy.
	assembler::GrowthElasticity ge;
	ge.set_size(3);
	Units units;
	ge.add_multimaterial(0, material_params(), units, io::FileSystemIO("."));
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

TEST_CASE("growth_law_threshold_from_state", "[growth][growth_law]")
{
	// Threshold from_state: phi built by the solver from the state (C_check at
	// theta_crit) and iterated to its fixed point within the increment. On the
	// shear-free cube f_gamma = 0 and the cube
	// stops exactly at vartheta^e = theta_crit, theta = lambda^2 / theta_crit, for
	// one increment or several (path independence).
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const int n_el = 6 * n * n * n;
	const auto dir = scratch_dir();
	const std::string mesh = write_cube_mesh(dir, n);
	const double lambda = std::sqrt(1.5), crit = 1.25;

	json args = cube_args(mesh, lambda, 469.0, 1, 1.0, "t");
	args["materials"]["growth_law"]["threshold"] = "from_state";
	args["materials"]["growth_law"]["relag_tol"] = 1e-4;
	args["materials"]["growth_law"]["theta_crit"] = crit;

	const GrowthRun r = run_growth(args);
	REQUIRE(r.sol.rows() == 4 * nv);
	const auto &vf = r.vf;
	for (int a = 0; a < nv; ++a)
		CHECK(r.sol(3 * nv + a) == Catch::Approx(lambda * lambda / crit).epsilon(1e-6));
	REQUIRE(vf->growth_attainment().size() > 0);
	CHECK(vf->growth_attainment().minCoeff() == Catch::Approx(1.0).epsilon(1e-6));
	CHECK(vf->growth_attainment().maxCoeff() == Catch::Approx(1.0).epsilon(1e-6));
	CHECK(vf->growth_lag_residual().cwiseAbs().maxCoeff() < 1e-3);
	CHECK(vf->growth_shear_supply().cwiseAbs().maxCoeff() < 1e-6);
	const double phi_par = vf->growth_threshold_parallel()(0);
	CHECK(phi_par > 0);
	// threshold split: no shear, so S = 0 and D = v_n PHI_PAR (theta - 1) per element
	REQUIRE(vf->growth_dissipation().size() > 0);
	CHECK(vf->growth_shear_supply_work().cwiseAbs().maxCoeff() < 1e-6);
	CHECK(vf->growth_dissipation().minCoeff() == Catch::Approx(phi_par * (lambda * lambda / crit - 1.0)).epsilon(1e-5));
	CHECK(vf->growth_dissipation().maxCoeff() == Catch::Approx(phi_par * (lambda * lambda / crit - 1.0)).epsilon(1e-5));

	SECTION("the converged threshold, given as a file, reproduces the solve (fixed point)")
	{
		REQUIRE(vf->growth_threshold().size() == n_el);
		json b = cube_args(mesh, lambda, 469.0, 1, 1.0, "t");
		b["materials"]["growth_law"]["threshold"] = {
			{"type", "per_element_file"},
			{"path", write_cell_field_vtk(dir / "threshold_fixed_point.vtk", "PHI", vf->growth_threshold())},
			{"field", "PHI"}};
		const GrowthRun r2 = run_growth(b);
		for (int a = 0; a < nv; ++a)
			CHECK(r2.sol(3 * nv + a) == Catch::Approx(lambda * lambda / crit).epsilon(1e-6));
	}

	SECTION("several increments reach the same state")
	{
		json c = args;
		c["materials"]["growth_law"]["n_steps"] = 4;
		const GrowthRun r4 = run_growth(c);
		for (int a = 0; a < nv; ++a)
			CHECK(r4.sol(3 * nv + a) == Catch::Approx(lambda * lambda / crit).epsilon(1e-6));
	}

	SECTION("capped updates: 1 holds the predictor threshold, a large cap is the uncapped fixed point")
	{
		json cap1 = args;
		cap1["materials"]["growth_law"]["relag_max_updates"] = 1;
		const GrowthRun r1 = run_growth(cap1);
		// cap 1 = the threshold built on the elastic predictor, held: a solve with
		// that threshold given as a file reproduces it
		json held = cube_args(mesh, lambda, 469.0, 1, 1.0, "t");
		held["materials"]["growth_law"]["threshold"] = {
			{"type", "per_element_file"},
			{"path", write_cell_field_vtk(dir / "threshold_predictor.vtk", "PHI", r1.vf->growth_threshold())},
			{"field", "PHI"}};
		const GrowthRun rh = run_growth(held);
		CHECK((r1.sol.bottomRows(nv) - rh.sol.bottomRows(nv)).cwiseAbs().maxCoeff() < 1e-6);

		json big = args;
		big["materials"]["growth_law"]["relag_max_updates"] = 100;
		const GrowthRun rb = run_growth(big);
		for (int a = 0; a < nv; ++a)
			CHECK(rb.sol(3 * nv + a) == Catch::Approx(lambda * lambda / crit).epsilon(1e-6));
	}

	SECTION("from_state needs theta_crit")
	{
		json bad = args;
		bad["materials"]["growth_law"].erase("theta_crit");
		State s2;
		CHECK_THROWS(s2.init(bad, /*strict_validation=*/true));
	}
}

TEST_CASE("growth_law_final_tolerance_pass", "[growth][growth_law]")
{
	// The confirmation pass re-solves the settled set with a tighter gradient
	// tolerance; the solution is unchanged up to that tolerance.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const std::string mesh = write_cube_mesh(scratch_dir(), n);
	const double phi = 469.0, lambda = std::sqrt(1.5);
	json args = cube_args(mesh, lambda, phi, 1, 1.0, "t");
	args["materials"]["growth_law"]["final_tol_factor"] = 0.01;
	const double vstar = activation_stretch(phi);
	const Eigen::VectorXd theta = solve_growth(args, nv);
	for (int a = 0; a < nv; ++a)
		CHECK(theta(a) == Catch::Approx(lambda * lambda / vstar).epsilon(1e-6));

	json bad = cube_args(mesh, lambda, phi, 1, 1.0, "t");
	bad["materials"]["growth_law"]["final_tol_factor"] = 0.0;
	State state;
	CHECK_THROWS(state.init(bad, /*strict_validation=*/true));
}

TEST_CASE("growth_law_increment_splitting", "[growth][growth_law]")
{
	// A large stretch in one increment with the Newton iterations capped (no
	// growth: the threshold is out of reach, so only the elastic difficulty
	// scales with the increment): the full increment fails, its halves
	// converge. Without splitting the run aborts; with it, it reaches the same
	// state as an uncapped single increment.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const std::string mesh = write_cube_mesh(scratch_dir(), n);
	const double lambda = 1.9;
	json ref = cube_args(mesh, lambda, 1e9, 1, 1.0, "t");

	State s_ref;
	s_ref.init(ref, /*strict_validation=*/true);
	s_ref.set_max_threads(1);
	s_ref.load_mesh();
	Eigen::MatrixXd sol_ref;
	s_ref.solve(sol_ref);

	json args = ref;
	args["solver"]["nonlinear"]["max_iterations"] = 5;
	args["solver"]["augmented_lagrangian"]["nonlinear"]["max_iterations"] = 5;

	json no_split = args;
	no_split["materials"]["growth_law"]["max_increment_splits"] = 0;
	{
		State state;
		state.init(no_split, /*strict_validation=*/true);
		state.set_max_threads(1);
		state.load_mesh();
		Eigen::MatrixXd sol;
		CHECK_THROWS(state.solve(sol));
	}

	args["materials"]["growth_law"]["max_increment_splits"] = 5; // margin: smaller steps need fewer iterations
	State state;
	state.init(args, /*strict_validation=*/true);
	state.set_max_threads(1);
	state.load_mesh();
	Eigen::MatrixXd sol;
	state.solve(sol);
	REQUIRE(sol.rows() == sol_ref.rows());
	CHECK((sol - sol_ref).cwiseAbs().maxCoeff() < 1e-8);
	for (int a = 0; a < nv; ++a)
		CHECK(sol(3 * nv + a) == 1.0);
}

namespace
{
	/// Pe = d psi_e / dF at a general F by central differences of psi_C(F^T F)
	Eigen::Matrix3d pk1_ref(const Eigen::Matrix3d &F)
	{
		const double h = 1e-6;
		Eigen::Matrix3d P;
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
			{
				Eigen::Matrix3d Fp = F, Fm = F;
				Fp(a, b) += h;
				Fm(a, b) -= h;
				P(a, b) = (psi_C(Fp.transpose() * Fp) - psi_C(Fm.transpose() * Fm)) / (2 * h);
			}
		return P;
	}

	/// Fg^-1 for n0 = z
	Eigen::Matrix3d growth_inverse(const double theta, const double vn)
	{
		return Eigen::Vector3d(1 / std::sqrt(theta), 1 / std::sqrt(theta), 1 / vn).asDiagonal();
	}

	void check_close(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B, const double tol)
	{
		CHECK((A - B).cwiseAbs().maxCoeff() < tol * std::max(1.0, B.cwiseAbs().maxCoeff()));
	}
} // namespace

TEST_CASE("growth_grown_stresses", "[growth][growth_stress]")
{
	// Stresses of J^g psi_e(F Fg^-1) against the independent scalar energy:
	// P = J^g Pe(Fe) Fg^-T, sigma = Pe(Fe) Fe^T / J^e, S = F^-1 P.
	assembler::GrowthElasticity ge;
	ge.set_size(3);
	Units units;
	const RowVectorNd pos = RowVectorNd::Zero(3);
	Eigen::Matrix3d F;
	F << 1.12, 0.04, 0.10,
		-0.03, 1.05, 0.06,
		0.08, -0.05, 0.88;
	Eigen::Matrix3d s, P, S;

	SECTION("no growth: the plain elastic stresses at F")
	{
		ge.add_multimaterial(0, material_params(), units, io::FileSystemIO("."));
		ge.grown_stresses(0, pos, 0, F, 1.0, s, P, S);
		const Eigen::Matrix3d P_ref = pk1_ref(F);
		check_close(P, P_ref, 1e-6);
		check_close(s, P_ref * F.transpose() / F.determinant(), 1e-6);
		check_close(S, F.inverse() * P_ref, 1e-6);
	}

	for (const double vn : {1.0, 1.1})
	{
		DYNAMIC_SECTION("grown, vn = " << vn)
		{
			json params = material_params();
			params["normal_growth"] = vn;
			ge.add_multimaterial(0, params, units, io::FileSystemIO("."));
			const double theta = 1.3, Jg = theta * vn;
			const Eigen::Matrix3d Fgi = growth_inverse(theta, vn);
			const Eigen::Matrix3d Fe = F * Fgi;
			const Eigen::Matrix3d Pe = pk1_ref(Fe);

			ge.grown_stresses(0, pos, 0, F, theta, s, P, S);
			check_close(P, Jg * Pe * Fgi.transpose(), 1e-6);
			check_close(s, Pe * Fe.transpose() / Fe.determinant(), 1e-6);
			check_close(S, F.inverse() * P, 1e-10);
			CHECK((s - s.transpose()).cwiseAbs().maxCoeff() < 1e-8 * s.cwiseAbs().maxCoeff());

			// F = Fg: the grown reference state is stress free
			ge.grown_stresses(0, pos, 0, Fgi.inverse(), theta, s, P, S);
			CHECK(s.cwiseAbs().maxCoeff() < 1e-8);
			CHECK(P.cwiseAbs().maxCoeff() < 1e-8);
			CHECK(S.cwiseAbs().maxCoeff() < 1e-8);
		}
	}
}

TEST_CASE("growth_law_stress_output", "[growth][growth_stress]")
{
	// Cube above threshold: F = diag(lambda, lambda, b), Fe = diag(a, a, b)
	// with a^2 = vartheta* and the top traction free, so sigma_zz = 0. The
	// exported stresses must be the grown ones; the ungrown material at the
	// total F (what the elastic base evaluates) has sigma_zz far from 0.
	const int n = 2;
	const int nv = (n + 1) * (n + 1) * (n + 1);
	const int n_el = 6 * n * n * n;
	const std::string mesh = write_cube_mesh(scratch_dir(), n);
	const double phi = 469.0, vstar = activation_stretch(phi);
	const double lambda = std::sqrt(1.5);

	State state;
	state.init(cube_args(mesh, lambda, phi, 1, 1.0, "t"), /*strict_validation=*/true);
	state.set_max_threads(1);
	state.load_mesh();
	Eigen::MatrixXd sol;
	state.solve(sol);
	REQUIRE(sol.rows() == 4 * nv);
	const auto vf = std::dynamic_pointer_cast<varform::GrowthElasticVarForm>(state.variational_formulation);
	REQUIRE(vf);

	// reference state
	const double theta = lambda * lambda / vstar, a = std::sqrt(vstar);
	const double b = bisect([&](double bb) { return dpsi_db(a, bb); }, 0.3, 1.8);
	const Eigen::Matrix3d F = Eigen::Vector3d(lambda, lambda, b).asDiagonal();
	const Eigen::Matrix3d Fe = Eigen::Vector3d(a, a, b).asDiagonal();
	const Eigen::Matrix3d Pe = pk1_ref(Fe);
	const Eigen::Matrix3d s_ref = Pe * Fe.transpose() / Fe.determinant();
	const Eigen::Matrix3d P_ref = theta * Pe * growth_inverse(theta, 1.0).transpose();
	const Eigen::Matrix3d S_ref = F.inverse() * P_ref;
	const Eigen::Matrix3d P_ungrown = pk1_ref(F);
	REQUIRE(std::abs((P_ungrown * F.transpose() / F.determinant())(2, 2)) > 100.0);

	// one sample at each element centroid
	io::OutputSample sample;
	sample.domain = io::OutputSample::Domain::Volume;
	sample.local_points = Eigen::MatrixXd::Constant(n_el, 3, 0.25);
	sample.element_ids = Eigen::VectorXi::LinSpaced(n_el, 0, n_el - 1);
	sample.points = Eigen::MatrixXd::Zero(n_el, 3);
	const std::vector<io::OutputField> fields = vf->output_fields(sample, sol, io::OutputFieldOptions{});

	const auto field = [&](const std::string &name) -> const Eigen::MatrixXd & {
		const auto it = std::find_if(fields.begin(), fields.end(), [&](const io::OutputField &f) { return f.name == name; });
		REQUIRE(it != fields.end());
		REQUIRE(it->values.rows() == n_el);
		return it->values;
	};
	// "<name>_k" holds column k of the tensor
	const auto tensor_at = [&](const std::string &name, const int row) {
		Eigen::Matrix3d T;
		for (int k = 0; k < 3; ++k)
			T.col(k) = field(name + "_" + std::to_string(k + 1)).row(row).transpose();
		return T;
	};

	const double tol = 1e-4;
	for (const std::string suffix : {"", "_avg"})
	{
		for (int i = 0; i < n_el; ++i)
		{
			const Eigen::Matrix3d s = tensor_at("cauchy_stess" + suffix, i);
			CHECK(std::abs(s(2, 2)) < tol * s_ref.cwiseAbs().maxCoeff());
			check_close(s, s_ref, tol);
			check_close(tensor_at("pk1_stess" + suffix, i), P_ref, tol);
			check_close(tensor_at("pk2_stess" + suffix, i), S_ref, tol);
			check_close(tensor_at("F" + suffix, i), F, tol);
			CHECK(field("von_mises" + suffix)(i, 0) == Catch::Approx(von_mises_stress_for_stress_tensor(s_ref)).epsilon(tol));
		}
	}
}
