// V1(a) unit checks for GrowthElasticity: finite-difference verification of
// the mixed gradient and Hessian (including the K_ug coupling quadrant), and
// the theta == 1 exact-zero identity of the additive correction.
//
// Follows the synthetic-element pattern of test_assembler.cpp: hand-built
// ElementAssemblyValues on a unit reference tet with the true P1 basis
// gradients and a single quadrature point, so the deformation gradient is
// exactly controllable through the displacement DOFs.

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/assembler/AssemblerData.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/GrowthElasticity.hpp>
#include <polyfem/basis/Basis.hpp>
#include <polyfem/utils/Types.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;

namespace
{
	// MixedNLAssembler's public assemble_* are the GLOBAL assembly entry
	// points; the per-element compute_* this test exercises are protected.
	// Re-raise them, following the Configurable* idiom of test_assembler.cpp.
	class TestableGrowthElasticity : public GrowthElasticity
	{
	public:
		using GrowthElasticity::compute_energy;
		using GrowthElasticity::compute_gradient;
		using GrowthElasticity::compute_hessian;
	};

	// Unit reference tet: nodes X0=(0,0,0), X1=e1, X2=e2, X3=e3.
	// P1 gradients: dN0 = (-1,-1,-1), dNi = ei. One quadrature point at the
	// centroid, jac_it = I, so grad_u = sum_i u_i dNi^T exactly and
	// F = I + grad_u is set exactly by the nodal displacements.
	struct TetSpaces
	{
		ElementAssemblyValues phi_vals; // displacement (vector P1, 4 bases)
		ElementAssemblyValues psi_vals; // growth (scalar P1, 4 bases)
		QuadratureVector da;
	};

	void fill_p1_tet(ElementAssemblyValues &vals, const int dim)
	{
		const Eigen::Vector3d centroid(0.25, 0.25, 0.25);
		Eigen::MatrixXd nodes(4, 3);
		nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
		Eigen::MatrixXd grads(4, 3);
		grads << -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1;
		const Eigen::Vector4d N(0.25, 0.25, 0.25, 0.25); // P1 at centroid

		vals.element_id = 0;
		vals.is_volume_ = true;
		vals.val = centroid.transpose();
		vals.quadrature.points = centroid.transpose();
		vals.quadrature.weights = Eigen::VectorXd::Ones(1);
		vals.det = Eigen::VectorXd::Ones(1);
		vals.jac_it = {Eigen::MatrixXd::Identity(dim, dim)};
		vals.basis_values.resize(4);
		for (int i = 0; i < 4; ++i)
		{
			AssemblyValues &basis = vals.basis_values[i];
			basis.global = {Local2Global(i, nodes.row(i), 1.0)};
			basis.val = Eigen::MatrixXd::Constant(1, 1, N(i));
			basis.grad = grads.row(i);
			basis.grad_t_m = basis.grad;
		}
	}

	TetSpaces make_tet_spaces()
	{
		TetSpaces s;
		fill_p1_tet(s.phi_vals, 3);
		fill_p1_tet(s.psi_vals, 3);
		s.da = QuadratureVector::Constant(1, 1.0 / 6.0); // tet volume
		return s;
	}

	// Nodal displacements realizing an exact homogeneous F: u_i = (F - I) X_i.
	Eigen::MatrixXd x_phi_for(const Eigen::Matrix3d &F)
	{
		Eigen::MatrixXd nodes(4, 3);
		nodes << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
		Eigen::MatrixXd x(12, 1);
		for (int i = 0; i < 4; ++i)
		{
			const Eigen::Vector3d u = (F - Eigen::Matrix3d::Identity()) * nodes.row(i).transpose();
			for (int d = 0; d < 3; ++d)
				x(i * 3 + d) = u(d);
		}
		return x;
	}

	json growth_material(const json &normal_growth = json())
	{
		json m = {
			{"type", "GrowthElasticity"},
			{"id", 0},
			{"displacement_space_id", 0},
			{"growth_space_id", 1},
			// constant wall normal, deliberately NOT axis-aligned so the
			// theta == 1 identity is exercised for a generic direction
			{"normal_direction", {0.3, -0.2, 0.9330951164939612}},
			{"elastic_material",
			 {{"type", "MaterialSum"},
			  {"models",
			   {// the fitted uterine composition (values from the fit)
				{{"type", "IsochoricNeoHookean"}, {"E", 2600.0}, {"nu", 0.26}},
				{{"type", "HGODispersion"},
				 {"k1", 2400.0},
				 {"k2", 0.01},
				 {"kappa", 0.24},
				 {"k_chi", 100.0},
				 // NOTE: constant-fiber syntax; adjust to GenericFiber's
				 // schema if the first run flags it
				 {"fiber_direction", {1.0, 0.0, 0.0}}},
				{{"type", "VolumePenalty"}, {"k", 1805.6}}}}}}};
		if (!normal_growth.is_null())
			m["normal_growth"] = normal_growth;
		return m;
	}

	std::unique_ptr<TestableGrowthElasticity> make_assembler(const json &material)
	{
		auto assembler = std::make_unique<TestableGrowthElasticity>();
		assembler->set_size(3);
		Units units;
		assembler->add_multimaterial(0, material, units, "");
		return assembler;
	}

	MixedNonLinearAssemblerData make_data(
		const TetSpaces &s, const Eigen::MatrixXd &x_phi, const Eigen::MatrixXd &x_psi,
		const Eigen::MatrixXd &x_phi_prev, const Eigen::MatrixXd &x_psi_prev)
	{
		// NOTE ctor order: psi_vals first (see AssemblerData.hpp)
		return MixedNonLinearAssemblerData(
			s.psi_vals, s.phi_vals, /*t=*/0.0, /*dt=*/0.0,
			x_phi, x_psi, x_phi_prev, x_psi_prev, s.da);
	}

	// A generic well-conditioned test state: stretch + shear, J > 0.
	Eigen::Matrix3d test_F()
	{
		Eigen::Matrix3d F;
		F << 1.30, 0.08, 0.03,
			0.05, 1.15, -0.04,
			0.02, 0.06, 0.92;
		return F;
	}

	void check_fd(const json &material, const Eigen::Matrix3d &F, const Eigen::Vector4d &theta_nodes)
	{
		const auto assembler = make_assembler(material);
		const TetSpaces s = make_tet_spaces();
		const Eigen::MatrixXd zero_phi = Eigen::MatrixXd::Zero(12, 1);
		const Eigen::MatrixXd zero_psi = Eigen::MatrixXd::Zero(4, 1);

		Eigen::MatrixXd x_phi = x_phi_for(F);
		Eigen::MatrixXd x_psi = theta_nodes;

		const auto energy_at = [&](const Eigen::MatrixXd &xp, const Eigen::MatrixXd &xs) {
			return assembler->compute_energy(make_data(s, xp, xs, zero_phi, zero_psi));
		};

		const Eigen::VectorXd grad =
			assembler->compute_gradient(make_data(s, x_phi, x_psi, zero_phi, zero_psi));
		const Eigen::MatrixXd hess =
			assembler->compute_hessian(make_data(s, x_phi, x_psi, zero_phi, zero_psi));

		REQUIRE(grad.size() == 16);
		REQUIRE(hess.rows() == 16);
		REQUIRE(hess.cols() == 16);

		// ---- gradient vs central FD of the energy, all 16 locals
		const double h = 1e-6;
		for (int k = 0; k < 16; ++k)
		{
			Eigen::MatrixXd xp_p = x_phi, xp_m = x_phi, xs_p = x_psi, xs_m = x_psi;
			if (k < 12)
			{
				xp_p(k) += h;
				xp_m(k) -= h;
			}
			else
			{
				xs_p(k - 12) += h;
				xs_m(k - 12) -= h;
			}
			const double fd = (energy_at(xp_p, xs_p) - energy_at(xp_m, xs_m)) / (2 * h);
			const double scale = std::max({1.0, std::abs(fd), std::abs(grad(k))});
			REQUIRE(grad(k) == Catch::Approx(fd).epsilon(5e-5).margin(5e-5 * scale));
		}

		// ---- hessian vs central FD of the gradient (checks ALL quadrants,
		// including the K_ug coupling block, rows 0-11 x cols 12-15)
		for (int k = 0; k < 16; ++k)
		{
			Eigen::MatrixXd xp_p = x_phi, xp_m = x_phi, xs_p = x_psi, xs_m = x_psi;
			if (k < 12)
			{
				xp_p(k) += h;
				xp_m(k) -= h;
			}
			else
			{
				xs_p(k - 12) += h;
				xs_m(k - 12) -= h;
			}
			const Eigen::VectorXd gp = assembler->compute_gradient(make_data(s, xp_p, xs_p, zero_phi, zero_psi));
			const Eigen::VectorXd gm = assembler->compute_gradient(make_data(s, xp_m, xs_m, zero_phi, zero_psi));
			const Eigen::VectorXd fd = (gp - gm) / (2 * h);
			for (int r = 0; r < 16; ++r)
			{
				const double scale = std::max({1.0, std::abs(fd(r)), std::abs(hess(r, k))});
				REQUIRE(hess(r, k) == Catch::Approx(fd(r)).epsilon(5e-4).margin(5e-4 * scale));
			}
		}

		// ---- symmetry (one energy, one autodiff pass: must be symmetric)
		for (int r = 0; r < 16; ++r)
			for (int c = r + 1; c < 16; ++c)
				REQUIRE(hess(r, c) == Catch::Approx(hess(c, r)).margin(1e-10 * std::max(1.0, std::abs(hess(r, c)))));
	}
} // namespace

TEST_CASE("growth_elasticity_theta_one_identity", "[assembler][growth]")
{
	// At theta == 1 (vn == 1) the correction must vanish EXACTLY -- zero
	// energy, zero gradient in every component, zero Hessian in the uu
	// quadrant -- because Fgi is the bitwise identity by construction
	// (identity-anchored form) and both energy calls see the same def_grad.
	// This is the unit-level half of the V1 bit-identity regression.
	const auto assembler = make_assembler(growth_material());
	const TetSpaces s = make_tet_spaces();
	const Eigen::MatrixXd zero_phi = Eigen::MatrixXd::Zero(12, 1);
	const Eigen::MatrixXd zero_psi = Eigen::MatrixXd::Zero(4, 1);
	const Eigen::MatrixXd x_psi = Eigen::MatrixXd::Ones(4, 1);

	for (const double scale : {0.0, 0.15, 0.4})
	{
		Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + scale * (test_F() - Eigen::Matrix3d::Identity());
		const Eigen::MatrixXd x_phi = x_phi_for(F);

		const double e = assembler->compute_energy(make_data(s, x_phi, x_psi, zero_phi, zero_psi));
		REQUIRE(e == 0.0); // exact, not approximate

		const Eigen::VectorXd g = assembler->compute_gradient(make_data(s, x_phi, x_psi, zero_phi, zero_psi));
		for (int k = 0; k < 12; ++k)
			REQUIRE(g(k) == 0.0); // u-gradient exactly zero
		// (theta-gradient is the growth driving force; nonzero by design)

		const Eigen::MatrixXd H = assembler->compute_hessian(make_data(s, x_phi, x_psi, zero_phi, zero_psi));
		for (int r = 0; r < 12; ++r)
			for (int c = 0; c < 12; ++c)
				REQUIRE(H(r, c) == 0.0); // uu quadrant exactly zero
	}
}

TEST_CASE("growth_elasticity_gradient_hessian_fd", "[assembler][growth]")
{
	// Uniform growth, generic deformation
	check_fd(growth_material(), test_F(), Eigen::Vector4d::Constant(1.7));
	// Non-uniform nodal growth (exercises the psi interpolation path)
	check_fd(growth_material(), test_F(), Eigen::Vector4d(1.2, 1.9, 1.5, 2.4));
	// Prescribed normal growth vn != 1 (the candidate-map slot)
	check_fd(growth_material(/*normal_growth=*/json(0.8)), test_F(),
			 Eigen::Vector4d::Constant(1.7));
}

TEST_CASE("growth_elasticity_positivity_guard", "[assembler][growth]")
{
	const auto assembler = make_assembler(growth_material());
	const TetSpaces s = make_tet_spaces();
	const Eigen::MatrixXd zero_phi = Eigen::MatrixXd::Zero(12, 1);
	const Eigen::MatrixXd zero_psi = Eigen::MatrixXd::Zero(4, 1);
	const Eigen::MatrixXd x_phi = x_phi_for(test_F());
	const Eigen::MatrixXd x_psi_bad = Eigen::MatrixXd::Constant(4, 1, -0.5);

	REQUIRE_THROWS(assembler->compute_energy(make_data(s, x_phi, x_psi_bad, zero_phi, zero_psi)));
}
