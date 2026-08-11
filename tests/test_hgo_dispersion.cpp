////////////////////////////////////////////////////////////////////////////////
// Constitutive unit test for HGODispersion.
//
// Differentiates the shipped HGODispersion energy through PolyFEM's own
// DScalar2 autodiff type at a prescribed deformation gradient, converts the
// resulting first Piola stress to Cauchy, and compares against the analytic
// fiber stress. This isolates the constitutive code from assembly, quadrature
// and nodal stress recovery, so it is not subject to the ~8.5 Pa hydrostatic
// recovery artifact seen in full solver runs near incompressibility.
//
// Deformation is the rotated-45 state used in the validation report:
//   F = R diag(1.5, 0.816889, 0.816889) R^T,  R = rotation of 45 deg about z
//   a0 = [1/sqrt(2), 1/sqrt(2), 0]  (aligned with the 1.5 stretch direction)
// giving I4 = 2.25, E4 = 0.4903, and fiber-only Cauchy stress
//   [[216.76, 164.56, 0], [164.56, 216.76, 0], [0, 0, 52.20]] Pa.
////////////////////////////////////////////////////////////////////////////////

#include <polyfem/Units.hpp>
#include <polyfem/assembler/HGODispersion.hpp>
#include <polyfem/utils/autodiff.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Dense>

using namespace polyfem;
using namespace polyfem::assembler;

namespace
{
	// Same second-order autodiff scalar GenericElastic uses for the Hessian path.
	typedef DScalar2<double,
					 Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>,
					 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9>>
		Diff2;
} // namespace

TEST_CASE("hgo-dispersion-fiber-stress", "[assembler]")
{
	// ---- material ----------------------------------------------------------
	HGODispersion mat;
	mat.set_size(3);

	Units units;
	json params = R"({
		"type": "HGODispersion",
		"k1": 100.0,
		"k2": 5.0,
		"kappa": 0.24,
		"k_chi": 100.0,
		"fiber_direction": [0.70710678, 0.70710678, 0.0]
	})"_json;
	mat.add_multimaterial(0, params, units, "");

	// ---- deformation gradient (rotated 45 deg about z) ---------------------
	Eigen::Matrix3d F;
	F << 1.15844444, 0.34155556, 0.0,
		0.34155556, 1.15844444, 0.0,
		0.0, 0.0, 0.81688889;

	// ---- seed autodiff variables ------------------------------------------
	DiffScalarBase::setVariableCount(9);

	DefGradMatrix<Diff2> F_ad(3, 3);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			F_ad(i, j) = Diff2(i * 3 + j, F(i, j));

	// ---- energy and its gradient ------------------------------------------
	RowVectorNd p(3);
	p.setZero(); // constant fiber_direction => evaluation point is irrelevant

	const Diff2 psi = mat.elastic_energy(p, /*t=*/0.0, /*el_id=*/0, F_ad);

	// P = d(psi)/dF, using the same index convention as the seeding above.
	Eigen::Matrix3d P;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			P(i, j) = psi.getGradient()(i * 3 + j);

	// sigma = J^{-1} P F^T
	const double J = F.determinant();
	const Eigen::Matrix3d sigma = (P * F.transpose()) / J;

	// ---- analytic reference ------------------------------------------------
	Eigen::Matrix3d expected;
	expected << 216.76, 164.56, 0.0,
		164.56, 216.76, 0.0,
		0.0, 0.0, 52.20;

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			CHECK(sigma(i, j) == Catch::Approx(expected(i, j)).margin(1e-2));

	// Cauchy must be symmetric.
	CHECK((sigma - sigma.transpose()).norm() == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("hgo-dispersion-invariants", "[assembler]")
{
	Units units;

	// Rigid rotation must give exactly zero energy (E4 = 0 at the unloaded
	// state, which is also where the logistic switch is centered).
	{
		HGODispersion mat;
		mat.set_size(3);
		json params = R"({
			"type": "HGODispersion",
			"k1": 100.0, "k2": 5.0, "kappa": 0.24, "k_chi": 100.0,
			"fiber_direction": [0.70710678, 0.70710678, 0.0]
		})"_json;
		mat.add_multimaterial(0, params, units, "");

		const double th = 0.7;
		Eigen::Matrix3d R;
		R << std::cos(th), -std::sin(th), 0.0,
			std::sin(th), std::cos(th), 0.0,
			0.0, 0.0, 1.0;

		DefGradMatrix<double> F_r(3, 3);
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				F_r(i, j) = R(i, j);

		RowVectorNd p(3);
		p.setZero();
		CHECK(mat.elastic_energy(p, 0.0, 0, F_r) == Catch::Approx(0.0).margin(1e-12));
	}

	// At kappa = 1/d the (1 - d*kappa) weight vanishes, so I4 drops out and the
	// response must be independent of the fiber direction.
	{
		Eigen::Matrix3d F;
		F << 1.15844444, 0.34155556, 0.0,
			0.34155556, 1.15844444, 0.0,
			0.0, 0.0, 0.81688889;

		DefGradMatrix<double> F_d(3, 3);
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				F_d(i, j) = F(i, j);

		RowVectorNd p(3);
		p.setZero();

		const std::vector<std::string> dirs = {
			"[1.0, 0.0, 0.0]", "[0.0, 1.0, 0.0]", "[0.57735, 0.57735, 0.57735]"};

		double reference = 0.0;
		for (size_t d = 0; d < dirs.size(); ++d)
		{
			HGODispersion mat;
			mat.set_size(3);
			json params = json::parse(
				R"({"type": "HGODispersion", "k1": 100.0, "k2": 5.0,
				    "kappa": 0.3333333333333333, "k_chi": 100.0,
				    "fiber_direction": )"
				+ dirs[d] + "}");
			mat.add_multimaterial(0, params, units, "");

			const double e = mat.elastic_energy(p, 0.0, 0, F_d);
			if (d == 0)
				reference = e;
			else
				CHECK(e == Catch::Approx(reference).epsilon(1e-10));
		}
	}
}
