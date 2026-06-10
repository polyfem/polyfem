////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/quadrature/LineQuadrature.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/quadrature/PrismQuadrature.hpp>
#include <polyfem/quadrature/PyramidQuadrature.hpp>
#include <polyfem/quadrature/PolyhedronQuadrature.hpp>
#include <polyfem/quadrature/QuadratureOrder.hpp>
#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/LinearElasticity.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/NavierStokes.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/assembler/Stokes.hpp>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::quadrature;

const double pi = 3.14159265358979323846264338327950288419717;

////////////////////////////////////////////////////////////////////////////////

namespace
{

	// double p01_exact() {
	// 	return pi * pi / 6.0;
	// }

	// Eigen::VectorXd p01_fun(const Eigen::MatrixXd &x) {
	// 	Eigen::VectorXd f = 1.0 / (1.0 - x.row(0).array() * x.row(1).array());
	// 	return f;
	// }

	// Eigen::AlignedBox2d p01_lim() {
	// 	Eigen::AlignedBox2d box;
	// 	box.min().setConstant(0.0);
	// 	box.max().setConstant(1.0);
	// 	return box;
	// }

	namespace
	{
		std::shared_ptr<State> get_state(
			const std::string &mesh, const int n_refs,
			const int basis_order,
			const int quadrature, const int mass_quadrature,
			const bool spline, const bool serendipity)
		{
			const std::string path = POLYFEM_DATA_DIR;
			json in_args = R"(
		{
			"materials": {
                "type": "Laplacian"
            },

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh",
				"surface_selection": 7,
				"n_refs": 0
			}],

			"space": {
				"discr_order": 0,
				"advanced": {
					"quadrature_order": -1,
					"mass_quadrature_order": -1
				}
			},

			"time": {
				"dt": 0.001,
				"tend": 1.0
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": 0
				}],
				"rhs": 10
			},

			"output": {
				"log": {
					"level": "warning"
				}
			}

		})"_json;
			in_args["geometry"][0]["mesh"] = path + "/quad_test/" + mesh;
			in_args["geometry"][0]["n_refs"] = n_refs;
			in_args["space"]["discr_order"] = basis_order;
			in_args["space"]["advanced"]["quadrature_order"] = quadrature;
			in_args["space"]["advanced"]["mass_quadrature_order"] = mass_quadrature;

			if (spline)
				in_args["space"]["basis_type"] = "Spline";
			else if (serendipity)
				in_args["space"]["basis_type"] = "Serendipity";

			auto state = std::make_shared<State>();
			state->set_max_threads(1);
			state->init(in_args, true);

			state->load_mesh();

			state->build_basis();
			state->assemble_rhs();
			state->assemble_mass_mat();

			return state;
		}

		void test_quadrature(const std::string &mesh, const int n_refs,
							 const int basis_order,
							 const bool spline, const bool serendipity)
		{
			const bool is_q = mesh.find("quad") != std::string::npos || mesh.find("hex") != std::string::npos;
			const int expected_quad = is_q ? 20 : 14;
			static const double margin = 1e-10;
			// spdlog::info("Reference quad={}", expected_quad);

			auto state = get_state(mesh, n_refs, basis_order, -1, -1, spline, serendipity);
			auto expected = get_state(mesh, n_refs, basis_order, expected_quad, expected_quad, spline, serendipity);
			StiffnessMatrix exp_st, st;
			state->build_stiffness_mat(st);
			expected->build_stiffness_mat(exp_st);

			StiffnessMatrix tmp = st - exp_st;
			const auto val = Catch::Approx(0).margin(margin);

			REQUIRE(tmp.rows() > 8);

			for (int k = 0; k < tmp.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
				{
					if (fabs(it.value()) > margin)
						spdlog::error("error: {} != 0", it.value());
					REQUIRE(it.value() == val);
				}
			}

			tmp = state->mass - expected->mass;

			for (int k = 0; k < tmp.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
				{
					if (fabs(it.value()) > margin)
						spdlog::error("error: {} != 0", it.value());
					REQUIRE(it.value() == val);
				}
			}
		}
	} // namespace

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

TEST_CASE("quadrature_order_hints", "[quadrature]")
{
	const WeakFormOrderHint stiffness{0, 2, 0};
	const WeakFormOrderHint mass{2, 0, 0};
	const GeometryBasisOrderHint linear2d{BasisFamily::SIMPLEX, 2, 1, 1};
	const GeometryBasisOrderHint linear_tensor2d{BasisFamily::TENSOR, 2, 1, 1};
	const GeometryBasisOrderHint quadratic2d{BasisFamily::SIMPLEX, 2, 2, 2};

	REQUIRE((QuadratureOrder{stiffness, BasisOrderHint{BasisFamily::SIMPLEX, 1, 1}, linear2d, -1}.order == 1));
	REQUIRE((QuadratureOrder{stiffness, BasisOrderHint{BasisFamily::SIMPLEX, 2, 2}, linear2d, -1}.order == 2));
	REQUIRE((QuadratureOrder{stiffness, BasisOrderHint{BasisFamily::SIMPLEX, 3, 3}, linear2d, -1}.order == 4));
	REQUIRE((QuadratureOrder{mass, BasisOrderHint{BasisFamily::SIMPLEX, 3, 3}, linear2d, -1}.order == 6));
	REQUIRE((QuadratureOrder{stiffness, BasisOrderHint{BasisFamily::TENSOR, 2, 2}, linear_tensor2d, -1}.order == 6));
	REQUIRE((QuadratureOrder{mass, BasisOrderHint{BasisFamily::TENSOR, 2, 2}, linear_tensor2d, -1}.order == 6));
	REQUIRE((QuadratureOrder{WeakFormOrderHint{0, 2, 3}, BasisOrderHint{BasisFamily::SIMPLEX, 2, 2}, linear2d, -1}.order == 5));
	REQUIRE((QuadratureOrder{stiffness, BasisOrderHint{BasisFamily::SIMPLEX, 2, 2}, quadratic2d, -1}.order == 4));
	REQUIRE((QuadratureOrder{stiffness, BasisOrderHint{BasisFamily::SIMPLEX, 2, 2}, linear2d, 7}.order == 7));
}

TEST_CASE("prism_quadrature_order_hints", "[quadrature]")
{
	const GeometryBasisOrderHint linear3d{BasisFamily::PRISM, 3, 1, 1};
	const QuadratureOrder mass_order{WeakFormOrderHint{2, 0, 0}, BasisOrderHint{BasisFamily::PRISM, 2, 3}, linear3d, -1};
	const QuadratureOrder stiffness_order{WeakFormOrderHint{0, 2, 0}, BasisOrderHint{BasisFamily::PRISM, 2, 3}, linear3d, -1};

	REQUIRE(mass_order.order == 7);
	REQUIRE(mass_order.height_order == 7);
	REQUIRE(stiffness_order.order == 7);
	REQUIRE(stiffness_order.height_order == 7);
}

TEST_CASE("assembler_quadrature_hints", "[quadrature]")
{
	const Mass mass;
	const HRZMass hrz_mass;
	const Laplacian laplacian;
	const StokesMixed stokes_mixed;
	const NavierStokesVelocity navier_stokes;
	const NeoHookeanElasticity neo_hookean;
	const LinearElasticity linear_elasticity;

	REQUIRE(mass.weak_form_order_hint().phi_count == 2);
	REQUIRE(mass.weak_form_order_hint().grad_phi_count == 0);
	REQUIRE(hrz_mass.weak_form_order_hint().phi_count == 2);
	REQUIRE(laplacian.weak_form_order_hint().grad_phi_count == 2);
	REQUIRE(stokes_mixed.weak_form_order_hint().phi_count == 1);
	REQUIRE(stokes_mixed.weak_form_order_hint().grad_phi_count == 1);
	REQUIRE(navier_stokes.weak_form_order_hint().phi_count == 2);
	REQUIRE(navier_stokes.weak_form_order_hint().grad_phi_count == 1);
	REQUIRE(neo_hookean.weak_form_order_hint().extra_order == 2);
	REQUIRE(linear_elasticity.weak_form_order_hint().extra_order == 0);
}

TEST_CASE("prism_quadrature_respects_height_order", "[quadrature]")
{
	PrismQuadrature prism;
	Quadrature low_height, high_height;
	prism.get_quadrature(2, 1, low_height);
	prism.get_quadrature(2, 4, high_height);

	REQUIRE(high_height.size() > low_height.size());
}

TEST_CASE("polyhedron_quadrature_respects_order", "[quadrature]")
{
	Eigen::MatrixXd V(4, 3);
	V << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;

	Eigen::MatrixXi F(4, 3);
	F << 0, 2, 1,
		0, 1, 3,
		1, 2, 3,
		2, 0, 3;

	Quadrature low_order, high_order;
	PolyhedronQuadrature::get_quadrature(V, F, Eigen::RowVector3d(0.25, 0.25, 0.25), 1, low_order);
	PolyhedronQuadrature::get_quadrature(V, F, Eigen::RowVector3d(0.25, 0.25, 0.25), 4, high_order);

	REQUIRE(high_order.size() > low_order.size());
}

////////////////////////////////////////////////////////////////////////////////

TEST_CASE("auto_quadrature", "[quadrature]")
{
	struct data
	{
		std::string mesh;
		int n_refs;
		int order;
		bool spline;
		bool serendipity;
	};

	std::vector<data> tests = {
		// P
		{"tri.obj", 2, 1, false, false},
		{"tri.obj", 1, 2, false, false},
		{"tri.obj", 1, 3, false, false},

		{"tet.msh", 0, 1, false, false},
		{"tet.msh", 0, 2, false, false},

		// Q
		{"quad.obj", 3, 1, false, false},
		{"quad.obj", 2, 2, false, false},

		{"hex.HYBRID", 0, 1, false, false},
		{"hex.HYBRID", 0, 2, false, false},

		// Spline
		{"quad.obj", 2, 2, true, false},
		{"hex.HYBRID", 0, 2, true, false},

		// serendipity
		{"quad.obj", 2, 2, false, true},
		{"hex.HYBRID", 0, 2, false, true},
	};

	for (const auto &d : tests)
	{
		spdlog::set_level(spdlog::level::info);
		spdlog::info("Running {} Order={}, spline={} serendipity={}", d.mesh, d.order, d.spline, d.serendipity);
		test_quadrature(d.mesh, d.n_refs, d.order, d.spline, d.serendipity);
	}
}

TEST_CASE("weights", "[quadrature]")
{
	// Segment
	for (int order = 1; order <= 64; ++order)
	{
		LineQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Triangle
	for (int order = 1; order < 16; ++order)
	{
		TriQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(0.5).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Tetrahedron
	for (int order = 1; order < 16; ++order)
	{
		TetQuadrature tri;
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0 / 6.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Triangle Corner
	for (int order = 1; order < 15; ++order)
	{
		TriQuadrature tri(true);
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(0.5).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Tetrahedron Corner
	for (int order = 1; order < 10; ++order)
	{
		TetQuadrature tri(true);
		Quadrature quadr;
		tri.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0 / 6.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Prism
	for (int order = 1; order < 16; ++order)
	{
		PrismQuadrature pri;
		Quadrature quadr;
		pri.get_quadrature(order, order + 1, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0 / 2.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}

	// Pyramid
	for (int order : std::vector<int>{1, 2, 3, 5})
	{
		PyramidQuadrature pyr;
		Quadrature quadr;
		pyr.get_quadrature(order, quadr);
		REQUIRE(quadr.weights.sum() == Catch::Approx(1.0 / 3.0).margin(1e-12));
		REQUIRE(quadr.points.minCoeff() >= 0.0);
		REQUIRE(quadr.points.maxCoeff() <= 1.0);
	}
}
