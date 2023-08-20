////////////////////////////////////////////////////////////////////////////////

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>

#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/quadrature/HexQuadrature.hpp>

#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/basis/barycentric/MVPolygonalBasis2d.hpp>
#include <polyfem/basis/barycentric/WSPolygonalBasis2d.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::quadrature;

/////////////////////////////////////////
constexpr std::array<std::array<int, 2>, 8> linear_tri_local_node = {{
	{{0, 0}}, // v0  = (0, 0)
	{{1, 0}}, // v1  = (1, 0)
	{{0, 1}}, // v2  = (0, 1)
}};
constexpr std::array<std::array<int, 2>, 6> quadr_tri_local_node = {{
	{{0, 0}}, // v0  = (0, 0)
	{{2, 0}}, // v1  = (1, 0)
	{{0, 2}}, // v2  = (0, 1)
	{{1, 0}}, // e0  = (0.5,   0)
	{{1, 1}}, // e1  = (0.5, 0.5)
	{{0, 1}}, // e3  = (  0, 0.5)
}};

constexpr std::array<std::array<int, 3>, 4> linear_tet_local_node = {{
	{{0, 0, 0}}, // v0  = (0, 0, 0)
	{{1, 0, 0}}, // v1  = (1, 0, 0)
	{{0, 1, 0}}, // v2  = (0, 1, 0)
	{{0, 0, 1}}, // v3  = (0, 0, 1)
}};

constexpr std::array<std::array<int, 3>, 10> quadr_tet_local_node = {{
	{{0, 0, 0}}, // v0  = (  0,   0,   0)
	{{2, 0, 0}}, // v1  = (  1,   0,   0)
	{{0, 2, 0}}, // v2  = (  0,   1,   0)
	{{0, 0, 2}}, // v3  = (  0,   0,   1)
	{{1, 0, 0}}, // e0  = (0.5,   0,   0)
	{{1, 1, 0}}, // e1  = (0.5, 0.5,   0)
	{{0, 1, 0}}, // e2  = (  0, 0.5,   0)
	{{0, 0, 1}}, // e3  = (  0,   0, 0.5)
	{{1, 0, 1}}, // e4  = (0.5,   0, 0.5)
	{{0, 1, 1}}, // e5  = (  0, 0.5, 0.5)
}};

void linear_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();
	switch (local_index)
	{
	case 0:
		val = 1 - u - v;
		break;
	case 1:
		val = u;
		break;
	case 2:
		val = v;
		break;
	default:
		assert(false);
	}
}

void linear_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	val.resize(uv.rows(), uv.cols());
	switch (local_index)
	{
	case 0:
		val.col(0).setConstant(-1);
		val.col(1).setConstant(-1);
		break;
	case 1:
		val.col(0).setConstant(1);
		val.col(1).setConstant(0);
		break;
	case 2:
		val.col(0).setConstant(0);
		val.col(1).setConstant(1);
		break;
	default:
		assert(false);
	}
}

void quadr_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();
	switch (local_index)
	{
	case 0:
		val = (1 - u - v) * (1 - 2 * u - 2 * v);
		break;
	case 1:
		val = u * (2 * u - 1);
		break;
	case 2:
		val = v * (2 * v - 1);
		break;

	case 3:
		val = 4 * u * (1 - u - v);
		break;
	case 4:
		val = 4 * u * v;
		break;
	case 5:
		val = 4 * v * (1 - u - v);
		break;
	default:
		assert(false);
	}
}

void quadr_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();
	val.resize(uv.rows(), uv.cols());
	switch (local_index)
	{
	case 0:
		val.col(0) = 4 * u + 4 * v - 3;
		val.col(1) = 4 * u + 4 * v - 3;
		break;

	case 1:
		val.col(0) = 4 * u - 1;
		val.col(1).setZero();
		break;

	case 2:
		val.col(0).setZero();
		val.col(1) = 4 * v - 1;
		break;

	case 3:
		val.col(0) = 4 - 8 * u - 4 * v;
		val.col(1) = -4 * u;
		break;

	case 4:
		val.col(0) = 4 * v;
		val.col(1) = 4 * u;
		break;

	case 5:
		val.col(0) = -4 * v;
		val.col(1) = 4 - 4 * u - 8 * v;
		break;
	default:
		assert(false);
	}
}

void linear_tet_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	switch (local_index)
	{
	case 0:
		val = 1 - x - n - e;
		break;
	case 1:
		val = x;
		break;
	case 2:
		val = n;
		break;
	case 3:
		val = e;
		break;
	default:
		assert(false);
	}
}

void linear_tet_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	val.resize(xne.rows(), xne.cols());

	switch (local_index)
	{
	case 0:
		val.col(0).setConstant(-1);
		val.col(1).setConstant(-1);
		val.col(2).setConstant(-1);
		break;

	case 1:
		val.col(0).setConstant(1);
		val.col(1).setConstant(0);
		val.col(2).setConstant(0);
		break;

	case 2:
		val.col(0).setConstant(0);
		val.col(1).setConstant(1);
		val.col(2).setConstant(0);
		break;

	case 3:
		val.col(0).setConstant(0);
		val.col(1).setConstant(0);
		val.col(2).setConstant(1);
		break;

	default:
		assert(false);
	}
}

void quadr_tet_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	switch (local_index)
	{
	case 0:
		val = (1 - 2 * x - 2 * n - 2 * e) * (1 - x - n - e);
		break;
	case 1:
		val = (2 * x - 1) * x;
		break;
	case 2:
		val = (2 * n - 1) * n;
		break;
	case 3:
		val = (2 * e - 1) * e;
		break;

	case 4:
		val = 4 * x * (1 - x - n - e);
		break;
	case 5:
		val = 4 * x * n;
		break;
	case 6:
		val = 4 * (1 - x - n - e) * n;
		break;

	case 7:
		val = 4 * (1 - x - n - e) * e;
		break;
	case 8:
		val = 4 * x * e;
		break;
	case 9:
		val = 4 * n * e;
		break;
	default:
		assert(false);
	}
}

void quadr_tet_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	val.resize(xne.rows(), xne.cols());

	switch (local_index)
	{
	case 0:
		val.col(0) = -3 + 4 * x + 4 * n + 4 * e;
		val.col(1) = -3 + 4 * x + 4 * n + 4 * e;
		val.col(2) = -3 + 4 * x + 4 * n + 4 * e;
		break;
	case 1:
		val.col(0) = 4 * x - 1;
		val.col(1).setZero();
		val.col(2).setZero();
		break;
	case 2:
		val.col(0).setZero();
		val.col(1) = 4 * n - 1;
		val.col(2).setZero();
		break;
	case 3:
		val.col(0).setZero();
		val.col(1).setZero();
		val.col(2) = 4 * e - 1;
		break;

	case 4:
		val.col(0) = 4 - 8 * x - 4 * n - 4 * e;
		val.col(1) = -4 * x;
		val.col(2) = -4 * x;
		break;
	case 5:
		val.col(0) = 4 * n;
		val.col(1) = 4 * x;
		val.col(2).setZero();
		break;
	case 6:
		val.col(0) = -4 * n;
		val.col(1) = -8 * n + 4 - 4 * x - 4 * e;
		val.col(2) = -4 * n;
		break;

	case 7:
		val.col(0) = -4 * e;
		val.col(1) = -4 * e;
		val.col(2) = -8 * e + 4 - 4 * x - 4 * n;
		break;
	case 8:
		val.col(0) = 4 * e;
		val.col(1).setZero();
		val.col(2) = 4 * x;
		break;
	case 9:
		val.col(0).setZero();
		val.col(1) = 4 * e;
		val.col(2) = 4 * n;
		break;
	default:
		assert(false);
	}
}

/////////////////////////////////////////
/////////////////////////////////////////

constexpr std::array<std::array<int, 2>, 4> linear_quad_local_node = {{
	{{0, 0}}, // v0  = (0, 0)
	{{1, 0}}, // v1  = (1, 0)
	{{1, 1}}, // v2  = (1, 1)
	{{0, 1}}, // v3  = (0, 1)
}};

constexpr std::array<std::array<int, 2>, 9> quadr_quad_local_node = {{
	{{0, 0}}, // v0  = (  0,   0)
	{{2, 0}}, // v1  = (  1,   0)
	{{2, 2}}, // v2  = (  1,   1)
	{{0, 2}}, // v3  = (  0,   1)
	{{1, 0}}, // e0  = (0.5,   0)
	{{2, 1}}, // e1  = (  1, 0.5)
	{{1, 2}}, // e2  = (0.5,   1)
	{{0, 1}}, // e3  = (  0, 0.5)
	{{1, 1}}, // f0  = (0.5, 0.5)
}};

constexpr std::array<std::array<int, 3>, 8> linear_hex_local_node = {{
	{{0, 0, 0}}, // v0  = (0, 0, 0)
	{{1, 0, 0}}, // v1  = (1, 0, 0)
	{{1, 1, 0}}, // v2  = (1, 1, 0)
	{{0, 1, 0}}, // v3  = (0, 1, 0)
	{{0, 0, 1}}, // v4  = (0, 0, 1)
	{{1, 0, 1}}, // v5  = (1, 0, 1)
	{{1, 1, 1}}, // v6  = (1, 1, 1)
	{{0, 1, 1}}, // v7  = (0, 1, 1)
}};

constexpr std::array<std::array<int, 3>, 27> quadr_hex_local_node = {{
	{{0, 0, 0}}, // v0  = (  0,   0,   0)
	{{2, 0, 0}}, // v1  = (  1,   0,   0)
	{{2, 2, 0}}, // v2  = (  1,   1,   0)
	{{0, 2, 0}}, // v3  = (  0,   1,   0)
	{{0, 0, 2}}, // v4  = (  0,   0,   1)
	{{2, 0, 2}}, // v5  = (  1,   0,   1)
	{{2, 2, 2}}, // v6  = (  1,   1,   1)
	{{0, 2, 2}}, // v7  = (  0,   1,   1)
	{{1, 0, 0}}, // e0  = (0.5,   0,   0)
	{{2, 1, 0}}, // e1  = (  1, 0.5,   0)
	{{1, 2, 0}}, // e2  = (0.5,   1,   0)
	{{0, 1, 0}}, // e3  = (  0, 0.5,   0)
	{{0, 0, 1}}, // e4  = (  0,   0, 0.5)
	{{2, 0, 1}}, // e5  = (  1,   0, 0.5)
	{{2, 2, 1}}, // e6  = (  1,   1, 0.5)
	{{0, 2, 1}}, // e7  = (  0,   1, 0.5)
	{{1, 0, 2}}, // e8  = (0.5,   0,   1)
	{{2, 1, 2}}, // e9  = (  1, 0.5,   1)
	{{1, 2, 2}}, // e10 = (0.5,   1,   1)
	{{0, 1, 2}}, // e11 = (  0, 0.5,   1)
	{{0, 1, 1}}, // f0  = (  0, 0.5, 0.5)
	{{2, 1, 1}}, // f1  = (  1, 0.5, 0.5)
	{{1, 0, 1}}, // f2  = (0.5,   0, 0.5)
	{{1, 2, 1}}, // f3  = (0.5,   1, 0.5)
	{{1, 1, 0}}, // f4  = (0.5, 0.5,   0)
	{{1, 1, 2}}, // f5  = (0.5, 0.5,   1)
	{{1, 1, 1}}, // c0  = (0.5, 0.5, 0.5)
}};

template <typename T>
Eigen::MatrixXd alpha2d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return (1 - t);
	case 1:
		return t;
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd dalpha2d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return -1 + 0 * t;
	case 1:
		return 1 + 0 * t;
		;
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd theta2d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return (1 - t) * (1 - 2 * t);
	case 1:
		return 4 * t * (1 - t);
	case 2:
		return t * (2 * t - 1);
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd dtheta2d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return -3 + 4 * t;
	case 1:
		return 4 - 8 * t;
	case 2:
		return -1 + 4 * t;
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd alpha3d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return (1 - t);
	case 1:
		return t;
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd dalpha3d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return -1 + 0 * t;
	case 1:
		return 1 + 0 * t;
		;
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd theta3d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return (1 - t) * (1 - 2 * t);
	case 1:
		return 4 * t * (1 - t);
	case 2:
		return t * (2 * t - 1);
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template <typename T>
Eigen::MatrixXd dtheta3d(int i, T &t)
{
	switch (i)
	{
	case 0:
		return -3 + 4 * t;
	case 1:
		return 4 - 8 * t;
	case 2:
		return -1 + 4 * t;
	default:
		assert(false);
	}
	throw std::runtime_error("Invalid index");
}

void linear_quad_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();

	std::array<int, 2> idx = linear_quad_local_node[local_index];
	val = alpha2d(idx[0], u).array() * alpha2d(idx[1], v).array();
}

void linear_quad_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();

	std::array<int, 2> idx = linear_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dalpha2d(idx[0], u).array() * alpha2d(idx[1], v).array();
	val.col(1) = alpha2d(idx[0], u).array() * dalpha2d(idx[1], v).array();
}

void quadr_quad_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];
	val = theta2d(idx[0], u).array() * theta2d(idx[1], v).array();
}

void quadr_quad_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u = uv.col(0).array();
	auto v = uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dtheta2d(idx[0], u).array() * theta2d(idx[1], v).array();
	val.col(1) = theta2d(idx[0], u).array() * dtheta2d(idx[1], v).array();
}

void linear_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	std::array<int, 3> idx = linear_hex_local_node[local_index];
	val = alpha3d(idx[0], x).array() * alpha3d(idx[1], n).array() * alpha3d(idx[2], e).array();
}

void linear_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	std::array<int, 3> idx = linear_hex_local_node[local_index];

	val.resize(xne.rows(), 3);
	val.col(0) = dalpha3d(idx[0], x).array() * alpha3d(idx[1], n).array() * alpha3d(idx[2], e).array();
	val.col(1) = alpha3d(idx[0], x).array() * dalpha3d(idx[1], n).array() * alpha3d(idx[2], e).array();
	val.col(2) = alpha3d(idx[0], x).array() * alpha3d(idx[1], n).array() * dalpha3d(idx[2], e).array();
}

void quadr_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	std::array<int, 3> idx = quadr_hex_local_node[local_index];
	val = theta3d(idx[0], x).array() * theta3d(idx[1], n).array() * theta3d(idx[2], e).array();
}

void quadr_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
{
	auto x = xne.col(0).array();
	auto n = xne.col(1).array();
	auto e = xne.col(2).array();

	std::array<int, 3> idx = quadr_hex_local_node[local_index];

	val.resize(xne.rows(), 3);
	val.col(0) = dtheta3d(idx[0], x).array() * theta3d(idx[1], n).array() * theta3d(idx[2], e).array();
	val.col(1) = theta3d(idx[0], x).array() * dtheta3d(idx[1], n).array() * theta3d(idx[2], e).array();
	val.col(2) = theta3d(idx[0], x).array() * theta3d(idx[1], n).array() * dtheta3d(idx[2], e).array();
}

TEST_CASE("P1_2d", "[bases]")
{
	TriQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 3; ++i)
	{
		linear_tri_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_2d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		linear_tri_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_2d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::p_nodes_2d(1, val);
	for (int i = 0; i < 3; ++i)
	{
		for (int d = 0; d < 2; ++d)
			REQUIRE(linear_tri_local_node[i][d] == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("P2_2d", "[bases]")
{
	TriQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 6; ++i)
	{
		quadr_tri_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_2d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		quadr_tri_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_2d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::p_nodes_2d(2, val);
	for (int i = 0; i < 6; ++i)
	{
		for (int d = 0; d < 2; ++d)
			REQUIRE(quadr_tri_local_node[i][d] / 2. == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("Pk_2d", "[bases]")
{

	Eigen::MatrixXd pts;
	for (int k = 1; k < polyfem::autogen::MAX_P_BASES; ++k)
	{
		polyfem::autogen::p_nodes_2d(k, pts);

		Eigen::MatrixXd val;
		for (int i = 0; i < pts.rows(); ++i)
		{
			polyfem::autogen::p_basis_value_2d(k, i, pts, val);

			// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for (int j = 0; j < val.size(); ++j)
			{
				if (i == j)
					REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
			}
		}
	}
}

TEST_CASE("P1_3d", "[bases]")
{
	TetQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(8, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 4; ++i)
	{
		linear_tet_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_3d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		linear_tet_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_3d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::p_nodes_3d(1, val);
	for (int i = 0; i < 4; ++i)
	{
		for (int d = 0; d < 3; ++d)
			REQUIRE(linear_tet_local_node[i][d] == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("P2_3d", "[bases]")
{
	TetQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(8, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 10; ++i)
	{
		quadr_tet_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_3d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		quadr_tet_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_3d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::p_nodes_3d(2, val);
	for (int i = 0; i < 10; ++i)
	{
		for (int d = 0; d < 3; ++d)
			REQUIRE(quadr_tet_local_node[i][d] / 2. == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("P3_2d", "[bases]")
{
	Eigen::MatrixXd pts(10, 2);
	pts << 0, 0,
		1, 0,
		0, 1,
		1. / 3., 0,
		2. / 3., 0,
		2. / 3., 1. / 3.,
		1. / 3., 2. / 3.,
		0, 2. / 3.,
		0, 1. / 3.,
		1. / 3., 1. / 3.;

	Eigen::MatrixXd val;

	for (int i = 0; i < pts.rows(); ++i)
	{
		polyfem::autogen::p_basis_value_2d(3, i, pts, val);

		// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

		for (int j = 0; j < val.size(); ++j)
		{
			if (i == j)
				REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
			else
				REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
		}
	}
}

TEST_CASE("Pk_3d", "[bases]")
{
	Eigen::MatrixXd pts;
	for (int k = 1; k < polyfem::autogen::MAX_P_BASES; ++k)
	{
		polyfem::autogen::p_nodes_3d(k, pts);

		Eigen::MatrixXd val;
		for (int i = 0; i < pts.rows(); ++i)
		{
			polyfem::autogen::p_basis_value_3d(k, i, pts, val);

			// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for (int j = 0; j < val.size(); ++j)
			{
				if (i == j)
					REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
			}
		}
	}
}

TEST_CASE("Q1_2d", "[bases]")
{
	QuadQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 4; ++i)
	{
		linear_quad_basis_value(i, quad.points, expected);
		polyfem::autogen::q_basis_value_2d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		linear_quad_basis_grad(i, quad.points, expected);
		polyfem::autogen::q_grad_basis_value_2d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::q_nodes_2d(1, val);
	for (int i = 0; i < 4; ++i)
	{
		for (int d = 0; d < 2; ++d)
			REQUIRE(linear_quad_local_node[i][d] == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("Q2_2d", "[bases]")
{
	QuadQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 9; ++i)
	{
		quadr_quad_basis_value(i, quad.points, expected);
		polyfem::autogen::q_basis_value_2d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		quadr_quad_basis_grad(i, quad.points, expected);
		polyfem::autogen::q_grad_basis_value_2d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::q_nodes_2d(2, val);
	for (int i = 0; i < 9; ++i)
	{
		for (int d = 0; d < 2; ++d)
			REQUIRE(quadr_quad_local_node[i][d] / 2. == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("Qk_2d", "[bases]")
{

	Eigen::MatrixXd pts;
	for (int k = 1; k < polyfem::autogen::MAX_Q_BASES; ++k)
	{
		polyfem::autogen::q_nodes_2d(k, pts);

		Eigen::MatrixXd val;
		for (int i = 0; i < pts.rows(); ++i)
		{
			polyfem::autogen::q_basis_value_2d(k, i, pts, val);

			// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for (int j = 0; j < val.size(); ++j)
			{
				if (i == j)
					REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
			}
		}
	}

	int k = -2;
	polyfem::autogen::q_nodes_2d(k, pts);

	Eigen::MatrixXd val;
	for (int i = 0; i < pts.rows(); ++i)
	{
		polyfem::autogen::q_basis_value_2d(k, i, pts, val);

		for (int j = 0; j < val.size(); ++j)
		{
			if (i == j)
				REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
			else
				REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
		}
	}
}

TEST_CASE("Q1_3d", "[bases]")
{
	HexQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 8; ++i)
	{
		linear_hex_basis_value(i, quad.points, expected);
		polyfem::autogen::q_basis_value_3d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		linear_hex_basis_grad(i, quad.points, expected);
		polyfem::autogen::q_grad_basis_value_3d(1, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::q_nodes_3d(1, val);
	for (int i = 0; i < 8; ++i)
	{
		for (int d = 0; d < 3; ++d)
			REQUIRE(linear_hex_local_node[i][d] == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("Q2_3d", "[bases]")
{
	HexQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for (int i = 0; i < 27; ++i)
	{
		quadr_hex_basis_value(i, quad.points, expected);
		polyfem::autogen::q_basis_value_3d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));

		quadr_hex_basis_grad(i, quad.points, expected);
		polyfem::autogen::q_grad_basis_value_3d(2, i, quad.points, val);

		for (int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Catch::Approx(val(j)).margin(1e-10));
	}

	// Check nodes
	polyfem::autogen::q_nodes_3d(2, val);
	for (int i = 0; i < 27; ++i)
	{
		for (int d = 0; d < 3; ++d)
			REQUIRE(quadr_hex_local_node[i][d] / 2. == Catch::Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("Qk_3d", "[bases]")
{

	Eigen::MatrixXd pts;
	for (int k = 1; k < polyfem::autogen::MAX_Q_BASES; ++k)
	{
		polyfem::autogen::q_nodes_3d(k, pts);

		Eigen::MatrixXd val;
		for (int i = 0; i < pts.rows(); ++i)
		{
			polyfem::autogen::q_basis_value_3d(k, i, pts, val);

			// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for (int j = 0; j < val.size(); ++j)
			{
				if (i == j)
					REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
			}
		}
	}

	int k = -2;
	polyfem::autogen::q_nodes_3d(k, pts);

	Eigen::MatrixXd val;
	for (int i = 0; i < pts.rows(); ++i)
	{
		polyfem::autogen::q_basis_value_3d(k, i, pts, val);

		// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

		for (int j = 0; j < val.size(); ++j)
		{
			if (i == j)
				REQUIRE(val(j) == Catch::Approx(1).margin(1e-10));
			else
				REQUIRE(val(j) == Catch::Approx(0).margin(1e-10));
		}
	}
}

TEST_CASE("MV_2d", "[bases]")
{
	Eigen::MatrixXd b, b_prime, b_dx, b_dy;
	const double eps = 1e-10;

	Eigen::MatrixXd polygon(6, 2);
	polygon.row(0) << 0, 0;
	polygon.row(1) << 1, 0;
	polygon.row(2) << 1, 1;
	polygon.row(3) << 1, 2;
	polygon.row(4) << 0, 2;
	polygon.row(5) << -1, 1;

	for (int i = 0; i < polygon.rows(); ++i)
	{
		MVPolygonalBasis2d::meanvalue(polygon, polygon.row(i), b, eps);

		for (int j = 0; j < b.size(); ++j)
		{
			if (i == j)
				REQUIRE(b(j) == Catch::Approx(1).margin(1e-10));
			else
				REQUIRE(b(j) == Catch::Approx(0).margin(1e-10));
		}
	}

	Eigen::MatrixXd pts(3, 2);
	pts.row(0) << 0.5, 0.5;
	pts.row(1) << 0, 0.5;
	// pts.row(2) << 0.5, 0;
	pts.row(2) << -0.5, 0;
	// pts.row(4) << 0.5, 1e-11;

	const double delta = 1e-6;

	for (int i = 0; i < pts.rows(); ++i)
	{
		Eigen::RowVector2d pt;
		MVPolygonalBasis2d::meanvalue(polygon, pts.row(i), b, eps);

		pt = pts.row(i);
		pt(0) += delta;
		MVPolygonalBasis2d::meanvalue(polygon, pt, b_dx, eps);

		pt = pts.row(i);
		pt(1) += delta;
		MVPolygonalBasis2d::meanvalue(polygon, pt, b_dy, eps);

		MVPolygonalBasis2d::meanvalue_derivative(polygon, pts.row(i), b_prime, eps);
		for (int j = 0; j < b.size(); ++j)
		{
			REQUIRE(!std::isnan(b(j)));
			REQUIRE(!std::isnan(b_prime(j, 0)));
			REQUIRE(!std::isnan(b_prime(j, 1)));

			const double dx = (b_dx(j) - b(j)) / delta;
			const double dy = (b_dy(j) - b(j)) / delta;

			// std::cout<<j<<": "<<dx<<" "<<dy<<" -> "<<b_prime(j, 0) <<" "<<b_prime(j, 1) <<std::endl;

			REQUIRE(b_prime(j, 0) == Catch::Approx(dx).margin(delta * 10));
			REQUIRE(b_prime(j, 1) == Catch::Approx(dy).margin(delta * 10));
		}
	}
}

TEST_CASE("WS_2d", "[bases]")
{
	Eigen::MatrixXd b, b_prime, b_dx, b_dy;
	const double eps = 1e-10;

	Eigen::MatrixXd polygon(6, 2);
	polygon.row(0) << 0, 0;
	polygon.row(1) << 1, 0;
	polygon.row(2) << 1, 1;
	polygon.row(3) << 0.5, 2;
	polygon.row(4) << 0, 2;
	polygon.row(5) << -1, 1;

	for (int i = 0; i < polygon.rows(); ++i)
	{
		WSPolygonalBasis2d::wachspress(polygon, polygon.row(i), b, eps);

		for (int j = 0; j < b.size(); ++j)
		{
			if (i == j)
				REQUIRE(b(j) == Catch::Approx(1).margin(1e-10));
			else
				REQUIRE(b(j) == Catch::Approx(0).margin(1e-10));
		}
	}

	Eigen::MatrixXd pts(3, 2);
	pts.row(0) << 0.5, 0.5;
	pts.row(1) << 0, 0.5;
	// pts.row(2) << 0.5, 0;
	pts.row(2) << 0.5, 1.5;
	// pts.row(4) << 0.5, 1e-11;

	const double delta = 1e-6;

	for (int i = 0; i < pts.rows(); ++i)
	{
		Eigen::RowVector2d pt;
		WSPolygonalBasis2d::wachspress(polygon, pts.row(i), b, eps);

		pt = pts.row(i);
		pt(0) += delta;
		WSPolygonalBasis2d::wachspress(polygon, pt, b_dx, eps);

		pt = pts.row(i);
		pt(1) += delta;
		WSPolygonalBasis2d::wachspress(polygon, pt, b_dy, eps);

		WSPolygonalBasis2d::wachspress_derivative(polygon, pts.row(i), b_prime, eps);
		for (int j = 0; j < b.size(); ++j)
		{
			REQUIRE(!std::isnan(b(j)));
			REQUIRE(!std::isnan(b_prime(j, 0)));
			REQUIRE(!std::isnan(b_prime(j, 1)));

			const double dx = (b_dx(j) - b(j)) / delta;
			const double dy = (b_dy(j) - b(j)) / delta;

			// std::cout<<j<<": "<<dx<<" "<<dy<<" -> "<<b_prime(j, 0) <<" "<<b_prime(j, 1) <<std::endl;

			REQUIRE(b_prime(j, 0) == Catch::Approx(dx).margin(delta * 10));
			REQUIRE(b_prime(j, 1) == Catch::Approx(dy).margin(delta * 10));
		}
	}
}
