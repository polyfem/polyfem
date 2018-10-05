////////////////////////////////////////////////////////////////////////////////

#include <polyfem/TriQuadrature.hpp>
#include <polyfem/TetQuadrature.hpp>

#include <polyfem/QuadQuadrature.hpp>
#include <polyfem/HexQuadrature.hpp>

#include <polyfem/FEBasis3d.hpp>
#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <catch.hpp>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;


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

void linear_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();
	switch(local_index)
	{
		case 0: val = 1 - u - v; break;
		case 1: val = u; break;
		case 2: val = v; break;
		default: assert(false);
	}
}

void linear_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	val.resize(uv.rows(), uv.cols());
	switch(local_index)
	{
		case 0: val.col(0).setConstant(-1); val.col(1).setConstant(-1); break;
		case 1: val.col(0).setConstant( 1); val.col(1).setConstant( 0); break;
		case 2: val.col(0).setConstant( 0); val.col(1).setConstant( 1); break;
		default: assert(false);
	}
}

void quadr_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();
	switch(local_index)
	{
		case 0: val = (1 - u - v)*(1-2*u-2*v); break;
		case 1: val = u*(2*u-1); break;
		case 2: val = v*(2*v-1); break;

		case 3: val = 4*u*(1-u-v); break;
		case 4: val = 4*u*v; break;
		case 5: val = 4*v*(1-u-v); break;
		default: assert(false);
	}
}

void quadr_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();
	val.resize(uv.rows(), uv.cols());
	switch(local_index)
	{
		case 0:
		val.col(0) = 4*u+4*v-3;
		val.col(1) = 4*u+4*v-3;
		break;

		case 1:
		val.col(0) = 4*u -1;
		val.col(1).setZero();
		break;

		case 2:
		val.col(0).setZero();
		val.col(1) = 4*v - 1;
		break;


		case 3:
		val.col(0) = 4 - 8*u - 4*v;
		val.col(1) = -4*u;
		break;

		case 4:
		val.col(0) = 4*v;
		val.col(1) = 4*u;
		break;

		case 5:
		val.col(0) = -4*v;
		val.col(1) = 4 - 4*u - 8*v;
		break;
		default: assert(false);
	}
}


void linear_tet_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	switch(local_index)
	{
		case 0: val = 1 - x - n - e; break;
		case 1: val = x; break;
		case 2: val = n; break;
		case 3: val = e; break;
		default: assert(false);
	}
}

void linear_tet_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	val.resize(xne.rows(), xne.cols());

	switch(local_index)
	{
		case 0:
		val.col(0).setConstant(-1);
		val.col(1).setConstant(-1);
		val.col(2).setConstant(-1);
		break;

		case 1:
		val.col(0).setConstant( 1);
		val.col(1).setConstant( 0);
		val.col(2).setConstant( 0);
		break;

		case 2:
		val.col(0).setConstant( 0);
		val.col(1).setConstant( 1);
		val.col(2).setConstant( 0);
		break;

		case 3:
		val.col(0).setConstant( 0);
		val.col(1).setConstant( 0);
		val.col(2).setConstant( 1);
		break;

		default: assert(false);
	}
}

void quadr_tet_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	switch(local_index)
	{
		case 0: val = (1 - 2*x - 2*n - 2*e)*(1 - x - n - e); break;
		case 1: val = (2*x-1)*x; break;
		case 2: val = (2*n-1)*n; break;
		case 3: val = (2*e-1)*e; break;

		case 4: val = 4*x * (1 - x - n - e); break;
		case 5: val = 4 * x * n; break;
		case 6: val = 4 * (1 - x - n - e) * n; break;

		case 7: val = 4*(1 - x - n - e)*e; break;
		case 8: val = 4*x*e; break;
		case 9: val = 4*n*e; break;
		default: assert(false);
	}
}

void quadr_tet_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	val.resize(xne.rows(), xne.cols());

	switch(local_index)
	{
		case 0:
		val.col(0) = -3+4*x+4*n+4*e;
		val.col(1) = -3+4*x+4*n+4*e;
		val.col(2) = -3+4*x+4*n+4*e;
		break;
		case 1:
		val.col(0) = 4*x-1;
		val.col(1).setZero();
		val.col(2).setZero();
		break;
		case 2:
		val.col(0).setZero();
		val.col(1) = 4*n-1;
		val.col(2).setZero();
		break;
		case 3:
		val.col(0).setZero();
		val.col(1).setZero();
		val.col(2) = 4*e-1;
		break;

		case 4:
		val.col(0) = 4-8*x-4*n-4*e;
		val.col(1) = -4*x;
		val.col(2) = -4*x;
		break;
		case 5:
		val.col(0) = 4*n;
		val.col(1) = 4*x;
		val.col(2).setZero();
		break;
		case 6:
		val.col(0) = -4*n;
		val.col(1) = -8*n+4-4*x-4*e;
		val.col(2) = -4*n;
		break;

		case 7:
		val.col(0) = -4*e;
		val.col(1) = -4*e;
		val.col(2) = -8*e+4-4*x-4*n;
		break;
		case 8:
		val.col(0) = 4*e;
		val.col(1).setZero();
		val.col(2) = 4*x;
		break;
		case 9:
		val.col(0).setZero();
		val.col(1) = 4*e;
		val.col(2) = 4*n;
		break;
		default: assert(false);
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



template<typename T>
Eigen::MatrixXd alpha(int i, T &t) {
	switch (i) {
		case 0: return (1-t);
		case 1: return t;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd dalpha(int i, T &t) {
	switch (i) {
		case 0: return -1+0*t;
		case 1: return 1+0*t;;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd theta(int i, T &t) {
	switch (i) {
		case 0: return (1 - t) * (1 - 2 * t);
		case 1: return 4 * t * (1 - t);
		case 2: return t * (2 * t - 1);
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd dtheta(int i, T &t) {
	switch (i) {
		case 0: return -3+4*t;
		case 1: return 4-8*t;
		case 2: return -1+4*t;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

void linear_quad_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = linear_quad_local_node[local_index];
	val = alpha(idx[0], u).array() * alpha(idx[1], v).array();
}

void linear_quad_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = linear_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dalpha(idx[0], u).array() * alpha(idx[1], v).array();
	val.col(1) = alpha(idx[0], u).array() * dalpha(idx[1], v).array();
}


void quadr_quad_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];
	val = theta(idx[0], u).array() * theta(idx[1], v).array();
}


void quadr_quad_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dtheta(idx[0], u).array() * theta(idx[1], v).array();
	val.col(1) = theta(idx[0], u).array() * dtheta(idx[1], v).array();
}
/////////////////////////////////////////







TEST_CASE("P1_2d", "[bases]") {
	TriQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for(int i = 0; i < 3; ++i){
		linear_tri_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_2d(1, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

		linear_tri_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_2d(1, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
	}

	//Check nodes
	polyfem::autogen::p_nodes_2d(1, val);
	for(int i = 0; i < 3; ++i){
		for(int d = 0; d < 2; ++d)
			REQUIRE(linear_tri_local_node[i][d] == Approx(val(i, d)).margin(1e-10));
	}

}

TEST_CASE("P2_2d", "[bases]") {
	TriQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for(int i = 0; i < 6; ++i){
		quadr_tri_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_2d(2, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

		quadr_tri_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_2d(2, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
	}

	//Check nodes
	polyfem::autogen::p_nodes_2d(2, val);
	for(int i = 0; i < 6; ++i){
		for(int d = 0; d < 2; ++d)
			REQUIRE(quadr_tri_local_node[i][d]/2. == Approx(val(i, d)).margin(1e-10));
	}
}


TEST_CASE("P1_3d", "[bases]") {
	TetQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(8, quad);

	Eigen::MatrixXd expected, val;
	for(int i = 0; i < 4; ++i){
		linear_tet_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_3d(1, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

		linear_tet_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_3d(1, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
	}

}

TEST_CASE("P2_3d", "[bases]") {
	TetQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(8, quad);

	Eigen::MatrixXd expected, val;
	for(int i = 0; i < 10; ++i){
		quadr_tet_basis_value(i, quad.points, expected);
		polyfem::autogen::p_basis_value_3d(2, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

		quadr_tet_basis_grad(i, quad.points, expected);
		polyfem::autogen::p_grad_basis_value_3d(2, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
	}
}


TEST_CASE("P3_2d", "[bases]") {
	Eigen::MatrixXd pts(10, 2);
	pts <<
	0, 0,
	1, 0,
	0, 1,
	1./3., 0,
	2./3., 0,
	2./3., 1./3.,
	1./3., 2./3.,
	0, 2./3.,
	0, 1./3.,
	1./3., 1./3.;

	Eigen::MatrixXd val;

	for(int i = 0; i < pts.rows(); ++i){
		polyfem::autogen::p_basis_value_2d(3, i, pts, val);

		// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

		for(int j = 0; j < val.size(); ++j){
			if(i == j)
				REQUIRE(val(j) == Approx(1).margin(1e-10));
			else
				REQUIRE(val(j) == Approx(0).margin(1e-10));
		}
	}
}

TEST_CASE("Pk_2d", "[bases]") {

	Eigen::MatrixXd pts;
	for(int k = 1; k < polyfem::autogen::MAX_P_BASES; ++k)
	{
		polyfem::autogen::p_nodes_2d(k, pts);

		Eigen::MatrixXd val;
		for(int i = 0; i < pts.rows(); ++i){
			polyfem::autogen::p_basis_value_2d(k, i, pts, val);

		// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for(int j = 0; j < val.size(); ++j){
				if(i == j)
					REQUIRE(val(j) == Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Approx(0).margin(1e-10));
			}
		}
	}
}

TEST_CASE("Pk_3d", "[bases]") {
	Eigen::MatrixXd pts;
	for(int k = 1; k < polyfem::autogen::MAX_P_BASES; ++k)
	{
		polyfem::autogen::p_nodes_3d(k, pts);

		Eigen::MatrixXd val;
		for(int i = 0; i < pts.rows(); ++i){
			polyfem::autogen::p_basis_value_3d(k, i, pts, val);

		// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for(int j = 0; j < val.size(); ++j){
				if(i == j)
					REQUIRE(val(j) == Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Approx(0).margin(1e-10));
			}
		}
	}
}




TEST_CASE("Q1_2d", "[bases]") {
	QuadQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for(int i = 0; i < 4; ++i){
		linear_quad_basis_value(i, quad.points, expected);
		polyfem::autogen::q_basis_value_2d(1, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

		linear_quad_basis_grad(i, quad.points, expected);
		polyfem::autogen::q_grad_basis_value_2d(1, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
	}

	//Check nodes
	polyfem::autogen::q_nodes_2d(1, val);
	for(int i = 0; i < 4; ++i){
		for(int d = 0; d < 2; ++d)
			REQUIRE(linear_quad_local_node[i][d] == Approx(val(i, d)).margin(1e-10));
	}

}

TEST_CASE("Q2_2d", "[bases]") {
	QuadQuadrature rule;
	Quadrature quad;
	rule.get_quadrature(12, quad);

	Eigen::MatrixXd expected, val;
	for(int i = 0; i < 9; ++i){
		quadr_quad_basis_value(i, quad.points, expected);
		polyfem::autogen::q_basis_value_2d(2, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));

		quadr_quad_basis_grad(i, quad.points, expected);
		polyfem::autogen::q_grad_basis_value_2d(2, i, quad.points, val);

		for(int j = 0; j < val.size(); ++j)
			REQUIRE(expected(j) == Approx(val(j)).margin(1e-10));
	}

	//Check nodes
	polyfem::autogen::q_nodes_2d(2, val);
	for(int i = 0; i < 9; ++i){
		for(int d = 0; d < 2; ++d)
			REQUIRE(quadr_quad_local_node[i][d]/2. == Approx(val(i, d)).margin(1e-10));
	}
}

TEST_CASE("Qk_2d", "[bases]") {

	Eigen::MatrixXd pts;
	for(int k = 1; k < polyfem::autogen::MAX_Q_BASES; ++k)
	{
		polyfem::autogen::q_nodes_2d(k, pts);

		Eigen::MatrixXd val;
		for(int i = 0; i < pts.rows(); ++i){
			polyfem::autogen::q_basis_value_2d(k, i, pts, val);

		// std::cout<<i<<"\n"<<val<<"\n\n\n"<<std::endl;

			for(int j = 0; j < val.size(); ++j){
				if(i == j)
					REQUIRE(val(j) == Approx(1).margin(1e-10));
				else
					REQUIRE(val(j) == Approx(0).margin(1e-10));
			}
		}
	}
}
