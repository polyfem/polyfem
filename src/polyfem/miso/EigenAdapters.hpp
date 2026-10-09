#pragma once
#include <Eigen/Core>

#include "generated/P1TriCGV.hpp"
#include "generated/P2TriVal.hpp"
#include "generated/P2TriCGV.hpp"
#include "generated/P3TriVal.hpp"
#include "generated/P3TriCGV.hpp"
#include "generated/P4TriVal.hpp"
#include "generated/P4TriCGV.hpp"
#include "generated/P1TetCGV.hpp"
#include "generated/P2TetVal.hpp"
#include "generated/P2TetCGV.hpp"
#include "generated/P3TetVal.hpp"
#include "generated/P3TetCGV.hpp"

#include "generated/JacEval_P1Tri.hpp"
#include "generated/JacEval_P2Tri.hpp"
#include "generated/JacEval_P3Tri.hpp"
#include "generated/JacEval_P4Tri.hpp"
#include "generated/JacEval_P1Tet.hpp"
#include "generated/JacEval_P2Tet.hpp"
#include "generated/JacEval_P3Tet.hpp"

namespace miso {

// Extract N rows from column `col` of `m` starting at `offset`,
// applying an optional index permutation for node reordering.
// perm[i] = which source row (relative to offset) feeds miso slot i.
template<int N>
inline RealVector<N> rv(const Eigen::MatrixXd &m, int offset, int col, const int *perm = nullptr)
{
	RealVector<N> v;
	for (int i = 0; i < N; ++i)
		v[i] = m(offset + (perm ? perm[i] : i), col);
	return v;
}

#ifndef MISO_FP_ARITHMETIC
// Displacements given as boxes [lo, hi] rather than points: the continuous
// check certifies every segment between the two boxes, which is how it covers
// the rounding of the step the solver finally takes (see max_time_step).
struct Box
{
	const Eigen::MatrixXd &lo;
	const Eigen::MatrixXd &hi;
};

template<int N>
inline RealVector<N> rv(const Box &m, int offset, int col, const int *perm = nullptr)
{
	RealVector<N> v;
	for (int i = 0; i < N; ++i)
	{
		const int r = offset + (perm ? perm[i] : i);
		v[i] = Real(m.lo(r, col), m.hi(r, col));
	}
	return v;
}
#endif

// Node-order permutations: polyfem's p_nodes ordering -> miso's eval_points ordering.
// perm[i] = polyfem row index that feeds miso node i.
static constexpr int tri1_perm[3] = {0, 2, 1};
static constexpr int tri2_perm[6] = {0, 5, 2, 3, 4, 1};
static constexpr int tri3_perm[10] = {0, 8, 7, 2, 3, 9, 6, 4, 5, 1};
static constexpr int tri4_perm[15] = {0, 11, 10, 9, 2, 3, 12, 13, 8, 4, 14, 7, 5, 6, 1};
static constexpr int tet1_perm[4] = {0, 3, 2, 1};
static constexpr int tet2_perm[10] = {0, 7, 3, 6, 9, 2, 4, 8, 5, 1};
static constexpr int tet3_perm[20] = {0, 10, 11, 3, 9, 19, 15, 8, 14, 2, 4, 17, 13, 16, 18, 7, 5, 12, 6, 1};

// Control points enter as rest positions r and displacements u, separately:
// the generated code forms r + u in interval arithmetic, so the displacement
// reaches it exactly as the solver computed it.

// ---- Val (static validity) factories ----

// P1: the Jacobian is constant, so the static check is one evaluation.
inline Real eval_p1tri_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return JacEval_P1Tri(rv<3>(r,o,0,tri1_perm), rv<3>(r,o,1,tri1_perm), rv<3>(u,o,0,tri1_perm), rv<3>(u,o,1,tri1_perm)); }

inline Real eval_p1tet_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return JacEval_P1Tet(rv<4>(r,o,0,tet1_perm), rv<4>(r,o,1,tet1_perm), rv<4>(r,o,2,tet1_perm), rv<4>(u,o,0,tet1_perm), rv<4>(u,o,1,tet1_perm), rv<4>(u,o,2,tet1_perm)); }

inline P2TriVal make_p2tri_val(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<6>(r,o,0,tri2_perm), rv<6>(r,o,1,tri2_perm), rv<6>(u,o,0,tri2_perm), rv<6>(u,o,1,tri2_perm)}; }

inline P3TriVal make_p3tri_val(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<10>(r,o,0,tri3_perm), rv<10>(r,o,1,tri3_perm), rv<10>(u,o,0,tri3_perm), rv<10>(u,o,1,tri3_perm)}; }

inline P4TriVal make_p4tri_val(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<15>(r,o,0,tri4_perm), rv<15>(r,o,1,tri4_perm), rv<15>(u,o,0,tri4_perm), rv<15>(u,o,1,tri4_perm)}; }

inline P2TetVal make_p2tet_val(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<10>(r,o,0,tet2_perm), rv<10>(r,o,1,tet2_perm), rv<10>(r,o,2,tet2_perm), rv<10>(u,o,0,tet2_perm), rv<10>(u,o,1,tet2_perm), rv<10>(u,o,2,tet2_perm)}; }

inline P3TetVal make_p3tet_val(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<20>(r,o,0,tet3_perm), rv<20>(r,o,1,tet3_perm), rv<20>(r,o,2,tet3_perm), rv<20>(u,o,0,tet3_perm), rv<20>(u,o,1,tet3_perm), rv<20>(u,o,2,tet3_perm)}; }

// ---- Pointwise evaluators (the elastic energy's det J) ----
// Bind one element's control points once, then evaluate at any reference
// point (row k of uv). P1 is constant: use eval_p1tri_jac / eval_p1tet_jac.

struct P2TriJac
{
	RealVector<6> rx, ry, ux, uy;
	Real operator()(const Eigen::MatrixXd &uv, int k) const { return JacEval_P2Tri(rx, ry, ux, uy, Real(uv(k, 0)), Real(uv(k, 1))); }
};
inline P2TriJac make_p2tri_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<6>(r,o,0,tri2_perm), rv<6>(r,o,1,tri2_perm), rv<6>(u,o,0,tri2_perm), rv<6>(u,o,1,tri2_perm)}; }

struct P3TriJac
{
	RealVector<10> rx, ry, ux, uy;
	Real operator()(const Eigen::MatrixXd &uv, int k) const { return JacEval_P3Tri(rx, ry, ux, uy, Real(uv(k, 0)), Real(uv(k, 1))); }
};
inline P3TriJac make_p3tri_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<10>(r,o,0,tri3_perm), rv<10>(r,o,1,tri3_perm), rv<10>(u,o,0,tri3_perm), rv<10>(u,o,1,tri3_perm)}; }

struct P4TriJac
{
	RealVector<15> rx, ry, ux, uy;
	Real operator()(const Eigen::MatrixXd &uv, int k) const { return JacEval_P4Tri(rx, ry, ux, uy, Real(uv(k, 0)), Real(uv(k, 1))); }
};
inline P4TriJac make_p4tri_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<15>(r,o,0,tri4_perm), rv<15>(r,o,1,tri4_perm), rv<15>(u,o,0,tri4_perm), rv<15>(u,o,1,tri4_perm)}; }

struct P2TetJac
{
	RealVector<10> rx, ry, rz, ux, uy, uz;
	Real operator()(const Eigen::MatrixXd &uv, int k) const { return JacEval_P2Tet(rx, ry, rz, ux, uy, uz, Real(uv(k, 0)), Real(uv(k, 1)), Real(uv(k, 2))); }
};
inline P2TetJac make_p2tet_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<10>(r,o,0,tet2_perm), rv<10>(r,o,1,tet2_perm), rv<10>(r,o,2,tet2_perm), rv<10>(u,o,0,tet2_perm), rv<10>(u,o,1,tet2_perm), rv<10>(u,o,2,tet2_perm)}; }

struct P3TetJac
{
	RealVector<20> rx, ry, rz, ux, uy, uz;
	Real operator()(const Eigen::MatrixXd &uv, int k) const { return JacEval_P3Tet(rx, ry, rz, ux, uy, uz, Real(uv(k, 0)), Real(uv(k, 1)), Real(uv(k, 2))); }
};
inline P3TetJac make_p3tet_jac(const Eigen::MatrixXd &r, const Eigen::MatrixXd &u, int o)
{ return {rv<20>(r,o,0,tet3_perm), rv<20>(r,o,1,tet3_perm), rv<20>(r,o,2,tet3_perm), rv<20>(u,o,0,tet3_perm), rv<20>(u,o,1,tet3_perm), rv<20>(u,o,2,tet3_perm)}; }

// ---- CGV (continuous geometric validity) factories ----

// P1 is a one-variable problem in t: the Jacobian is constant in space.

template<typename U>
inline P1TriCGV make_p1tri_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<3>(r,o,0,tri1_perm), rv<3>(r,o,1,tri1_perm), rv<3>(u0,o,0,tri1_perm), rv<3>(u0,o,1,tri1_perm), rv<3>(u1,o,0,tri1_perm), rv<3>(u1,o,1,tri1_perm)}; }

template<typename U>
inline P2TriCGV make_p2tri_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<6>(r,o,0,tri2_perm), rv<6>(r,o,1,tri2_perm), rv<6>(u0,o,0,tri2_perm), rv<6>(u0,o,1,tri2_perm), rv<6>(u1,o,0,tri2_perm), rv<6>(u1,o,1,tri2_perm)}; }

template<typename U>
inline P3TriCGV make_p3tri_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<10>(r,o,0,tri3_perm), rv<10>(r,o,1,tri3_perm), rv<10>(u0,o,0,tri3_perm), rv<10>(u0,o,1,tri3_perm), rv<10>(u1,o,0,tri3_perm), rv<10>(u1,o,1,tri3_perm)}; }

template<typename U>
inline P4TriCGV make_p4tri_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<15>(r,o,0,tri4_perm), rv<15>(r,o,1,tri4_perm), rv<15>(u0,o,0,tri4_perm), rv<15>(u0,o,1,tri4_perm), rv<15>(u1,o,0,tri4_perm), rv<15>(u1,o,1,tri4_perm)}; }

template<typename U>
inline P1TetCGV make_p1tet_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<4>(r,o,0,tet1_perm), rv<4>(r,o,1,tet1_perm), rv<4>(r,o,2,tet1_perm), rv<4>(u0,o,0,tet1_perm), rv<4>(u0,o,1,tet1_perm), rv<4>(u0,o,2,tet1_perm), rv<4>(u1,o,0,tet1_perm), rv<4>(u1,o,1,tet1_perm), rv<4>(u1,o,2,tet1_perm)}; }

template<typename U>
inline P2TetCGV make_p2tet_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<10>(r,o,0,tet2_perm), rv<10>(r,o,1,tet2_perm), rv<10>(r,o,2,tet2_perm), rv<10>(u0,o,0,tet2_perm), rv<10>(u0,o,1,tet2_perm), rv<10>(u0,o,2,tet2_perm), rv<10>(u1,o,0,tet2_perm), rv<10>(u1,o,1,tet2_perm), rv<10>(u1,o,2,tet2_perm)}; }

template<typename U>
inline P3TetCGV make_p3tet_cgv(const Eigen::MatrixXd &r, const U &u0, const U &u1, int o)
{ return {rv<20>(r,o,0,tet3_perm), rv<20>(r,o,1,tet3_perm), rv<20>(r,o,2,tet3_perm), rv<20>(u0,o,0,tet3_perm), rv<20>(u0,o,1,tet3_perm), rv<20>(u0,o,2,tet3_perm), rv<20>(u1,o,0,tet3_perm), rv<20>(u1,o,1,tet3_perm), rv<20>(u1,o,2,tet3_perm)}; }

} // namespace miso
