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

// Node-order permutations: polyfem's p_nodes ordering -> miso's eval_points ordering.
// perm[i] = polyfem row index that feeds miso node i.
static constexpr int tri1_perm[3] = {0, 2, 1};
static constexpr int tri2_perm[6] = {0, 5, 2, 3, 4, 1};
static constexpr int tri3_perm[10] = {0, 8, 7, 2, 3, 9, 6, 4, 5, 1};
static constexpr int tri4_perm[15] = {0, 11, 10, 9, 2, 3, 12, 13, 8, 4, 14, 7, 5, 6, 1};
static constexpr int tet1_perm[4] = {0, 3, 2, 1};
static constexpr int tet2_perm[10] = {0, 7, 3, 6, 9, 2, 4, 8, 5, 1};
static constexpr int tet3_perm[20] = {0, 10, 11, 3, 9, 19, 15, 8, 14, 2, 4, 17, 13, 16, 18, 7, 5, 12, 6, 1};

// ---- Val (static validity) factories ----

inline P2TriVal make_p2tri_val(const Eigen::MatrixXd &cp, int o)
{ return {rv<6>(cp,o,0,tri2_perm), rv<6>(cp,o,1,tri2_perm)}; }

inline P3TriVal make_p3tri_val(const Eigen::MatrixXd &cp, int o)
{ return {rv<10>(cp,o,0,tri3_perm), rv<10>(cp,o,1,tri3_perm)}; }

inline P4TriVal make_p4tri_val(const Eigen::MatrixXd &cp, int o)
{ return {rv<15>(cp,o,0,tri4_perm), rv<15>(cp,o,1,tri4_perm)}; }

inline P2TetVal make_p2tet_val(const Eigen::MatrixXd &cp, int o)
{ return {rv<10>(cp,o,0,tet2_perm), rv<10>(cp,o,1,tet2_perm), rv<10>(cp,o,2,tet2_perm)}; }

inline P3TetVal make_p3tet_val(const Eigen::MatrixXd &cp, int o)
{ return {rv<20>(cp,o,0,tet3_perm), rv<20>(cp,o,1,tet3_perm), rv<20>(cp,o,2,tet3_perm)}; }

// ---- CGV (continuous geometric validity) factories ----

inline P1TriCGV make_p1tri_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<3>(cp1,o,0,tri1_perm), rv<3>(cp1,o,1,tri1_perm), rv<3>(cp2,o,0,tri1_perm), rv<3>(cp2,o,1,tri1_perm)}; }

inline P2TriCGV make_p2tri_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<6>(cp1,o,0,tri2_perm), rv<6>(cp1,o,1,tri2_perm), rv<6>(cp2,o,0,tri2_perm), rv<6>(cp2,o,1,tri2_perm)}; }

inline P3TriCGV make_p3tri_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<10>(cp1,o,0,tri3_perm), rv<10>(cp1,o,1,tri3_perm), rv<10>(cp2,o,0,tri3_perm), rv<10>(cp2,o,1,tri3_perm)}; }

inline P4TriCGV make_p4tri_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<15>(cp1,o,0,tri4_perm), rv<15>(cp1,o,1,tri4_perm), rv<15>(cp2,o,0,tri4_perm), rv<15>(cp2,o,1,tri4_perm)}; }

inline P1TetCGV make_p1tet_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<4>(cp1,o,0,tet1_perm), rv<4>(cp1,o,1,tet1_perm), rv<4>(cp1,o,2,tet1_perm),
          rv<4>(cp2,o,0,tet1_perm), rv<4>(cp2,o,1,tet1_perm), rv<4>(cp2,o,2,tet1_perm)}; }

inline P2TetCGV make_p2tet_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<10>(cp1,o,0,tet2_perm), rv<10>(cp1,o,1,tet2_perm), rv<10>(cp1,o,2,tet2_perm),
          rv<10>(cp2,o,0,tet2_perm), rv<10>(cp2,o,1,tet2_perm), rv<10>(cp2,o,2,tet2_perm)}; }

inline P3TetCGV make_p3tet_cgv(const Eigen::MatrixXd &cp1, const Eigen::MatrixXd &cp2, int o)
{ return {rv<20>(cp1,o,0,tet3_perm), rv<20>(cp1,o,1,tet3_perm), rv<20>(cp1,o,2,tet3_perm),
          rv<20>(cp2,o,0,tet3_perm), rv<20>(cp2,o,1,tet3_perm), rv<20>(cp2,o,2,tet3_perm)}; }

} // namespace miso
