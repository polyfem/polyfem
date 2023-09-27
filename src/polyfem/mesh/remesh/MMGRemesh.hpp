// Original source from cellogram (https://github.com/cellogram/cellogram/blob/master/src/cellogram/remesh_adaptive.h)
// Authors: Tobias Lendenmann, Teseo Schneider, Jérémie Dumas, Marco Tarini
// License: MIT (https://github.com/cellogram/cellogram/blob/master/LICENSE)

#pragma once

#ifdef POLYFEM_WITH_MMG

////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem::mesh
{

	// See MmgTools documentation for interpreation
	// https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options
	//
	struct MmgOptions
	{
		/* Remeshing */
		bool angle_detection = true;
		double angle_value = 45.;
		double hausd = 0.01;
		double hsiz = 0.0; /* using hmin and hmax if set to 0 */
		double hmin = 0.01;
		double hmax = 2.;
		double hgrad = 1.105171;
		bool enable_anisotropy = false;
		bool optim = false;
		bool optimLES = false;
		bool opnbdy = false;
		bool noinsert = false;
		bool noswap = false;
		bool nomove = false;
		bool nosurf = false;
		std::string metric_attribute = "no_metric";
		/* Level set extraction */
		bool level_set = false;
		std::string ls_attribute = "no_ls";
		double ls_value = 0.;
	};

	///
	/// Remesh a 2d triangle mesh adaptively following to the given scalar field.
	///
	/// @param[in]  V     { #V x (2|3) input mesh vertices }
	/// @param[in]  F     { #F x 3 input mesh triangles }
	/// @param[in]  S     { #V x 1 per-vertex scalar field to follow }
	/// @param[out] OV    { #OV x (2|3) output mesh vertices }
	/// @param[out] OF    { #OF x 3 output mesh triangles }
	///
	void remesh_adaptive_2d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXd &S,
							Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, MmgOptions opt = MmgOptions());

	///
	/// Remesh a 3d tet-mesh adaptively following to the given scalar field.
	///
	/// @param[in]  V     { #V x 3 input mesh vertices }
	/// @param[in]  T     { #T x 4 input mesh tetrahedra }
	/// @param[in]  S     { #V x 1 per-vertex scalar field to follow }
	/// @param[out] OV    { #OV x 3 output mesh vertices }
	/// @param[out] OF    { #OF x F output mesh triangles }
	/// @param[out] OT    { #OT x 4 output mesh tetrahedra }
	///
	void remesh_adaptive_3d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::VectorXd &S,
							Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OT, MmgOptions opt = MmgOptions());

} // namespace polyfem::mesh

#endif
