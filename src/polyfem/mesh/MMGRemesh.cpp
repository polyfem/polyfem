// Original source from cellogram (https://github.com/cellogram/cellogram/blob/master/src/cellogram/remesh_adaptive.h)
// Authors: Tobias Lendenmann, Teseo Schneider, Jérémie Dumas, Marco Tarini
// License: MIT (https://github.com/cellogram/cellogram/blob/master/LICENSE)

#ifdef POLYFEM_WITH_MMG

////////////////////////////////////////////////////////////////////////////////
#include "MMGRemesh.hpp"
// #include <cellogram/MeshUtils.h>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <igl/boundary_facets.h>
#include <mmg/libmmg.h>

#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/Logger.hpp>
////////////////////////////////////////////////////////////////////////////////
//
// Wrapper for 3D remeshing comes from:
// https://github.com/mxncr/mmgig
//

namespace polyfem::mesh
{
	namespace
	{
		// Flip triangle/tet if there signed volume is negative.
		bool orient_simplices(const Eigen::MatrixXd &vertices, Eigen::MatrixXi &elements)
		{
			for (int e = 0; e < elements.rows(); ++e)
			{
				double vol = 0;
				if (elements.cols() == 3)
				{
					vol = utils::triangle_area_2D(
						vertices.row(elements(e, 0)),
						vertices.row(elements(e, 1)),
						vertices.row(elements(e, 2)));
				}
				else
				{
					assert(elements.cols() == 4);
					vol = utils::tetrahedron_volume(
						vertices.row(elements(e, 0)),
						vertices.row(elements(e, 1)),
						vertices.row(elements(e, 2)),
						vertices.row(elements(e, 3)));
				}

				if (vol == 0)
				{
					logger().error("MMG produced a degenerate simplex {}", e);
					return false;
				}
				if (vol < 0)
					std::swap(elements(e, 1), elements(e, 2));
			}
			return true;
		}

		/// Simplicial mesh exchanged with MMG (vertices are 3D, z = 0 in 2D).
		struct SimplexMesh
		{
			Eigen::MatrixXd V; ///< #V x 3 vertex positions
			Eigen::MatrixXi F; ///< #F x 3 triangles
			Eigen::MatrixXi T; ///< #T x 4 tetrahedra
		};

		SimplexMesh make_simplex_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXi &T)
		{
			SimplexMesh M;
			M.V.resize(V.rows(), 3);
			for (int i = 0; i < V.rows(); ++i)
			{
				M.V(i, 0) = V(i, 0);
				M.V(i, 1) = V(i, 1);
				M.V(i, 2) = (V.cols() == 2 ? 0 : V(i, 2));
			}
			if (F.cols() != 3 && F.rows() != 0)
				throw std::runtime_error("Mesh faces not supported");
			M.F = F;
			M.F.conservativeResize(F.rows(), 3);
			if (T.cols() != 4 && T.rows() != 0)
				throw std::runtime_error("Mesh cells not supported");
			M.T = T;
			M.T.conservativeResize(T.rows(), 4);
			return M;
		}

		bool mmg_to_simplex_mesh(const MMG5_pMesh mmg, SimplexMesh &M)
		{
			logger().trace("converting MMG5_pMesh to Eigen ...");
			/* Notes:
			 * - indexing seems to start at 1 in MMG */

			assert(mmg->dim == 2 || mmg->dim == 3);
			M.V.resize(mmg->np, 3);
			for (int v = 0; v < mmg->np; ++v)
			{
				for (int d = 0; d < mmg->dim; ++d)
				{
					M.V(v, d) = mmg->point[v + 1].c[d];
				}
				// Vertices are 3D, init z coordinate to zero for 2D MMG mesh.
				if (mmg->dim == 2)
					M.V(v, 2) = 0.;
			}
			M.F.resize(mmg->nt, 3);
			for (int t = 0; t < mmg->nt; ++t)
			{
				M.F(t, 0) = mmg->tria[t + 1].v[0] - 1;
				M.F(t, 1) = mmg->tria[t + 1].v[1] - 1;
				M.F(t, 2) = mmg->tria[t + 1].v[2] - 1;
			}
			M.T.resize(mmg->dim == 3 ? mmg->ne : 0, 4);
			for (int c = 0; c < M.T.rows(); ++c)
			{
				M.T(c, 0) = mmg->tetra[c + 1].v[0] - 1;
				M.T(c, 1) = mmg->tetra[c + 1].v[1] - 1;
				M.T(c, 2) = mmg->tetra[c + 1].v[2] - 1;
				M.T(c, 3) = mmg->tetra[c + 1].v[3] - 1;
			}

			return true;
		}

		bool simplex_mesh_to_mmg3d(const SimplexMesh &M, MMG5_pMesh &mmg, MMG5_pSol &sol)
		{
			logger().trace("converting Eigen mesh to MMG5_pMesh ...");

			MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);

			if (MMG3D_Set_meshSize(mmg, (int)M.V.rows(), (int)M.T.rows(), 0, /* nb prisms */
								   (int)M.F.rows(), 0,                       /* nb quad */
								   0                                         /* nb edges */
								   )
				!= 1)
			{
				logger().error("failed to MMG3D_Set_meshSize");
				return false;
			}

			for (int v = 0; v < mmg->np; ++v)
			{
				for (int d = 0; d < 3; ++d)
				{
					mmg->point[v + 1].c[d] = M.V(v, d);
				}
			}
			for (int t = 0; t < mmg->nt; ++t)
			{
				mmg->tria[t + 1].v[0] = M.F(t, 0) + 1;
				mmg->tria[t + 1].v[1] = M.F(t, 1) + 1;
				mmg->tria[t + 1].v[2] = M.F(t, 2) + 1;
			}
			for (int c = 0; c < mmg->ne; ++c)
			{
				mmg->tetra[c + 1].v[0] = M.T(c, 0) + 1;
				mmg->tetra[c + 1].v[1] = M.T(c, 1) + 1;
				mmg->tetra[c + 1].v[2] = M.T(c, 2) + 1;
				mmg->tetra[c + 1].v[3] = M.T(c, 3) + 1;
			}

			if (MMG3D_Set_solSize(mmg, sol, MMG5_Vertex, (int)M.V.rows(), MMG5_Scalar) != 1)
			{
				logger().error("failed to MMG3D_Set_solSize");
				return false;
			}
			for (int v = 0; v < M.V.rows(); ++v)
			{
				sol->m[v + 1] = 1.;
			}
			if (MMG3D_Chk_meshData(mmg, sol) != 1)
			{
				logger().error("error in mmg: inconsistent mesh and sol");
				return false;
			}

			MMG3D_Set_handGivenMesh(mmg); /* because we don't use the API functions */

			return true;
		}

		bool simplex_mesh_to_mmg2d(const SimplexMesh &M, MMG5_pMesh &mmg, MMG5_pSol &sol)
		{
			MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
			if (!MMG2D_Set_meshSize(mmg, M.V.rows(), M.F.rows(), 0, 0))
			{
				logger().error("mmg2d_remesh: failed to allocate the MMG mesh");
				return false;
			}

			for (int v = 0; v < mmg->np; ++v)
			{
				mmg->point[v + 1].c[0] = M.V(v, 0);
				mmg->point[v + 1].c[1] = M.V(v, 1);
			}
			for (int t = 0; t < mmg->nt; ++t)
			{
				// MMG use 1 based indexing.
				mmg->tria[t + 1].v[0] = M.F(t, 0) + 1;
				mmg->tria[t + 1].v[1] = M.F(t, 1) + 1;
				mmg->tria[t + 1].v[2] = M.F(t, 2) + 1;
			}

			if (!MMG2D_Set_solSize(mmg, sol, MMG5_Vertex, M.V.rows(), MMG5_Scalar))
			{
				logger().error("failed to MMG2D_Set_solSize");
				return false;
			}

			for (int v = 0; v < M.V.rows(); ++v)
				sol->m[v + 1] = 1.;

			if (!MMG2D_Chk_meshData(mmg, sol))
			{
				logger().error("error in mmg: inconsistent mesh and sol");
				return false;
			}
			return true;
		}

		void mmg2d_free(MMG5_pMesh mmg, MMG5_pSol sol)
		{
			MMG2D_Free_all(MMG5_ARG_start,
						   MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
		}

		void mmg3d_free(MMG5_pMesh mmg, MMG5_pSol sol)
		{
			MMG3D_Free_all(MMG5_ARG_start,
						   MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
		}

		/// @param metric Per-vertex size used when opt.metric_attribute is set, or nullptr.
		bool mmg2d_tri_remesh(
			const SimplexMesh &M,
			const Eigen::VectorXd *metric,
			const std::vector<int> *pinned_vertices,
			SimplexMesh &M_out,
			const MmgOptions &opt)
		{
			MMG5_pMesh mesh = nullptr;
			MMG5_pSol met = nullptr;
			bool ok = simplex_mesh_to_mmg2d(M, mesh, met);
			if (!ok)
			{
				logger().error("mmg2d_remesh: failed to convert mesh to MMG5_pMesh");
				mmg2d_free(mesh, met);
				return false;
			}

			// Pin vertices.
			if (pinned_vertices != nullptr)
			{
				for (const int v : *pinned_vertices)
				{
					assert(v >= 0 && v < M.V.rows());
					MMG2D_Set_requiredVertex(mesh, v + 1);
				}
			}

			/* Set remeshing options */
			MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_angleDetection, opt.angle_value);
			if (opt.enable_anisotropy)
			{
				MMG2D_Set_solSize(mesh, met, MMG5_Vertex, 0, MMG5_Tensor);
			}
			const bool with_metric = opt.metric_attribute != "no_metric";
			if (opt.hsiz == 0. || with_metric)
			{
				if (with_metric || !opt.optim)
				{
					MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hmin, opt.hmin);
					MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hmax, opt.hmax);
				}
				else
				{
					met->np = 0;
				}
			}
			else
			{
				met->np = 0;
				MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hsiz, opt.hsiz);
			}
			MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hausd, opt.hausd);
			MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hgrad, opt.hgrad);
			MMG2D_Set_iparameter(mesh, met, MMG2D_IPARAM_angle, int(opt.angle_detection));
			MMG2D_Set_iparameter(mesh, met, MMG2D_IPARAM_noswap, int(opt.noswap));
			MMG2D_Set_iparameter(mesh, met, MMG2D_IPARAM_noinsert, int(opt.noinsert));
			MMG2D_Set_iparameter(mesh, met, MMG2D_IPARAM_nomove, int(opt.nomove));
			MMG2D_Set_iparameter(mesh, met, MMG2D_IPARAM_nosurf, int(opt.nosurf));
			MMG2D_Set_iparameter(mesh, met, MMG2D_IPARAM_optim, int(opt.optim));
			if (with_metric)
			{
				if (metric == nullptr)
				{
					logger().error("mmg2D_remesh: {} is not a vertex attribute, cancel", opt.metric_attribute);
					mmg2d_free(mesh, met);
					return false;
				}
				for (int v = 0; v < M.V.rows(); ++v)
				{
					met->m[v + 1] = (*metric)(v);
				}
			}

			int ier = MMG2D_mmg2dlib(mesh, met);
			if (ier != MMG5_SUCCESS)
			{
				logger().error("mmg2d_remesh: failed to remesh");
				mmg2d_free(mesh, met);
				return false;
			}

			ok = mmg_to_simplex_mesh(mesh, M_out);

			mmg2d_free(mesh, met);
			return ok;
		}

		/// @param metric Per-vertex size used when opt.metric_attribute is set, or nullptr.
		bool mmg3d_tet_remesh(
			const SimplexMesh &M,
			const Eigen::VectorXd *metric,
			const std::vector<int> *pinned_vertices,
			SimplexMesh &M_out,
			const MmgOptions &opt)
		{
			MMG5_pMesh mesh = nullptr;
			MMG5_pSol met = nullptr;
			bool ok = simplex_mesh_to_mmg3d(M, mesh, met);
			if (!ok)
			{
				logger().error("mmg3d_remesh: failed to convert mesh to MMG5_pMesh");
				mmg3d_free(mesh, met);
				return false;
			}

			// Pin vertices and triangles.
			std::vector<bool> is_pinned(M.V.rows(), false);
			if (pinned_vertices != nullptr)
			{
				for (const int v : *pinned_vertices)
				{
					assert(v >= 0 && v < M.V.rows());
					is_pinned[v] = true;
					MMG3D_Set_requiredVertex(mesh, v + 1);
				}
			}
			for (int f = 0; f < M.F.rows(); ++f)
			{
				if (is_pinned[M.F(f, 0)]
					&& is_pinned[M.F(f, 1)]
					&& is_pinned[M.F(f, 2)])
				{
					MMG3D_Set_requiredTriangle(mesh, f + 1);
				}
			}

			/* Set remeshing options */
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_angleDetection, opt.angle_value);
			if (opt.enable_anisotropy)
			{
				MMG3D_Set_solSize(mesh, met, MMG5_Vertex, 0, MMG5_Tensor);
			}
			const bool with_metric = opt.metric_attribute != "no_metric";
			if (opt.hsiz == 0. || with_metric)
			{
				if (with_metric || !opt.optim)
				{
					MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmin, opt.hmin);
					MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmax, opt.hmax);
				}
				else
				{
					met->np = 0;
				}
			}
			else
			{
				met->np = 0;
				MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hsiz, opt.hsiz);
			}
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hausd, opt.hausd);
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hgrad, opt.hgrad);
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_angle, int(opt.angle_detection));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_noswap, int(opt.noswap));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_noinsert, int(opt.noinsert));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_nomove, int(opt.nomove));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_nosurf, int(opt.nosurf));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_opnbdy, int(opt.opnbdy));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_optim, int(opt.optim));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_optimLES, int(opt.optimLES));
			if (with_metric)
			{
				if (metric == nullptr)
				{
					logger().error("mmg3D_remesh: {} is not a vertex attribute, cancel", opt.metric_attribute);
					mmg3d_free(mesh, met);
					return false;
				}
				for (int v = 0; v < M.V.rows(); ++v)
				{
					met->m[v + 1] = (*metric)(v);
				}
			}

			int ier = MMG3D_mmg3dlib(mesh, met);
			if (ier != MMG5_SUCCESS)
			{
				logger().error("mmg3d_remesh: failed to remesh");
				mmg3d_free(mesh, met);
				return false;
			}

			ok = mmg_to_simplex_mesh(mesh, M_out);

			mmg3d_free(mesh, met);
			return ok;
		}

	} // anonymous namespace

	////////////////////////////////////////////////////////////////////////////////

	bool remesh_2d(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		Eigen::MatrixXd &OV,
		Eigen::MatrixXi &OF,
		MmgOptions opt,
		const std::vector<int> *pinned_vertices)
	{
		assert(V.cols() == 2);
		assert(F.cols() == 3);
		const SimplexMesh M = make_simplex_mesh(V, F, Eigen::MatrixXi(0, 4));
		SimplexMesh M_out;
		if (!mmg2d_tri_remesh(M, nullptr, pinned_vertices, M_out, opt))
			return false;
		OV = M_out.V;
		OF = M_out.F;
		// The exchanged vertices are 3D, truncate unused third column.
		OV.conservativeResize(OV.rows(), 2);
		// MMG does not preserve orientation. Ensure singed area is positive.
		return orient_simplices(OV, OF);
	}

	void remesh_adaptive_2d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXd &S,
							Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, MmgOptions opt)
	{
		assert(V.cols() == 2 || V.cols() == 3);
		assert(V.rows() == S.size());
		assert(F.cols() == 3);
		const SimplexMesh M = make_simplex_mesh(V, F, Eigen::MatrixXi(0, 4));
		SimplexMesh M_out;

		opt.metric_attribute = "scalar";
		if (!mmg2d_tri_remesh(M, &S, nullptr, M_out, opt))
			return;
		OV = M_out.V;
		OF = M_out.F;
		OV.conservativeResize(OV.rows(), 2);
		orient_simplices(OV, OF);
	}

	bool remesh_3d(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &T,
		Eigen::MatrixXd &OV,
		Eigen::MatrixXi &OF,
		Eigen::MatrixXi &OT,
		MmgOptions opt,
		const std::vector<int> *pinned_vertices)
	{
		assert(V.cols() == 3);
		assert(T.cols() == 4);
		Eigen::MatrixXi boundary;
		igl::boundary_facets(T, boundary);
		const SimplexMesh M = make_simplex_mesh(V, boundary, T);
		SimplexMesh M_out;
		if (!mmg3d_tet_remesh(M, nullptr, pinned_vertices, M_out, opt))
			return false;
		OV = M_out.V;
		OF = M_out.F;
		OT = M_out.T;
		// MMG does not preserve orientation. Ensure singed area is positive.
		return orient_simplices(OV, OT);
	}

	void remesh_adaptive_3d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::VectorXd &S,
							Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OT, MmgOptions opt)
	{
		assert(V.cols() == 3);
		assert(V.rows() == S.size());
		const SimplexMesh M = make_simplex_mesh(V, Eigen::MatrixXi(0, 3), T);
		SimplexMesh M_out;

		// Remeshing options
		opt.metric_attribute = "scalar";

		// Remesh volume
		if (!mmg3d_tet_remesh(M, &S, nullptr, M_out, opt))
			return;

		// Convert output
		OV = M_out.V;
		OF = M_out.F;
		OT = M_out.T;
		orient_simplices(OV, OT);
	}

} // namespace polyfem::mesh

#endif
