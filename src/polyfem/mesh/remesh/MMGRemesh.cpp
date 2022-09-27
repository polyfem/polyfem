// Original source from cellogram (https://github.com/cellogram/cellogram/blob/master/src/cellogram/remesh_adaptive.h)
// Authors: Tobias Lendenmann, Teseo Schneider, Jérémie Dumas, Marco Tarini
// License: MIT (https://github.com/cellogram/cellogram/blob/master/LICENSE)

#ifdef POLYFEM_WITH_MMG

////////////////////////////////////////////////////////////////////////////////
#include "MMGRemesh.hpp"
// #include <cellogram/MeshUtils.h>
#include <iomanip>
#include <cassert>
#include <geogram/basic/attributes.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <mmg/libmmg.h>

#include <polyfem/utils/Logger.hpp>
////////////////////////////////////////////////////////////////////////////////
//
// Wrapper for 3D remeshing comes from:
// https://github.com/mxncr/mmgig
//

#ifdef WIN32
typedef unsigned int uint;
#endif // WIN32

namespace polyfem::mesh
{
	namespace
	{
		void to_geogram_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, GEO::Mesh &M)
		{
			M.clear();
			// Setup vertices
			M.vertices.create_vertices((int)V.rows());
			for (int i = 0; i < (int)M.vertices.nb(); ++i)
			{
				GEO::vec3 &p = M.vertices.point(i);
				p[0] = V(i, 0);
				p[1] = V(i, 1);
				p[2] = (V.cols() == 2 ? 0 : V(i, 2));
			}
			// Setup faces
			if (F.cols() == 3)
			{
				M.facets.create_triangles((int)F.rows());
			}
			else if (F.cols() == 4)
			{
				M.facets.create_quads((int)F.rows());
			}
			else
			{
				throw std::runtime_error("Mesh faces not supported");
			}
			for (int c = 0; c < (int)M.facets.nb(); ++c)
			{
				for (int lv = 0; lv < F.cols(); ++lv)
				{
					M.facets.set_vertex(c, lv, F(c, lv));
				}
			}
			M.facets.connect();
		}

		void to_geogram_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXi &T, GEO::Mesh &M)
		{
			to_geogram_mesh(V, F, M);
			if (T.cols() == 4)
			{
				M.cells.create_tets((int)T.rows());
			}
			else if (T.rows() != 0)
			{
				throw std::runtime_error("Mesh cells not supported");
			}
			for (int c = 0; c < (int)M.cells.nb(); ++c)
			{
				for (int lv = 0; lv < T.cols(); ++lv)
				{
					M.cells.set_vertex(c, lv, T(c, lv));
				}
			}
			M.cells.connect();
		}

		void from_geogram_mesh(const GEO::Mesh &M, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &T)
		{
			V.resize(M.vertices.nb(), 3);
			for (int i = 0; i < (int)M.vertices.nb(); ++i)
			{
				GEO::vec3 p = M.vertices.point(i);
				V.row(i) << p[0], p[1], p[2];
			}
			assert(M.facets.are_simplices());
			F.resize(M.facets.nb(), 3);
			for (int c = 0; c < (int)M.facets.nb(); ++c)
			{
				for (int lv = 0; lv < 3; ++lv)
				{
					F(c, lv) = M.facets.vertex(c, lv);
				}
			}
			assert(M.cells.are_simplices());
			T.resize(M.cells.nb(), 4);
			for (int c = 0; c < (int)M.cells.nb(); ++c)
			{
				for (int lv = 0; lv < 4; ++lv)
				{
					T(c, lv) = M.cells.vertex(c, lv);
				}
			}
		}

		bool mmg_to_geo(const MMG5_pMesh mmg, GEO::Mesh &M)
		{
			logger().trace("converting MMG5_pMesh to GEO::Mesh ...");
			/* Notes:
			 * - indexing seems to start at 1 in MMG */

			assert(mmg->dim == 3);
			M.clear();
			M.vertices.create_vertices((uint)mmg->np);
			M.edges.create_edges((uint)mmg->na);
			M.facets.create_triangles((uint)mmg->nt);
			M.cells.create_tets((uint)mmg->ne);

			for (uint v = 0; v < M.vertices.nb(); ++v)
			{
				for (uint d = 0; d < (uint)mmg->dim; ++d)
				{
					M.vertices.point_ptr(v)[d] = mmg->point[v + 1].c[d];
				}
			}
			for (uint e = 0; e < M.edges.nb(); ++e)
			{
				M.edges.set_vertex(e, 0, (uint)mmg->edge[e + 1].a - 1);
				M.edges.set_vertex(e, 1, (uint)mmg->edge[e + 1].b - 1);
			}
			for (uint t = 0; t < M.facets.nb(); ++t)
			{
				M.facets.set_vertex(t, 0, (uint)mmg->tria[t + 1].v[0] - 1);
				M.facets.set_vertex(t, 1, (uint)mmg->tria[t + 1].v[1] - 1);
				M.facets.set_vertex(t, 2, (uint)mmg->tria[t + 1].v[2] - 1);
			}
			for (uint c = 0; c < M.cells.nb(); ++c)
			{
				M.cells.set_vertex(c, 0, (uint)mmg->tetra[c + 1].v[0] - 1);
				M.cells.set_vertex(c, 1, (uint)mmg->tetra[c + 1].v[1] - 1);
				M.cells.set_vertex(c, 2, (uint)mmg->tetra[c + 1].v[2] - 1);
				M.cells.set_vertex(c, 3, (uint)mmg->tetra[c + 1].v[3] - 1);
			}
			M.facets.connect();
			M.cells.connect();

			return true;
		}

		bool geo_to_mmg(const GEO::Mesh &M, MMG5_pMesh &mmg, MMG5_pSol &sol, bool volume_mesh = true)
		{
			logger().trace("converting GEO::M to MMG5_pMesh ...");
			assert(M.vertices.dimension() == 3);
			if (M.facets.nb() > 0)
				assert(M.facets.are_simplices());
			if (M.cells.nb() > 0)
				assert(M.cells.are_simplices());

			if (volume_mesh)
			{
				MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
			}
			else
			{
				MMGS_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
			}

			if (volume_mesh && MMG3D_Set_meshSize(mmg, (int)M.vertices.nb(), (int)M.cells.nb(), 0, /* nb prisms */
												  (int)M.facets.nb(), 0,                           /* nb quad */
												  (int)M.edges.nb()                                /* nb edges */
												  )
								   != 1)
			{
				logger().error("failed to MMG3D_Set_meshSize");
				return false;
			}
			else if (!volume_mesh && MMGS_Set_meshSize(mmg, (int)M.vertices.nb(), (int)M.facets.nb(), (int)M.edges.nb() /* nb edges */
													   )
										 != 1)
			{
				logger().error("failed to MMGS_Set_meshSize");
				return false;
			}

			for (uint v = 0; v < (uint)mmg->np; ++v)
			{
				for (uint d = 0; d < M.vertices.dimension(); ++d)
				{
					mmg->point[v + 1].c[d] = M.vertices.point_ptr(v)[d];
				}
			}
			for (uint e = 0; e < (uint)mmg->na; ++e)
			{
				mmg->edge[e + 1].a = (int)M.edges.vertex(e, 0) + 1;
				mmg->edge[e + 1].b = (int)M.edges.vertex(e, 1) + 1;
			}
			for (uint t = 0; t < (uint)mmg->nt; ++t)
			{
				mmg->tria[t + 1].v[0] = (int)M.facets.vertex(t, 0) + 1;
				mmg->tria[t + 1].v[1] = (int)M.facets.vertex(t, 1) + 1;
				mmg->tria[t + 1].v[2] = (int)M.facets.vertex(t, 2) + 1;
			}
			if (volume_mesh)
			{
				for (uint c = 0; c < (uint)mmg->ne; ++c)
				{
					mmg->tetra[c + 1].v[0] = (int)M.cells.vertex(c, 0) + 1;
					mmg->tetra[c + 1].v[1] = (int)M.cells.vertex(c, 1) + 1;
					mmg->tetra[c + 1].v[2] = (int)M.cells.vertex(c, 2) + 1;
					mmg->tetra[c + 1].v[3] = (int)M.cells.vertex(c, 3) + 1;
				}
			}

			if (volume_mesh && MMG3D_Set_solSize(mmg, sol, MMG5_Vertex, (int)M.vertices.nb(), MMG5_Scalar) != 1)
			{
				logger().error("failed to MMG3D_Set_solSize");
				return false;
			}
			else if (!volume_mesh && MMGS_Set_solSize(mmg, sol, MMG5_Vertex, (int)M.vertices.nb(), MMG5_Scalar) != 1)
			{
				logger().error("failed to MMGS_Set_solSize");
				return false;
			}
			for (uint v = 0; v < M.vertices.nb(); ++v)
			{
				sol->m[v + 1] = 1.;
			}
			if (volume_mesh && MMG3D_Chk_meshData(mmg, sol) != 1)
			{
				logger().error("error in mmg: inconsistent mesh and sol");
				return false;
			}
			else if (!volume_mesh && MMGS_Chk_meshData(mmg, sol) != 1)
			{
				logger().error("error in mmg: inconsistent mesh and sol");
				return false;
			}

			if (volume_mesh)
			{
				MMG3D_Set_handGivenMesh(mmg); /* because we don't use the API functions */
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

		void mmgs_free(MMG5_pMesh mmg, MMG5_pSol sol)
		{
			MMGS_Free_all(MMG5_ARG_start,
						  MMG5_ARG_ppMesh, &mmg, MMG5_ARG_ppMet, &sol, MMG5_ARG_end);
		}

		bool mmg_wrapper_test_geo2mmg2geo(const GEO::Mesh &M_in, GEO::Mesh &M_out)
		{
			MMG5_pMesh mmg = nullptr;
			MMG5_pSol sol = nullptr;
			bool ok = geo_to_mmg(M_in, mmg, sol);
			if (!ok)
				return false;
			ok = mmg_to_geo(mmg, M_out);
			mmg3d_free(mmg, sol);
			return ok;
		}

		bool mmg2d_tri_remesh(const GEO::Mesh &M, GEO::Mesh &M_out, const MmgOptions &opt)
		{
			MMG5_pMesh mesh = nullptr;
			MMG5_pSol met = nullptr;
			bool ok = geo_to_mmg(M, mesh, met, false);
			if (!ok)
			{
				logger().error("mmg2d_remesh: failed to convert mesh to MMG5_pMesh");
				mmg2d_free(mesh, met);
				return false;
			}

			/* Set remeshing options */
			MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_angleDetection, opt.angle_value);
			if (opt.enable_anisotropy)
			{
				MMG2D_Set_solSize(mesh, met, MMG5_Vertex, 0, MMG5_Tensor);
			}
			if (opt.hsiz == 0. || opt.metric_attribute != "no_metric")
			{
				MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hmin, opt.hmin);
				MMG2D_Set_dparameter(mesh, met, MMG2D_DPARAM_hmax, opt.hmax);
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
			if (opt.metric_attribute != "no_metric")
			{
				if (!M.vertices.attributes().is_defined(opt.metric_attribute))
				{
					logger().error("mmg2D_remesh: {} is not a vertex attribute, cancel", opt.metric_attribute);
					return false;
				}
				GEO::Attribute<double> h_local(M.vertices.attributes(), opt.metric_attribute);
				for (uint v = 0; v < M.vertices.nb(); ++v)
				{
					met->m[v + 1] = h_local[v];
				}
			}

			int ier = MMG2D_mmg2dlib(mesh, met);
			if (ier != MMG5_SUCCESS)
			{
				logger().error("mmg2d_remesh: failed to remesh");
				mmg2d_free(mesh, met);
				return false;
			}

			ok = mmg_to_geo(mesh, M_out);

			mmg2d_free(mesh, met);
			return ok;
		}

		bool mmgs_tri_remesh(const GEO::Mesh &M, GEO::Mesh &M_out, const MmgOptions &opt)
		{
			MMG5_pMesh mesh = nullptr;
			MMG5_pSol met = nullptr;
			bool ok = geo_to_mmg(M, mesh, met, false);
			if (!ok)
			{
				logger().error("mmgs_remesh: failed to convert mesh to MMG5_pMesh");
				mmgs_free(mesh, met);
				return false;
			}

			/* Set remeshing options */
			MMGS_Set_dparameter(mesh, met, MMGS_DPARAM_angleDetection, opt.angle_value);
			if (opt.enable_anisotropy)
			{
				MMGS_Set_solSize(mesh, met, MMG5_Vertex, 0, MMG5_Tensor);
			}
			if (opt.hsiz == 0. || opt.metric_attribute != "no_metric")
			{
				MMGS_Set_dparameter(mesh, met, MMGS_DPARAM_hmin, opt.hmin);
				MMGS_Set_dparameter(mesh, met, MMGS_DPARAM_hmax, opt.hmax);
			}
			else
			{
				met->np = 0;
				MMGS_Set_dparameter(mesh, met, MMGS_DPARAM_hsiz, opt.hsiz);
			}
			MMGS_Set_dparameter(mesh, met, MMGS_DPARAM_hausd, opt.hausd);
			MMGS_Set_dparameter(mesh, met, MMGS_DPARAM_hgrad, opt.hgrad);
			MMGS_Set_iparameter(mesh, met, MMGS_IPARAM_angle, int(opt.angle_detection));
			MMGS_Set_iparameter(mesh, met, MMGS_IPARAM_noswap, int(opt.noswap));
			MMGS_Set_iparameter(mesh, met, MMGS_IPARAM_noinsert, int(opt.noinsert));
			MMGS_Set_iparameter(mesh, met, MMGS_IPARAM_nomove, int(opt.nomove));
			if (opt.metric_attribute != "no_metric")
			{
				if (!M.vertices.attributes().is_defined(opt.metric_attribute))
				{
					logger().error("mmgs_remesh: {} is not a vertex attribute, cancel", opt.metric_attribute);
					return false;
				}
				GEO::Attribute<double> h_local(M.vertices.attributes(), opt.metric_attribute);
				for (uint v = 0; v < M.vertices.nb(); ++v)
				{
					met->m[v + 1] = h_local[v];
				}
			}

			int ier = MMGS_mmgslib(mesh, met);
			if (ier != MMG5_SUCCESS)
			{
				logger().error("mmgs_remesh: failed to remesh");
				mmgs_free(mesh, met);
				return false;
			}

			ok = mmg_to_geo(mesh, M_out);

			mmgs_free(mesh, met);
			return ok;
		}

		bool mmg3d_tet_remesh(const GEO::Mesh &M, GEO::Mesh &M_out, const MmgOptions &opt)
		{
			MMG5_pMesh mesh = nullptr;
			MMG5_pSol met = nullptr;
			bool ok = geo_to_mmg(M, mesh, met, true);
			if (!ok)
			{
				logger().error("mmg3d_remesh: failed to convert mesh to MMG5_pMesh");
				mmg3d_free(mesh, met);
				return false;
			}

			/* Set remeshing options */
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_angleDetection, opt.angle_value);
			if (opt.enable_anisotropy)
			{
				MMG3D_Set_solSize(mesh, met, MMG5_Vertex, 0, MMG5_Tensor);
			}
			if (opt.hsiz == 0. || opt.metric_attribute != "no_metric")
			{
				MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmin, opt.hmin);
				MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmax, opt.hmax);
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
			if (opt.metric_attribute != "no_metric")
			{
				if (!M.vertices.attributes().is_defined(opt.metric_attribute))
				{
					logger().error("mmg3D_remesh: {} is not a vertex attribute, cancel", opt.metric_attribute);
					return false;
				}
				GEO::Attribute<double> h_local(M.vertices.attributes(), opt.metric_attribute);
				for (uint v = 0; v < M.vertices.nb(); ++v)
				{
					met->m[v + 1] = h_local[v];
				}
			}

			int ier = MMG3D_mmg3dlib(mesh, met);
			if (ier != MMG5_SUCCESS)
			{
				logger().error("mmg3d_remesh: failed to remesh");
				mmg3d_free(mesh, met);
				return false;
			}

			ok = mmg_to_geo(mesh, M_out);

			mmg3d_free(mesh, met);
			return ok;
		}

		bool mmg3d_extract_iso(const GEO::Mesh &M, GEO::Mesh &M_out, const MmgOptions &opt)
		{
			if (!opt.level_set || opt.ls_attribute == "no_ls" || !M.vertices.attributes().is_defined(opt.ls_attribute))
			{
				logger().error("mmg3D_iso: {} is not a vertex attribute, cancel", opt.ls_attribute);
				return false;
			}
			if (opt.angle_detection)
			{
				logger().warn("mmg3D_iso: angle_detection shoud probably be disabled because level set functions are smooth");
			}

			MMG5_pMesh mesh = nullptr;
			MMG5_pSol met = nullptr;
			bool ok = geo_to_mmg(M, mesh, met, true);
			if (!ok)
			{
				logger().error("mmg3d_remesh: failed to convert mesh to MMG5_pMesh");
				mmg3d_free(mesh, met);
				return false;
			}
			GEO::Attribute<double> ls(M.vertices.attributes(), opt.ls_attribute);
			for (uint v = 0; v < M.vertices.nb(); ++v)
			{
				met->m[v + 1] = ls[v];
			}

			/* Flag border for future deletion */
			// std::vector<bool> on_border(M.vertices.nb(), false);
			// for (index_t t = 0; t < M.cells.nb(); ++t) {
			//     for (index_t lf = 0; lf < M.cells.nb_facets(t); ++lf) {
			//         if (M.cells.adjacent(t,lf) != GEO::NO_CELL) continue;
			//         for (index_t lv = 0; lv < M.cells.facet_nb_vertices(t,lf); ++lv) {
			//             on_border[M.cells.facet_vertex(t,lf,lv)] = true;
			//         }
			//     }
			// }

			/* Set remeshing options */
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_iso, 1);
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_ls, opt.ls_value);
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_angleDetection, opt.angle_value);
			if (opt.hsiz == 0.)
			{
				MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmin, opt.hmin);
				MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmax, opt.hmax);
			}
			else
			{
				logger().error("mmg3d_iso: should not use hsiz parameter for level set mode");
				mmg3d_free(mesh, met);
				return false;
			}
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hausd, opt.hausd);
			MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hgrad, opt.hgrad);
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_angle, int(opt.angle_detection));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_noswap, int(opt.noswap));
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_noinsert, 1);
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_nomove, 1);
			MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_nosurf, 1);

			// TODO: Check this is correct
			// Used to be
			// int ier = MMG3D_mmg3dls(mesh, met);
			int ier = MMG3D_mmg3dls(mesh, met, nullptr);
			if (ier != MMG5_SUCCESS)
			{
				logger().error("mmg3d_iso: failed to remesh isovalue");
				mmg3d_free(mesh, met);
				return false;
			}

			/* Convert back */
			ok = mmg_to_geo(mesh, M_out);
			GEO::Attribute<double> ls_out(M_out.vertices.attributes(), opt.ls_attribute);
			for (uint v = 0; v < M_out.vertices.nb(); ++v)
			{
				ls_out[v] = met->m[v + 1];
			}
			/* Extract only the border */
			// M_out.cells.clear(false,false);
			// M_out.vertices.remove_isolated();
			// GEO::vector<index_t> to_del(M_out.facets.nb(), 0);
			// for (index_t f = 0; f < M_out.facets.nb(); ++f) {
			//     double d = 0;
			//     bool f_on_border = true;
			//     for (index_t lv = 0; lv < M_out.facets.nb_vertices(f); ++lv) {
			//         d = geo_max(d,std::abs(ls_out[M_out.facets.vertex(f,0)] - opt.ls_value));
			//         if (M_out.facets.vertex(f,lv) < M.vertices.nb()) {
			//             if (!on_border[M_out.facets.vertex(f,lv)]) f_on_border = false;
			//         } else {
			//             f_on_border = false;
			//         }
			//     }
			//     // if (d > 1.1 * opt.hmin) {
			//     //     to_del[f] = 1;
			//     // }
			//     if (f_on_border) to_del[f] = 1;
			// }
			// M_out.facets.delete_elements(to_del, true);

			mmg3d_free(mesh, met);
			return ok;
		}

	} // anonymous namespace

	////////////////////////////////////////////////////////////////////////////////

	void remesh_adaptive_2d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXd &S,
							Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, MmgOptions opt)
	{
		assert(V.cols() == 2 || V.cols() == 3);
		assert(V.rows() == S.size());
		GEO::Mesh M, M_out;
		to_geogram_mesh(V, F, M);

		// Remeshing options
		opt.metric_attribute = "scalar";

		GEO::Attribute<double> scalar(M.vertices.attributes(), opt.metric_attribute);
		for (int v = 0; v < M.vertices.nb(); ++v)
		{
			scalar[v] = S(v);
		}

		// Remesh surface
		mmg2d_tri_remesh(M, M_out, opt);

		// Convert output
		Eigen::MatrixXi OT;
		from_geogram_mesh(M_out, OV, OF, OT);

		// Promised a 2D mesh, so drop z-coordinate
		OV.conservativeResize(OV.rows(), 2);
	}

	void remesh_adaptive_3d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, const Eigen::VectorXd &S,
							Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OT, MmgOptions opt)
	{
		assert(V.cols() == 3);
		assert(V.rows() == S.size());
		GEO::Mesh M, M_out;
		to_geogram_mesh(V, Eigen::MatrixXi(0, 3), T, M);

		// Remeshing options
		opt.metric_attribute = "scalar";

		GEO::Attribute<double> scalar(M.vertices.attributes(), opt.metric_attribute);
		for (int v = 0; v < M.vertices.nb(); ++v)
		{
			scalar[v] = S(v);
		}

		// Remesh volume
		mmg3d_tet_remesh(M, M_out, opt);

		// Convert output
		from_geogram_mesh(M_out, OV, OF, OT);
	}

} // namespace polyfem::mesh

#endif