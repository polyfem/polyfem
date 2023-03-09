////////////////////////////////////////////////////////////////////////////////
#include "PolygonalBasis3d.hpp"
#include "LagrangeBasis3d.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/quadrature/PolyhedronQuadrature.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh2D/Refinement.hpp>
#include <polyfem/utils/RefElementSampler.hpp>
#include "function/RBFWithLinear.hpp"
#include "function/RBFWithQuadratic.hpp"
#include "function/RBFWithQuadraticLagrange.hpp"
#include <polyfem/utils/Logger.hpp>

#include <polyfem/autogen/auto_q_bases.hpp>

#include <igl/per_vertex_normals.h>
#include <random>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace quadrature;
	using namespace utils;

	namespace basis
	{
		namespace
		{

			const int max_num_kernels = 300;

			// -----------------------------------------------------------------------------

			std::vector<int> compute_nonzero_bases_ids(const Mesh3D &mesh, const int c,
													   const std::vector<ElementBases> &bases,
													   const std::map<int, InterfaceData> &poly_face_to_data)
			{
				std::vector<int> local_to_global;

				for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
				{
					auto index = mesh.get_index_from_element(c, lf, 0);
					const int c2 = mesh.switch_element(index).element;
					assert(c2 >= 0); // no boundary polytope
					assert(poly_face_to_data.count(index.face) > 0);
					const InterfaceData &bdata = poly_face_to_data.at(index.face);
					const ElementBases &b = bases[c2];
					for (int other_local_basis_id : bdata.local_indices)
					{
						for (const auto &x : b.bases[other_local_basis_id].global())
						{
							const int global_node_id = x.index;
							local_to_global.push_back(global_node_id);
						}
					}
				}

				std::sort(local_to_global.begin(), local_to_global.end());
				auto it = std::unique(local_to_global.begin(), local_to_global.end());
				local_to_global.resize(std::distance(local_to_global.begin(), it));

				return local_to_global;
			}

			// -----------------------------------------------------------------------------

			// Canonical triangle mesh in parametric domain
			void compute_canonical_pattern(int n_samples_per_edge, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
			{
				regular_2d_grid(n_samples_per_edge, false, V, F);

				// igl::opengl::glfw::Viewer viewer;
				// viewer.data().set_mesh(V, F);
				// viewer.launch();
			}

			// -----------------------------------------------------------------------------

			// Needs to be consistent between `evalFunc` and `compute_quad_mesh_from_cell`
			constexpr int lv0 = 3;

			// Assemble the surface quad mesh (V, F) corresponding to the polyhedron c
			GetAdjacentLocalEdge compute_quad_mesh_from_cell(
				const Mesh3D &mesh, int c, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
			{
				std::vector<std::array<int, 4>> quads(mesh.n_cell_faces(c));
				typedef std::tuple<int, int, bool> QuadLocalEdge;
				std::vector<std::array<QuadLocalEdge, 4>> adj(mesh.n_cell_faces(c));
				int num_vertices = 0;
				std::map<int, int> vertex_g2l;
				std::map<int, int> face_g2l;
				for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
				{
					face_g2l.emplace(mesh.get_index_from_element(c, lf, lv0).face, lf);
				}
				for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
				{
					auto index = mesh.get_index_from_element(c, lf, lv0);
					assert(mesh.n_face_vertices(index.face) == 4);
					for (int lv = 0; lv < 4; ++lv)
					{
						if (!vertex_g2l.count(index.vertex))
						{
							vertex_g2l.emplace(index.vertex, num_vertices++);
						}
						quads[lf][lv] = vertex_g2l.at(index.vertex);

						// Set adjacency info
						auto index2 = mesh.switch_face(index);
						int lf2 = face_g2l.at(index2.face);
						std::get<0>(adj[lf][lv]) = lf2;
						auto index3 = mesh.get_index_from_element(c, lf2, lv0);
						for (int lv2 = 0; lv2 < 4; ++lv2)
						{
							if (index3.edge == index2.edge)
							{
								std::get<1>(adj[lf][lv]) = lv2;
								if (index2.vertex != index3.vertex)
								{
									assert(mesh.switch_vertex(index3).vertex == index2.vertex);
									std::get<2>(adj[lf][lv]) = true;
								}
								else
								{
									std::get<2>(adj[lf][lv]) = false;
								}
							}
							index3 = mesh.next_around_face(index3);
						}

						index = mesh.next_around_face(index);
					}
				}
				V.resize(num_vertices, 3);
				for (const auto &kv : vertex_g2l)
				{
					V.row(kv.second) = mesh.point(kv.first);
				}
				F.resize(quads.size(), 4);
				int f = 0;
				for (auto q : quads)
				{
					F.row(f++) << q[0], q[1], q[2], q[3];
				}

				return [adj](int q, int lv) {
					return adj[q][lv];
				};
			}

			// -----------------------------------------------------------------------------

			void compute_offset_kernels(const Eigen::MatrixXd &QV, const Eigen::MatrixXi &QF,
										int n_kernels_per_edge, double eps, Eigen::MatrixXd &kernel_centers,
										Eigen::MatrixXd &KV, Eigen::MatrixXi &KF,
										EvalParametersFunc evalFuncGeom, GetAdjacentLocalEdge getAdjLocalEdge)
			{
				Eigen::MatrixXd PV, KN;
				Eigen::MatrixXi PF;
				Eigen::VectorXd D;
				compute_canonical_pattern(n_kernels_per_edge, PV, PF);
				instantiate_pattern(QV, QF, PV, PF, KV, KF, nullptr, evalFuncGeom, getAdjLocalEdge);
				orient_closed_surface(KV, KF);
				double volume = std::pow(signed_volume(KV, KF), 1.0 / 3.0);

				if (true || KV.rows() < max_num_kernels)
				{
					igl::per_vertex_normals(KV, KF, KN);
					kernel_centers = KV;
				}
				else
				{
					// std::cout << "fancy sampling" << std::endl;
					polyfem::sample_surface(KV, KF, max_num_kernels, kernel_centers, &KN, 10, 10);
					// std::cout << "size: "<< kernel_centers.size() << std::endl;
				}
				// std::cout << "eps: " << eps << std::endl;
				// std::cout << eps * volume << std::endl;
				kernel_centers += eps * volume * KN;

				// std::default_random_engine gen;
				// std::uniform_real_distribution<double> dist(-1.0, 1.0);
				// for (int v = 0; v < kernel_centers.rows(); ++v) {
				// 	kernel_centers.row(v) = KV.row(v) + dist(gen) * KN.row(v);
				// }
				assert(kernel_centers.cols() == 3);
				signed_squared_distances(KV, KF, kernel_centers, D);
				std::vector<Eigen::RowVector3d> remap;
				std::vector<Eigen::RowVector3d> rejected;
				for (int v = 0; v < kernel_centers.rows(); ++v)
				{
					if (std::sqrt(D(v)) > 0.8 * eps * volume)
					{
						remap.push_back(kernel_centers.row(v));
					}
					else
					{
						// rejected.push_back(kernel_centers.row(v));
					}
				}
				kernel_centers.resize(remap.size(), 3);
				for (int v = 0; v < kernel_centers.rows(); ++v)
				{
					kernel_centers.row(v) = remap[v];
				}
				// Eigen::MatrixXd rej(rejected.size(), 3);
				// for (int v = 0; v < (int) rejected.size(); ++v) {
				// 	rej.row(v) = rejected[v];
				// }
				// igl::write_triangle_mesh("foo_medium.obj", KV, KF);
				// std::cout << "nkernels: " << KV.rows() << std::endl;
				// igl::write_triangle_mesh("foo.obj", KV, KF);

				// igl::opengl::glfw::Viewer viewer;
				// viewer.data().set_mesh(KV, KF);
				// viewer.data().add_points(kernel_centers, Eigen::RowVector3d(0,1,1));
				// viewer.data().add_points(rej, Eigen::RowVector3d(1,0,0));
				// viewer.launch();
			}

			// -----------------------------------------------------------------------------

			/// @brief Compute boundary sample points + centers of harmonic bases for the polygonal element
			void sample_polyhedra(
				const int element_index,
				const int n_quadrature_vertices_per_edge,
				const int n_kernels_per_edge,
				int n_samples_per_edge,
				const int quadrature_order,
				const int mass_quadrature_order,
				const Mesh3D &mesh,
				const std::map<int, InterfaceData> &poly_face_to_data,
				const std::vector<ElementBases> &bases,
				const std::vector<ElementBases> &gbases,
				const double eps,
				std::vector<int> &local_to_global,
				Eigen::MatrixXd &collocation_points,
				Eigen::MatrixXd &kernel_centers,
				Eigen::MatrixXd &rhs,
				Eigen::MatrixXd &triangulated_vertices,
				Eigen::MatrixXi &triangulated_faces,
				Quadrature &quadrature,
				Quadrature &mass_quadrature,
				double &scaling,
				Eigen::RowVector3d &translation)
			{
				// Local ids of nonzero bases over the polygon
				local_to_global = compute_nonzero_bases_ids(mesh, element_index, bases, poly_face_to_data);

				// Compute the image of the canonical pattern vertices through the geometric mapping
				// of the given local face
				auto evalFunc = [&](const Eigen::MatrixXd &uv, Eigen::MatrixXd &mapped, int lf) {
					const auto &u = uv.col(0).array();
					const auto &v = uv.col(1).array();
					auto index = mesh.get_index_from_element(element_index, lf, lv0);
					index = mesh.switch_element(index);
					// Eigen::MatrixXd abcd = LagrangeBasis3d::linear_hex_face_local_nodes_coordinates(mesh, index);
					const auto indices = LagrangeBasis3d::hex_face_local_nodes(false, 1, mesh, index);
					assert(indices.size() == 4);
					Eigen::MatrixXd abcd;
					polyfem::autogen::q_nodes_3d(1, abcd);
					Eigen::RowVector3d a = abcd.row(indices(0));
					Eigen::RowVector3d b = abcd.row(indices(1));
					Eigen::RowVector3d c = abcd.row(indices(2));
					Eigen::RowVector3d d = abcd.row(indices(3));
					mapped = ((1 - u) * (1 - v)).matrix() * a + (u * (1 - v)).matrix() * b + (u * v).matrix() * c + ((1 - u) * v).matrix() * d;
					mapped = mapped.array().max(0.0).min(1.0);
					assert(mapped.maxCoeff() >= 0.0);
					assert(mapped.maxCoeff() <= 1.0);
				};
				auto evalFuncGeom = [&](const Eigen::MatrixXd &uv, Eigen::MatrixXd &mapped, int lf) {
					Eigen::MatrixXd samples;
					evalFunc(uv, samples, lf);
					auto index = mesh.get_index_from_element(element_index, lf, lv0);
					index = mesh.switch_element(index);
					const ElementBases &gb = gbases[index.element];
					gb.eval_geom_mapping(samples, mapped);
				};

				Eigen::MatrixXd QV, KV;
				Eigen::MatrixXi QF, KF;
				auto getAdjLocalEdge = compute_quad_mesh_from_cell(mesh, element_index, QV, QF);

				// Compute kernel centers
				compute_offset_kernels(QV, QF, n_kernels_per_edge, eps, kernel_centers, KV, KF,
									   evalFuncGeom, getAdjLocalEdge);
				// if (KV.rows() >= max_num_kernels) { n_samples_per_edge = 5; }

				// Compute collocation points
				Eigen::MatrixXd PV, UV;
				Eigen::MatrixXi PF, CF, UF;
				Eigen::VectorXi uv_sources, uv_ranges;
				compute_canonical_pattern(n_samples_per_edge, PV, PF);
				instantiate_pattern(QV, QF, PV, PF, UV, UF, &uv_sources, evalFunc, getAdjLocalEdge);
				orient_closed_surface(UV, UF);
				instantiate_pattern(QV, QF, PV, PF, collocation_points, CF, nullptr, evalFuncGeom, getAdjLocalEdge);
				orient_closed_surface(collocation_points, CF);
				reorder_mesh(collocation_points, CF, uv_sources, uv_ranges);
				reorder_mesh(UV, UF, uv_sources, uv_ranges);
				assert(uv_ranges.size() == mesh.n_cell_faces(element_index) + 1);

				// Compute coarse surface surface for visualization
				compute_canonical_pattern(n_quadrature_vertices_per_edge, PV, PF);
				instantiate_pattern(QV, QF, PV, PF, triangulated_vertices, triangulated_faces,
									nullptr, evalFuncGeom, getAdjLocalEdge);
				orient_closed_surface(triangulated_vertices, triangulated_faces);

				// for (int f = 0; f < KF.rows(); ++f) {
				// 	triangulated_faces.row(f) = KF.row(f).reverse();
				// }

				// {
				// 	Eigen::MatrixXd V;
				// 	evalFuncGeom(PV, V, 0);
				// igl::write_triangle_mesh("foo_dense.obj", collocation_points, CF);
				// igl::write_triangle_mesh("foo_small.obj", triangulated_vertices, triangulated_faces);
				// igl::opengl::glfw::Viewer viewer;
				// viewer.data().set_points(kernel_centers, Eigen::RowVector3d(1,0,1));
				// viewer.data().set_mesh(KV, KF);
				// viewer.launch();
				// }

				// igl::opengl::glfw::Viewer viewer;
				// viewer.data().set_mesh(collocation_points, CF);
				// viewer.data().add_points(kernel_centers, Eigen::RowVector3d(0,1,1));
				// for (int lf = 0; lf < mesh.n_cell_faces(element_index); ++lf) {
				// 	Eigen::MatrixXd samples;
				// 	samples = UV.middleRows(uv_ranges(lf), uv_ranges(lf+1) - uv_ranges(lf));
				// 	Eigen::RowVector3d c = Eigen::RowVector3d::Random();
				// 	viewer.data().add_points(samples, c);
				// }
				// viewer.launch();

				// Compute right-hand side constraints for setting the harmonic kernels
				Eigen::MatrixXd samples;
				std::vector<AssemblyValues> basis_val;
				rhs.resize(UV.rows(), local_to_global.size());
				rhs.setZero();
				for (int lf = 0; lf < mesh.n_cell_faces(element_index); ++lf)
				{
					auto index = mesh.get_index_from_element(element_index, lf, 0);
					const int c2 = mesh.switch_element(index).element;
					assert(c2 >= 0); // no boundary polytope

					const InterfaceData &bdata = poly_face_to_data.at(index.face);
					const ElementBases &b = bases[c2];

					samples = UV.middleRows(uv_ranges(lf), uv_ranges(lf + 1) - uv_ranges(lf));
					b.evaluate_bases(samples, basis_val);

					// Evaluate field basis and set up the rhs
					for (int other_local_basis_id : bdata.local_indices)
					{
						// b.bases[other_local_basis_id].basis(samples, basis_val);

						for (const auto &x : b.bases[other_local_basis_id].global())
						{
							const int global_node_id = x.index;
							const double weight = x.val;

							const int poly_local_basis_id = std::distance(local_to_global.begin(),
																		  std::find(local_to_global.begin(), local_to_global.end(), global_node_id));
							rhs.block(uv_ranges(lf), poly_local_basis_id, basis_val[other_local_basis_id].val.size(), 1) += basis_val[other_local_basis_id].val * weight;
						}
					}
				}

				// Compute quadrature points + normalize kernels and collocation points
				Eigen::MatrixXd NV = triangulated_vertices;
				// scaling = (NV.colwise().maxCoeff() - NV.colwise().minCoeff()).maxCoeff();
				// translation = NV.colwise().minCoeff();
				scaling = 1.0;
				translation.setZero();
				// NV = (NV.rowwise() - translation) / scaling;
				PolyhedronQuadrature::get_quadrature(NV, triangulated_faces, mesh.kernel(element_index),
													 quadrature_order, quadrature);
				PolyhedronQuadrature::get_quadrature(NV, triangulated_faces, mesh.kernel(element_index),
													 mass_quadrature_order, mass_quadrature);

				// Normalization
				// collocation_points = (collocation_points.rowwise() - translation) / scaling;
				// kernel_centers = (kernel_centers.rowwise() - translation) / scaling;
				// KV = (KV.rowwise() - translation) / scaling;

				triangulated_vertices = KV;
				triangulated_faces = KF;

				// std::cout << "volume: " << signed_volume(KV, KF) << std::endl;
			}

		} // anonymous namespace

		////////////////////////////////////////////////////////////////////////////////

		// Compute the integral constraints for each basis of the mesh
		void PolygonalBasis3d::compute_integral_constraints(
			const Assembler &assembler,
			const Mesh3D &mesh,
			const int n_bases,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			Eigen::MatrixXd &basis_integrals)
		{
			assert(mesh.is_volume());

			basis_integrals.resize(n_bases, 9);
			basis_integrals.setZero();
			Eigen::MatrixXd rhs(n_bases, 9);
			rhs.setZero();

			const int n_elements = mesh.n_elements();
			for (int e = 0; e < n_elements; ++e)
			{
				if (mesh.is_polytope(e))
				{
					continue;
				}
				// ElementAssemblyValues vals = values[e];
				// const ElementAssemblyValues &gvals = gvalues[e];
				ElementAssemblyValues vals;
				vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

				// Computes the discretized integral of the PDE over the element
				const int n_local_bases = int(vals.basis_values.size());
				for (int j = 0; j < n_local_bases; ++j)
				{
					const AssemblyValues &v = vals.basis_values[j];
					const double integral_100 = (v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
					const double integral_010 = (v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
					const double integral_001 = (v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

					const double integral_110 = ((vals.val.col(1).array() * v.grad_t_m.col(0).array() + vals.val.col(0).array() * v.grad_t_m.col(1).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
					const double integral_011 = ((vals.val.col(2).array() * v.grad_t_m.col(1).array() + vals.val.col(1).array() * v.grad_t_m.col(2).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
					const double integral_101 = ((vals.val.col(0).array() * v.grad_t_m.col(2).array() + vals.val.col(2).array() * v.grad_t_m.col(0).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();

					const double integral_200 = 2 * (vals.val.col(0).array() * v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
					const double integral_020 = 2 * (vals.val.col(1).array() * v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
					const double integral_002 = 2 * (vals.val.col(2).array() * v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

					const double area = (v.val.array() * vals.det.array() * vals.quadrature.weights.array()).sum();

					for (size_t ii = 0; ii < v.global.size(); ++ii)
					{
						basis_integrals(v.global[ii].index, 0) += integral_100 * v.global[ii].val;
						basis_integrals(v.global[ii].index, 1) += integral_010 * v.global[ii].val;
						basis_integrals(v.global[ii].index, 2) += integral_001 * v.global[ii].val;

						basis_integrals(v.global[ii].index, 3) += integral_110 * v.global[ii].val;
						basis_integrals(v.global[ii].index, 4) += integral_011 * v.global[ii].val;
						basis_integrals(v.global[ii].index, 5) += integral_101 * v.global[ii].val;

						basis_integrals(v.global[ii].index, 6) += integral_200 * v.global[ii].val;
						basis_integrals(v.global[ii].index, 7) += integral_020 * v.global[ii].val;
						basis_integrals(v.global[ii].index, 8) += integral_002 * v.global[ii].val;

						rhs(v.global[ii].index, 6) += -2.0 * area * v.global[ii].val;
						rhs(v.global[ii].index, 7) += -2.0 * area * v.global[ii].val;
						rhs(v.global[ii].index, 8) += -2.0 * area * v.global[ii].val;
					}
				}
			}

			basis_integrals -= rhs;
		}

		// -----------------------------------------------------------------------------

		// Distance from harmonic kernels to polygon boundary
		double compute_epsilon(const Mesh3D &mesh, int e)
		{
			// double area = 0;
			// const int n_edges = mesh.n_element_vertices(e);
			// for (int i = 0; i < n_edges; ++i) {
			// 	const int ip = (i + 1) % n_edges;

			// 	Eigen::RowVector2d p0 = mesh.point(mesh.vertex_global_index(e, i));
			// 	Eigen::RowVector2d p1 = mesh.point(mesh.vertex_global_index(e, ip));

			// 	Eigen::Matrix2d det_mat;
			// 	det_mat.row(0) = p0;
			// 	det_mat.row(1) = p1;

			// 	area += det_mat.determinant();
			// }
			// area = std::fabs(area);
			// // const double eps = use_harmonic ? (0.08*area) : 0;
			// const double eps = 0.08*area;

			return 0.1; // will be relative to the volume of the poly
		}

		// -----------------------------------------------------------------------------

		// namespace {

		// void add_spheres(igl::opengl::glfw::Viewer &viewer0, const Eigen::MatrixXd &P, double radius) {
		// 	Eigen::MatrixXd V = viewer0.data().V, VS, VN;
		// 	Eigen::MatrixXi F = viewer0.data().F, FS;
		// 	igl::read_triangle_mesh(POLYFEM_MESH_PATH "sphere.ply", VS, FS);

		// 	Eigen::RowVector3d minV = VS.colwise().minCoeff();
		// 	Eigen::RowVector3d maxV = VS.colwise().maxCoeff();
		// 	VS.rowwise() -= minV + 0.5 * (maxV - minV);
		// 	VS /= (maxV - minV).maxCoeff();
		// 	VS *= 2.0 * radius;
		// 	// std::cout << V.colwise().minCoeff() << std::endl;
		// 	// std::cout << V.colwise().maxCoeff() << std::endl;

		// 	Eigen::MatrixXd C = viewer0.data().F_material_ambient.leftCols(3);
		// 	C *= 10;

		// 	int nv = V.rows();
		// 	int nf = F.rows();
		// 	V.conservativeResize(V.rows() + P.rows() * VS.rows(), V.cols());
		// 	F.conservativeResize(F.rows() + P.rows() * FS.rows(), F.cols());
		// 	C.conservativeResize(C.rows() + P.rows() * FS.rows(), C.cols());
		// 	for (int i = 0; i < P.rows(); ++i) {
		// 		V.middleRows(nv, VS.rows()) = VS.rowwise() + P.row(i);
		// 		F.middleRows(nf, FS.rows()) = FS.array() + nv;
		// 		C.middleRows(nf, FS.rows()).rowwise() = Eigen::RowVector3d(142, 68, 173)/255.;
		// 		nv += VS.rows();
		// 		nf += FS.rows();
		// 	}

		// 	igl::per_corner_normals(V, F, 20.0, VN);

		// 	std::cout << C.topRows(10) << std::endl;
		// 	std::cout << C.bottomRows(10) << std::endl;

		// 	igl::opengl::glfw::Viewer viewer;
		// 	viewer.data().set_mesh(V, F);
		// 	// viewer.data().add_points(P, Eigen::Vector3d(0,1,1).transpose());
		// 	viewer.data().set_normals(VN);
		// 	viewer.data().set_face_based(false);
		// 	viewer.data().set_colors(C);
		// 	viewer.data().lines = viewer0.data().lines;
		// 	viewer.data().show_lines = false;
		// 	viewer.data().line_width = 5;
		// 	viewer.core.background_color.setOnes();
		// 	viewer.core.set_rotation_type(igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL);

		// 	// #ifdef IGL_VIEWER_WITH_NANOGUI
		// 	// viewer.callback_init = [&](igl::opengl::glfw::Viewer& viewer_) {
		// 	// 	viewer_.ngui->addButton("Save screenshot", [&] {
		// 	// 		// Allocate temporary buffers
		// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(6400, 4000);
		// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(6400, 4000);
		// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(6400, 4000);
		// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(6400, 4000);

		// 	// 		// Draw the scene in the buffers
		// 	// 		viewer_.core.draw_buffer(viewer.data,viewer.opengl,false,R,G,B,A);
		// 	// 		A.setConstant(255);

		// 	// 		// Save it to a PNG
		// 	// 		igl::png::writePNG(R,G,B,A,"foo.png");
		// 	// 	});
		// 	// 	viewer_.screen->performLayout();
		// 	// 	return false;
		// 	// };
		// 	// #endif

		// 	viewer.launch();
		// }

		// } // anonymous namespace

		// -----------------------------------------------------------------------------

		int PolygonalBasis3d::build_bases(
			const LinearAssembler &assembler,
			const int nn_samples_per_edge,
			const Mesh3D &mesh,
			const int n_bases,
			const int quadrature_order,
			const int mass_quadrature_order,
			const int integral_constraints,
			std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const std::map<int, InterfaceData> &poly_face_to_data,
			std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &mapped_boundary)
		{
			assert(mesh.is_volume());
			if (poly_face_to_data.empty())
			{
				return 0;
			}
			int n_kernels_per_edge = 4; //(int) std::round(n_samples_per_edge / 3.0);
			int n_samples_per_edge = 3 * n_kernels_per_edge;

			// Step 1: Compute integral constraints
			Eigen::MatrixXd basis_integrals;
			compute_integral_constraints(assembler, mesh, n_bases, bases, gbases, basis_integrals);

			// Step 2: Compute the rest =)
			for (int e = 0; e < mesh.n_elements(); ++e)
			{
				if (!mesh.is_polytope(e))
				{
					continue;
				}
				// No boundary polytope
				// assert(element_type[e] != ElementType::BOUNDARY_POLYTOPE);

				// Kernel distance to polygon boundary
				const double eps = compute_epsilon(mesh, e);

				std::vector<int> local_to_global; // map local basis id (the ones that are nonzero on the polygon boundary) to global basis id
				Eigen::MatrixXd collocation_points, kernel_centers, triangulated_vertices;
				Eigen::MatrixXi triangulated_faces;
				Eigen::MatrixXd rhs; // 1 row per collocation point, 1 column per basis that is nonzero on the polygon boundary

				ElementBases &b = bases[e];
				b.has_parameterization = false;

				Quadrature tmp_quadrature, tmp_mass_quadrature;
				double scaling;
				Eigen::RowVector3d translation;
				sample_polyhedra(e, 2, n_kernels_per_edge, n_samples_per_edge,
								 quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler.name(), 2, AssemblerUtils::BasisType::POLY, 3),
								 mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 2, AssemblerUtils::BasisType::POLY, 3),
								 mesh, poly_face_to_data, bases, gbases, eps, local_to_global,
								 collocation_points, kernel_centers, rhs, triangulated_vertices,
								 triangulated_faces, tmp_quadrature, tmp_mass_quadrature, scaling, translation);

				b.set_quadrature([tmp_quadrature](Quadrature &quad) { quad = tmp_quadrature; });
				b.set_mass_quadrature([tmp_mass_quadrature](Quadrature &quad) { quad = tmp_mass_quadrature; });
				// b.scaling_ = scaling;
				// b.translation_ = translation;

				// igl::opengl::glfw::Viewer & viewer = UIState::ui_state().viewer;
				// viewer.data().clear();
				// viewer.data().set_mesh(triangulated_vertices, triangulated_faces);
				// viewer.data().add_points(kernel_centers, Eigen::Vector3d(0,1,1).transpose());
				// add_spheres(viewer, kernel_centers, 0.005);

				// Eigen::MatrixXd pts = triangulated_vertices, normals;
				// Eigen::MatrixXi tris = triangulated_faces;
				// igl::per_corner_normals(pts, tris, 20, normals);
				// viewer.data().set_normals(normals);
				// viewer.data().set_face_based(false);
				// viewer.launch();

				// for(int a = 0; rhs.cols();++a)
				// 	{
				// 	igl::opengl::glfw::Viewer viewer;
				// 	Eigen::MatrixXd asd(collocation_points.rows(), 3);
				// 	asd.col(0)=collocation_points.col(0);
				// 	asd.col(1)=collocation_points.col(1);
				// 	asd.col(2)=collocation_points.col(2);
				// 	Eigen::VectorXd S = rhs.col(a);
				// 	Eigen::MatrixXd C;
				// 	igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, S, true, C);
				// 	viewer.data().add_points(asd, C);
				// 	viewer.launch();
				// }

				// for(int asd = 0; asd < collocation_points.rows(); ++asd) {
				//     viewer.data().add_label(collocation_points.row(asd), std::to_string(asd));
				// }

				// Compute the weights of the RBF kernels
				Eigen::MatrixXd local_basis_integrals(rhs.cols(), basis_integrals.cols());
				for (long k = 0; k < rhs.cols(); ++k)
				{
					local_basis_integrals.row(k) = -basis_integrals.row(local_to_global[k]);
				}
				auto set_rbf = [&b](auto rbf) {
					b.set_bases_func([rbf](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val) {
						Eigen::MatrixXd tmp;
						rbf->bases_values(uv, tmp);
						val.resize(tmp.cols());
						assert(tmp.rows() == uv.rows());

						for (size_t i = 0; i < tmp.cols(); ++i)
						{
							val[i].val = tmp.col(i);
						}
					});
					b.set_grads_func([rbf](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val) {
						Eigen::MatrixXd tmpx, tmpy, tmpz;

						rbf->bases_grads(0, uv, tmpx);
						rbf->bases_grads(1, uv, tmpy);
						rbf->bases_grads(2, uv, tmpz);

						val.resize(tmpx.cols());
						assert(tmpx.cols() == tmpy.cols());
						assert(tmpx.cols() == tmpz.cols());
						assert(tmpx.rows() == uv.rows());
						for (size_t i = 0; i < tmpx.cols(); ++i)
						{
							val[i].grad.resize(uv.rows(), uv.cols());
							val[i].grad.col(0) = tmpx.col(i);
							val[i].grad.col(1) = tmpy.col(i);
							val[i].grad.col(2) = tmpz.col(i);
						}
					});
				};
				if (integral_constraints == 0)
				{
					set_rbf(std::make_shared<RBFWithLinear>(
						kernel_centers, collocation_points, local_basis_integrals, tmp_quadrature, rhs, false));
				}
				else if (integral_constraints == 1)
				{
					set_rbf(std::make_shared<RBFWithLinear>(
						kernel_centers, collocation_points, local_basis_integrals, tmp_quadrature, rhs));
				}
				else if (integral_constraints == 2)
				{
					set_rbf(std::make_shared<RBFWithQuadratic>(
						// set_rbf(std::make_shared<RBFWithQuadraticLagrange>(
						assembler, kernel_centers, collocation_points, local_basis_integrals, tmp_quadrature, rhs));
				}
				else
				{
					throw std::runtime_error(fmt::format("Unsupported constraint order: {:d}", integral_constraints));
				}

				// Set the bases which are nonzero inside the polygon
				const int n_poly_bases = int(local_to_global.size());
				b.bases.resize(n_poly_bases);
				for (int i = 0; i < n_poly_bases; ++i)
				{
					b.bases[i].init(-2, local_to_global[i], i, Eigen::MatrixXd::Constant(1, 3, std::nan("")));
				}

				// Polygon boundary after geometric mapping from neighboring elements
				orient_closed_surface(triangulated_vertices, triangulated_faces, false); // stupid viewer is flipping all the faces
				mapped_boundary[e].first = triangulated_vertices;
				mapped_boundary[e].second = triangulated_faces;
			}

			return 0;
		}
	} // namespace basis
} // namespace polyfem
