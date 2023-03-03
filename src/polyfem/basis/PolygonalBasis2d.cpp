////////////////////////////////////////////////////////////////////////////////
#include "PolygonalBasis2d.hpp"
#include "LagrangeBasis2d.hpp"

#include <polyfem/quadrature/PolygonQuadrature.hpp>
#include <polyfem/mesh/mesh2D/PolygonUtils.hpp>
#include "function/RBFWithLinear.hpp"
#include "function/RBFWithQuadratic.hpp"
#include "function/RBFWithQuadraticLagrange.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/autogen/auto_q_bases.hpp>

#include <random>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace quadrature;

	namespace basis
	{

		namespace
		{

			// -----------------------------------------------------------------------------

			std::vector<int> compute_nonzero_bases_ids(const Mesh2D &mesh, const int element_index, const std::vector<ElementBases> &bases, const std::map<int, InterfaceData> &poly_edge_to_data)
			{
				std::vector<int> local_to_global;

				const int n_edges = mesh.n_face_vertices(element_index);
				Navigation::Index index = mesh.get_index_from_face(element_index);
				for (int i = 0; i < n_edges; ++i)
				{
					const int f2 = mesh.switch_face(index).face;
					assert(f2 >= 0); // no boundary polygons
					const InterfaceData &bdata = poly_edge_to_data.at(index.edge);
					const ElementBases &b = bases[f2];
					for (int other_local_basis_id : bdata.local_indices)
					{
						for (const auto &x : b.bases[other_local_basis_id].global())
						{
							const int global_node_id = x.index;
							local_to_global.push_back(global_node_id);
						}
					}

					index = mesh.next_around_face(index);
				}

				std::sort(local_to_global.begin(), local_to_global.end());
				auto it = std::unique(local_to_global.begin(), local_to_global.end());
				local_to_global.resize(std::distance(local_to_global.begin(), it));

				// assert(int(local_to_global.size()) <= n_edges);
				return local_to_global;
			}

			// -----------------------------------------------------------------------------

			void sample_parametric_edge(const Mesh2D &mesh, Navigation::Index index, int n_samples, Eigen::MatrixXd &samples)
			{
				// Eigen::MatrixXd endpoints = LagrangeBasis2d::linear_quad_edge_local_nodes_coordinates(mesh, index);
				const auto indices = LagrangeBasis2d::quad_edge_local_nodes(1, mesh, index);
				assert(mesh.is_cube(index.face));
				assert(indices.size() == 2);
				Eigen::MatrixXd tmp;
				autogen::q_nodes_2d(1, tmp);
				Eigen::Matrix2d endpoints;
				endpoints.row(0) = tmp.row(indices(0));
				endpoints.row(1) = tmp.row(indices(1));
				const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
				samples.resize(n_samples, endpoints.cols());
				for (int c = 0; c < 2; ++c)
				{
					samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
				}
			}

			// -----------------------------------------------------------------------------

			void compute_offset_kernels(const Eigen::MatrixXd &polygon, int n_kernels, double eps, Eigen::MatrixXd &kernel_centers)
			{
				Eigen::MatrixXd offset, samples;
				std::vector<bool> inside;
				offset_polygon(polygon, offset, eps);
				sample_polygon(offset, n_kernels, samples);
				int n_inside = is_inside(polygon, samples, inside);
				assert(n_inside == 0);
				kernel_centers = samples;
				// kernel_centers.resize(samples.rows() - n_inside, samples.cols());
				// for (int i = 0, j = 0; i < samples.rows(); ++i) {
				// 	if (!inside[i]) {
				// 		kernel_centers.row(j++) = samples.row(i);
				// 	}
				// }

				// igl::opengl::glfw::Viewer &viewer = UIState::ui_state().viewer;
				// viewer.data().add_points(samples, Eigen::Vector3d(0,1,1).transpose());
			}

			// -----------------------------------------------------------------------------

			/// @brief Compute boundary sample points + centers of harmonic bases for the polygonal element
			void sample_polygon(const int element_index, const int n_samples_per_edge, const Mesh2D &mesh, const std::map<int, InterfaceData> &poly_edge_to_data,
								const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases, const double eps, std::vector<int> &local_to_global,
								Eigen::MatrixXd &collocation_points, Eigen::MatrixXd &kernel_centers, Eigen::MatrixXd &rhs)
			{
				const int n_edges = mesh.n_face_vertices(element_index);

				const int n_kernel_per_edges = (n_samples_per_edge - 1) / 3;
				const int n_collocation_points = (n_samples_per_edge - 1) * n_edges;
				const int n_kernels = n_kernel_per_edges * n_edges;

				// Local ids of nonzero bases over the polygon
				local_to_global = compute_nonzero_bases_ids(mesh, element_index, bases, poly_edge_to_data);

				collocation_points.resize(n_collocation_points, 2);
				collocation_points.setZero();

				rhs.resize(n_collocation_points, local_to_global.size());
				rhs.setZero();

				Eigen::MatrixXd samples, mapped;
				std::vector<AssemblyValues> basis_val;
				auto index = mesh.get_index_from_face(element_index);
				for (int i = 0; i < n_edges; ++i)
				{
					const int f2 = mesh.switch_face(index).face;
					assert(f2 >= 0); // no boundary polygons

					const InterfaceData &bdata = poly_edge_to_data.at(index.edge);
					const ElementBases &b = bases[f2];
					const ElementBases &gb = gbases[f2];

					// Sample collocation points on the boundary edge
					sample_parametric_edge(mesh, mesh.switch_face(index), n_samples_per_edge, samples);
					samples.conservativeResize(samples.rows() - 1, samples.cols());
					gb.eval_geom_mapping(samples, mapped);
					assert(mapped.rows() == (n_samples_per_edge - 1));
					collocation_points.block(i * (n_samples_per_edge - 1), 0, mapped.rows(), mapped.cols()) = mapped;

					b.evaluate_bases(samples, basis_val);
					// Evaluate field basis and set up the rhs
					for (int other_local_basis_id : bdata.local_indices)
					{
						// b.bases[other_local_basis_id].basis(samples, basis_val);

						for (const auto &x : b.bases[other_local_basis_id].global())
						{
							const int global_node_id = x.index;
							const double weight = x.val;

							const int poly_local_basis_id = std::distance(local_to_global.begin(), std::find(local_to_global.begin(), local_to_global.end(), global_node_id));
							rhs.block(i * (n_samples_per_edge - 1), poly_local_basis_id, basis_val[other_local_basis_id].val.size(), 1) += basis_val[other_local_basis_id].val * weight;
						}
					}

					index = mesh.next_around_face(index);
				}

				compute_offset_kernels(collocation_points, n_kernels, eps, kernel_centers);
			}

		} // anonymous namespace

		////////////////////////////////////////////////////////////////////////////////

		// Compute the integral constraints for each basis of the mesh
		void PolygonalBasis2d::compute_integral_constraints(const LinearAssembler &assembler, const Mesh2D &mesh, const int n_bases,
															const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases, Eigen::MatrixXd &basis_integrals)
		{
			assert(!mesh.is_volume());

			const int dim = assembler.is_tensor() ? 2 : 1;

			basis_integrals.resize(n_bases, RBFWithQuadratic::index_mapping(dim - 1, dim - 1, 4, dim) + 1);
			basis_integrals.setZero();

			std::array<Eigen::MatrixXd, 5> strong;

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
				vals.compute(e, false, bases[e], gbases[e]);

				const auto &quadr = vals.quadrature;
				const QuadratureVector da = vals.det.array() * quadr.weights.array();

				// Computes the discretized integral of the PDE over the element
				const int n_local_bases = int(vals.basis_values.size());

				// add monomials
				vals.basis_values.resize(n_local_bases + 5);
				RBFWithQuadratic::setup_monomials_vals_2d(n_local_bases, vals.val, vals);
				RBFWithQuadratic::setup_monomials_strong_2d(dim, assembler, vals.val, da, strong);

				for (int j = 0; j < n_local_bases; ++j)
				{
					const AssemblyValues &v = vals.basis_values[j];

					for (int d = 0; d < 5; ++d)
					{
						const auto tmp = assembler.assemble(LinearAssemblerData(vals, n_local_bases + d, j, da));

						for (size_t ii = 0; ii < v.global.size(); ++ii)
						{
							for (int alpha = 0; alpha < dim; ++alpha)
							{
								for (int beta = 0; beta < dim; ++beta)
								{
									const int loc_index = alpha * dim + beta;
									const int r = RBFWithQuadratic::index_mapping(alpha, beta, d, dim);

									basis_integrals(v.global[ii].index, r) += tmp(loc_index) + (strong[d].row(loc_index).transpose().array() * v.val.array()).sum();
								}
							}
						}
					}
				}
			}
		}

		// -----------------------------------------------------------------------------

		// Distance from harmonic kernels to polygon boundary
		double compute_epsilon(const Mesh2D &mesh, int e)
		{
			double area = 0;
			const int n_edges = mesh.n_face_vertices(e);
			Navigation::Index index = mesh.get_index_from_face(e);

			for (int i = 0; i < n_edges; ++i)
			{
				Eigen::Matrix2d det_mat;
				det_mat.row(0) = mesh.point(index.vertex);
				det_mat.row(1) = mesh.point(mesh.switch_vertex(index).vertex);

				area += det_mat.determinant();

				index = mesh.next_around_face(index);
			}
			area = std::fabs(area);
			// const double eps = use_harmonic ? (0.08*area) : 0;
			const double eps = 0.08 * area;

			return eps;
		}

		// namespace {

		// void add_spheres(igl::opengl::glfw::Viewer &viewer0, const Eigen::MatrixXd &PP, double radius) {
		// 	Eigen::MatrixXd V = viewer0.data().V, VS, VN;
		// 	Eigen::MatrixXi F = viewer0.data().F, FS;
		// 	igl::read_triangle_mesh(POLYFEM_MESH_PATH "sphere.ply", VS, FS);

		// 	Eigen::RowVector3d minV = VS.colwise().minCoeff();
		// 	Eigen::RowVector3d maxV = VS.colwise().maxCoeff();
		// 	VS.rowwise() -= minV + 0.5 * (maxV - minV);
		// 	VS /= (maxV - minV).maxCoeff();
		// 	VS *= 2.0 * radius;
		// 	std::cout << V.colwise().minCoeff() << std::endl;
		// 	std::cout << V.colwise().maxCoeff() << std::endl;

		// 	Eigen::MatrixXd C = viewer0.data().F_material_ambient.leftCols(3);
		// 	C *= 10 / 2.0;

		// 	Eigen::MatrixXd P(PP.rows(), 3);
		// 	P.leftCols(2) = PP;
		// 	P.col(2).setZero();

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

		// 	viewer.data().F_material_specular.setZero();
		// 	viewer.data().V_material_specular.setZero();
		// 	viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE;
		// 	viewer.data().V_material_ambient *= 2;
		// 	viewer.data().F_material_ambient *= 2;

		// 	viewer.core.align_camera_center(V);
		// 	viewer.launch();
		// }

		// } // anonymous namespace
		// -----------------------------------------------------------------------------

		int PolygonalBasis2d::build_bases(const LinearAssembler &assembler, const int n_samples_per_edge, const Mesh2D &mesh, const int n_bases,
										  const int quadrature_order, const int mass_quadrature_order, const int integral_constraints, std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases,
										  const std::map<int, InterfaceData> &poly_edge_to_data, std::map<int, Eigen::MatrixXd> &mapped_boundary)
		{
			assert(!mesh.is_volume());
			if (poly_edge_to_data.empty())
			{
				return 0;
			}

			const int dim = assembler.is_tensor() ? 2 : 1;

			// Step 1: Compute integral constraints
			Eigen::MatrixXd basis_integrals;
			compute_integral_constraints(assembler, mesh, n_bases, bases, gbases, basis_integrals);

			// Step 2: Compute the rest =)
			PolygonQuadrature poly_quadr;
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
				Eigen::MatrixXd collocation_points, kernel_centers;
				Eigen::MatrixXd rhs; // 1 row per collocation point, 1 column per basis that is nonzero on the polygon boundary

				sample_polygon(e, n_samples_per_edge, mesh, poly_edge_to_data, bases, gbases, eps, local_to_global, collocation_points, kernel_centers, rhs);

				// igl::opengl::glfw::Viewer viewer;
				// viewer.data().add_points(kernel_centers, Eigen::Vector3d(0,1,1).transpose());

				// Eigen::MatrixXd asd(collocation_points.rows(), 3);
				// asd.col(0)=collocation_points.col(0);
				// asd.col(1)=collocation_points.col(1);
				// asd.col(2)=rhs.col(0);
				// viewer.data().add_points(asd, Eigen::Vector3d(1,0,1).transpose());

				// for(int asd = 0; asd < collocation_points.rows(); ++asd) {
				//     viewer.data().add_label(collocation_points.row(asd), std::to_string(asd));
				// }

				// viewer.launch();

				// igl::opengl::glfw::Viewer & viewer = UIState::ui_state().viewer;
				// viewer.data().clear();
				// viewer.data().set_mesh(triangulated_vertices, triangulated_faces);
				// viewer.data().add_points(kernel_centers, Eigen::Vector3d(0,1,1).transpose());
				// add_spheres(viewer, kernel_centers, 0.01);

				ElementBases &b = bases[e];
				b.has_parameterization = false;

				// Compute quadrature points for the polygon
				Quadrature tmp_quadrature;
				poly_quadr.get_quadrature(collocation_points, quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler.name(), 2, AssemblerUtils::BasisType::POLY, 2), tmp_quadrature);

				Quadrature tmp_mass_quadrature;
				poly_quadr.get_quadrature(collocation_points, mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 2, AssemblerUtils::BasisType::POLY, 2), tmp_mass_quadrature);

				b.set_quadrature([tmp_quadrature](Quadrature &quad) { quad = tmp_quadrature; });
				b.set_mass_quadrature([tmp_mass_quadrature](Quadrature &quad) { quad = tmp_mass_quadrature; });

				// Compute the weights of the harmonic kernels
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
						Eigen::MatrixXd tmpx, tmpy;

						rbf->bases_grads(0, uv, tmpx);
						rbf->bases_grads(1, uv, tmpy);

						val.resize(tmpx.cols());
						assert(tmpx.cols() == tmpy.cols());
						assert(tmpx.rows() == uv.rows());
						for (size_t i = 0; i < tmpx.cols(); ++i)
						{
							val[i].grad.resize(uv.rows(), uv.cols());
							val[i].grad.col(0) = tmpx.col(i);
							val[i].grad.col(1) = tmpy.col(i);
						}
					});
				};
				if (integral_constraints == 0)
				{
					set_rbf(std::make_shared<RBFWithLinear>(kernel_centers, collocation_points, local_basis_integrals, tmp_quadrature, rhs, false));
				}
				else if (integral_constraints == 1)
				{
					set_rbf(std::make_shared<RBFWithLinear>(kernel_centers, collocation_points, local_basis_integrals, tmp_quadrature, rhs));
				}
				else if (integral_constraints == 2)
				{
					set_rbf(std::make_shared<RBFWithQuadraticLagrange>(assembler, kernel_centers, collocation_points, local_basis_integrals, tmp_quadrature, rhs));
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
					b.bases[i].init(-2, local_to_global[i], i, Eigen::MatrixXd::Constant(1, 2, std::nan("")));
				}

				// Polygon boundary after geometric mapping from neighboring elements
				mapped_boundary[e] = collocation_points;
			}

			return 0;
		}
	} // namespace basis
} // namespace polyfem
