////////////////////////////////////////////////////////////////////////////////
#include "PolygonalBasis3d.hpp"
// #include "PolyhedronQuadrature.hpp"
#include "FEBasis3d.hpp"
#include "MeshUtils.hpp"
#include "Refinement.hpp"
#include "Harmonic.hpp"
#include "UIState.hpp"
#include <igl/triangle/triangulate.h>
////////////////////////////////////////////////////////////////////////////////

namespace poly_fem {
namespace {

// -----------------------------------------------------------------------------

std::vector<int> compute_nonzero_bases_ids(const Mesh3D &mesh, const int c,
	const std::map<int, InterfaceData> &poly_face_to_data)
{
	std::vector<int> local_to_global;

	for (int lf = 0; lf < mesh.n_element_faces(c); ++lf) {
		int f = mesh.get_index_from_element(c, lf, 0).face;
		const InterfaceData &bdata = poly_face_to_data.at(f);
		local_to_global.insert(local_to_global.end(), bdata.node_id.begin(), bdata.node_id.end());
	}

	std::sort(local_to_global.begin(), local_to_global.end());
	auto it = std::unique(local_to_global.begin(), local_to_global.end());
	local_to_global.resize(std::distance(local_to_global.begin(), it));

    return local_to_global;
}

// -----------------------------------------------------------------------------

// Canonical triangle mesh in parametric domain
void compute_canonical_pattern(int n_samples_per_edge, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
	Eigen::Matrix<double, 4, 2> corners;
	corners <<
		0, 0,
		1, 0,
		1, 1,
		0, 1;
	const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples_per_edge, 0, 1).head(n_samples_per_edge - 1);
	Eigen::MatrixXd P(4*(n_samples_per_edge-1), 2); // Contour of the face
	Eigen::MatrixXi E(P.rows(), 2);
	for (int lv = 0; lv < 4; ++lv) {
		int offset = lv*(n_samples_per_edge-1);
		for (int d = 0; d < 2; ++d) {
			P.block(offset, d, n_samples_per_edge-1, 1) =
				(1.0 - t.array()).matrix() * corners(lv, d) + t * corners((lv+1)%4, d);
		}
	}
	for (int i = 0; i < P.rows(); ++i) {
		E.row(i) << i, (i+1)%P.rows();
	}

	assert(n_samples_per_edge > 1);
	double spacing = 1.0/(n_samples_per_edge-1);
	double area = 1.0 * std::sqrt(3.0)/4.0*spacing*spacing;
	std::string flags = "QpYq30a" + std::to_string(area);
	igl::triangle::triangulate(P, E, Eigen::MatrixXd(0, 2), flags, V, F);

	// igl::viewer::Viewer viewer;
	// viewer.data.set_mesh(V, F);
	// viewer.launch();
}

// -----------------------------------------------------------------------------

// Assemble the surface quad mesh (V, F) corresponding to the polyhedron c
void compute_quad_mesh_from_cell(const Mesh3D &mesh, int c, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
	std::vector<std::array<int, 4>> quads(mesh.n_element_faces(c));
	std::map<int, int> global_to_local;
	int num_vertices = 0;
	for (int lf = 0; lf < mesh.n_element_faces(c); ++lf) {
		auto index = mesh.get_index_from_element(c, lf, 0);
		assert(mesh.n_face_vertices(index.face) == 4);
		for (int lv = 0; lv < 4; ++lv) {
			if (!global_to_local.count(index.vertex)) {
				global_to_local.emplace(index.vertex, num_vertices++);
			}
			quads[lf][lv] = global_to_local.at(index.vertex);
			index = mesh.next_around_face_of_element(index);
		}
	}
	V.resize(num_vertices, 3);
	for (const auto &kv : global_to_local) {
		V.row(kv.second) = mesh.point(kv.first);
	}
	F.resize(quads.size(), 4);
	int f = 0;
	for (auto q : quads) {
		F.row(f++) << q[0], q[1], q[2], q[3];
	}
}

// -----------------------------------------------------------------------------

void compute_offset_kernels(const Eigen::MatrixXd &polygon, int n_kernels, double eps,
	Eigen::MatrixXd &kernel_centers)
{
	// Eigen::MatrixXd offset, samples;
	// std::vector<bool> inside;
	// offset_polygon(polygon, offset, eps);
	// sample_polygon(offset, n_kernels, samples);
	// int n_inside = is_inside(polygon, samples, inside);
	// kernel_centers.resize(samples.rows() - n_inside, samples.cols());
	// for (int i = 0, j = 0; i < samples.rows(); ++i) {
	// 	if (!inside[i]) {
	// 		kernel_centers.row(j++) = samples.row(i);
	// 	}
	// }

	// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
	// viewer.data.add_points(samples, Eigen::Vector3d(0,1,1).transpose());
}

// -----------------------------------------------------------------------------

///
/// @brief      { Compute boundary sample points + centers of harmonic bases for
///             the polygonal element }
///
void sample_polyhedra(
	const int element_index,
	const int n_samples_per_edge,
	const Mesh3D &mesh,
	const std::map<int, InterfaceData> &poly_face_to_data,
	const std::vector< ElementBases > &bases,
	const std::vector< ElementBases > &gbases,
	const double eps,
	std::vector<int> &local_to_global,
	Eigen::MatrixXd &collocation_points,
	Eigen::MatrixXi &collocation_faces,
	Eigen::MatrixXd &kernel_centers,
	Eigen::MatrixXd &rhs)
{
	// Local ids of nonzero bases over the polygon
	local_to_global = compute_nonzero_bases_ids(mesh, element_index, poly_face_to_data);

	// Compute the image of the canonical pattern vertices through the geometric mapping
	// of the given local face
	auto evalFunc = [&] (const Eigen::MatrixXd &uv, Eigen::MatrixXd &mapped, int lf) {
		const auto & u = uv.col(0).array();
		const auto & v = uv.col(1).array();
		auto index = mesh.get_index_from_element(element_index, lf, 0);
		index = mesh.switch_element(index);
		Eigen::MatrixXd abcd = FEBasis3d::linear_hex_face_local_nodes_coordinates(mesh, index);
		Eigen::RowVector3d a = abcd.row(0);
		Eigen::RowVector3d b = abcd.row(1);
		Eigen::RowVector3d c = abcd.row(2);
		Eigen::RowVector3d d = abcd.row(3);
		mapped = ((1-u)*(1-v)).matrix()*a
			+ (u*(1-v)).matrix()*b
			+ (u*v).matrix()*c
			+ ((1-u)*v).matrix()*d;
	};
	auto evalFuncGeom = [&] (const Eigen::MatrixXd &uv, Eigen::MatrixXd &mapped, int lf) {
		Eigen::MatrixXd samples;
		evalFunc(uv, samples, lf);
		auto index = mesh.get_index_from_element(element_index, lf, 0);
		index = mesh.switch_element(index);
		const ElementBases &gb=gbases[index.element];
		gb.eval_geom_mapping(samples, mapped);
	};

	// Compute collocation points
	Eigen::MatrixXd IV, PV, OV, UV;
	Eigen::MatrixXi IF, PF, OF, UF;
	Eigen::VectorXi uv_sources, uv_ranges;
	compute_quad_mesh_from_cell(mesh, element_index, IV, IF);
	compute_canonical_pattern(n_samples_per_edge, PV, PF);
	instanciate_pattern(IV, IF, PV, PF, UV, UF, &uv_sources, evalFunc);
	instanciate_pattern(IV, IF, PV, PF, collocation_points, collocation_faces, nullptr, evalFuncGeom);
	reorder_mesh(UV, UF, uv_sources, uv_ranges);
	assert(uv_ranges.size() == mesh.n_element_faces(element_index) + 1);

	// Compute kernel centers
	// compute_offset_kernels(collocation_points, n_kernels, eps, kernel_centers);

	// igl::viewer::Viewer viewer;
	// viewer.data.set_mesh(UV, UF);
	// for (int lf = 0; lf < mesh.n_element_faces(element_index); ++lf) {
	// 	Eigen::MatrixXd samples;
	// 	samples = UV.middleRows(uv_ranges(lf), uv_ranges(lf+1) - uv_ranges(lf));
	// 	Eigen::RowVector3d c = Eigen::RowVector3d::Random();
	// 	viewer.data.add_points(samples, c);
	// }
	// viewer.launch();

	// Compute right-hand side constraints for setting the harmonic kernels
	Eigen::MatrixXd samples, basis_val;
	rhs.resize(UV.rows(), local_to_global.size());
	rhs.setZero();
	for (int lf = 0; lf < mesh.n_element_faces(element_index); ++lf) {
		auto index = mesh.get_index_from_element(element_index, lf, 0);
		assert(mesh.switch_element(index).element >= 0); // no boundary polygons

		const InterfaceData &bdata = poly_face_to_data.at(index.face);
		const ElementBases &b=bases[bdata.element_id];

		assert(bdata.element_id == mesh.switch_element(index).element);

		// Evaluate field basis and set up the rhs
		for(size_t bi = 0; bi < bdata.node_id.size(); ++bi) {
			const int local_index = bdata.local_indices[bi];
			const long local_basis_id = std::distance(local_to_global.begin(),
				std::find(local_to_global.begin(), local_to_global.end(), bdata.node_id[bi]));

			samples = UV.middleRows(uv_ranges(lf), uv_ranges(lf+1) - uv_ranges(lf));
			b.bases[local_index].basis(samples, basis_val);

			MatrixXd m = basis_val * bdata.vals[bi];
			rhs.block(uv_ranges(lf), local_basis_id, basis_val.rows(), 1) += basis_val * bdata.vals[bi];
		}
	}
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

// Compute the integral constraints for each basis of the mesh
void PolygonalBasis3d::compute_integral_constraints(
	const Mesh3D &mesh,
	const int n_bases,
	const std::vector< ElementAssemblyValues > &values,
	const std::vector< ElementAssemblyValues > &gvalues,
	Eigen::MatrixXd &basis_integrals)
{
	assert(mesh.is_volume());

	basis_integrals.resize(n_bases, 3);
	basis_integrals.setZero();

	const int n_elements = mesh.n_elements();
	for(int e = 0; e < n_elements; ++e) {
		const ElementAssemblyValues &vals = values[e];
		const ElementAssemblyValues &gvals = gvalues[e];

		// Computes the discretized integral of the PDE over the element
		const int n_local_bases = int(vals.basis_values.size());
		for(int j = 0; j < n_local_bases; ++j) {
			const AssemblyValues &v=vals.basis_values[j];
			const double integralx = (v.grad_t_m.col(0).array() * gvals.det.array() * vals.quadrature.weights.array()).sum();
			const double integraly = (v.grad_t_m.col(1).array() * gvals.det.array() * vals.quadrature.weights.array()).sum();
			const double integralz = (v.grad_t_m.col(2).array() * gvals.det.array() * vals.quadrature.weights.array()).sum();

			for(size_t ii = 0; ii < v.global.size(); ++ii) {
				basis_integrals(v.global[ii].index, 0) += integralx * v.global[ii].val;
				basis_integrals(v.global[ii].index, 1) += integraly * v.global[ii].val;
				basis_integrals(v.global[ii].index, 2) += integralz * v.global[ii].val;
			}
		}
	}
}

// -----------------------------------------------------------------------------

// Distance from harmonic kernels to polygon boundary
double compute_epsilon(const Mesh3D &mesh, int e) {
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

	return 0.01;
}

// -----------------------------------------------------------------------------

void PolygonalBasis3d::build_bases(
	const int n_samples_per_edge,
	const Mesh3D &mesh,
	const int n_bases,
	const std::vector<ElementType> &element_type,
	const int quadrature_order,
	const std::vector< ElementAssemblyValues > &values,
	const std::vector< ElementAssemblyValues > &gvalues,
	std::vector< ElementBases > &bases,
	const std::vector< ElementBases > &gbases,
	const std::map<int, InterfaceData> &poly_face_to_data,
	std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi> > &mapped_boundary)
{
	assert(mesh.is_volume());
	// if (poly_face_to_data.empty()) {
	// 	return;
	// }

	// Step 1: Compute integral constraints
	Eigen::MatrixXd basis_integrals;
	compute_integral_constraints(mesh, n_bases, values, gvalues, basis_integrals);

	// Step 2: Compute the rest =)
	// PolyhedronQuadrature poly_quadr;
	for (int e = 0; e < mesh.n_elements(); ++e) {
		if (element_type[e] != ElementType::InteriorPolytope && element_type[e] != ElementType::BoundaryPolytope) {
			continue;
		}
		// No boundary polytope
		assert(element_type[e] != ElementType::BoundaryPolytope);

		// Kernel distance to polygon boundary
		const double eps = compute_epsilon(mesh, e);

		std::vector<int> local_to_global; // map local basis id (the ones that are nonzero on the polygon boundary) to global basis id
		Eigen::MatrixXd collocation_points, kernel_centers;
		Eigen::MatrixXi collocation_faces;
		Eigen::MatrixXd rhs; // 1 row per collocation point, 1 column per basis that is nonzero on the polygon boundary

		sample_polyhedra(e, n_samples_per_edge, mesh, poly_face_to_data, bases, gbases,
			eps, local_to_global, collocation_points, collocation_faces, kernel_centers, rhs);

		// igl::viewer::Viewer viewer;
		// viewer.data.set_mesh(collocation_points, collocation_faces);
		// viewer.launch();
		// viewer.data.add_points(kernel_centers, Eigen::Vector3d(0,1,1).transpose());

		// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
		// Eigen::MatrixXd asd(collocation_points.rows(), 3);
		// asd.col(0)=collocation_points.col(0);
		// asd.col(1)=collocation_points.col(1);
		// asd.col(2)=rhs.col(0);
		// viewer.data.add_points(asd, Eigen::Vector3d(1,0,1).transpose());

		// for(int asd = 0; asd < collocation_points.rows(); ++asd) {
		//     viewer.data.add_label(collocation_points.row(asd), std::to_string(asd));
		// }

		ElementBases &b=bases[e];
		b.has_parameterization = false;

		// Compute quadrature points for the polygon
		// poly_quadr.get_quadrature(collocation_points, quadrature_order, b.quadrature);

		// Compute the weights of the harmonic kernels
		Eigen::MatrixXd local_basis_integrals(rhs.cols(), basis_integrals.cols());
		for (long k = 0; k < rhs.cols(); ++k) {
			local_basis_integrals.row(k) = -basis_integrals.row(local_to_global[k]);
		}
		// Harmonic harmonic(kernel_centers, collocation_points, local_basis_integrals, b.quadrature, rhs);

		// Set the bases which are nonzero inside the polygon
		const int n_poly_bases = int(local_to_global.size());
		b.bases.resize(n_poly_bases);
		for (int i = 0; i < n_poly_bases; ++i) {
			b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd::Zero(1, 2));
			// b.bases[i].set_basis([harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			// 	{ harmonic.basis(i, uv, val); });
			// b.bases[i].set_grad( [harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			// 	{ harmonic.grad(i, uv, val); });
		}

		// Polygon boundary after geometric mapping from neighboring elements
		mapped_boundary[e].first = collocation_points;
		mapped_boundary[e].second = collocation_faces;
	}
}

} // namespace poly_fem
