////////////////////////////////////////////////////////////////////////////////
#include "PolygonalBasis2d.hpp"
#include "PolygonQuadrature.hpp"
#include "PolygonUtils.hpp"
#include "FEBasis2d.hpp"
#include "Harmonic.hpp"
#include "Biharmonic.hpp"
#include "UIState.hpp"
////////////////////////////////////////////////////////////////////////////////

namespace poly_fem {
namespace {

// -----------------------------------------------------------------------------

std::vector<int> compute_nonzero_bases_ids(const Mesh2D &mesh, const int element_index,
	const std::map<int, InterfaceData> &poly_edge_to_data)
{
	std::vector<int> local_to_global;

	Navigation::Index index = mesh.get_index_from_face(element_index);
	const int n_edges = mesh.n_element_vertices(element_index);
	for(int i = 0; i < n_edges; ++i) {
		const InterfaceData &bdata = poly_edge_to_data.at(index.edge);
		local_to_global.insert(local_to_global.end(), bdata.node_id.begin(), bdata.node_id.end());

		index = mesh.next_around_face(index);
	}

	std::sort(local_to_global.begin(), local_to_global.end());
	auto it = std::unique(local_to_global.begin(), local_to_global.end());
	local_to_global.resize(std::distance(local_to_global.begin(), it));

    // assert(int(local_to_global.size()) <= n_edges);
    return local_to_global;
}

// -----------------------------------------------------------------------------

void sample_edge(const Mesh2D &mesh, Navigation::Index index, int n_samples, Eigen::MatrixXd &samples) {
	Eigen::MatrixXd endpoints = FEBasis2d::linear_quad_edge_local_nodes_coordinates(mesh, index);
	const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
	samples.resize(n_samples, endpoints.cols());
	for (int c = 0; c < 2; ++c) {
		samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
	}
}

// -----------------------------------------------------------------------------

void compute_offset_kernels(const Eigen::MatrixXd &polygon, int n_kernels, double eps,
	Eigen::MatrixXd &kernel_centers)
{
	Eigen::MatrixXd offset, samples;
	std::vector<bool> inside;
	offset_polygon(polygon, offset, eps);
	sample_polygon(offset, n_kernels, samples);
	int n_inside = is_inside(polygon, samples, inside);
	kernel_centers.resize(samples.rows() - n_inside, samples.cols());
	for (int i = 0, j = 0; i < samples.rows(); ++i) {
		if (!inside[i]) {
			kernel_centers.row(j++) = samples.row(i);
		}
	}

	// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
	// viewer.data.add_points(samples, Eigen::Vector3d(0,1,1).transpose());
}

// -----------------------------------------------------------------------------

///
/// @brief      { Compute boundary sample points + centers of harmonic bases for
///             the polygonal element }
///
void sample_polygon(
	const int element_index,
	const int n_samples_per_edge,
	const Mesh2D &mesh,
	const std::map<int, InterfaceData> &poly_edge_to_data,
	const std::vector< ElementBases > &bases,
	const std::vector< ElementBases > &gbases,
	const double eps,
	std::vector<int> &local_to_global,
	Eigen::MatrixXd &collocation_points,
	Eigen::MatrixXd &kernel_centers,
	Eigen::MatrixXd &rhs)
{
	const int n_edges = mesh.n_element_vertices(element_index);

	const int n_kernel_per_edges = (n_samples_per_edge - 1)/3;
	const int n_collocation_points = (n_samples_per_edge - 1) * n_edges;
	const int n_kernels = n_kernel_per_edges * n_edges;

	// Local ids of nonzero bases over the polygon
	local_to_global = compute_nonzero_bases_ids(mesh, element_index, poly_edge_to_data);

	collocation_points.resize(n_collocation_points, 2);
	collocation_points.setZero();

	rhs.resize(n_collocation_points, local_to_global.size());
	rhs.setZero();

	Eigen::MatrixXd samples, mapped, basis_val;
	auto index = mesh.get_index_from_face(element_index);
	for(int i = 0; i < n_edges; ++i) {
		assert(mesh.switch_face(index).face >= 0); // no boundary polygons

		const InterfaceData &bdata = poly_edge_to_data.at(index.edge);
		const ElementBases &b=bases[bdata.element_id];
		const ElementBases &gb=gbases[bdata.element_id];

		assert(bdata.element_id == mesh.switch_face(index).face);

		sample_edge(mesh, mesh.switch_face(index), n_samples_per_edge, samples);
		samples.conservativeResize(samples.rows() - 1, samples.cols());
		gb.eval_geom_mapping(samples, mapped);

		for(size_t bi = 0; bi < bdata.node_id.size(); ++bi) {
			const int local_index = bdata.local_indices[bi];
			const long local_basis_id = std::distance(local_to_global.begin(), std::find(local_to_global.begin(), local_to_global.end(), bdata.node_id[bi]));

			b.bases[local_index].basis(samples, basis_val);

			rhs.block(i*(n_samples_per_edge-1), local_basis_id, basis_val.rows(), 1) += basis_val * bdata.vals[bi];
		}

		assert(mapped.rows() == (n_samples_per_edge-1));
		collocation_points.block(i*(n_samples_per_edge-1), 0, mapped.rows(), mapped.cols()) = mapped;

		index = mesh.next_around_face(index);
	}

	compute_offset_kernels(collocation_points, n_kernels, eps, kernel_centers);
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

// Compute the integral constraints for each basis of the mesh
void PolygonalBasis2d::compute_integral_constraints(
	const Mesh2D &mesh,
	const int n_bases,
	const std::vector< ElementAssemblyValues > &values,
	const std::vector< ElementAssemblyValues > &gvalues,
	Eigen::MatrixXd &basis_integrals)
{
	assert(!mesh.is_volume());

	basis_integrals.resize(n_bases, 2);
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

			for(size_t ii = 0; ii < v.global.size(); ++ii) {
				basis_integrals(v.global[ii].index, 0) += integralx * v.global[ii].val;
				basis_integrals(v.global[ii].index, 1) += integraly * v.global[ii].val;
			}
		}
	}
}

// -----------------------------------------------------------------------------

// Distance from harmonic kernels to polygon boundary
double compute_epsilon(const Mesh2D &mesh, int e) {
	double area = 0;
	const int n_edges = mesh.n_element_vertices(e);
	for (int i = 0; i < n_edges; ++i) {
		const int ip = (i + 1) % n_edges;

		Eigen::RowVector2d p0 = mesh.point(mesh.vertex_global_index(e, i));
		Eigen::RowVector2d p1 = mesh.point(mesh.vertex_global_index(e, ip));

		Eigen::Matrix2d det_mat;
		det_mat.row(0) = p0;
		det_mat.row(1) = p1;

		area += det_mat.determinant();
	}
	area = std::fabs(area);
	// const double eps = use_harmonic ? (0.08*area) : 0;
	const double eps = 0.08*area;

	return eps;
}

// -----------------------------------------------------------------------------

void PolygonalBasis2d::build_bases(
	const int n_samples_per_edge,
	const Mesh2D &mesh,
	const int n_bases,
	const std::vector<ElementType> &element_type,
	const int quadrature_order,
	const std::vector< ElementAssemblyValues > &values,
	const std::vector< ElementAssemblyValues > &gvalues,
	std::vector< ElementBases > &bases,
	const std::vector< ElementBases > &gbases,
	const std::map<int, InterfaceData> &poly_edge_to_data,
	std::map<int, Eigen::MatrixXd> &mapped_boundary)
{
	assert(!mesh.is_volume());
	if (poly_edge_to_data.empty()) {
		return;
	}

	// Step 1: Compute integral constraints
	Eigen::MatrixXd basis_integrals;
	compute_integral_constraints(mesh, n_bases, values, gvalues, basis_integrals);

	PolygonQuadrature poly_quadr;
	Eigen::Matrix2d det_mat;
	Eigen::MatrixXd p0, p1;

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
		Eigen::MatrixXd rhs; // 1 row per collocation point, 1 column per basis that is nonzero on the polygon boundary

		sample_polygon(e, n_samples_per_edge, mesh, poly_edge_to_data, bases, gbases,
			eps, local_to_global, collocation_points, kernel_centers, rhs);

		// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
		// viewer.data.add_points(kernel_centers, Eigen::Vector3d(0,1,1).transpose());

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
		poly_quadr.get_quadrature(collocation_points, quadrature_order, b.quadrature);

		// Compute the weights of the harmonic kernels
		Eigen::MatrixXd local_basis_integrals(rhs.cols(), basis_integrals.cols());
		for (long k = 0; k < rhs.cols(); ++k) {
			local_basis_integrals.row(k) = -basis_integrals.row(local_to_global[k]);
		}
		Harmonic harmonic(kernel_centers, collocation_points, local_basis_integrals, b.quadrature, rhs);

		// Set the bases which are nonzero inside the polygon
		const int n_poly_bases = int(local_to_global.size());
		b.bases.resize(n_poly_bases);
		for (int i = 0; i < n_poly_bases; ++i) {
			b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd::Zero(1, 2));
			b.bases[i].set_basis([harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
				{ harmonic.basis(i, uv, val); });
			b.bases[i].set_grad( [harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
				{ harmonic.grad(i, uv, val); });
		}

		// Polygon boundary after geometric mapping from neighboring elements
		mapped_boundary[e] = collocation_points;
	}
}

} // namespace poly_fem
