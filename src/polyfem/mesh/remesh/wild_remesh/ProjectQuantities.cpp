#include <polyfem/mesh/remesh/WildRemeshing.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>

namespace polyfem::mesh
{
	void WildRemeshing::project_quantities()
	{
		// Assume the rest positions and elements have been updated
		const Eigen::MatrixXd proposed_rest_positions = rest_positions();
		const Eigen::MatrixXi proposed_elements = elements();

		// --------------------------------------------------------------------

		// Assume isoparametric
		std::vector<polyfem::basis::ElementBases> bases_before;
		Eigen::VectorXi vertex_to_basis_before;
		int n_bases_before = build_bases(
			global_cache.rest_positions_before, global_cache.elements_before,
			state.formulation(), bases_before, vertex_to_basis_before);
		const std::vector<polyfem::basis::ElementBases> &geom_bases_before = bases_before;
		n_bases_before += state.obstacle.n_vertices();

		// Old values of independent variables
		Eigen::MatrixXd y(n_bases_before * dim(), 1 + n_quantities());
		y.col(0) = utils::flatten(utils::reorder_matrix(
			global_cache.positions_before - global_cache.rest_positions_before,
			vertex_to_basis_before, n_bases_before));
		y.rightCols(n_quantities()) = utils::reorder_matrix(
			global_cache.projected_quantities_before, vertex_to_basis_before,
			n_bases_before, dim());

		// --------------------------------------------------------------------

		const int num_vertices = proposed_rest_positions.rows();

		std::vector<polyfem::basis::ElementBases> bases;
		Eigen::VectorXi vertex_to_basis;
		int n_bases = build_bases(
			proposed_rest_positions, proposed_elements, state.formulation(),
			bases, vertex_to_basis);
		const std::vector<polyfem::basis::ElementBases> &geom_bases = bases;
		n_bases += state.obstacle.n_vertices();

		const Eigen::MatrixXd target_x = utils::flatten(utils::reorder_matrix(
			displacements(), vertex_to_basis, n_bases));

		std::vector<int> boundary_nodes = this->boundary_nodes();
		for (int &boundary_node : boundary_nodes)
			boundary_node = vertex_to_basis[boundary_node];
		std::sort(boundary_nodes.begin(), boundary_nodes.end());

		// --------------------------------------------------------------------

		// L2 Projection
		Eigen::MatrixXd x;
		L2_projection(
			is_volume(), /*size=*/dim(),
			n_bases_before, bases_before, geom_bases_before, // from
			n_bases, bases, geom_bases,                      // to
			boundary_nodes, state.obstacle, target_x,
			y, x, /*lump_mass_matrix=*/false);

		// --------------------------------------------------------------------

		set_positions(proposed_rest_positions + utils::unreorder_matrix( //
						  utils::unflatten(x.col(0), dim()), vertex_to_basis, num_vertices));
		set_projected_quantities(utils::unreorder_matrix(
			x.rightCols(n_quantities()), vertex_to_basis, num_vertices, dim()));
	}

} // namespace polyfem::mesh