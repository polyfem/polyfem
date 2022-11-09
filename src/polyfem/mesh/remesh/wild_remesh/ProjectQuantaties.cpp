#include <polyfem/mesh/remesh/WildRemesh2D.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>

namespace polyfem::mesh
{
	void WildRemeshing2D::cache_before()
	{
		global_cache.rest_positions_before = rest_positions();
		global_cache.positions_before = positions();
		global_cache.projected_quantities_before = projected_quantities();
		global_cache.triangles_before = triangles();
	}

	void WildRemeshing2D::project_quantities()
	{
		// Assume the rest positions and triangles have been updated
		const Eigen::MatrixXd proposed_rest_positions = rest_positions();
		const Eigen::MatrixXi proposed_triangles = triangles();

		// --------------------------------------------------------------------

		// Assume isoparametric
		std::vector<polyfem::basis::ElementBases> bases_before;
		Eigen::VectorXi vertex_to_basis_before;
		int n_bases_before = build_bases(
			global_cache.rest_positions_before, global_cache.triangles_before,
			state.formulation(), bases_before, vertex_to_basis_before);
		const std::vector<polyfem::basis::ElementBases> &geom_bases_before = bases_before;
		n_bases_before += state.obstacle.n_vertices();

		// Old values of independent variables
		Eigen::MatrixXd y(n_bases_before * DIM, 1 + n_quantities);
		y.col(0) = utils::flatten(utils::reorder_matrix(
			global_cache.positions_before - global_cache.rest_positions_before,
			vertex_to_basis_before, n_bases_before));
		y.rightCols(n_quantities) = utils::reorder_matrix(
			global_cache.projected_quantities_before, vertex_to_basis_before,
			n_bases_before, DIM);

		// --------------------------------------------------------------------

		const int num_vertices = proposed_rest_positions.rows();

		std::vector<polyfem::basis::ElementBases> bases;
		Eigen::VectorXi vertex_to_basis;
		int n_bases = build_bases(
			proposed_rest_positions, proposed_triangles, state.formulation(),
			bases, vertex_to_basis);
		const std::vector<polyfem::basis::ElementBases> &geom_bases = bases;
		n_bases += state.obstacle.n_vertices();

		Eigen::MatrixXd target_x = Eigen::MatrixXd::Zero(n_bases, DIM);
		for (int i = 0; i < num_vertices; i++)
		{
			const int j = vertex_to_basis[i];
			if (j < 0)
				continue;
			target_x.row(j) = vertex_attrs[i].displacement().transpose();
		}
		target_x = utils::flatten(target_x);

		std::vector<int> boundary_nodes = this->boundary_nodes();
		for (int &boundary_node : boundary_nodes)
			boundary_node = vertex_to_basis[boundary_node];
		std::sort(boundary_nodes.begin(), boundary_nodes.end());

		// --------------------------------------------------------------------

		// L2 Projection
		Eigen::MatrixXd x;
		L2_projection(
			/*is_volume=*/DIM == 3, /*size=*/DIM,
			n_bases_before, bases_before, geom_bases_before, // from
			n_bases, bases, geom_bases,                      // to
			boundary_nodes, state.obstacle, target_x,
			y, x, /*lump_mass_matrix=*/false);

		// --------------------------------------------------------------------

		set_positions(proposed_rest_positions + utils::unreorder_matrix( //
						  utils::unflatten(x.col(0), DIM), vertex_to_basis, num_vertices));
		set_projected_quantities(utils::unreorder_matrix(
			x.rightCols(n_quantities), vertex_to_basis, num_vertices, DIM));

		write_rest_obj("proposed_rest_mesh.obj");
		write_deformed_obj("proposed_deformed_mesh.obj");
	}

} // namespace polyfem::mesh