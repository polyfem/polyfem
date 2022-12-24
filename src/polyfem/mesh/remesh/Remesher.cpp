#include "Remesher.hpp"

#include <polyfem/mesh/remesh/L2Projection.hpp>

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/VTUWriter.hpp>

#include <igl/boundary_facets.h>

namespace polyfem::mesh
{
	Remesher::Remesher(const State &state,
					   const Eigen::MatrixXd &obstacle_displacements,
					   const Eigen::MatrixXd &obstacle_quantities,
					   const double current_time,
					   const double starting_energy)
		: state(state),
		  m_obstacle_displacements(obstacle_displacements),
		  m_obstacle_quantities(obstacle_quantities),
		  current_time(current_time),
		  starting_energy(starting_energy)
	{
	}

	void Remesher::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &elements,
		const Eigen::MatrixXd &projection_quantities,
		const BoundaryMap<int> &boundary_to_id,
		const std::vector<int> &body_ids)
	{
		assert(elements.size() > 0);

		// --------------------------------------------------------------------
		// Determine which vertices are on the boundary

		// Partition mesh by body_ids
		assert(body_ids.size() == elements.rows());
		std::unordered_map<int, std::vector<int>> body_elements;
		for (int i = 0; i < elements.rows(); ++i)
		{
			const int body_id = body_ids[i];
			if (body_elements.find(body_id) == body_elements.end())
				body_elements[body_id] = std::vector<int>();
			body_elements[body_id].push_back(i);
		}

		// Determine boundary vertices
		std::vector<bool> is_boundary_vertex(positions.rows(), false);
		for (const auto &[body_id, rows] : body_elements)
		{
			Eigen::MatrixXi boundary_facets;
			igl::boundary_facets(elements(rows, Eigen::all), boundary_facets);

			for (int i = 0; i < boundary_facets.rows(); ++i)
				for (int j = 0; j < boundary_facets.cols(); ++j)
					is_boundary_vertex[boundary_facets(i, j)] = true;
		}

		// --------------------------------------------------------------------

		// Initialize the mesh atributes and connectivity
		init_attributes_and_connectivity(positions.rows(), elements);

		// Save the vertex position in the vertex attributes
		set_rest_positions(rest_positions);
		set_positions(positions);
		set_projection_quantities(projection_quantities);
		set_fixed(is_boundary_vertex);
		set_boundary_ids(boundary_to_id);
		set_body_ids(body_ids);
	}

	void Remesher::project_quantities()
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
			global_cache.projection_quantities_before, vertex_to_basis_before,
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
		set_projection_quantities(utils::unreorder_matrix(
			x.rightCols(n_quantities()), vertex_to_basis, num_vertices, dim()));
	}

	int Remesher::build_bases(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::string &assembler_formulation,
		std::vector<polyfem::basis::ElementBases> &bases,
		Eigen::VectorXi &vertex_to_basis)
	{
		using namespace polyfem::basis;
		const int dim = V.cols();

		int n_bases;
		std::vector<LocalBoundary> local_boundary;
		std::map<int, basis::InterfaceData> poly_edge_to_data;
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;
		if (dim == 2)
		{
			CMesh2D mesh;
			mesh.build_from_matrices(V, F);
			n_bases = LagrangeBasis2d::build_bases(
				mesh, assembler_formulation, /*quadrature_order=*/1,
				/*mass_quadrature_order=*/2, /*discr_order=*/1,
				/*serendipity=*/false, /*has_polys=*/false,
				/*is_geom_bases=*/false, bases, local_boundary,
				poly_edge_to_data, mesh_nodes);
		}
		else
		{
			assert(dim == 3);
			CMesh3D mesh;
			mesh.build_from_matrices(V, F);
			n_bases = LagrangeBasis3d::build_bases(
				mesh, assembler_formulation, /*quadrature_order=*/1,
				/*mass_quadrature_order=*/2, /*discr_order=*/1,
				/*serendipity=*/false, /*has_polys=*/false,
				/*is_geom_bases=*/false, bases, local_boundary,
				poly_edge_to_data, mesh_nodes);
		}

		// TODO: use mesh_nodes to build vertex_to_basis
		vertex_to_basis.setConstant(V.rows(), -1);
		for (const ElementBases &elm : bases)
		{
			for (const Basis &basis : elm.bases)
			{
				assert(basis.global().size() == 1);
				const int basis_id = basis.global()[0].index;
				const RowVectorNd v = basis.global()[0].node;

				for (int i = 0; i < V.rows(); i++)
				{
					if ((V.row(i).array() == v.array()).all())
					{
						if (vertex_to_basis[i] == -1)
							vertex_to_basis[i] = basis_id;
						assert(vertex_to_basis[i] == basis_id);
						break;
					}
				}
			}
		}

		return n_bases;
	}

	assembler::AssemblerUtils Remesher::create_assembler(
		const std::vector<int> &body_ids) const
	{
		POLYFEM_SCOPED_TIMER(timings.create_assembler);
		assembler::AssemblerUtils new_assembler = state.assembler;
		assert(utils::is_param_valid(state.args, "materials"));
		new_assembler.set_materials(body_ids, state.args["materials"]);
		return new_assembler;
	}

	void Remesher::cache_before()
	{
		global_cache.rest_positions_before = rest_positions();
		global_cache.positions_before = positions();
		global_cache.projection_quantities_before = projection_quantities();
		global_cache.elements_before = elements();
	}

	void Remesher::write_mesh(const std::string &path, bool deformed) const
	{
		if (utils::StringUtils::endswith(path, ".obj"))
		{
			assert(dim() == 2); // OBJ does not support tets
			io::OBJWriter::write(path, deformed ? positions() : rest_positions(), elements());
		}
		else if (utils::StringUtils::endswith(path, ".vtu"))
		{
			io::VTUWriter writer;
			const Eigen::MatrixXd projection_quantities = this->projection_quantities();

			Eigen::MatrixXd rest_positions = this->rest_positions();
			Eigen::MatrixXd displacements = this->displacements();
			Eigen::MatrixXd prev_displacements = utils::unflatten(projection_quantities.leftCols(1), dim());
			Eigen::MatrixXd friction_gradient = utils::unflatten(projection_quantities.rightCols(1), dim());
			Eigen::MatrixXi elements = this->elements();
			std::vector<std::vector<int>> all_elements(elements.rows(), std::vector<int>(elements.cols()));
			for (int i = 0; i < elements.rows(); i++)
				for (int j = 0; j < elements.cols(); j++)
					all_elements[i][j] = elements(i, j);

			const int offset = rest_positions.rows();
			if (obstacle().n_vertices() > 0)
			{
				utils::append_rows(rest_positions, obstacle().v());
				utils::append_rows(displacements, obstacle_displacements());
				utils::append_rows(prev_displacements, utils::unflatten(obstacle_quantities().leftCols(1), dim()));
				utils::append_rows(friction_gradient, utils::unflatten(obstacle_quantities().leftCols(1), dim()));

				const Eigen::MatrixXi obstacle_edges = obstacle().e().array() + offset;
				all_elements.resize(all_elements.size() + obstacle().n_edges());
				for (int i = 0; i < obstacle().n_edges(); i++)
				{
					all_elements[i + elements.rows()] = std::vector<int>(2);
					for (int j = 0; j < 2; j++)
						all_elements[i + elements.rows()][j] = obstacle_edges(i, j);
				}

				// TODO: add obstacle triangles
			}

			writer.add_field("prev_displacement", prev_displacements);
			writer.add_field("friction_gradient", friction_gradient);
			writer.add_field("displacement", displacements);
			writer.write_mesh(path, rest_positions, all_elements, /*is_simplicial=*/true);
		}
		else
		{
			log_and_throw_error("Unsupported mesh file format: {}", path);
		}
	}

	Eigen::MatrixXd Remesher::combine_time_integrator_quantities(
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> &time_integrator)
	{
		if (time_integrator == nullptr)
			return Eigen::MatrixXd();

		// not including current displacement as this will be handled as positions
		Eigen::MatrixXd projection_quantities(
			time_integrator->x_prev().size(), 3 * time_integrator->steps());
		int i = 0;
		for (const Eigen::VectorXd &x : time_integrator->x_prevs())
			projection_quantities.col(i++) = x;
		for (const Eigen::VectorXd &v : time_integrator->v_prevs())
			projection_quantities.col(i++) = v;
		for (const Eigen::VectorXd &a : time_integrator->a_prevs())
			projection_quantities.col(i++) = a;
		assert(i == projection_quantities.cols());

		return projection_quantities;
	}

	void Remesher::split_time_integrator_quantities(
		const Eigen::MatrixXd &quantities,
		const int dim,
		std::vector<Eigen::VectorXd> &x_prevs,
		std::vector<Eigen::VectorXd> &v_prevs,
		std::vector<Eigen::VectorXd> &a_prevs)
	{
		if (quantities.size() == 0)
			return;

		const int ndof = quantities.rows();
		const int n_vertices = ndof / dim;
		assert(ndof % dim == 0);

		const std::array<std::vector<Eigen::VectorXd> *, 3> all_prevs{{&x_prevs, &v_prevs, &a_prevs}};
		const int n_steps = quantities.cols() / 3;
		assert(quantities.cols() % 3 == 0);

		int offset = 0;
		for (std::vector<Eigen::VectorXd> *prevs : all_prevs)
		{
			prevs->clear();
			for (int i = 0; i < n_steps; ++i)
				prevs->push_back(quantities.col(offset + i));
			offset += n_steps;
		}

		assert(offset == quantities.cols());
	}

} // namespace polyfem::mesh