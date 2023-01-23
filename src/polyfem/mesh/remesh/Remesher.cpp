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
#include <igl/edges.h>

namespace polyfem::mesh
{
	Remesher::Remesher(const State &state,
					   const Eigen::MatrixXd &obstacle_displacements,
					   const Eigen::MatrixXd &obstacle_quantities,
					   const double current_time,
					   const double starting_energy)
		: state(state),
		  args(state.args["space"]["remesh"]),
		  m_obstacle_displacements(obstacle_displacements),
		  m_obstacle_quantities(obstacle_quantities),
		  current_time(current_time),
		  starting_energy(starting_energy),
		  assembler(state.assembler)
	{
	}

	void Remesher::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &elements,
		const Eigen::MatrixXd &projection_quantities,
		const BoundaryMap<int> &boundary_to_id,
		const std::vector<int> &body_ids,
		const EdgeMap<double> &elastic_energy,
		const EdgeMap<double> &contact_energy)
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

	void Remesher::cache_before()
	{
		global_projection_cache.rest_positions = rest_positions();
		global_projection_cache.elements = elements();
		global_projection_cache.projection_quantities = projection_quantities();
	}

	void Remesher::project_quantities()
	{
		POLYFEM_REMESHER_SCOPED_TIMER("Project quantities");

		using namespace polyfem::assembler;
		using namespace polyfem::basis;
		using namespace polyfem::utils;

		std::vector<ElementBases> from_bases;
		Eigen::VectorXi from_vertex_to_basis;
		int n_from_basis = build_bases(
			global_projection_cache.rest_positions,
			global_projection_cache.elements,
			state.formulation(),
			from_bases,
			from_vertex_to_basis);

		// Old values of independent variables
		Eigen::MatrixXd from_projection_quantities = reorder_matrix(
			global_projection_cache.projection_quantities,
			from_vertex_to_basis, n_from_basis, dim());
		append_rows(from_projection_quantities, obstacle_quantities());
		n_from_basis += obstacle().n_vertices();
		assert(dim() * n_from_basis == from_projection_quantities.rows());

		// --------------------------------------------------------------------

		Eigen::MatrixXd rest_positions = this->rest_positions();
		Eigen::MatrixXi elements = this->elements();

		std::vector<ElementBases> to_bases;
		Eigen::VectorXi to_vertex_to_basis;
		int n_to_basis = build_bases(
			rest_positions, elements, state.formulation(),
			to_bases, to_vertex_to_basis);

		rest_positions = reorder_matrix(rest_positions, to_vertex_to_basis, n_to_basis);
		append_rows(rest_positions, obstacle().v());
		n_to_basis += obstacle().n_vertices();
		assert(rest_positions.rows() == n_to_basis);
		assert(rest_positions.cols() == dim());

		// Interpolated values of independent variables
		Eigen::MatrixXd to_projection_quantities = reorder_matrix(
			projection_quantities(), to_vertex_to_basis, n_to_basis, dim());
		append_rows(to_projection_quantities, obstacle_quantities());
		assert(dim() * n_to_basis == to_projection_quantities.rows());

		// --------------------------------------------------------------------

		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			MassMatrixAssembler assembler;
			Density no_density; // Density of one (i.e., no scaling of mass matrix)
			AssemblyValsCache cache;

			assembler.assemble(
				is_volume(), dim(),
				n_to_basis, no_density, to_bases, to_bases,
				cache, M);
			assert(M.rows() == to_projection_quantities.rows());

			// if (lump_mass_matrix)
			// 	M = lump_matrix(M);

			assembler.assemble_cross(
				is_volume(), dim(),
				n_from_basis, from_bases, from_bases,
				n_to_basis, to_bases, to_bases,
				cache, A);
			assert(A.rows() == to_projection_quantities.rows());
			assert(A.cols() == from_projection_quantities.rows());
		}

		// --------------------------------------------------------------------

		ipc::CollisionMesh collision_mesh;
		{
			Eigen::MatrixXd V_rest = this->rest_positions();
			utils::append_rows(V_rest, obstacle().v());
			collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				V_rest, boundary_edges(), boundary_faces());
		}

		assert(rest_positions.size() == to_projection_quantities.rows());

		// --------------------------------------------------------------------

		Eigen::MatrixXd projected_quantities(to_projection_quantities.rows(), n_quantities());
		const int n_constrained_quantaties = n_quantities() / 3;
		const int n_unconstrained_quantaties = n_quantities() - n_constrained_quantaties;

		const std::vector<int> boundary_nodes = this->boundary_nodes(to_vertex_to_basis);
		for (int i = 0; i < n_constrained_quantaties; ++i)
		{
			projected_quantities.col(i) = constrained_L2_projection(
				state.make_nl_solver<polyfem::solver::NLProblem>(),
				// L2 projection form
				M, A, /*y=*/from_projection_quantities.col(i),
				// Inversion-free form
				rest_positions, elements, dim(),
				// Contact form
				collision_mesh, state.args["contact"]["dhat"],
				state.solve_data.contact_form
					? state.solve_data.contact_form->barrier_stiffness()
					: 1.0,
				state.args["contact"]["use_convergent_formulation"],
				state.args["solver"]["contact"]["CCD"]["broad_phase"],
				state.args["solver"]["contact"]["CCD"]["tolerance"],
				state.args["solver"]["contact"]["CCD"]["max_iterations"],
				// Augmented lagrangian form
				boundary_nodes, obstacle(), to_projection_quantities.col(i),
				// Initial guess
				to_projection_quantities.col(i));
		}

		// NOTE: no need for to_projection_quantities.rightCols(n_unconstrained_quantaties)
		projected_quantities.rightCols(n_unconstrained_quantaties) = unconstrained_L2_projection(
			M, A, from_projection_quantities.rightCols(n_unconstrained_quantaties));

		// --------------------------------------------------------------------

		assert(projected_quantities.rows() == dim() * n_to_basis);
		set_projection_quantities(unreorder_matrix(
			projected_quantities, to_vertex_to_basis, to_vertex_to_basis.size(), dim()));
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

	assembler::AssemblerUtils &Remesher::init_assembler(
		const std::vector<int> &body_ids) const
	{
		POLYFEM_REMESHER_SCOPED_TIMER("Create assembler");
		// assembler::AssemblerUtils new_assembler = state.assembler;
		assert(utils::is_param_valid(state.args, "materials"));
		assembler.set_materials(body_ids, state.args["materials"]);
		return assembler;
	}

	void Remesher::write_mesh(const std::string &path) const
	{
		assert(utils::StringUtils::endswith(path, ".vtu"));

		io::VTUWriter writer;

		Eigen::MatrixXd rest_positions = this->rest_positions();
		Eigen::MatrixXd displacements = this->displacements();

		const Eigen::MatrixXd projection_quantities = this->projection_quantities();
		std::vector<Eigen::MatrixXd> unflattened_projection_quantities;
		for (const auto &q : projection_quantities.colwise())
			unflattened_projection_quantities.push_back(utils::unflatten(q, dim()));

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
			for (int i = 0; i < unflattened_projection_quantities.size(); ++i)
				utils::append_rows(
					unflattened_projection_quantities[i],
					utils::unflatten(obstacle_quantities().col(i), dim()));

			if (obstacle().n_edges() > 0)
			{
				const Eigen::MatrixXi obstacle_edges = obstacle().e().array() + offset;
				all_elements.resize(all_elements.size() + obstacle_edges.rows());
				for (int i = 0; i < obstacle().n_edges(); i++)
				{
					all_elements[i + elements.rows()] = std::vector<int>(obstacle_edges.cols());
					for (int j = 0; j < obstacle_edges.cols(); j++)
						all_elements[i + elements.rows()][j] = obstacle_edges(i, j);
				}
			}

			if (obstacle().n_faces() > 0)
			{
				const Eigen::MatrixXi obstacle_faces = obstacle().f().array() + offset;
				all_elements.resize(all_elements.size() + obstacle_faces.rows());
				for (int i = 0; i < obstacle().n_edges(); i++)
				{
					all_elements[i + elements.rows()] = std::vector<int>(obstacle_faces.cols());
					for (int j = 0; j < obstacle_faces.cols(); j++)
						all_elements[i + elements.rows()][j] = obstacle_faces(i, j);
				}
			}
		}

		for (int i = 0; i < unflattened_projection_quantities.size(); ++i)
			writer.add_field(fmt::format("projection_quantity_{:d}", i), unflattened_projection_quantities[i]);
		writer.add_field("displacement", displacements);
		writer.write_mesh(path, rest_positions, all_elements, /*is_simplicial=*/true);
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

	void Remesher::log_timings()
	{
		logger().debug("Total time: {:.3g}s", total_time);
		double sum = 0;
		for (const auto &[name, time] : timings)
		{
			logger().debug("{}: {:.3g}s {:.1f}%", name, time, time / total_time * 100);
			sum += time;
		}
		logger().debug("Miscellaneous: {:.3g}s {:.1f}%", total_time - sum, (total_time - sum) / total_time * 100);
		if (num_solves > 0)
			logger().debug("Avg. # DOF per solve: {}", total_ndofs / double(num_solves));
	}

	// Static members must be initialized in the source file:
	std::map<std::string, double> Remesher::timings;
	double Remesher::total_time = 0;
	size_t Remesher::num_solves = 0;
	size_t Remesher::total_ndofs = 0;

} // namespace polyfem::mesh