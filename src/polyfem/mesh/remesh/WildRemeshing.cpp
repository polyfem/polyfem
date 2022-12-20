#include "WildRemeshing.hpp"

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/VTUWriter.hpp>

#include <igl/predicates/predicates.h>
#include <igl/boundary_facets.h>

namespace polyfem::mesh
{
	static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

	void WildRemeshing::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &elements,
		const Eigen::MatrixXd &projection_quantities,
		const EdgeMap<int> &edge_to_boundary_id,
		const std::vector<int> &body_ids)
	{
		assert(elements.size() > 0);
		// Only support triangles in 2D and tetrahedra in 3D
		assert(rest_positions.cols() == 2 && elements.cols() == 3
			   || rest_positions.cols() == 3 && elements.cols() == 4);

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
		create_mesh(positions.rows(), elements);

		// Save the vertex position in the vertex attributes
		set_rest_positions(rest_positions);
		set_positions(positions);
		set_projected_quantities(projection_quantities);
		set_fixed(is_boundary_vertex);
		set_boundary_ids(edge_to_boundary_id);
		set_body_ids(body_ids);
	}

	int WildRemeshing::build_bases(
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

	assembler::AssemblerUtils WildRemeshing::create_assembler(
		const std::vector<int> &body_ids) const
	{
		POLYFEM_SCOPED_TIMER(timings.create_assembler);
		assembler::AssemblerUtils new_assembler = state.assembler;
		assert(utils::is_param_valid(state.args, "materials"));
		new_assembler.set_materials(body_ids, state.args["materials"]);
		return new_assembler;
	}

	void WildRemeshing::cache_before()
	{
		global_cache.rest_positions_before = rest_positions();
		global_cache.positions_before = positions();
		global_cache.projected_quantities_before = projected_quantities();
		global_cache.elements_before = elements();
	}

	void WildRemeshing::write_mesh(const std::string &path, bool deformed) const
	{
		if (utils::StringUtils::endswith(path, ".obj"))
		{
			assert(dim() == 2); // OBJ does not support tets
			io::OBJWriter::write(path, deformed ? positions() : rest_positions(), elements());
		}
		else if (utils::StringUtils::endswith(path, ".vtu"))
		{
			io::VTUWriter writer;
			const Eigen::MatrixXd projected_quantities = this->projected_quantities();

			Eigen::MatrixXd rest_positions = this->rest_positions();
			Eigen::MatrixXd displacements = this->displacements();
			Eigen::MatrixXd prev_displacements = utils::unflatten(projected_quantities.leftCols(1), dim());
			Eigen::MatrixXd friction_gradient = utils::unflatten(projected_quantities.rightCols(1), dim());
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
				utils::append_rows(prev_displacements, obstacle_prev_displacement());
				utils::append_rows(friction_gradient, obstacle_friction_gradient());

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

} // namespace polyfem::mesh