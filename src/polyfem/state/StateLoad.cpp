#include <polyfem/State.hpp>

#include <polyfem/assembler/Mass.hpp>

#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/utils/Selection.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <igl/Timer.h>
namespace polyfem
{
	using namespace basis;
	using namespace mesh;
	using namespace utils;

	void State::reset_mesh()
	{
		bases.clear();
		pressure_bases.clear();
		geom_bases_.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		obstacle.clear();

		mass.resize(0, 0);
		rhs.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;
	}

	void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool non_conforming, bool skip_boundary_sideset)
	{
		reset_mesh();

		igl::Timer timer;
		timer.start();
		logger().info("Loading mesh...");
		mesh = Mesh::create(meshin, non_conforming);
		if (!mesh)
		{
			logger().error("Unable to load the mesh");
			return;
		}

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		logger().info("mesh bb min [{}], max [{}]", min, max);

		if (!skip_boundary_sideset)
			mesh->compute_boundary_ids(boundary_marker);

		std::vector<std::shared_ptr<assembler::Assembler>> assemblers;
		assemblers.push_back(assembler);
		assemblers.push_back(mass_matrix_assembler);
		if (mixed_assembler != nullptr)
			// TODO: assemblers.push_back(mixed_assembler);
			mixed_assembler->set_size(mesh->dimension());
		if (pressure_assembler != nullptr)
			assemblers.push_back(pressure_assembler);
		set_materials(assemblers);

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		timer.start();
		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
			units,
			args["geometry"],
			utils::json_as_array(args["boundary_conditions"]["obstacle_displacements"]),
			utils::json_as_array(args["boundary_conditions"]["dirichlet_boundary"]),
			args["root_path"], mesh->dimension());

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		out_geom.init_sampler(*mesh, args["output"]["paraview"]["vismesh_rel_area"]);
	}

	void State::load_mesh(bool non_conforming,
						  const std::vector<std::string> &names,
						  const std::vector<Eigen::MatrixXi> &cells,
						  const std::vector<Eigen::MatrixXd> &vertices)
	{
		assert(names.size() == cells.size());
		assert(vertices.size() == cells.size());

		reset_mesh();

		igl::Timer timer;
		timer.start();

		logger().info("Loading mesh ...");
		if (mesh == nullptr)
		{
			assert(is_param_valid(args, "geometry"));
			mesh = mesh::read_fem_geometry(
				units,
				args["geometry"], args["root_path"],
				names, vertices, cells, non_conforming);
		}

		if (mesh == nullptr)
		{
			log_and_throw_error("unable to load the mesh!");
		}

		// if(!flipped_elements.empty())
		// {
		// 	mesh->compute_elements_tag();
		// 	for(auto el_id : flipped_elements)
		// 		mesh->set_tag(el_id, ElementType::INTERIOR_POLYTOPE);
		// }

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		logger().info("mesh bb min [{}], max [{}]", min, max);

		std::vector<std::shared_ptr<assembler::Assembler>> assemblers;
		assemblers.push_back(assembler);
		assemblers.push_back(mass_matrix_assembler);
		if (mixed_assembler != nullptr)
			// TODO: assemblers.push_back(mixed_assembler);
			mixed_assembler->set_size(mesh->dimension());
		if (pressure_assembler != nullptr)
			assemblers.push_back(pressure_assembler);
		set_materials(assemblers);

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		out_geom.init_sampler(*mesh, args["output"]["paraview"]["vismesh_rel_area"]);

		timer.start();
		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
			units,
			args["geometry"],
			utils::json_as_array(args["boundary_conditions"]["obstacle_displacements"]),
			utils::json_as_array(args["boundary_conditions"]["dirichlet_boundary"]),
			args["root_path"], mesh->dimension(), names, vertices, cells);
		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());
	}

	void State::build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F)
	{
		assert(bases.size() == mesh->n_elements());
		const size_t n_vertices = n_bases - obstacle.n_vertices();
		const int dim = mesh->dimension();

		V.resize(n_vertices, dim);
		F.resize(bases.size(), dim + 1); // TODO: this only works for triangles and tetrahedra

		for (int i = 0; i < bases.size(); i++)
		{
			const basis::ElementBases &element = bases[i];
			for (int j = 0; j < element.bases.size(); j++)
			{
				const basis::Basis &basis = element.bases[j];
				assert(basis.global().size() == 1);
				V.row(basis.global()[0].index) = basis.global()[0].node;
				if (j < F.cols()) // Only grab the corners of the triangles/tetrahedra
					F(i, j) = basis.global()[0].index;
			}
		}
	}
} // namespace polyfem
