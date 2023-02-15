#include <polyfem/State.hpp>

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

		stiffness.resize(0, 0);
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

		// TODO: renable this
		// if (args["normalize_mesh"])
		// 	mesh->normalize();
		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		logger().info("mesh bb min [{}], max [{}]", min, max);

		assembler.set_size(formulation(), mesh->dimension());

		// TODO: renable this
		// int n_refs = args["n_refs"];
		// if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		// {
		// 	if (args["force_no_ref_for_harmonic"])
		// 		logger().warn("Using harmonic bases without refinement");
		// 	else
		// 		n_refs = 1;
		// }
		// if (n_refs > 0)
		// 	mesh->refine(n_refs, args["refinement_location"], parent_elements);

		if (!skip_boundary_sideset)
			mesh->compute_boundary_ids(boundary_marker);
		// TODO: renable this
		// BoxSetter::set_sidesets(args, *mesh);
		set_materials();

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		timer.start();
		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
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

		assembler.set_size(formulation(), mesh->dimension());
		set_materials();

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		out_geom.init_sampler(*mesh, args["output"]["paraview"]["vismesh_rel_area"]);

		timer.start();
		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
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
			assert(element.bases.size() == F.cols());
			for (int j = 0; j < element.bases.size(); j++)
			{
				const basis::Basis &basis = element.bases[j];
				assert(basis.global().size() == 1);
				V.row(basis.global()[0].index) = basis.global()[0].node;
				F(i, j) = basis.global()[0].index;
			}
		}
	}
} // namespace polyfem
