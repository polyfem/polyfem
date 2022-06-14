#include <polyfem/State.hpp>

#include <polyfem/utils/FEBioReader.hpp>

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

	void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool non_conforming, bool skip_boundary_sideset)
	{
		bases.clear();
		pressure_bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		parent_elements.clear();
		obstacle.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		igl::Timer timer;
		timer.start();
		logger().info("Loading mesh...");
		mesh = Mesh::create(meshin, non_conforming);
		if (!mesh)
		{
			logger().error("Unable to load the mesh");
			return;
		}

		//TODO
		// if (args["normalize_mesh"])
		// {
		// 	mesh->normalize();
		// }
		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		if (min.size() == 2)
			logger().info("mesh bb min [{} {}], max [{} {}]", min(0), min(1), max(0), max(1));
		else
			logger().info("mesh bb min [{} {} {}], max [{} {} {}]", min(0), min(1), min(2), max(0), max(1), max(2));

		assembler.set_size(mesh->dimension());

		//TODO
		// int n_refs = args["n_refs"];

		// if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		// {
		// 	if (args["force_no_ref_for_harmonic"])
		// 		logger().warn("Using harmonic bases without refinement");
		// 	else
		// 		n_refs = 1;
		// }

		// if (n_refs > 0)
		// 	mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

		// if (!skip_boundary_sideset)
		// 	mesh->compute_boundary_ids(boundary_marker);
		// BoxSetter::set_sidesets(args, *mesh);
		// set_multimaterial([&](const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus, const Eigen::MatrixXd &rhos) {
		// 	assembler.init_multimaterial(mesh->is_volume(), Es, nus);
		// 	density.init_multimaterial(rhos);
		// });

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		timer.start();
		logger().info("Loading obstacles...");
		//TODO
		// if (args.contains("obstacles"))
		// 	obstacle.init(args["obstacles"], args["root_path"], mesh->dimension());
		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		ref_element_sampler.init(mesh->is_volume(), mesh->n_elements(), args["output"]["paraview"]["vismesh_rel_area"]);
	}

	void State::load_mesh(bool non_conforming,
						  const std::vector<std::string> &names,
						  const std::vector<Eigen::MatrixXi> &cells,
						  const std::vector<Eigen::MatrixXd> &vertices)
	{
		assert(names.size() == cells.size());
		assert(vertices.size() == cells.size());

		bases.clear();
		pressure_bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		parent_elements.clear();
		obstacle.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		igl::Timer timer;
		timer.start();

		if (mesh == nullptr)
		{

			// TODO fix me for hdf5
			// {
			// 	int index = -1;
			// 	for (int i = 0; i < names.size(); ++i)
			// 	{
			// 		if (names[i] == args["meshes"])
			// 		{
			// 			index = i;
			// 			break;
			// 		}
			// 	}

			// 	assert(index >= 0);

			// 	if (vertices[index].cols() == 2)
			// 		mesh = std::make_unique<polyfem::CMesh2D>();
			// 	else
			// 		mesh = std::make_unique<polyfem::Mesh3D>();
			// 	mesh->build_from_matrices(vertices[index], cells[index]);
			// }

			assert(is_param_valid(args, "geometry"));
			logger().info("Loading geometry ...");
			load_geometry(
				args["geometry"], args["root_path"], mesh, obstacle,
				names, vertices, cells, non_conforming);
		}
		else
		{
			logger().info("Loading mesh ...");
		}

		if (mesh == nullptr)
		{
			logger().error("unable to load the mesh!");
			return;
		}

		// if(!flipped_elements.empty())
		// {
		// 	mesh->compute_elements_tag();
		// 	for(auto el_id : flipped_elements)
		// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
		// }

		// TODO: fix me @zfergus
		// if (args["normalize_mesh"])
		// 	mesh->normalize();

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		if (min.size() == 2)
			logger().info("mesh bb min [{}, {}], max [{}, {}]", min(0), min(1), max(0), max(1));
		else
			logger().info("mesh bb min [{}, {}, {}], max [{}, {}, {}]", min(0), min(1), min(2), max(0), max(1), max(2));

		assembler.set_size(mesh->dimension());

		// TODO: fix me @zfergus
		// int n_refs = args["n_refs"];

		// if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		// {
		// 	if (args["force_no_ref_for_harmonic"])
		// 		logger().warn("Using harmonic bases without refinement");
		// 	else
		// 		n_refs = 1;
		// }

		// if (n_refs > 0)
		// 	mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

		// mesh->set_tag(1712, ElementType::InteriorPolytope);

		// TODO: fix me @zfergus
		const double boundary_id_threshold = mesh->is_volume() ? 1e-2 : 1e-7;
		mesh->compute_boundary_ids(boundary_id_threshold);

		// double boundary_id_threshold = args["boundary_id_threshold"];
		// if (boundary_id_threshold <= 0)
		// 	boundary_id_threshold = mesh->is_volume() ? 1e-2 : 1e-7;

		// if (!mesh->has_boundary_ids())
		// {
		// 	const std::string bc_tag_path = resolve_input_path(args["bc_tag"]);
		// 	if (bc_tag_path.empty())
		// 		mesh->compute_boundary_ids(boundary_id_threshold);
		// 	else
		// 		mesh->load_boundary_ids(bc_tag_path);
		// }
		// BoxSetter::set_sidesets(args, *mesh);
		// set_multimaterial([&](const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus, const Eigen::MatrixXd &rhos) {
		// 	assembler.init_multimaterial(mesh->is_volume(), Es, nus);
		// 	density.init_multimaterial(rhos);
		// });

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		ref_element_sampler.init(mesh->is_volume(), mesh->n_elements(), args["output"]["paraview"]["vismesh_rel_area"]);

		timer.start();
		// TODO: fix me @zfergus
		// logger().info("Loading obstacles...");
		// if (is_param_valid(args, "obstacles"))
		// 	obstacle.init(args["obstacles"], args["root_path"], mesh->dimension(), names, cells, vertices);
		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		// const double poly_percentage = 0.05;
		// const double poly_percentage = 0;
		// const double perturb_t = 0.3;

		// if(poly_percentage > 0)
		// {
		// 	const int n_poly = std::max(1., mesh->n_elements()*poly_percentage);
		// 	int counter = 0;
		// 	srand(11);

		// 	for(int trial = 0; trial < n_poly*10; ++trial)
		// 	{
		// 		int el_id = rand() % mesh->n_elements();

		// 		auto tags = mesh->elements_tag();

		// 		if(mesh->is_volume())
		// 		{
		// 			assert(false);
		// 		}
		// 		else
		// 		{
		// 			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
		// 			auto index = tmp_mesh.get_index_from_face(el_id);

		// 			bool stop = false;

		// 			for(int i = 0; i < tmp_mesh.n_face_vertices(el_id); ++i)
		// 			{
		// 				if(tmp_mesh.is_boundary_edge(index.edge))
		// 				{
		// 					stop = true;
		// 					break;
		// 				}

		// 				const auto neigh_index = tmp_mesh.switch_face(index);
		// 				if(tags[neigh_index.face] != ElementType::RegularInteriorCube)
		// 				{
		// 					stop = true;
		// 					break;
		// 				}

		// 				const auto f1 = tmp_mesh.switch_face(tmp_mesh.switch_edge(neigh_index						 )).face;
		// 				const auto f2 = tmp_mesh.switch_face(tmp_mesh.switch_edge(tmp_mesh.switch_vertex(neigh_index))).face;
		// 				if((f1 >= 0 && tags[f1] != ElementType::RegularInteriorCube) || (f2 >= 0 && tags[f2] != ElementType::RegularInteriorCube ))
		// 				{
		// 					stop = true;
		// 					break;
		// 				}

		// 				index = tmp_mesh.next_around_face(index);
		// 			}

		// 			if(stop) continue;
		// 		}

		// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
		// 		++counter;

		// 		mesh->update_elements_tag();

		// 		if(counter >= n_poly)
		// 			break;

		// 	}

		// 	if(perturb_t > 0)
		// 	{
		// 		if(mesh->is_volume())
		// 		{
		// 			assert(false);
		// 		}
		// 		else
		// 		{
		// 			Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
		// 			for(int el_id = 0; el_id < tmp_mesh.n_elements(); ++el_id)
		// 			{
		// 				if(!tmp_mesh.is_polytope(el_id))
		// 					continue;

		// 				const int rand_index = rand() % tmp_mesh.n_face_vertices(el_id);
		// 				auto index = tmp_mesh.get_index_from_face(el_id);
		// 				for(int r = 0; r < rand_index; ++r)
		// 					index = tmp_mesh.next_around_face(index);

		// 				const auto v1 = tmp_mesh.point(index.vertex);
		// 				const auto v2 = tmp_mesh.point(tmp_mesh.next_around_face(tmp_mesh.next_around_face(index)).vertex);

		// 				const double t = perturb_t + ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
		// 				const RowVectorNd v = t * v1 + (1-t) * v2;
		// 				tmp_mesh.set_point(index.vertex, v);
		// 			}
		// 		}
		// 	}
		// }
	}

	void State::load_febio(const std::string &path, const json &args_in)
	{
		FEBioReader::load(path, args_in, *this);

		// igl::Timer timer;
		// timer.start();
		// TODO: fix me @zfergus
		// logger().info("Loading obstacles...");
		// if (args.contains("obstacles"))
		// 	obstacle.init(args["obstacles"], args["root_path"], mesh->dimension());
		// timer.stop();
		// logger().info(" took {}s", timer.getElapsedTime());
	}

	void State::compute_mesh_stats()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		bases.clear();
		pressure_bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		simplex_count = 0;
		regular_count = 0;
		regular_boundary_count = 0;
		simple_singular_count = 0;
		multi_singular_count = 0;
		boundary_count = 0;
		non_regular_boundary_count = 0;
		non_regular_count = 0;
		undefined_count = 0;
		multi_singular_boundary_count = 0;

		const auto &els_tag = mesh->elements_tag();

		mesh->prepare_mesh();

		for (size_t i = 0; i < els_tag.size(); ++i)
		{
			const ElementType type = els_tag[i];

			switch (type)
			{
			case ElementType::Simplex:
				simplex_count++;
				break;
			case ElementType::RegularInteriorCube:
				regular_count++;
				break;
			case ElementType::RegularBoundaryCube:
				regular_boundary_count++;
				break;
			case ElementType::SimpleSingularInteriorCube:
				simple_singular_count++;
				break;
			case ElementType::MultiSingularInteriorCube:
				multi_singular_count++;
				break;
			case ElementType::SimpleSingularBoundaryCube:
				boundary_count++;
				break;
			case ElementType::InterfaceCube:
			case ElementType::MultiSingularBoundaryCube:
				multi_singular_boundary_count++;
				break;
			case ElementType::BoundaryPolytope:
				non_regular_boundary_count++;
				break;
			case ElementType::InteriorPolytope:
				non_regular_count++;
				break;
			case ElementType::Undefined:
				undefined_count++;
				break;
			}
		}

		logger().info("simplex_count: \t{}", simplex_count);
		logger().info("regular_count: \t{}", regular_count);
		logger().info("regular_boundary_count: \t{}", regular_boundary_count);
		logger().info("simple_singular_count: \t{}", simple_singular_count);
		logger().info("multi_singular_count: \t{}", multi_singular_count);
		logger().info("boundary_count: \t{}", boundary_count);
		logger().info("multi_singular_boundary_count: \t{}", multi_singular_boundary_count);
		logger().info("non_regular_count: \t{}", non_regular_count);
		logger().info("non_regular_boundary_count: \t{}", non_regular_boundary_count);
		logger().info("undefined_count: \t{}", undefined_count);
		logger().info("total count:\t {}", mesh->n_elements());
	}

} // namespace polyfem
