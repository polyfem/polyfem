#include <polyfem/State.hpp>

#include <polyfem/FEBioReader.hpp>

#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <polyfem/RefElementSampler.hpp>

#include <polyfem/BoxSetter.hpp>

#include <igl/Timer.h>
namespace polyfem
{

	void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool skip_boundary_sideset)
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

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		igl::Timer timer;
		timer.start();
		logger().info("Loading mesh...");
		mesh = Mesh::create(meshin);
		if (!mesh)
		{
			logger().error("Unable to load the mesh");
			return;
		}

		if (args["normalize_mesh"])
		{
			mesh->normalize();
		}
		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		if (min.size() == 2)
			logger().info("mesh bb min [{} {}], max [{} {}]", min(0), min(1), max(0), max(1));
		else
			logger().info("mesh bb min [{} {} {}], max [{} {} {}]", min(0), min(1), min(2), max(0), max(1), max(2));

		int n_refs = args["n_refs"];

		if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		{
			if (args["force_no_ref_for_harmonic"])
				logger().warn("Using harmonic bases without refinement");
			else
				n_refs = 1;
		}

		if (n_refs > 0)
			mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

		if (!skip_boundary_sideset)
			mesh->compute_boundary_ids(boundary_marker);
		BoxSetter::set_sidesets(args, *mesh);
		set_multimaterial([&](const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus, const Eigen::MatrixXd &rhos) {
			assembler.init_multimaterial(mesh->is_volume(), Es, nus);
			density.init_multimaterial(rhos);
		});

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements(), args["vismesh_rel_area"]);
	}

	void State::load_mesh()
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

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		igl::Timer timer;
		timer.start();
		logger().info("Loading mesh...");

		if (!mesh || !mesh_path().empty())
		{
			mesh = Mesh::create(mesh_path());
		}
		if (!mesh)
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

		if (args["normalize_mesh"])
			mesh->normalize();

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		if (min.size() == 2)
			logger().info("mesh bb min [{}, {}], max [{}, {}]", min(0), min(1), max(0), max(1));
		else
			logger().info("mesh bb min [{}, {}, {}], max [{}, {}, {}]", min(0), min(1), min(2), max(0), max(1), max(2));

		int n_refs = args["n_refs"];

		if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		{
			if (args["force_no_ref_for_harmonic"])
				logger().warn("Using harmonic bases without refinement");
			else
				n_refs = 1;
		}

		if (n_refs > 0)
			mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

		// mesh->set_tag(1712, ElementType::InteriorPolytope);

		const std::string bc_tag_path = args["bc_tag"];

		double boundary_id_threshold = args["boundary_id_threshold"];
		if (boundary_id_threshold <= 0)
			boundary_id_threshold = mesh->is_volume() ? 1e-2 : 1e-7;

		if (!mesh->has_boundary_ids())
		{
			if (bc_tag_path.empty())
				mesh->compute_boundary_ids(boundary_id_threshold);
			else
				mesh->load_boundary_ids(bc_tag_path);
		}
		BoxSetter::set_sidesets(args, *mesh);
		set_multimaterial([&](const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus, const Eigen::MatrixXd &rhos) {
			assembler.init_multimaterial(mesh->is_volume(), Es, nus);
			density.init_multimaterial(rhos);
		});

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements(), args["vismesh_rel_area"]);

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

	void State::load_febio(const std::string &path)
	{
		FEBioReader::load(path, *this);
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
