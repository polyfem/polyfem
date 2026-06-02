#include <polyfem/State.hpp>

#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/NonlinearElasticTransientVarForm.hpp>

#include <igl/Timer.h>
namespace polyfem
{
	using namespace basis;
	using namespace mesh;
	using namespace utils;

	void State::reset_mesh()
	{
	}

	void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &boundary_marker, bool non_conforming, bool skip_boundary_sideset)
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

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		assert(variational_formulation != nullptr);

		variational_formulation->load_mesh(*mesh, args);
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

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		assert(variational_formulation != nullptr);

		variational_formulation->load_mesh(*mesh, args);

#ifdef POLYFEM_WITH_BEZIER
		if (!mesh->is_simplicial())
#else
		if constexpr (true)
#endif
		{
			args["space"]["advanced"]["count_flipped_els_continuous"] = false;
			args["output"]["paraview"]["options"]["jacobian_validity"] = false;
			args["solver"]["advanced"]["check_inversion"] = "Discrete";
		}
		else if (args["solver"]["advanced"]["check_inversion"] != "Discrete")
		{
			args["space"]["advanced"]["use_corner_quadrature"] = true;
		}
	}

} // namespace polyfem
