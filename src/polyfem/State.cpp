#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/basis/FEBasis2d.hpp>
#include <polyfem/basis/FEBasis3d.hpp>

#include <polyfem/basis/SpectralBasis2d.hpp>

#include <polyfem/basis/SplineBasis2d.hpp>
#include <polyfem/basis/SplineBasis3d.hpp>

#include <polyfem/basis/MVPolygonalBasis2d.hpp>

#include <polyfem/basis/PolygonalBasis2d.hpp>
#include <polyfem/basis/PolygonalBasis3d.hpp>

#include <polyfem/utils/EdgeSampler.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/quadrature/HexQuadrature.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <igl/Timer.h>

#include <BVH.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <algorithm>
#include <memory>

#include <polyfem/utils/autodiff.h>
DECLARE_DIFFSCALAR_BASE();

using namespace Eigen;

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace utils;

	std::string State::formulation() const
	{
		assert(!args["materials"].is_null());
		if (args["materials"].is_null())
		{
			logger().error("specify some 'materials'");
			throw "invalid input";
		}

		if (args["materials"].is_array())
		{
			std::string current = "";
			for (const auto &m : args["materials"])
			{
				const std::string tmp = m["type"];
				if (current.empty())
					current = tmp;
				else if (current != tmp)
				{
					if (current == "LinearElasticity" || current == "NeoHookean" || current == "MultiModels")
					{
						if (tmp == "LinearElasticity" || tmp == "NeoHookean")
							current = "MultiModels";
						else
						{
							logger().error("Current material is {}, new material is {}, multimaterial supported only for LinearElasticity and NeoHookean", current, tmp);
							throw "invalid input";
						}
					}
				}
			}

			return current;
		}
		else
			return args["materials"]["type"];
	}

	void State::sol_to_pressure()
	{
		if (n_pressure_bases <= 0)
		{
			logger().error("No pressure bases defined!");
			return;
		}

		// assert(problem->is_mixed());
		assert(AssemblerUtils::is_mixed(formulation()));
		Eigen::MatrixXd tmp = sol;

		int fluid_offset = use_avg_pressure ? (AssemblerUtils::is_fluid(formulation()) ? 1 : 0) : 0;
		sol = tmp.block(0, 0, tmp.rows() - n_pressure_bases - fluid_offset, tmp.cols());
		assert(sol.size() == n_bases * (problem->is_scalar() ? 1 : mesh->dimension()));
		pressure = tmp.block(tmp.rows() - n_pressure_bases - fluid_offset, 0, n_pressure_bases, tmp.cols());
		assert(pressure.size() == n_pressure_bases);
	}

	void State::compute_mesh_size(const Mesh &mesh_in, const std::vector<ElementBases> &bases_in, const int n_samples)
	{
		Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1, p;

		mesh_size = 0;
		average_edge_length = 0;
		min_edge_length = std::numeric_limits<double>::max();

		if (!args["output"]["advanced"]["curved_mesh_size"])
		{
			mesh_in.get_edges(p0, p1);
			p = p0 - p1;
			min_edge_length = p.rowwise().norm().minCoeff();
			average_edge_length = p.rowwise().norm().mean();
			mesh_size = p.rowwise().norm().maxCoeff();

			logger().info("hmin: {}", min_edge_length);
			logger().info("hmax: {}", mesh_size);
			logger().info("havg: {}", average_edge_length);

			return;
		}

		if (mesh_in.is_volume())
		{
			EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_3d_cube(n_samples, samples_cube);
		}
		else
		{
			EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_2d_cube(n_samples, samples_cube);
		}

		int n = 0;
		for (size_t i = 0; i < bases_in.size(); ++i)
		{
			if (mesh_in.is_polytope(i))
				continue;
			int n_edges;

			if (mesh_in.is_simplex(i))
			{
				n_edges = mesh_in.is_volume() ? 6 : 3;
				bases_in[i].eval_geom_mapping(samples_simplex, mapped);
			}
			else
			{
				n_edges = mesh_in.is_volume() ? 12 : 4;
				bases_in[i].eval_geom_mapping(samples_cube, mapped);
			}

			for (int j = 0; j < n_edges; ++j)
			{
				double current_edge = 0;
				for (int k = 0; k < n_samples - 1; ++k)
				{
					p0 = mapped.row(j * n_samples + k);
					p1 = mapped.row(j * n_samples + k + 1);
					p = p0 - p1;

					current_edge += p.norm();
				}

				mesh_size = std::max(current_edge, mesh_size);
				min_edge_length = std::min(current_edge, min_edge_length);
				average_edge_length += current_edge;
				++n;
			}
		}

		average_edge_length /= n;

		logger().info("hmin: {}", min_edge_length);
		logger().info("hmax: {}", mesh_size);
		logger().info("havg: {}", average_edge_length);
	}

	void State::set_multimaterial(const std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &)> &setter)
	{
		if (!is_param_valid(args, "materials"))
			return;

		const auto &body_params = args["materials"];

		if (!body_params.is_array())
			return;

		//FIXME with the new stuff
		const json default_material = R"({
			"id": -1,
			"E": 100,
			"nu": 0.3,
			"rho": 1,
			"density": 1,
			"type": null
		})"_json;

		Eigen::MatrixXd Es(mesh->n_elements(), 1), nus(mesh->n_elements(), 1), rhos(mesh->n_elements(), 1);
		Es.setConstant(default_material["E"].get<double>());
		nus.setConstant(default_material["nu"].get<double>());
		rhos.setConstant(default_material["density"].get<double>());

		std::map<int, std::tuple<double, double, double>> materials;
		for (int i = 0; i < body_params.size(); ++i)
		{
			//TODO fix and check me
			check_for_unknown_args(default_material, body_params[i], fmt::format("/material[{}]", i));
			json mat = default_material;
			mat.merge_patch(body_params[i]);

			const int mid = mat["id"];
			Density d;
			d.init(mat);

			const double rho = d(0, 0, 0, 0, 0, 0, 0);
			const double E = mat["E"];
			const double nu = mat["nu"];

			materials[mid] = std::tuple<double, double, double>(E, nu, rho);
		}

		std::string missing = "";

		for (int e = 0; e < mesh->n_elements(); ++e)
		{
			const int bid = mesh->get_body_id(e);
			const auto it = materials.find(bid);
			if (it == materials.end())
			{
				missing += fmt::format("{:d}, ", bid);
				continue;
			}

			Es(e) = std::get<0>(it->second);
			nus(e) = std::get<1>(it->second);
			rhos(e) = std::get<2>(it->second);
			// std::cout << e << " " << Es(e) << " " << nus(e) << std::endl;
		}

		setter(Es, nus, rhos);
		if (missing.size() > 0)
			logger().warn("Missing parameters for {}", missing);
	}

	void compute_integral_constraints(
		const Mesh3D &mesh,
		const int n_bases,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		Eigen::MatrixXd &basis_integrals)
	{
		if (!mesh.is_volume())
		{
			logger().error("Works only on volumetric meshes!");
			return;
		}
		assert(mesh.is_volume());

		basis_integrals.resize(n_bases, 9);
		basis_integrals.setZero();
		Eigen::MatrixXd rhs(n_bases, 9);
		rhs.setZero();

		const int n_elements = mesh.n_elements();
		for (int e = 0; e < n_elements; ++e)
		{
			// if (mesh.is_polytope(e)) {
			// 	continue;
			// }
			// ElementAssemblyValues vals = values[e];
			// const ElementAssemblyValues &gvals = gvalues[e];
			ElementAssemblyValues vals;
			vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

			// Computes the discretized integral of the PDE over the element
			const int n_local_bases = int(vals.basis_values.size());
			for (int j = 0; j < n_local_bases; ++j)
			{
				const AssemblyValues &v = vals.basis_values[j];
				const double integral_100 = (v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_010 = (v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_001 = (v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

				const double integral_110 = ((vals.val.col(1).array() * v.grad_t_m.col(0).array() + vals.val.col(0).array() * v.grad_t_m.col(1).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_011 = ((vals.val.col(2).array() * v.grad_t_m.col(1).array() + vals.val.col(1).array() * v.grad_t_m.col(2).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_101 = ((vals.val.col(0).array() * v.grad_t_m.col(2).array() + vals.val.col(2).array() * v.grad_t_m.col(0).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();

				const double integral_200 = 2 * (vals.val.col(0).array() * v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_020 = 2 * (vals.val.col(1).array() * v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_002 = 2 * (vals.val.col(2).array() * v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

				const double area = (v.val.array() * vals.det.array() * vals.quadrature.weights.array()).sum();

				for (size_t ii = 0; ii < v.global.size(); ++ii)
				{
					basis_integrals(v.global[ii].index, 0) += integral_100 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 1) += integral_010 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 2) += integral_001 * v.global[ii].val;

					basis_integrals(v.global[ii].index, 3) += integral_110 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 4) += integral_011 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 5) += integral_101 * v.global[ii].val;

					basis_integrals(v.global[ii].index, 6) += integral_200 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 7) += integral_020 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 8) += integral_002 * v.global[ii].val;

					rhs(v.global[ii].index, 6) += -2.0 * area * v.global[ii].val;
					rhs(v.global[ii].index, 7) += -2.0 * area * v.global[ii].val;
					rhs(v.global[ii].index, 8) += -2.0 * area * v.global[ii].val;
				}
			}
		}

		basis_integrals -= rhs;
	}

	void State::build_basis()
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
		total_local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		sigma_avg = 0;
		sigma_max = 0;
		sigma_min = 0;

		disc_orders.resize(mesh->n_elements());

		if (!args["materials"].is_null() && !args["materials"].is_array())
		{
			assembler.set_parameters(args["materials"]);
			density.init(args["materials"]);
		}
		problem->init(*mesh);

		logger().info("Building {} basis...", (iso_parametric() ? "isoparametric" : "not isoparametric"));
		const bool has_polys = non_regular_count > 0 || non_regular_boundary_count > 0 || undefined_count > 0;

		local_boundary.clear();
		local_neumann_boundary.clear();
		std::map<int, InterfaceData> poly_edge_to_data_geom; // temp dummy variable

		const auto &tmp_json = args["space"]["discr_order"];
		if (tmp_json.is_number_integer())
			disc_orders.setConstant(tmp_json);
		else if (tmp_json.is_string())
		{
			const std::string discr_orders_path = tmp_json;
			Eigen::MatrixXi tmp;
			read_matrix(discr_orders_path, tmp);
			assert(tmp.size() == disc_orders.size());
			assert(tmp.cols() == 1);
			disc_orders = tmp;
		}
		else if (tmp_json.is_array())
		{
			const auto b_discr_orders = tmp_json;

			std::map<int, int> b_orders;
			for (size_t i = 0; i < b_discr_orders.size(); ++i)
			{
				b_orders[b_discr_orders[i]["id"]] = b_discr_orders[i]["order"];
				logger().trace("bid {}, discr {}", b_discr_orders[i]["id"], b_discr_orders[i]["order"]);
			}

			for (int e = 0; e < mesh->n_elements(); ++e)
			{
				const int bid = mesh->get_body_id(e);
				disc_orders[e] = b_orders.at(bid);
			}
		}
		else
		{
			logger().error("space/discr_order must be either a number a path or an array");
			throw "invalid json";
		}
		//TODO same for pressure!

		Eigen::MatrixXi geom_disc_orders;
		if (!iso_parametric())
		{
			if (mesh->orders().size() <= 0)
			{
				geom_disc_orders.resizeLike(disc_orders);
				geom_disc_orders.setConstant(1);
			}
			else
				geom_disc_orders = mesh->orders();
		}

		igl::Timer timer;
		timer.start();
		if (args["space"]["use_p_ref"])
		{
			if (mesh->is_volume())
				p_refinement(*dynamic_cast<Mesh3D *>(mesh.get()));
			else
				p_refinement(*dynamic_cast<Mesh2D *>(mesh.get()));

			logger().info("min p: {} max p: {}", disc_orders.minCoeff(), disc_orders.maxCoeff());
		}

		if (mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
			if (args["space"]["advanced"]["use_spline"])
			{
				// if (!iso_parametric())
				// {
				// 	logger().error("Splines must be isoparametric, ignoring...");
				// 	// FEBasis3d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], geom_disc_orders, has_polys, geom_bases, local_boundary, poly_edge_to_data_geom, primitive_to_node);
				// 	SplineBasis3d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], geom_bases, local_boundary, poly_edge_to_data);
				// }

				n_bases = SplineBasis3d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				// if (iso_parametric() && args["fit_nodes"])
				// 	SplineBasis3d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					FEBasis3d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], geom_disc_orders, false, has_polys, true, geom_bases, local_boundary, poly_edge_to_data_geom, primitive_to_node);

				n_bases = FEBasis3d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], disc_orders, args["space"]["advanced"]["serendipity"], has_polys, false, bases, local_boundary, poly_edge_to_data, primitive_to_node);
			}

			// if(problem->is_mixed())
			if (assembler.is_mixed(formulation()))
			{
				n_pressure_bases = FEBasis3d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], int(args["space"]["pressure_discr_order"]), false, has_polys, false, pressure_bases, local_boundary, poly_edge_to_data_geom, primitive_to_node);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			if (args["space"]["advanced"]["use_spline"])
			{
				//TODO?
				// if (!iso_parametric())
				// {
				// 	logger().error("Splines must be isoparametric, ignoring...");
				// 	// FEBasis2d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], disc_orders, has_polys, geom_bases, local_boundary, poly_edge_to_data_geom, primitive_to_node);
				// 	n_bases = SplineBasis2d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], geom_bases, local_boundary, poly_edge_to_data);
				// }

				n_bases = SplineBasis2d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				// if (iso_parametric() && args["fit_nodes"])
				// 	SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					FEBasis2d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], geom_disc_orders, false, has_polys, true, geom_bases, local_boundary, poly_edge_to_data_geom, primitive_to_node);

				n_bases = FEBasis2d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], disc_orders, args["space"]["advanced"]["serendipity"], has_polys, false, bases, local_boundary, poly_edge_to_data, primitive_to_node);
			}

			// if(problem->is_mixed())
			if (assembler.is_mixed(formulation()))
			{
				n_pressure_bases = FEBasis2d::build_bases(tmp_mesh, args["space"]["advanced"]["quadrature_order"], int(args["space"]["pressure_discr_order"]), false, has_polys, false, pressure_bases, local_boundary, poly_edge_to_data_geom, primitive_to_node);
			}
		}
		timer.stop();

		build_polygonal_basis();

		auto &gbases = iso_parametric() ? bases : geom_bases;

		for (const auto &lb : local_boundary)
			total_local_boundary.emplace_back(lb);

		n_flipped = 0;

		if (args["space"]["advanced"]["count_flipped_els"])
		{
			logger().info("Counting flipped elements...");
			const auto &els_tag = mesh->elements_tag();

			// flipped_elements.clear();
			for (size_t i = 0; i < gbases.size(); ++i)
			{
				if (mesh->is_polytope(i))
					continue;

				ElementAssemblyValues vals;
				if (!vals.is_geom_mapping_positive(mesh->is_volume(), gbases[i]))
				{
					++n_flipped;

					std::string type = "";
					switch (els_tag[i])
					{
					case ElementType::Simplex:
						type = "Simplex";
						break;
					case ElementType::RegularInteriorCube:
						type = "RegularInteriorCube";
						break;
					case ElementType::RegularBoundaryCube:
						type = "RegularBoundaryCube";
						break;
					case ElementType::SimpleSingularInteriorCube:
						type = "SimpleSingularInteriorCube";
						break;
					case ElementType::MultiSingularInteriorCube:
						type = "MultiSingularInteriorCube";
						break;
					case ElementType::SimpleSingularBoundaryCube:
						type = "SimpleSingularBoundaryCube";
						break;
					case ElementType::InterfaceCube:
						type = "InterfaceCube";
						break;
					case ElementType::MultiSingularBoundaryCube:
						type = "MultiSingularBoundaryCube";
						break;
					case ElementType::BoundaryPolytope:
						type = "BoundaryPolytope";
						break;
					case ElementType::InteriorPolytope:
						type = "InteriorPolytope";
						break;
					case ElementType::Undefined:
						type = "Undefined";
						break;
					}

					logger().error("element {} is flipped, type {}", i, type);
					throw "invalid mesh";

					// if(!parent_elements.empty())
					// 	flipped_elements.push_back(parent_elements[i]);
				}
			}

			logger().info(" done");
		}

		// dynamic_cast<Mesh3D *>(mesh.get())->save({56}, 1, "mesh.HYBRID");

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));

		const int prev_bases = n_bases;
		n_bases += obstacle.n_vertices();

		logger().info("Extracting boundary mesh...");
		build_collision_mesh();
		if (n_pressure_bases > 0)
		{
			extract_boundary_mesh(
				pressure_bases,
				boundary_nodes_pos_pressure,
				boundary_edges_pressure,
				boundary_triangles_pressure);
		}
		// const std::string export_surface = args["export"]["surface"];
		// if (!export_surface.empty())
		extract_vis_boundary_mesh();
		logger().info("Done!");

		const int prev_b_size = local_boundary.size();
		problem->setup_bc(*mesh, bases, pressure_bases, local_boundary, boundary_nodes, local_neumann_boundary, pressure_boundary_nodes);
		const bool has_neumann = local_neumann_boundary.size() > 0 || local_boundary.size() < prev_b_size;
		use_avg_pressure = !has_neumann;
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();

		for (int i = prev_bases; i < n_bases; ++i)
		{
			for (int d = 0; d < problem_dim; ++d)
				boundary_nodes.push_back(i * problem_dim + d);
		}

		const auto &curret_bases = iso_parametric() ? bases : geom_bases;
		const int n_samples = 10;
		compute_mesh_size(*mesh, curret_bases, n_samples);

		if (args["contact"]["enabled"])
		{
			if (!has_dhat && args["contact"]["dhat"] > min_edge_length)
			{
				args["contact"]["dhat"] = double(args["contact"]["dhat_percentage"]) * min_edge_length;
				logger().info("dhat set to {}", double(args["contact"]["dhat"]));
			}
			else
			{
				if (args["contact"]["dhat"] > min_edge_length)
					logger().warn("dhat larger than min edge, {} > {}", double(args["contact"]["dhat"]), min_edge_length);
			}
		}

		logger().info("n_bases {}", n_bases);

		building_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", building_basis_time);

		logger().info("flipped elements {}", n_flipped);
		logger().info("h: {}", mesh_size);
		logger().info("n bases: {}", n_bases);
		logger().info("n pressure bases: {}", n_pressure_bases);

		ass_vals_cache.clear();
		if (n_bases <= args["solver"]["advanced"]["cache_size"])
		{
			timer.start();
			logger().info("Building cache...");
			ass_vals_cache.init(mesh->is_volume(), bases, curret_bases);
			if (assembler.is_mixed(formulation()))
				pressure_ass_vals_cache.init(mesh->is_volume(), pressure_bases, curret_bases);

			logger().info(" took {}s", timer.getElapsedTime());
		}

		if (args["output"]["advanced"]["sol_on_grid"] > 0)
		{
			const double spacing = args["output"]["advanced"]["sol_on_grid"];
			RowVectorNd min, max;
			mesh->bounding_box(min, max);
			const RowVectorNd delta = max - min;
			const int nx = delta[0] / spacing + 1;
			const int ny = delta[1] / spacing + 1;
			const int nz = delta.cols() >= 3 ? (delta[2] / spacing + 1) : 1;
			const int n = nx * ny * nz;

			grid_points.resize(n, delta.cols());
			int index = 0;
			for (int i = 0; i < nx; ++i)
			{
				const double x = (delta[0] / (nx - 1)) * i + min[0];

				for (int j = 0; j < ny; ++j)
				{
					const double y = (delta[1] / (ny - 1)) * j + min[1];

					if (delta.cols() <= 2)
					{
						grid_points.row(index++) << x, y;
					}
					else
					{
						for (int k = 0; k < nz; ++k)
						{
							const double z = (delta[2] / (nz - 1)) * k + min[2];
							grid_points.row(index++) << x, y, z;
						}
					}
				}
			}

			assert(index == n);

			std::vector<std::array<Eigen::Vector3d, 2>> boxes;
			mesh->elements_boxes(boxes);

			BVH::BVH bvh;
			bvh.init(boxes);

			const double eps = 1e-6;

			grid_points_to_elements.resize(grid_points.rows(), 1);
			grid_points_to_elements.setConstant(-1);

			grid_points_bc.resize(grid_points.rows(), mesh->is_volume() ? 4 : 3);

			for (int i = 0; i < grid_points.rows(); ++i)
			{
				const Eigen::Vector3d min(
					grid_points(i, 0) - eps,
					grid_points(i, 1) - eps,
					(mesh->is_volume() ? grid_points(i, 2) : 0) - eps);

				const Eigen::Vector3d max(
					grid_points(i, 0) + eps,
					grid_points(i, 1) + eps,
					(mesh->is_volume() ? grid_points(i, 2) : 0) + eps);

				std::vector<unsigned int> candidates;

				bvh.intersect_box(min, max, candidates);

				for (const auto cand : candidates)
				{
					if (!mesh->is_simplex(cand))
					{
						logger().warn("Element {} is not simplex, skipping", cand);
						continue;
					}

					Eigen::MatrixXd coords;
					mesh->barycentric_coords(grid_points.row(i), cand, coords);

					for (int d = 0; d < coords.size(); ++d)
					{
						if (fabs(coords(d)) < 1e-8)
							coords(d) = 0;
						else if (fabs(coords(d) - 1) < 1e-8)
							coords(d) = 1;
					}

					if (coords.array().minCoeff() >= 0 && coords.array().maxCoeff() <= 1)
					{
						grid_points_to_elements(i) = cand;
						grid_points_bc.row(i) = coords;
						break;
					}
				}
			}
		}
	}

	void State::build_polygonal_basis()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		igl::Timer timer;
		timer.start();
		logger().info("Computing polygonal basis...");

		// std::sort(boundary_nodes.begin(), boundary_nodes.end());

		// mixed not supports polygonal bases
		assert(n_pressure_bases == 0 || poly_edge_to_data.size() == 0);

		int new_bases = 0;

		if (iso_parametric())
		{
			if (mesh->is_volume())
			{
				if (args["space"]["advanced"]["poly_bases"] == "MeanValue")
					logger().error("MeanValue bases not supported in 3D");
				new_bases = PolygonalBasis3d::build_bases(assembler, formulation(), args["space"]["advanced"]["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["space"]["advanced"]["quadrature_order"], args["space"]["advanced"]["integral_constraints"], bases, bases, poly_edge_to_data, polys_3d);
			}
			else
			{
				if (args["space"]["advanced"]["poly_bases"] == "MeanValue")
				{
					new_bases = MVPolygonalBasis2d::build_bases(formulation(), *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["space"]["advanced"]["quadrature_order"], bases, bases, poly_edge_to_data, local_boundary, polys);
				}
				else
					new_bases = PolygonalBasis2d::build_bases(assembler, formulation(), args["space"]["advanced"]["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["space"]["advanced"]["quadrature_order"], args["space"]["advanced"]["integral_constraints"], bases, bases, poly_edge_to_data, polys);
			}
		}
		else
		{
			if (mesh->is_volume())
			{
				if (args["space"]["advanced"]["poly_bases"] == "MeanValue")
				{
					logger().error("MeanValue bases not supported in 3D");
					throw "not implemented";
				}
				new_bases = PolygonalBasis3d::build_bases(assembler, formulation(), args["space"]["advanced"]["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["space"]["advanced"]["quadrature_order"], args["space"]["advanced"]["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys_3d);
			}
			else
			{
				if (args["space"]["advanced"]["poly_bases"] == "MeanValue")
					new_bases = MVPolygonalBasis2d::build_bases(formulation(), *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["space"]["advanced"]["quadrature_order"], bases, geom_bases, poly_edge_to_data, local_boundary, polys);
				else
					new_bases = PolygonalBasis2d::build_bases(assembler, formulation(), args["space"]["advanced"]["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["space"]["advanced"]["quadrature_order"], args["space"]["advanced"]["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys);
			}
		}

		timer.stop();
		computing_poly_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", computing_poly_basis_time);

		n_bases += new_bases;
	}

	void State::build_collision_mesh()
	{
		extract_boundary_mesh(
			bases, boundary_nodes_pos, boundary_edges, boundary_triangles);

		Eigen::VectorXi codimensional_nodes;
		if (obstacle.n_vertices() > 0)
		{
			// boundary_nodes_pos uses n_bases that already contains the obstacle
			const int n_v = boundary_nodes_pos.rows() - obstacle.n_vertices();

			if (obstacle.v().size())
				boundary_nodes_pos.bottomRows(obstacle.v().rows()) = obstacle.v();

			if (obstacle.codim_v().size())
			{
				codimensional_nodes.conservativeResize(codimensional_nodes.size() + obstacle.codim_v().size());
				codimensional_nodes.tail(obstacle.codim_v().size()) = obstacle.codim_v().array() + n_v;
			}

			if (obstacle.e().size())
			{
				boundary_edges.conservativeResize(boundary_edges.rows() + obstacle.e().rows(), 2);
				boundary_edges.bottomRows(obstacle.e().rows()) = obstacle.e().array() + n_v;
			}

			if (obstacle.f().size())
			{
				boundary_triangles.conservativeResize(boundary_triangles.rows() + obstacle.f().rows(), 3);
				boundary_triangles.bottomRows(obstacle.f().rows()) = obstacle.f().array() + n_v;
			}
		}

		std::vector<bool> is_on_surface = ipc::CollisionMesh::construct_is_on_surface(boundary_nodes_pos.rows(), boundary_edges);
		for (int i = 0; i < codimensional_nodes.size(); i++)
		{
			is_on_surface[codimensional_nodes[i]] = true;
		}

		collision_mesh = ipc::CollisionMesh(is_on_surface, boundary_nodes_pos, boundary_edges, boundary_triangles);

		collision_mesh.can_collide = [&](size_t vi, size_t vj) {
			// obstacles do not collide with other obstacles
			return !this->is_obstacle_vertex(collision_mesh.to_full_vertex_id(vi))
				   || !this->is_obstacle_vertex(collision_mesh.to_full_vertex_id(vj));
		};
	}

	void State::assemble_stiffness_mat()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		if (formulation() == "OperatorSplitting")
		{
			stiffness.resize(1, 1);
			assembling_stiffness_mat_time = 0;
			return;
		}

		stiffness.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);
		mass.resize(0, 0);
		avg_mass = 1;

		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");

		// if(problem->is_mixed())
		if (assembler.is_mixed(formulation()))
		{
			if (assembler.is_linear(formulation()))
			{
				StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
				assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, velocity_stiffness);
				assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
				assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache, pressure_stiffness);

				const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();

				AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure ? assembler.is_fluid(formulation()) : false,
													 velocity_stiffness, mixed_stiffness, pressure_stiffness,
													 stiffness);

				if (problem->is_time_dependent())
				{
					StiffnessMatrix velocity_mass;
					assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, velocity_mass);

					std::vector<Eigen::Triplet<double>> mass_blocks;
					mass_blocks.reserve(velocity_mass.nonZeros());

					for (int k = 0; k < velocity_mass.outerSize(); ++k)
					{
						for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
						{
							mass_blocks.emplace_back(it.row(), it.col(), it.value());
						}
					}

					mass.resize(stiffness.rows(), stiffness.cols());
					mass.setFromTriplets(mass_blocks.begin(), mass_blocks.end());
					mass.makeCompressed();
				}
			}
		}
		else
		{
			if (!args["contact"]["enabled"]) // collisions are non-linear
				assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, stiffness);
			if (problem->is_time_dependent())
			{
				assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, mass);
			}
		}

		if (mass.size() > 0)
		{
			for (int k = 0; k < mass.outerSize(); ++k)
			{

				for (StiffnessMatrix::InnerIterator it(mass, k); it; ++it)
				{
					assert(it.col() == k);
					avg_mass += it.value();
				}
			}

			avg_mass /= mass.rows();
			logger().info("avgerage mass {}", avg_mass);

			if (args["solver"]["advanced"]["lump_mass_matrix"])
			{

				std::vector<Eigen::Triplet<double>> lumped;

				for (int k = 0; k < mass.outerSize(); ++k)
				{
					for (StiffnessMatrix::InnerIterator it(mass, k); it; ++it)
					{
						lumped.emplace_back(it.row(), it.row(), it.value());
					}
				}

				mass.resize(mass.rows(), mass.cols());
				mass.setFromTriplets(lumped.begin(), lumped.end());
				mass.makeCompressed();
			}
		}

		timer.stop();
		assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", assembling_stiffness_mat_time);

		nn_zero = stiffness.nonZeros();
		num_dofs = stiffness.rows();
		mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		logger().info("sparsity: {}/{}", nn_zero, mat_size);
	}

	void State::assemble_rhs()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}

		igl::Timer timer;
		// std::string rhs_path = "";
		// if (args["boundary_conditions"]["rhs"].is_string())
		// 	rhs_path = resolve_input_path(args["boundary_conditions"]["rhs"]);

		json p_params = {};
		p_params["formulation"] = formulation();
		{
			RowVectorNd min, max, delta;
			mesh->bounding_box(min, max);
			delta = (max - min) / 2. + min;
			if (mesh->is_volume())
				p_params["bbox_center"] = {delta(0), delta(1), delta(2)};
			else
				p_params["bbox_center"] = {delta(0), delta(1)};
		}
		problem->set_parameters(p_params);

		// stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		step_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle,
			n_bases, size,
			bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
			formulation(), *problem,
			args["space"]["advanced"]["bc_method"],
			args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], rhs_solver_params);

		// if (!rhs_path.empty() || rhs_in.size() > 0)
		// {
		// 	logger().debug("Loading rhs...");

		// 	if (rhs_in.size())
		// 		rhs = rhs_in;
		// 	else
		// 		read_matrix(rhs_path, rhs);

		// 	StiffnessMatrix tmp_mass;
		// 	assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, tmp_mass);
		// 	rhs = tmp_mass * rhs;
		// 	logger().debug("done!");
		// }
		// else
		{
			step_data.rhs_assembler->assemble(density, rhs);
			rhs *= -1;
		}

		// if(problem->is_mixed())
		if (assembler.is_mixed(formulation()))
		{
			const int prev_size = rhs.size();
			const int n_larger = n_pressure_bases + (use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0);
			rhs.conservativeResize(prev_size + n_larger, rhs.cols());
			if (formulation() == "OperatorSplitting")
			{
				assigning_rhs_time = 0;
				return;
			}
			// Divergence free rhs
			if (formulation() != "Bilaplacian" || local_neumann_boundary.empty())
			{
				rhs.block(prev_size, 0, n_larger, rhs.cols()).setZero();
			}
			else
			{
				Eigen::MatrixXd tmp(n_pressure_bases, 1);
				tmp.setZero();

				RhsAssembler tmp_rhs_assembler(
					assembler, *mesh, obstacle,
					n_pressure_bases, size,
					pressure_bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache,
					formulation(), *problem,
					args["space"]["advanced"]["bc_method"],
					args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], rhs_solver_params);

				tmp_rhs_assembler.set_bc(std::vector<LocalBoundary>(), std::vector<int>(), n_boundary_samples(), local_neumann_boundary, tmp);
				rhs.block(prev_size, 0, n_larger, rhs.cols()) = tmp;
			}
		}

		timer.stop();
		assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", assigning_rhs_time);
	}

	void State::init_timesteps()
	{
		const double t0 = args["time"]["t0"];
		double tend = args["time"]["tend"];          // default=1
		int time_steps = args["time"]["time_steps"]; // default=10 set in State::State()
		double dt = args["time"]["dt"];

		if (tend > 0)
		{
			if (dt > 0) // Explicit timestep param. has priority
			{
				time_steps = int(ceil((tend - t0) / dt));
			}
			else
			{
				dt = (tend - t0) / time_steps;
			}
		}
		else if (dt > 0) // Compute tend from dt and time_steps
		{
			tend = dt * time_steps + t0;
		}
		else // Use default tend
		{
			tend = 1;
			dt = (tend - t0) / time_steps;
		}
		assert(tend > 0 && dt > 0 && time_steps > 0);

		if (tend <= t0)
		{
			tend = t0 + time_steps * dt;
		}

		// Store these for possible use later
		args["time"]["tend"] = tend;
		args["time"]["dt"] = dt;
		args["time"]["time_steps"] = time_steps;

		logger().info("t0={}, dt={}, tend={}", t0, dt, tend);
	}

	void State::solve_problem()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}

		if (assembler.is_linear(formulation()) && !args["contact"]["enabled"] && stiffness.rows() <= 0)
		{
			logger().error("Assemble the stiffness matrix first!");
			return;
		}
		if (rhs.size() <= 0)
		{
			logger().error("Assemble the rhs first!");
			return;
		}

		sol.resize(0, 0);
		pressure.resize(0, 0);
		spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", formulation());

		const std::string full_mat_path = args["output"]["data"]["full_mat"];
		if (!full_mat_path.empty())
		{
			Eigen::saveMarket(stiffness, full_mat_path);
		}

		if (problem->is_time_dependent())
		{
			init_timesteps();
			const double t0 = args["time"]["t0"];
			const int time_steps = args["time"]["time_steps"];
			const double dt = args["time"]["dt"];

			// Pre log the output path for easier watching
			if (args["output"]["advanced"]["save_time_sequence"])
			{
				logger().info("Time sequence of simulation will be written to: {}",
							  resolve_output_path(args["output"]["paraview"]["file_name"]));
			}

			Eigen::VectorXd c_sol;
			init_transient(c_sol);
			RhsAssembler &rhs_assembler = *(step_data.rhs_assembler);

			if (formulation() == "NavierStokes")
				solve_transient_navier_stokes(time_steps, t0, dt, rhs_assembler, c_sol);
			else if (formulation() == "OperatorSplitting")
				solve_transient_navier_stokes_split(time_steps, dt, rhs_assembler);
			else if (problem->is_scalar() || assembler.is_mixed(formulation()))
				solve_transient_scalar(time_steps, t0, dt, rhs_assembler, c_sol);
			else if (assembler.is_linear(formulation()) && !args["contact"]["enabled"]) // Collisions add nonlinearity to the problem
				solve_transient_tensor_linear(time_steps, t0, dt, rhs_assembler);
			else
				solve_transient_tensor_non_linear(time_steps, t0, dt, rhs_assembler);
		}
		else // if(!problem->is_time_dependent())
		{
			if (formulation() == "NavierStokes")
				solve_navier_stokes();
			else if (assembler.is_linear(formulation()) && !args["contact"]["enabled"])
				solve_linear();
			else
				solve_non_linear();
		}

		timer.stop();
		solving_time = timer.getElapsedTime();
		logger().info(" took {}s", solving_time);
	}

	void State::compute_errors()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
		if (rhs.size() <= 0)
		{
			logger().error("Assemble the rhs first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		if (!args["output"]["advanced"]["compute_error"])
			return;

		int actual_dim = 1;
		if (!problem->is_scalar())
			actual_dim = mesh->dimension();

		igl::Timer timer;
		timer.start();
		logger().info("Computing errors...");
		using std::max;

		const int n_el = int(bases.size());

		MatrixXd v_exact, v_approx;
		MatrixXd v_exact_grad(0, 0), v_approx_grad;

		l2_err = 0;
		h1_err = 0;
		grad_max_err = 0;
		h1_semi_err = 0;
		linf_err = 0;
		lp_err = 0;
		// double pred_norm = 0;

		static const int p = 8;

		// Eigen::MatrixXd err_per_el(n_el, 5);
		ElementAssemblyValues vals;

		double tend;
		if (!args["time"].is_null())
		{
			tend = args["time"].value("tend", 1.0);
			if (tend <= 0)
				tend = 1;
		}
		else
			tend = 0;

		for (int e = 0; e < n_el; ++e)
		{
			// const auto &vals    = values[e];
			// const auto &gvalues = iso_parametric() ? values[e] : geom_values[e];

			if (iso_parametric())
				vals.compute(e, mesh->is_volume(), bases[e], bases[e]);
			else
				vals.compute(e, mesh->is_volume(), bases[e], geom_bases[e]);

			if (problem->has_exact_sol())
			{
				problem->exact(vals.val, tend, v_exact);
				problem->exact_grad(vals.val, tend, v_exact_grad);
			}

			v_approx.resize(vals.val.rows(), actual_dim);
			v_approx.setZero();

			v_approx_grad.resize(vals.val.rows(), mesh->dimension() * actual_dim);
			v_approx_grad.setZero();

			const int n_loc_bases = int(vals.basis_values.size());

			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto &val = vals.basis_values[i];

				for (size_t ii = 0; ii < val.global.size(); ++ii)
				{
					for (int d = 0; d < actual_dim; ++d)
					{
						v_approx.col(d) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.val;
						v_approx_grad.block(0, d * val.grad_t_m.cols(), v_approx_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.grad_t_m;
					}
				}
			}

			const auto err = problem->has_exact_sol() ? (v_exact - v_approx).eval().rowwise().norm().eval() : (v_approx).eval().rowwise().norm().eval();
			const auto err_grad = problem->has_exact_sol() ? (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval() : (v_approx_grad).eval().rowwise().norm().eval();

			// for(long i = 0; i < err.size(); ++i)
			// errors.push_back(err(i));

			linf_err = max(linf_err, err.maxCoeff());
			grad_max_err = max(linf_err, err_grad.maxCoeff());

			// {
			// 	const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
			// 	const auto v0 = mesh3d.point(mesh3d.cell_vertex(e, 0));
			// 	const auto v1 = mesh3d.point(mesh3d.cell_vertex(e, 1));
			// 	const auto v2 = mesh3d.point(mesh3d.cell_vertex(e, 2));
			// 	const auto v3 = mesh3d.point(mesh3d.cell_vertex(e, 3));

			// 	Eigen::Matrix<double, 6, 3> ee;
			// 	ee.row(0) = v0 - v1;
			// 	ee.row(1) = v1 - v2;
			// 	ee.row(2) = v2 - v0;

			// 	ee.row(3) = v0 - v3;
			// 	ee.row(4) = v1 - v3;
			// 	ee.row(5) = v2 - v3;

			// 	Eigen::Matrix<double, 6, 1> en = ee.rowwise().norm();

			// 	// Eigen::Matrix<double, 3*4, 1> alpha;
			// 	// alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
			// 	// alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
			// 	// alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
			// 	// alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

			// 	const double S = (ee.row(0).cross(ee.row(1)).norm() + ee.row(0).cross(ee.row(4)).norm() + ee.row(4).cross(ee.row(1)).norm() + ee.row(2).cross(ee.row(5)).norm()) / 2;
			// 	const double V = std::abs(ee.row(3).dot(ee.row(2).cross(-ee.row(0))))/6;
			// 	const double rho = 3 * V / S;
			// 	const double hp = en.maxCoeff();
			// 	const int pp = disc_orders(e);
			// 	const int p_ref = args["space"]["discr_order"];

			// 	err_per_el(e, 0) = err.mean();
			// 	err_per_el(e, 1) = err.maxCoeff();
			// 	err_per_el(e, 2) = std::pow(hp, pp+1)/(rho/hp); // /std::pow(average_edge_length, p_ref+1) * (sqrt(6)/12);
			// 	err_per_el(e, 3) = rho/hp;
			// 	err_per_el(e, 4) = (vals.det.array() * vals.quadrature.weights.array()).sum();

			// 	// pred_norm += (pow(std::pow(hp, pp+1)/(rho/hp),p) * vals.det.array() * vals.quadrature.weights.array()).sum();
			// }

			l2_err += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			h1_err += (err_grad.array() * err_grad.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(p) * vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		h1_semi_err = sqrt(fabs(h1_err));
		h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
		l2_err = sqrt(fabs(l2_err));

		lp_err = pow(fabs(lp_err), 1. / p);

		// pred_norm = pow(fabs(pred_norm), 1./p);

		timer.stop();
		computing_errors_time = timer.getElapsedTime();
		logger().info(" took {}s", computing_errors_time);

		logger().info("-- L2 error: {}", l2_err);
		logger().info("-- Lp error: {}", lp_err);
		logger().info("-- H1 error: {}", h1_err);
		logger().info("-- H1 semi error: {}", h1_semi_err);
		// logger().info("-- Perd norm: {}", pred_norm);

		logger().info("-- Linf error: {}", linf_err);
		logger().info("-- grad max error: {}", grad_max_err);

		logger().info("total time: {}s", (building_basis_time + assembling_stiffness_mat_time + solving_time));

		// {
		// 	std::ofstream out("errs.txt");
		// 	out<<err_per_el;
		// 	out.close();
		// }
	}

} // namespace polyfem
