#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/StringUtils.hpp>

#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <polyfem/FEBasis2d.hpp>
#include <polyfem/FEBasis3d.hpp>

#include <polyfem/SpectralBasis2d.hpp>

#include <polyfem/SplineBasis2d.hpp>
#include <polyfem/SplineBasis3d.hpp>

#include <polyfem/MVPolygonalBasis2d.hpp>

#include <polyfem/PolygonalBasis2d.hpp>
#include <polyfem/PolygonalBasis3d.hpp>

#include <polyfem/EdgeSampler.hpp>

#include <polyfem/RhsAssembler.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/NLProblem.hpp>
#include <polyfem/LbfgsSolver.hpp>
#include <polyfem/SparseNewtonDescentSolver.hpp>
#include <polyfem/NavierStokesSolver.hpp>
#include <polyfem/TransientNavierStokesSolver.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <polyfem/HexQuadrature.hpp>
#include <polyfem/QuadQuadrature.hpp>
#include <polyfem/TetQuadrature.hpp>
#include <polyfem/TriQuadrature.hpp>
#include <polyfem/BDF.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <algorithm>
#include <memory>

#include <polyfem/autodiff.h>
DECLARE_DIFFSCALAR_BASE();

using namespace Eigen;

namespace polyfem
{
	using namespace polysolve;

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

		if (!args["curved_mesh_size"])
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
		if (args.find("body_params") == args.end())
			return;

		const auto &body_params = args["body_params"];
		assert(body_params.is_array());
		Eigen::MatrixXd Es(mesh->n_elements(), 1), nus(mesh->n_elements(), 1), rhos(mesh->n_elements(), 1);
		Es.setConstant(100);
		nus.setConstant(0.3);
		rhos.setOnes();

		std::map<int, std::tuple<double, double, double>> materials;
		for (int i = 0; i < body_params.size(); ++i)
		{
			const auto &mat = body_params[i];
			const int mid = mat["id"];
			Density d;
			d.init(mat);

			const double rho = d(0, 0, 0, 0);
			const double E = mat["E"];
			const double nu = mat["nu"];

			materials[mid] = std::tuple<double, double, double>(E, nu, rho);
			// std::cout << mid << " " << E << " " << nu << " " << rho << " " << std::endl;
		}

		std::string missing = "";

		for (int e = 0; e < mesh->n_elements(); ++e)
		{
			const int bid = mesh->get_body_id(e);
			const auto it = materials.find(bid);
			if (it == materials.end())
			{
				missing += std::to_string(bid) + ", ";
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
		disc_orders.setConstant(args["discr_order"]);

		const auto params = build_json_params();
		assembler.set_parameters(params);
		density.init(params);
		problem->init(*mesh);

		logger().info("Building {} basis...", (iso_parametric() ? "isoparametric" : "not isoparametric"));
		const bool has_polys = non_regular_count > 0 || non_regular_boundary_count > 0 || undefined_count > 0;

		local_boundary.clear();
		local_neumann_boundary.clear();
		std::map<int, InterfaceData> poly_edge_to_data_geom; //temp dummy variable

		const int base_p = args["discr_order"];
		disc_orders.setConstant(base_p);

		Eigen::MatrixXi geom_disc_orders;
		if (!iso_parametric())
		{
			if (args["force_linear_geometry"] || mesh->orders().size() <= 0)
			{
				geom_disc_orders.resizeLike(disc_orders);
				geom_disc_orders.setConstant(1);
			}
			else
				geom_disc_orders = mesh->orders();
		}

		igl::Timer timer;
		timer.start();
		if (args["use_p_ref"])
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
			if (args["use_spline"])
			{
				if (!iso_parametric())
				{
					logger().error("Splines must be isoparametric, ignoring...");
					// FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], geom_disc_orders, has_polys, geom_bases, local_boundary, poly_edge_to_data_geom);
					SplineBasis3d::build_bases(tmp_mesh, args["quadrature_order"], geom_bases, local_boundary, poly_edge_to_data);
				}

				n_bases = SplineBasis3d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				if (iso_parametric() && args["fit_nodes"])
					SplineBasis3d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], geom_disc_orders, false, has_polys, true, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, args["serendipity"], has_polys, false, bases, local_boundary, poly_edge_to_data);
			}

			// if(problem->is_mixed())
			if (assembler.is_mixed(formulation()))
			{
				n_pressure_bases = FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], int(args["pressure_discr_order"]), false, has_polys, false, pressure_bases, local_boundary, poly_edge_to_data_geom);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			if (args["use_spline"])
			{

				if (!iso_parametric())
				{
					logger().error("Splines must be isoparametric, ignoring...");
					// FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, has_polys, geom_bases, local_boundary, poly_edge_to_data_geom);
					n_bases = SplineBasis2d::build_bases(tmp_mesh, args["quadrature_order"], geom_bases, local_boundary, poly_edge_to_data);
				}

				n_bases = SplineBasis2d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				if (iso_parametric() && args["fit_nodes"])
					SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], geom_disc_orders, false, has_polys, true, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, args["serendipity"], has_polys, false, bases, local_boundary, poly_edge_to_data);
			}

			// if(problem->is_mixed())
			if (assembler.is_mixed(formulation()))
			{
				n_pressure_bases = FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], int(args["pressure_discr_order"]), false, has_polys, false, pressure_bases, local_boundary, poly_edge_to_data_geom);
			}
		}
		timer.stop();

		build_polygonal_basis();

		auto &gbases = iso_parametric() ? bases : geom_bases;

		n_flipped = 0;

		if (args["count_flipped_els"])
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

					logger().info("element {} is flipped, type {}", i, type);

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

		logger().info("Extracting boundary mesh...");
		extract_boundary_mesh();
		const std::string export_surface = args["export"]["surface"];
		if (!export_surface.empty())
			extract_vis_boundary_mesh();
		logger().info("Done!");

		problem->setup_bc(*mesh, bases, local_boundary, boundary_nodes, local_neumann_boundary);

		//add a pressure node to avoid singular solution
		if (assembler.is_mixed(formulation())) // && !assembler.is_fluid(formulation()))
		{
			if (!use_avg_pressure)
			{
				const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
				const bool has_neumann = args["has_neumann"];
				if (!has_neumann)
					boundary_nodes.push_back(n_bases * problem_dim + 0);

				// boundary_nodes.push_back(n_bases * problem_dim + 1);
				// boundary_nodes.push_back(n_bases * problem_dim + 2);
				// boundary_nodes.push_back(n_bases * problem_dim + 3);
				// boundary_nodes.push_back(n_bases * problem_dim + 3);
				// boundary_nodes.push_back(n_bases * problem_dim + 215);
			}
		}

		const auto &curret_bases = iso_parametric() ? bases : geom_bases;
		const int n_samples = 10;
		compute_mesh_size(*mesh, curret_bases, n_samples);

		building_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", building_basis_time);

		logger().info("flipped elements {}", n_flipped);
		logger().info("h: {}", mesh_size);
		logger().info("n bases: {}", n_bases);
		logger().info("n pressure bases: {}", n_pressure_bases);

		if (bases.size() <= args["cache_size"])
		{
			ass_vals_cache.init(mesh->is_volume(), bases, curret_bases);
			if (assembler.is_mixed(formulation()))
				pressure_ass_vals_cache.init(mesh->is_volume(), pressure_bases, curret_bases);
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

		//mixed not supports polygonal bases
		assert(n_pressure_bases == 0 || poly_edge_to_data.size() == 0);

		int new_bases = 0;

		if (iso_parametric())
		{
			if (mesh->is_volume())
			{
				if (args["poly_bases"] == "MeanValue")
					logger().error("MeanValue bases not supported in 3D");
				new_bases = PolygonalBasis3d::build_bases(assembler, formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, bases, poly_edge_to_data, polys_3d);
			}
			else
			{
				if (args["poly_bases"] == "MeanValue")
				{
					new_bases = MVPolygonalBasis2d::build_bases(formulation(), *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], bases, bases, poly_edge_to_data, local_boundary, polys);
				}
				else
					new_bases = PolygonalBasis2d::build_bases(assembler, formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, bases, poly_edge_to_data, polys);
			}
		}
		else
		{
			if (mesh->is_volume())
			{
				if (args["poly_bases"] == "MeanValue")
					logger().error("MeanValue bases not supported in 3D");
				new_bases = PolygonalBasis3d::build_bases(assembler, formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys_3d);
			}
			else
			{
				if (args["poly_bases"] == "MeanValue")
					new_bases = MVPolygonalBasis2d::build_bases(formulation(), *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], bases, geom_bases, poly_edge_to_data, local_boundary, polys);
				else
					new_bases = PolygonalBasis2d::build_bases(assembler, formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys);
			}
		}

		timer.stop();
		computing_poly_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", computing_poly_basis_time);

		n_bases += new_bases;
	}

	json State::build_json_params()
	{
		json params = args["params"];
		params["size"] = mesh->dimension();

		return params;
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
			logger().trace("avg mass {}", avg_mass);

			if (args["lump_mass_matrix"])
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
		const std::string rhs_path = args["rhs_path"];

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

		// const auto params = build_json_params();
		// assembler.set_parameters(params);

		// stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

		if (!rhs_path.empty() || rhs_in.size() > 0)
		{
			logger().debug("Loading rhs...");

			if (rhs_in.size())
				rhs = rhs_in;
			else
				read_matrix(args["rhs_path"], rhs);

			StiffnessMatrix tmp_mass;
			assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, tmp_mass);
			rhs = tmp_mass * rhs;
			logger().debug("done!");
		}
		else
		{
			rhs_assembler.assemble(density, rhs);
			rhs *= -1;
		}

		// if(problem->is_mixed())
		if (assembler.is_mixed(formulation()))
		{
			const int prev_size = rhs.size();
			const int n_larger = n_pressure_bases + (use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0);
			rhs.conservativeResize(prev_size + n_larger, rhs.cols());
			//Divergence free rhs
			if (formulation() != "Bilaplacian" || local_neumann_boundary.empty())
			{
				rhs.block(prev_size, 0, n_larger, rhs.cols()).setZero();
			}
			else
			{
				Eigen::MatrixXd tmp(n_pressure_bases, 1);
				tmp.setZero();

				RhsAssembler rhs_assembler1(assembler, *mesh,
											n_pressure_bases, size,
											pressure_bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache,
											formulation(), *problem,
											args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);
				rhs_assembler1.set_bc(std::vector<LocalBoundary>(), std::vector<int>(), args["n_boundary_samples"], local_neumann_boundary, tmp);
				rhs.block(prev_size, 0, n_larger, rhs.cols()) = tmp;
			}
		}

		timer.stop();
		assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", assigning_rhs_time);
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

		if (assembler.is_linear(formulation()) && stiffness.rows() <= 0)
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
		logger().info("Solving {} with", formulation());

		const json &params = solver_params();

		const std::string full_mat_path = args["export"]["full_mat"];
		if (!full_mat_path.empty())
		{
			Eigen::saveMarket(stiffness, full_mat_path);
		}

		if (problem->is_time_dependent())
		{

			const double tend = args["tend"];
			const int time_steps = args["time_steps"];
			const double dt = tend / time_steps;
			logger().info("dt={}", dt);

			const auto &gbases = iso_parametric() ? bases : geom_bases;
			json rhs_solver_params = args["rhs_solver_params"];
			rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

			RhsAssembler rhs_assembler(assembler, *mesh,
									   n_bases, problem->is_scalar() ? 1 : mesh->dimension(),
									   bases, gbases, ass_vals_cache,
									   formulation(), *problem,
									   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);
			rhs_assembler.initial_solution(sol);

			Eigen::MatrixXd current_rhs = rhs;

			if (formulation() == "NavierStokes")
			{
				StiffnessMatrix velocity_mass;
				assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, velocity_mass);

				StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;

				const int prev_size = sol.size();
				sol.conservativeResize(rhs.size(), sol.cols());
				//Zero initial pressure
				sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
				sol(sol.size() - 1) = 0;
				Eigen::VectorXd c_sol = sol;

				Eigen::VectorXd prev_sol;

				int BDF_order = args["BDF_order"];
				// int aux_steps = BDF_order-1;
				BDF bdf(BDF_order);
				bdf.new_solution(c_sol);

				sol = c_sol;
				sol_to_pressure();
				if (args["save_time_sequence"])
				{
					if (!solve_export_to_file)
						solution_frames.emplace_back();
					save_vtu("step_" + std::to_string(0) + ".vtu", 0);
					save_wire("step_" + std::to_string(0) + ".obj");
				}

				assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, velocity_stiffness);
				assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
				assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, pressure_stiffness);

				TransientNavierStokesSolver ns_solver(solver_params(), build_json_params(), solver_type(), precond_type());
				const int n_larger = n_pressure_bases + (use_avg_pressure ? 1 : 0);

				for (int t = 1; t <= time_steps; ++t)
				{
					double time = t * dt;
					double current_dt = dt;

					logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, current_dt, time);

					bdf.rhs(prev_sol);
					rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
					rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

					const int prev_size = current_rhs.size();
					if (prev_size != rhs.size())
					{
						current_rhs.conservativeResize(prev_size + n_larger, current_rhs.cols());
						current_rhs.block(prev_size, 0, n_larger, current_rhs.cols()).setZero();
					}

					assembler.clear_cache();
					ns_solver.minimize(*this, bdf.alpha(), current_dt, prev_sol,
									   velocity_stiffness, mixed_stiffness, pressure_stiffness,
									   velocity_mass, current_rhs, c_sol);
					bdf.new_solution(c_sol);
					sol = c_sol;
					sol_to_pressure();

					if (args["save_time_sequence"])
					{
						if (!solve_export_to_file)
							solution_frames.emplace_back();
						save_vtu("step_" + std::to_string(t) + ".vtu", time);
						save_wire("step_" + std::to_string(t) + ".obj");
					}
				}
			}
			else //if (formulation() != "NavierStokes")
			{
				if (assembler.is_mixed(formulation()))
				{
					pressure.resize(n_pressure_bases, 1);
					pressure.setZero();
				}

				auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
				solver->setParameters(params);
				logger().info("{}...", solver->name());

				if (args["save_time_sequence"])
				{
					if (!solve_export_to_file)
						solution_frames.emplace_back();
					save_vtu("step_" + std::to_string(0) + ".vtu", 0);
					save_wire("step_" + std::to_string(0) + ".obj");
				}

				if (assembler.is_mixed(formulation()))
				{
					pressure.resize(0, 0);
					const int prev_size = sol.size();
					sol.conservativeResize(rhs.size(), sol.cols());
					//Zero initial pressure
					sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
					sol(sol.size() - 1) = 0;
				}

				if (problem->is_scalar() || assembler.is_mixed(formulation()))
				{
					StiffnessMatrix A;
					Eigen::VectorXd b, x;

					const int BDF_order = args["BDF_order"];
					// const int aux_steps = BDF_order-1;
					BDF bdf(BDF_order);
					x = sol;
					bdf.new_solution(x);

					const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
					const int precond_num = problem_dim * n_bases;

					for (int t = 1; t <= time_steps; ++t)
					{
						double time = t * dt;
						double current_dt = dt;

						logger().info("{}/{} {}s", t, time_steps, time);
						rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

						if (assembler.is_mixed(formulation()))
						{
							//divergence free
							int fluid_offset = use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0;

							current_rhs.block(current_rhs.rows() - n_pressure_bases - use_avg_pressure, 0, n_pressure_bases + use_avg_pressure, current_rhs.cols()).setZero();
						}

						A = (bdf.alpha() / current_dt) * mass + stiffness;
						bdf.rhs(x);
						b = (mass * x) / current_dt;
						for (int i : boundary_nodes)
							b[i] = 0;
						b += current_rhs;

						spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == time_steps && args["export"]["spectrum"]);
						bdf.new_solution(x);
						sol = x;

						if (assembler.is_mixed(formulation()))
						{
							sol_to_pressure();
						}

						if (args["save_time_sequence"])
						{
							if (!solve_export_to_file)
								solution_frames.emplace_back();

							save_vtu("step_" + std::to_string(t) + ".vtu", time);
							save_wire("step_" + std::to_string(t) + ".obj");
						}
					}
				}
				else //tensor time dependent
				{
					Eigen::MatrixXd velocity, acceleration;
					rhs_assembler.initial_velocity(velocity);
					rhs_assembler.initial_acceleration(acceleration);

					const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
					const int precond_num = problem_dim * n_bases;

					if (assembler.is_linear(formulation()) && !args["has_collision"])
					{
						//Newmark
						const double gamma = 0.5;
						const double beta = 0.25;
						// makes the algorithm implicit and equivalent to the trapezoidal rule (unconditionally stable).

						Eigen::MatrixXd temp, b;
						StiffnessMatrix A;
						Eigen::VectorXd x, btmp;

						for (int t = 1; t <= time_steps; ++t)
						{
							const double dt2 = dt * dt;

							const Eigen::MatrixXd aOld = acceleration;
							const Eigen::MatrixXd vOld = velocity;
							const Eigen::MatrixXd uOld = sol;

							if (!problem->is_linear_in_time())
							{
								rhs_assembler.assemble(density, current_rhs, dt * t);
								current_rhs *= -1;
							}
							temp = -(uOld + dt * vOld + ((1 / 2. - beta) * dt2) * aOld);
							b = stiffness * temp + current_rhs;

							rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, b, dt * t);

							A = stiffness * beta * dt2 + mass;
							btmp = b;
							spectrum = dirichlet_solve(*solver, A, btmp, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == 1 && args["export"]["spectrum"]);
							acceleration = x;

							sol += dt * vOld + dt2 * ((1 / 2.0 - beta) * aOld + beta * acceleration);
							velocity += dt * ((1 - gamma) * aOld + gamma * acceleration);

							rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, dt * t);
							rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, dt * t);
							rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, dt * t);

							if (args["save_time_sequence"])
							{
								if (!solve_export_to_file)
									solution_frames.emplace_back();
								save_vtu("step_" + std::to_string(t) + ".vtu", dt * t);
								save_wire("step_" + std::to_string(t) + ".obj");
							}

							logger().info("{}/{}", t, time_steps);
						}
					}
					else //if (!assembler.is_linear(formulation()) || collision)
					{
						// {
						// 	boundary_nodes.clear();
						// 	local_boundary.clear();
						// 	local_neumann_boundary.clear();
						// 	NLProblem nl_problem(*this, rhs_assembler, 0, args["dhat"]);
						// 	Eigen::MatrixXd tmp_sol = rhs;

						// 	// tmp_sol.setRandom();
						// 	tmp_sol.setOnes();
						// 	// tmp_sol /=10000.;

						// 	velocity.setZero();
						// 	VectorXd xxx=tmp_sol;
						// 	velocity = tmp_sol;
						// 	nl_problem.init_timestep(xxx, velocity, dt);

						// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
						// 	nl_problem.gradient(tmp_sol, actual_grad);

						// 	StiffnessMatrix hessian;
						// 	Eigen::MatrixXd expected_hessian;
						// 	nl_problem.hessian(tmp_sol, hessian);

						// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
						// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

						// 	for (int i = 0; i < actual_hessian.rows(); ++i)
						// 	{
						// 		double hhh = 1e-6;
						// 		VectorXd xp = tmp_sol; xp(i) += hhh;
						// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

						// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
						// 		nl_problem.gradient(xp, tmp_grad_p);

						// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
						// 		nl_problem.gradient(xm, tmp_grad_m);

						// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/(hhh*2.);

						// 		const double vp = nl_problem.value(xp);
						// 		const double vm = nl_problem.value(xm);

						// 		const double fd = (vp-vm)/(hhh*2.);
						// 		const double  diff = std::abs(actual_grad(i) - fd);
						// 		if(diff > 1e-6)
						// 			std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<" rrr: "<<actual_grad(i)/fd<<std::endl;

						// 		for(int j = 0; j < actual_hessian.rows(); ++j)
						// 		{
						// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

						// 			if(diff > 1e-5)
						// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<" rrr: "<<actual_hessian(i,j)/fd_h(j)<<std::endl;

						// 		}
						// 	}

						// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
						// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
						// 	exit(0);
						// }

						const int full_size = n_bases * mesh->dimension();
						const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
						VectorXd tmp_sol;

						NLProblem nl_problem(*this, rhs_assembler, dt, args["dhat"], args["project_to_psd"]);
						nl_problem.init_timestep(sol, velocity, acceleration, dt);
						nl_problem.full_to_reduced(sol, tmp_sol);

						for (int t = 1; t <= time_steps; ++t)
						{
							cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
							nlsolver.setLineSearch(args["line_search"]);
							nl_problem.init(sol);
							nlsolver.minimize(nl_problem, tmp_sol);

							if (nlsolver.error_code() == -10)
							{
								double substep_delta = 0.5;
								double substep = substep_delta;
								bool solved = false;

								while (substep_delta > 1e-4 && !solved)
								{
									logger().debug("Substepping {}/{}, dt={}", (t - 1 + substep) * dt, t * dt, substep_delta);
									nl_problem.substepping((t - 1 + substep) * dt);
									nl_problem.full_to_reduced(sol, tmp_sol);
									nlsolver.minimize(nl_problem, tmp_sol);

									if (nlsolver.error_code() == -10)
									{
										substep -= substep_delta;
										substep_delta /= 2;
									}
									else
									{
										logger().trace("Done {}/{}, dt={}", (t - 1 + substep) * dt, t * dt, substep_delta);
										nl_problem.reduced_to_full(tmp_sol, sol);
										substep_delta *= 2;
									}

									solved = substep >= 1;

									substep += substep_delta;
									if (substep >= 1)
									{
										substep_delta -= substep - 1;
										substep = 1;
									}
								}
							}

							if (nlsolver.error_code() == -10)
							{
								logger().error("Unable to solve t={}", t * dt);
								save_vtu("stop.vtu", dt * t);
								break;
							}

							logger().debug("Step solved!");

							nlsolver.getInfo(solver_info);
							nl_problem.reduced_to_full(tmp_sol, sol);
							if (assembler.is_mixed(formulation()))
							{
								sol_to_pressure();
							}

							// rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, dt * t);

							nl_problem.update_quantities((t + 1) * dt, sol);

							if (args["save_time_sequence"])
							{
								if (!solve_export_to_file)
									solution_frames.emplace_back();
								save_vtu("step_" + std::to_string(t) + ".vtu", dt * t);
								save_wire("step_" + std::to_string(t) + ".obj");
							}

							logger().info("{}/{}", t, time_steps);
						}
					}
				}
			}
		}
		else //if(!problem->is_time_dependent())
		{
			if (assembler.is_linear(formulation()) && !args["has_collision"])
			{
				auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
				solver->setParameters(params);
				StiffnessMatrix A;
				Eigen::VectorXd b;
				logger().info("{}...", solver->name());
				json rhs_solver_params = args["rhs_solver_params"];
				rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
				const int size = problem->is_scalar() ? 1 : mesh->dimension();
				RhsAssembler rhs_assembler(assembler, *mesh,
										   n_bases, size,
										   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
										   formulation(), *problem,
										   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

				if (formulation() != "Bilaplacian")
					rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
				else
					rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], std::vector<LocalBoundary>(), rhs);

				const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
				const int precond_num = problem_dim * n_bases;

				A = stiffness;
				Eigen::VectorXd x;
				b = rhs;
				spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"]);
				sol = x;
				solver->getInfo(solver_info);

				logger().debug("Solver error: {}", (A * sol - b).norm());

				if (assembler.is_mixed(formulation()))
				{
					sol_to_pressure();
				}
			}
			else //if (!assembler.is_linear(formulation()) ||  args["has_collision"]))
			{
				if (formulation() == "NavierStokes")
				{
					auto params = build_json_params();
					const double viscosity = params.count("viscosity") ? double(params["viscosity"]) : 1.;
					NavierStokesSolver ns_solver(viscosity, solver_params(), build_json_params(), solver_type(), precond_type());
					Eigen::VectorXd x;
					assembler.clear_cache();
					json rhs_solver_params = args["rhs_solver_params"];
					rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

					RhsAssembler rhs_assembler(assembler, *mesh,
											   n_bases, mesh->dimension(),
											   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
											   formulation(), *problem,
											   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);
					rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
					ns_solver.minimize(*this, rhs, x);
					sol = x;
					sol_to_pressure();
				}
				else
				{
					const int full_size = n_bases * mesh->dimension();
					const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
					const double tend = args["tend"];

					const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
					const int precond_num = problem_dim * n_bases;

					int steps = args["nl_solver_rhs_steps"];
					if (steps <= 0)
					{
						RowVectorNd min, max;
						mesh->bounding_box(min, max);
						steps = problem->n_incremental_load_steps((max - min).norm());
					}
					steps = std::max(steps, 1);

					json rhs_solver_params = args["rhs_solver_params"];
					rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
					const int size = problem->is_scalar() ? 1 : mesh->dimension();
					RhsAssembler rhs_assembler(assembler, *mesh,
											   n_bases, size,
											   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
											   formulation(), *problem,
											   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

					StiffnessMatrix nlstiffness;
					auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
					Eigen::VectorXd x, b;
					Eigen::MatrixXd grad;
					Eigen::MatrixXd prev_rhs;

					VectorXd tmp_sol;

					double step_t = 1.0 / steps;
					double t = step_t;
					double prev_t = 0;

					sol.resizeLike(rhs);
					sol.setZero();

					prev_rhs.resizeLike(rhs);
					prev_rhs.setZero();

					x.resizeLike(sol);
					x.setZero();

					b.resizeLike(sol);
					b.setZero();

					if (args["save_solve_sequence"])
					{
						if (!solve_export_to_file)
							solution_frames.emplace_back();
						save_vtu("step_" + std::to_string(prev_t) + ".vtu", tend);
						save_wire("step_" + std::to_string(prev_t) + ".obj");
					}

					const auto &gbases = iso_parametric() ? bases : geom_bases;
					igl::Timer update_timer;
					while (t <= 1)
					{
						if (step_t < 1e-10)
						{
							logger().error("Step too small, giving up");
							break;
						}

						logger().info("t: {} prev: {} step: {}", t, prev_t, step_t);

						NLProblem nl_problem(*this, rhs_assembler, t, args["dhat"], args["project_to_psd"]);

						logger().debug("Updating starting point...");
						update_timer.start();

						{
							nl_problem.hessian_full(sol, nlstiffness);
							nl_problem.gradient_no_rhs(sol, grad);
							rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, grad, t);

							b = grad;
							for (int bId : boundary_nodes)
								b(bId) = -(nl_problem.current_rhs()(bId) - prev_rhs(bId));
							dirichlet_solve(*solver, nlstiffness, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"]);
							logger().trace("Checking step");
							const bool valid = nl_problem.is_step_valid(sol, (sol - x).eval());
							if (valid)
								x = sol - x;
							else
								x = sol;
							logger().trace("Done checking step, was {}valid", valid ? "" : "in");
							// logger().debug("Solver error: {}", (nlstiffness * sol - b).norm());
						}

						nl_problem.full_to_reduced(x, tmp_sol);
						update_timer.stop();
						logger().debug("done!, took {}s", update_timer.getElapsedTime());

						if (args["save_solve_sequence_debug"])
						{
							Eigen::MatrixXd xxx = sol;
							sol = x;
							if (assembler.is_mixed(formulation()))
								sol_to_pressure();
							if (!solve_export_to_file)
								solution_frames.emplace_back();

							save_vtu("step_s_" + std::to_string(t) + ".vtu", tend);
							save_wire("step_s_" + std::to_string(t) + ".obj");

							sol = xxx;
						}

						bool has_nan = false;
						for (int k = 0; k < tmp_sol.size(); ++k)
						{
							if (std::isnan(tmp_sol[k]))
							{
								has_nan = true;
								break;
							}
						}

						if (has_nan)
						{
							do
							{
								step_t /= 2;
								t = prev_t + step_t;
							} while (t >= 1);
							continue;
						}

						assembler.clear_cache();

						if (args["nl_solver"] == "newton")
						{
							cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
							nlsolver.setLineSearch(args["line_search"]);
							nl_problem.init(x);
							nlsolver.minimize(nl_problem, tmp_sol);

							if (nlsolver.error_code() == -10) //Nan
							{
								do
								{
									step_t /= 2;
									t = prev_t + step_t;
								} while (t >= 1);
								continue;
							}
							else
							{
								prev_t = t;
								step_t *= 2;
							}

							if (step_t > 1.0 / steps)
								step_t = 1.0 / steps;

							nlsolver.getInfo(solver_info);
						}
						else if (args["nl_solver"] == "lbfgs")
						{
							cppoptlib::LbfgsSolverL2<NLProblem> nlsolver;
							nlsolver.setLineSearch(args["line_search"]);
							nlsolver.setDebug(cppoptlib::DebugLevel::High);
							nlsolver.minimize(nl_problem, tmp_sol);

							prev_t = t;
						}
						else
						{
							throw std::invalid_argument("[State] invalid solver type for non-linear problem");
						}

						t = prev_t + step_t;
						if ((prev_t < 1 && t > 1) || abs(t - 1) < 1e-10)
							t = 1;

						nl_problem.reduced_to_full(tmp_sol, sol);

						// std::ofstream of("sol.txt");
						// of<<sol<<std::endl;
						// of.close();
						prev_rhs = nl_problem.current_rhs();
						if (args["save_solve_sequence"])
						{
							if (!solve_export_to_file)
								solution_frames.emplace_back();
							save_vtu("step_" + std::to_string(prev_t) + ".vtu", tend);
							save_wire("step_" + std::to_string(prev_t) + ".obj");
						}
					}

					if (assembler.is_mixed(formulation()))
					{
						sol_to_pressure();
					}

					// {
					// 	boundary_nodes.clear();
					// 	NLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"]);
					// 	tmp_sol = rhs;

					// 	// tmp_sol.setRandom();
					// 	tmp_sol.setOnes();
					// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
					// 	nl_problem.gradient(tmp_sol, actual_grad);

					// 	StiffnessMatrix hessian;
					// 	// Eigen::MatrixXd expected_hessian;
					// 	nl_problem.hessian(tmp_sol, hessian);
					// 	// nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

					// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
					// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

					// 	for (int i = 0; i < actual_hessian.rows(); ++i)
					// 	{
					// 		double hhh = 1e-6;
					// 		VectorXd xp = tmp_sol; xp(i) += hhh;
					// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

					// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
					// 		nl_problem.gradient(xp, tmp_grad_p);

					// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
					// 		nl_problem.gradient(xm, tmp_grad_m);

					// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/(hhh*2.);

					// 		const double vp = nl_problem.value(xp);
					// 		const double vm = nl_problem.value(xm);

					// 		const double fd = (vp-vm)/(hhh*2.);
					// 		const double  diff = std::abs(actual_grad(i) - fd);
					// 		if(diff > 1e-5)
					// 			std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<" rrr: "<<actual_grad(i)/fd<<std::endl;

					// 		for(int j = 0; j < actual_hessian.rows(); ++j)
					// 		{
					// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

					// 			if(diff > 1e-4)
					// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<" rrr: "<<actual_hessian(i,j)/fd_h(j)<<std::endl;

					// 		}
					// 	}

					// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
					// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
					// 	exit(0);
					// }

					// NLProblem::reduced_to_full_aux(full_size, reduced_size, tmp_sol, rhs, sol);
				}
			}
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

		const double tend = args["tend"];

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
			// 	const int p_ref = args["discr_order"];

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
