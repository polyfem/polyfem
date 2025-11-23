#include "VariationalForm.hpp"

namespace polyfem
{
	// TODO varform
	/*
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


		//set materials
		std::vector<std::shared_ptr<assembler::Assembler>> assemblers;
		assemblers.push_back(assembler);
		assemblers.push_back(mass_matrix_assembler);
		if (mixed_assembler != nullptr)
			// TODO: assemblers.push_back(mixed_assembler);
			mixed_assembler->set_size(mesh->dimension());
		if (pressure_assembler != nullptr)
			assemblers.push_back(pressure_assembler);
		set_materials(assemblers);

	*/

	void VariationalForm::compute_errors(const Eigen::MatrixXd &sol)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return;

		double tend = 0;

		if (!args["time"].is_null())
		{
			tend = args["time"]["tend"];
		}

		stats.compute_errors(n_bases, bases, geom_bases(), *mesh, *problem, tend, sol);
	}

	std::vector<int> VariationalForm::primitive_to_node() const
	{
		auto indices = iso_parametric() ? mesh_nodes->primitive_to_node() : geom_mesh_nodes->primitive_to_node();
		indices.resize(mesh->n_vertices());
		return indices;
	}

	std::vector<int> VariationalForm::node_to_primitive() const
	{
		auto p2n = primitive_to_node();
		std::vector<int> indices;
		indices.resize(n_geom_bases);
		for (int i = 0; i < p2n.size(); i++)
			indices[p2n[i]] = i;
		return indices;
	}

	void VariationalForm::build_node_mapping()
	{
		if (args["space"]["basis_type"] == "Spline")
		{
			logger().warn("Node ordering disabled, it dosent work for splines!");
			return;
		}

		if (disc_orders.maxCoeff() >= 4 || disc_orders.maxCoeff() != disc_orders.minCoeff())
		{
			logger().warn("Node ordering disabled, it works only for p < 4 and uniform order!");
			return;
		}

		if (!mesh->is_conforming())
		{
			logger().warn("Node ordering disabled, not supported for non-conforming meshes!");
			return;
		}

		if (mesh->has_poly())
		{
			logger().warn("Node ordering disabled, not supported for polygonal meshes!");
			return;
		}

		if (mesh->in_ordered_vertices().size() <= 0 || mesh->in_ordered_edges().size() <= 0 || (mesh->is_volume() && mesh->in_ordered_faces().size() <= 0))
		{
			logger().warn("Node ordering disabled, input vertices/edges/faces not computed!");
			return;
		}

		const int num_vertex_nodes = mesh_nodes->num_vertex_nodes();
		const int num_edge_nodes = mesh_nodes->num_edge_nodes();
		const int num_face_nodes = mesh_nodes->num_face_nodes();
		const int num_cell_nodes = mesh_nodes->num_cell_nodes();

		const int num_nodes = num_vertex_nodes + num_edge_nodes + num_face_nodes + num_cell_nodes;

		const long n_vertices = num_vertex_nodes;
		const int num_in_primitives = n_vertices + mesh->n_edges() + mesh->n_faces() + mesh->n_cells();
		const int num_primitives = mesh->n_vertices() + mesh->n_edges() + mesh->n_faces() + mesh->n_cells();

		igl::Timer timer;

		logger().trace("Building in-node to in-primitive mapping...");
		timer.start();
		Eigen::VectorXi in_node_to_in_primitive;
		Eigen::VectorXi in_node_offset;
		build_in_node_to_in_primitive(*mesh, *mesh_nodes, in_node_to_in_primitive, in_node_offset);
		timer.stop();
		logger().trace("Done (took {}s)", timer.getElapsedTime());

		logger().trace("Building in-primitive to primitive mapping...");
		timer.start();
		bool ok = build_in_primitive_to_primitive(
			*mesh, *mesh_nodes,
			mesh->in_ordered_vertices(),
			mesh->in_ordered_edges(),
			mesh->in_ordered_faces(),
			in_primitive_to_primitive);
		timer.stop();
		logger().trace("Done (took {}s)", timer.getElapsedTime());

		if (!ok)
		{
			in_node_to_node.resize(0);
			in_primitive_to_primitive.resize(0);
			return;
		}
		const auto &tmp = mesh_nodes->in_ordered_vertices();
		int max_tmp = -1;
		for (auto v : tmp)
			max_tmp = std::max(max_tmp, v);

		in_node_to_node.resize(max_tmp + 1);
		for (int i = 0; i < tmp.size(); ++i)
		{
			if (tmp[i] >= 0)
				in_node_to_node[tmp[i]] = i;
		}
	}

	std::string VariationalForm::formulation() const
	{
		if (args["materials"].is_null())
		{
			logger().error("specify some 'materials'");
			assert(!args["materials"].is_null());
			throw std::runtime_error("invalid input");
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
					if (AssemblerUtils::is_elastic_material(current))
					{
						if (AssemblerUtils::is_elastic_material(tmp))
							current = "MultiModels";
						else
						{
							logger().error("Current material is {}, new material is {}, multimaterial supported only for LinearElasticity and NeoHookean", current, tmp);
							throw std::runtime_error("invalid input");
						}
					}
					else
					{
						logger().error("Current material is {}, new material is {}, multimaterial supported only for LinearElasticity and NeoHookean", current, tmp);
						throw std::runtime_error("invalid input");
					}
				}
			}

			return current;
		}
		else
			return args["materials"]["type"];
	}

	bool VariationalForm::iso_parametric() const
	{
		if (mesh->has_poly())
			return true;

		if (args["space"]["basis_type"] == "Bernstein")
			return false;

		if (args["space"]["basis_type"] == "Spline")
			return true;

		if (mesh->is_rational())
			return false;

		if (args["space"]["use_p_ref"])
			return false;

		if (has_periodic_bc())
			return false;

		if (optimization_enabled == solver::CacheLevel::Derivatives)
			return false;

		if (mesh->orders().size() <= 0)
		{
			if (args["space"]["discr_order"] == 1)
				return true;
			else
				return args["space"]["advanced"]["isoparametric"];
		}

		if (mesh->orders().minCoeff() != mesh->orders().maxCoeff())
			return false;

		if (args["space"]["discr_order"] == mesh->orders().minCoeff())
			return true;

		// TODO:
		// if (args["space"]["discr_order"] == 1 && args["force_linear_geometry"])
		// 	return true;

		return args["space"]["advanced"]["isoparametric"];
	}

	void VariationalForm::build_basis()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		mesh->prepare_mesh();

		bases.clear();
		pressure_bases.clear();
		geom_bases_.clear();
		boundary_nodes.clear();
		dirichlet_nodes.clear();
		neumann_nodes.clear();
		local_boundary.clear();
		total_local_boundary.clear();
		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		polys.clear();
		poly_edge_to_data.clear();
		rhs.resize(0, 0);
		basis_nodes_to_gbasis_nodes.resize(0, 0);

		if (assembler::MultiModel *mm = dynamic_cast<assembler::MultiModel *>(assembler.get()))
		{
			assert(args["materials"].is_array());

			std::vector<std::string> materials(mesh->n_elements());

			std::map<int, std::string> mats;

			for (const auto &m : args["materials"])
				mats[m["id"].get<int>()] = m["type"];

			for (int i = 0; i < materials.size(); ++i)
				materials[i] = mats.at(mesh->get_body_id(i));

			mm->init_multimodels(materials);
		}

		n_bases = 0;
		n_geom_bases = 0;
		n_pressure_bases = 0;

		stats.reset();

		disc_orders.resize(mesh->n_elements());

		problem->init(*mesh);
		logger().info("Building {} basis...", (iso_parametric() ? "isoparametric" : "not isoparametric"));
		const bool has_polys = mesh->has_poly();

		local_boundary.clear();
		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		std::map<int, basis::InterfaceData> poly_edge_to_data_geom; // temp dummy variable

		const auto &tmp_json = args["space"]["discr_order"];
		if (tmp_json.is_number_integer())
		{
			disc_orders.setConstant(tmp_json);
		}
		else if (tmp_json.is_string())
		{
			const std::string discr_orders_path = utils::resolve_path(tmp_json, root_path());
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
				assert(b_discr_orders[i]["id"].is_array() || b_discr_orders[i]["id"].is_number_integer());

				const int order = b_discr_orders[i]["order"];
				for (const int id : json_as_array<int>(b_discr_orders[i]["id"]))
				{
					b_orders[id] = order;
					logger().trace("bid {}, discr {}", id, order);
				}
			}

			for (int e = 0; e < mesh->n_elements(); ++e)
			{
				const int bid = mesh->get_body_id(e);
				const auto order = b_orders.find(bid);
				if (order == b_orders.end())
				{
					logger().debug("Missing discretization order for body {}; using 1", bid);
					b_orders[bid] = 1;
					disc_orders[e] = 1;
				}
				else
				{
					disc_orders[e] = order->second;
				}
			}
		}
		else
		{
			logger().error("space/discr_order must be either a number a path or an array");
			throw std::runtime_error("invalid json");
		}
		// TODO: same for pressure!

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

		// TODO: implement prism geometric order
		Eigen::MatrixXi geom_disc_ordersq = geom_disc_orders;
		disc_ordersq = disc_orders;
		// disc_ordersq.setConstant(2);

		igl::Timer timer;
		timer.start();
		if (args["space"]["use_p_ref"])
		{
			refinement::APriori::p_refine(
				*mesh,
				args["space"]["advanced"]["B"],
				args["space"]["advanced"]["h1_formula"],
				args["space"]["discr_order"],
				args["space"]["advanced"]["discr_order_max"],
				stats,
				disc_orders);

			logger().info("min p: {} max p: {}", disc_orders.minCoeff(), disc_orders.maxCoeff());
		}

		int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();
		if (mixed_assembler != nullptr)
		{
			const int disc_order = disc_orders.maxCoeff();
			if (disc_order - disc_orders.minCoeff() != 0)
			{
				logger().error("p refinement not supported in mixed formulation!");
				return;
			}
		}

		// shape optimization needs continuous geometric basis
		// const bool use_continuous_gbasis = optimization_enabled == solver::CacheLevel::Derivatives;
		const bool use_continuous_gbasis = true;
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];

		if (mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
			if (args["space"]["basis_type"] == "Spline")
			{
				// if (!iso_parametric())
				// {
				// 	logger().error("Splines must be isoparametric, ignoring...");
				// 	// LagrangeBasis3d::build_bases(tmp_mesh, quadrature_order, geom_disc_orders, has_polys, geom_bases_, local_boundary, poly_edge_to_data_geom, mesh_nodes);
				// 	SplineBasis3d::build_bases(tmp_mesh, quadrature_order, geom_bases_, local_boundary, poly_edge_to_data);
				// }

				n_bases = basis::SplineBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, bases, local_boundary, poly_edge_to_data);

				// if (iso_parametric() && args["fit_nodes"])
				// 	SplineBasis3d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					n_geom_bases = basis::LagrangeBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, geom_disc_orders, geom_disc_ordersq, false, false, has_polys, !use_continuous_gbasis, use_corner_quadrature, geom_bases_, local_boundary, poly_edge_to_data_geom, geom_mesh_nodes);

				n_bases = basis::LagrangeBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, disc_orders, disc_ordersq, args["space"]["basis_type"] == "Bernstein", args["space"]["basis_type"] == "Serendipity", has_polys, false, use_corner_quadrature, bases, local_boundary, poly_edge_to_data, mesh_nodes);
			}

			// if(problem->is_mixed())
			if (mixed_assembler != nullptr)
			{
				const int order = args["space"]["pressure_discr_order"];
				// todo prism
				const int orderq = order;

				n_pressure_bases = basis::LagrangeBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, order, orderq, args["space"]["basis_type"] == "Bernstein", false, has_polys, false, use_corner_quadrature, pressure_bases, local_boundary, poly_edge_to_data_geom, pressure_mesh_nodes);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			if (args["space"]["basis_type"] == "Spline")
			{
				// TODO:
				// if (!iso_parametric())
				// {
				// 	logger().error("Splines must be isoparametric, ignoring...");
				// 	// LagrangeBasis2d::build_bases(tmp_mesh, quadrature_order, disc_orders, has_polys, geom_bases_, local_boundary, poly_edge_to_data_geom, mesh_nodes);
				// 	n_bases = SplineBasis2d::build_bases(tmp_mesh, quadrature_order, geom_bases_, local_boundary, poly_edge_to_data);
				// }

				n_bases = basis::SplineBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, bases, local_boundary, poly_edge_to_data);

				// if (iso_parametric() && args["fit_nodes"])
				// 	SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					n_geom_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, geom_disc_orders, false, false, has_polys, !use_continuous_gbasis, use_corner_quadrature, geom_bases_, local_boundary, poly_edge_to_data_geom, geom_mesh_nodes);

				n_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, disc_orders, args["space"]["basis_type"] == "Bernstein", args["space"]["basis_type"] == "Serendipity", has_polys, false, use_corner_quadrature, bases, local_boundary, poly_edge_to_data, mesh_nodes);
			}

			// if(problem->is_mixed())
			if (mixed_assembler != nullptr)
			{
				n_pressure_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, int(args["space"]["pressure_discr_order"]), args["space"]["basis_type"] == "Bernstein", false, has_polys, false, use_corner_quadrature, pressure_bases, local_boundary, poly_edge_to_data_geom, pressure_mesh_nodes);
			}
		}

		if (mixed_assembler != nullptr)
		{
			assert(bases.size() == pressure_bases.size());
			for (int i = 0; i < pressure_bases.size(); ++i)
			{
				quadrature::Quadrature b_quad;
				bases[i].compute_quadrature(b_quad);
				pressure_bases[i].set_quadrature([b_quad](quadrature::Quadrature &quad) { quad = b_quad; });
			}
		}

		timer.stop();

		build_polygonal_basis();

		if (n_geom_bases == 0)
			n_geom_bases = n_bases;

		auto &gbases = geom_bases();

		if (optimization_enabled == solver::CacheLevel::Derivatives)
		{
			std::map<std::array<int, 2>, double> pairs;
			for (int e = 0; e < gbases.size(); e++)
			{
				const auto &gbs = gbases[e].bases;
				const auto &bs = bases[e].bases;

				Eigen::MatrixXd local_pts;
				const int order = bs.front().order();
				if (mesh->is_volume())
				{
					if (mesh->is_simplex(e))
						autogen::p_nodes_3d(order, local_pts);
					else
						autogen::q_nodes_3d(order, local_pts);
				}
				else
				{
					if (mesh->is_simplex(e))
						autogen::p_nodes_2d(order, local_pts);
					else
						autogen::q_nodes_2d(order, local_pts);
				}

				ElementAssemblyValues vals;
				vals.compute(e, mesh->is_volume(), local_pts, gbases[e], gbases[e]);

				for (int i = 0; i < bs.size(); i++)
				{
					for (int j = 0; j < gbs.size(); j++)
					{
						if (std::abs(vals.basis_values[j].val(i)) > 1e-7)
						{
							std::array<int, 2> index = {{gbs[j].global()[0].index, bs[i].global()[0].index}};
							pairs.insert({index, vals.basis_values[j].val(i)});
						}
					}
				}
			}

			const int dim = mesh->dimension();
			std::vector<Eigen::Triplet<double>> coeffs;
			coeffs.clear();
			for (const auto &iter : pairs)
				for (int d = 0; d < dim; d++)
					coeffs.emplace_back(iter.first[0] * dim + d, iter.first[1] * dim + d, iter.second);

			basis_nodes_to_gbasis_nodes.resize(n_geom_bases * mesh->dimension(), n_bases * mesh->dimension());
			basis_nodes_to_gbasis_nodes.setFromTriplets(coeffs.begin(), coeffs.end());
		}

		for (const auto &lb : local_boundary)
			total_local_boundary.emplace_back(lb);

		const int dim = mesh->dimension();
		const int problem_dim = problem->is_scalar() ? 1 : dim;

		// handle periodic bc
		if (has_periodic_bc())
		{
			// collect periodic directions
			json directions = args["boundary_conditions"]["periodic_boundary"]["correspondence"];
			Eigen::MatrixXd tile_offset = Eigen::MatrixXd::Identity(dim, dim);

			if (directions.size() > 0)
			{
				Eigen::VectorXd tmp;
				for (int d = 0; d < dim; d++)
				{
					tmp = directions[d];
					if (tmp.size() != dim)
						log_and_throw_error("Invalid size of periodic directions!");
					tile_offset.col(d) = tmp;
				}
			}

			periodic_bc = std::make_shared<PeriodicBoundary>(problem->is_scalar(), n_bases, bases, mesh_nodes, tile_offset, args["boundary_conditions"]["periodic_boundary"]["tolerance"].get<double>());

			macro_strain_constraint.init(dim, args["boundary_conditions"]["periodic_boundary"]);
		}

		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(*mesh, geom_bases());

		const int prev_bases = n_bases;
		n_bases += obstacle.n_vertices();

		{
			igl::Timer timer2;
			logger().debug("Building node mapping...");
			timer2.start();
			build_node_mapping();
			problem->update_nodes(in_node_to_node);
			mesh->update_nodes(in_node_to_node);
			timer2.stop();
			logger().debug("Done (took {}s)", timer2.getElapsedTime());
		}

		logger().info("Building collision mesh...");
		build_collision_mesh();
		if (periodic_bc && args["contact"]["periodic"])
			build_periodic_collision_mesh();
		logger().info("Done!");

		const int prev_b_size = local_boundary.size();
		problem->setup_bc(*mesh, n_bases - obstacle.n_vertices(),
						  bases, geom_bases(), pressure_bases,
						  local_boundary,
						  boundary_nodes,
						  local_neumann_boundary,
						  local_pressure_boundary,
						  local_pressure_cavity,
						  pressure_boundary_nodes,
						  dirichlet_nodes, neumann_nodes);

		// setp nodal values
		{
			dirichlet_nodes_position.resize(dirichlet_nodes.size());
			for (int n = 0; n < dirichlet_nodes.size(); ++n)
			{
				const int n_id = dirichlet_nodes[n];
				bool found = false;
				for (const auto &bs : bases)
				{
					for (const auto &b : bs.bases)
					{
						for (const auto &lg : b.global())
						{
							if (lg.index == n_id)
							{
								dirichlet_nodes_position[n] = lg.node;
								found = true;
								break;
							}
						}

						if (found)
							break;
					}

					if (found)
						break;
				}

				assert(found);
			}

			neumann_nodes_position.resize(neumann_nodes.size());
			for (int n = 0; n < neumann_nodes.size(); ++n)
			{
				const int n_id = neumann_nodes[n];
				bool found = false;
				for (const auto &bs : bases)
				{
					for (const auto &b : bs.bases)
					{
						for (const auto &lg : b.global())
						{
							if (lg.index == n_id)
							{
								neumann_nodes_position[n] = lg.node;
								found = true;
								break;
							}
						}

						if (found)
							break;
					}

					if (found)
						break;
				}

				assert(found);
			}
		}

		const bool has_neumann = local_neumann_boundary.size() > 0 || local_boundary.size() < prev_b_size;
		use_avg_pressure = !has_neumann;

		for (int i = prev_bases; i < n_bases; ++i)
		{
			for (int d = 0; d < problem_dim; ++d)
				boundary_nodes.push_back(i * problem_dim + d);
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.resize(std::distance(boundary_nodes.begin(), it));

		// for elastic pure periodic problem, find an internal node and force zero dirichlet
		if ((!problem->is_time_dependent() || args["time"]["quasistatic"]) && boundary_nodes.size() == 0 && has_periodic_bc())
		{
			// find an internal node to force zero dirichlet
			std::vector<bool> isboundary(n_bases, false);
			for (const auto &lb : total_local_boundary)
			{
				const int e = lb.element_id();
				for (int i = 0; i < lb.size(); ++i)
				{
					const auto nodes = bases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *mesh);

					for (int n : nodes)
						isboundary[bases[e].bases[n].global()[0].index] = true;
				}
			}
			int i = 0;
			for (; i < n_bases; i++)
				if (!isboundary[i]) // (!periodic_bc->is_periodic_dof(i))
					break;
			if (i >= n_bases)
				log_and_throw_error("Failed to find a non-periodic node!");
			const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
			for (int d = 0; d < actual_dim; d++)
			{
				boundary_nodes.push_back(i * actual_dim + d);
			}
			logger().info("Fix solution at node {} to remove singularity due to periodic BC", i);
		}

		const auto &curret_bases = geom_bases();
		const int n_samples = 10;
		stats.compute_mesh_size(*mesh, curret_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);
		if (starting_min_edge_length < 0)
		{
			starting_min_edge_length = stats.min_edge_length;
		}
		if (starting_max_edge_length < 0)
		{
			starting_max_edge_length = stats.mesh_size;
		}

		if (is_contact_enabled())
		{
			min_boundary_edge_length = std::numeric_limits<double>::max();
			for (const auto &edge : collision_mesh.edges().rowwise())
			{
				const VectorNd v0 = collision_mesh.rest_positions().row(edge(0));
				const VectorNd v1 = collision_mesh.rest_positions().row(edge(1));
				min_boundary_edge_length = std::min(min_boundary_edge_length, (v1 - v0).norm());
			}

			double dhat = Units::convert(args["contact"]["dhat"], units.length());
			args["contact"]["epsv"] = Units::convert(args["contact"]["epsv"], units.velocity());
			args["contact"]["dhat"] = dhat;

			if (!has_dhat && dhat > min_boundary_edge_length)
			{
				args["contact"]["dhat"] = double(args["contact"]["dhat_percentage"]) * min_boundary_edge_length;
				logger().info("dhat set to {}", double(args["contact"]["dhat"]));
			}
			else
			{
				if (dhat > min_boundary_edge_length)
					logger().warn("dhat larger than min boundary edge, {} > {}", dhat, min_boundary_edge_length);
			}
		}

		logger().info("n_bases {}", n_bases);

		timings.building_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.building_basis_time);

		logger().info("flipped elements {}", stats.n_flipped);
		logger().info("h: {}", stats.mesh_size);
		logger().info("n bases: {}", n_bases);
		logger().info("n pressure bases: {}", n_pressure_bases);

		ass_vals_cache.clear();
		mass_ass_vals_cache.clear();
		if (n_bases <= args["solver"]["advanced"]["cache_size"])
		{
			timer.start();
			logger().info("Building cache...");
			ass_vals_cache.init(mesh->is_volume(), bases, curret_bases);
			mass_ass_vals_cache.init(mesh->is_volume(), bases, curret_bases, true);
			if (mixed_assembler != nullptr)
				pressure_ass_vals_cache.init(mesh->is_volume(), pressure_bases, curret_bases);

			logger().info(" took {}s", timer.getElapsedTime());
		}

		out_geom.build_grid(*mesh, args["output"]["advanced"]["sol_on_grid"]);

		if ((!problem->is_time_dependent() || args["time"]["quasistatic"]) && boundary_nodes.empty())
		{
			if (has_periodic_bc())
				logger().warn("(Quasi-)Static problem without Dirichlet nodes, will fix solution at one node to find a unique solution!");
			else
			{
				if (args["constraints"]["hard"].empty())
					log_and_throw_error("Static problem need to have some Dirichlet nodes!");
				else
					logger().warn("Relying on hard constraints to avoid infinite solutions");
			}
		}
	}

	void VariationalForm::build_polygonal_basis()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		rhs.resize(0, 0);

		if (poly_edge_to_data.empty() && polys.empty())
		{
			timings.computing_poly_basis_time = 0;
			return;
		}

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
				if (args["space"]["poly_basis_type"] == "MeanValue" || args["space"]["poly_basis_type"] == "Wachspress")
					logger().error("Barycentric bases not supported in 3D");
				assert(assembler->is_linear());
				new_bases = basis::PolygonalBasis3d::build_bases(
					*dynamic_cast<LinearAssembler *>(assembler.get()),
					args["space"]["advanced"]["n_harmonic_samples"],
					*dynamic_cast<Mesh3D *>(mesh.get()),
					n_bases,
					args["space"]["advanced"]["quadrature_order"],
					args["space"]["advanced"]["mass_quadrature_order"],
					args["space"]["advanced"]["integral_constraints"],
					bases,
					bases,
					poly_edge_to_data,
					polys_3d);
			}
			else
			{
				if (args["space"]["poly_basis_type"] == "MeanValue")
				{
					new_bases = basis::MVPolygonalBasis2d::build_bases(
						assembler->name(),
						assembler->is_tensor() ? 2 : 1,
						*dynamic_cast<Mesh2D *>(mesh.get()),
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else if (args["space"]["poly_basis_type"] == "Wachspress")
				{
					new_bases = basis::WSPolygonalBasis2d::build_bases(
						assembler->name(),
						assembler->is_tensor() ? 2 : 1,
						*dynamic_cast<Mesh2D *>(mesh.get()),
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else
				{
					assert(assembler->is_linear());
					new_bases = basis::PolygonalBasis2d::build_bases(
						*dynamic_cast<LinearAssembler *>(assembler.get()),
						args["space"]["advanced"]["n_harmonic_samples"],
						*dynamic_cast<Mesh2D *>(mesh.get()),
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						args["space"]["advanced"]["integral_constraints"],
						bases,
						bases,
						poly_edge_to_data,
						polys);
				}
			}
		}
		else
		{
			if (mesh->is_volume())
			{
				if (args["space"]["poly_basis_type"] == "MeanValue" || args["space"]["poly_basis_type"] == "Wachspress")
				{
					logger().error("Barycentric bases not supported in 3D");
					throw std::runtime_error("not implemented");
				}
				else
				{
					assert(assembler->is_linear());
					new_bases = basis::PolygonalBasis3d::build_bases(
						*dynamic_cast<LinearAssembler *>(assembler.get()),
						args["space"]["advanced"]["n_harmonic_samples"],
						*dynamic_cast<Mesh3D *>(mesh.get()),
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						args["space"]["advanced"]["integral_constraints"],
						bases,
						geom_bases_,
						poly_edge_to_data,
						polys_3d);
				}
			}
			else
			{
				if (args["space"]["poly_basis_type"] == "MeanValue")
				{
					new_bases = basis::MVPolygonalBasis2d::build_bases(
						assembler->name(),
						assembler->is_tensor() ? 2 : 1,
						*dynamic_cast<Mesh2D *>(mesh.get()),
						n_bases, args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else if (args["space"]["poly_basis_type"] == "Wachspress")
				{
					new_bases = basis::WSPolygonalBasis2d::build_bases(
						assembler->name(),
						assembler->is_tensor() ? 2 : 1,
						*dynamic_cast<Mesh2D *>(mesh.get()),
						n_bases, args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else
				{
					assert(assembler->is_linear());
					new_bases = basis::PolygonalBasis2d::build_bases(
						*dynamic_cast<LinearAssembler *>(assembler.get()),
						args["space"]["advanced"]["n_harmonic_samples"],
						*dynamic_cast<Mesh2D *>(mesh.get()),
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						args["space"]["advanced"]["integral_constraints"],
						bases,
						geom_bases_,
						poly_edge_to_data,
						polys);
				}
			}
		}

		timer.stop();
		timings.computing_poly_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.computing_poly_basis_time);

		n_bases += new_bases;
	}

	void VariationalForm::build_periodic_collision_mesh()
	{
		assert(!mesh->is_volume());
		const int dim = mesh->dimension();
		const int n_tiles = 2;

		if (mesh->dimension() != 2)
			log_and_throw_error("Periodic collision mesh is only implemented in 2D!");

		Eigen::MatrixXd V(n_bases, dim);
		for (const auto &bs : bases)
			for (const auto &b : bs.bases)
				for (const auto &g : b.global())
					V.row(g.index) = g.node;

		Eigen::MatrixXi E = collision_mesh.edges();
		for (int i = 0; i < E.size(); i++)
			E(i) = collision_mesh.to_full_vertex_id(E(i));

		Eigen::MatrixXd bbox(V.cols(), 2);
		bbox.col(0) = V.colwise().minCoeff();
		bbox.col(1) = V.colwise().maxCoeff();

		// remove boundary edges on periodic BC, buggy
		{
			std::vector<int> ind;
			for (int i = 0; i < E.rows(); i++)
			{
				if (!periodic_bc->is_periodic_dof(E(i, 0)) || !periodic_bc->is_periodic_dof(E(i, 1)))
					ind.push_back(i);
			}

			E = E(ind, Eigen::all).eval();
		}

		Eigen::MatrixXd Vtmp, Vnew;
		Eigen::MatrixXi Etmp, Enew;
		Vtmp.setZero(V.rows() * n_tiles * n_tiles, V.cols());
		Etmp.setZero(E.rows() * n_tiles * n_tiles, E.cols());

		Eigen::MatrixXd tile_offset = periodic_bc->get_affine_matrix();

		for (int i = 0, idx = 0; i < n_tiles; i++)
		{
			for (int j = 0; j < n_tiles; j++)
			{
				Eigen::Vector2d block_id;
				block_id << i, j;

				Vtmp.middleRows(idx * V.rows(), V.rows()) = V;
				// Vtmp.block(idx * V.rows(), 0, V.rows(), 1).array() += tile_offset(0) * i;
				// Vtmp.block(idx * V.rows(), 1, V.rows(), 1).array() += tile_offset(1) * j;
				for (int vid = 0; vid < V.rows(); vid++)
					Vtmp.block(idx * V.rows() + vid, 0, 1, 2) += (tile_offset * block_id).transpose();

				Etmp.middleRows(idx * E.rows(), E.rows()) = E.array() + idx * V.rows();
				idx += 1;
			}
		}

		// clean duplicated vertices
		Eigen::VectorXi indices;
		{
			std::vector<int> tmp;
			for (int i = 0; i < V.rows(); i++)
			{
				if (periodic_bc->is_periodic_dof(i))
					tmp.push_back(i);
			}

			indices.resize(tmp.size() * n_tiles * n_tiles);
			for (int i = 0; i < n_tiles * n_tiles; i++)
			{
				indices.segment(i * tmp.size(), tmp.size()) = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(tmp.data(), tmp.size());
				indices.segment(i * tmp.size(), tmp.size()).array() += i * V.rows();
			}
		}

		Eigen::VectorXi potentially_duplicate_mask(Vtmp.rows());
		potentially_duplicate_mask.setZero();
		potentially_duplicate_mask(indices).array() = 1;

		Eigen::MatrixXd candidates = Vtmp(indices, Eigen::all);

		Eigen::VectorXi SVI;
		std::vector<int> SVJ;
		SVI.setConstant(Vtmp.rows(), -1);
		int id = 0;
		const double eps = (bbox.col(1) - bbox.col(0)).maxCoeff() * args["boundary_conditions"]["periodic_boundary"]["tolerance"].get<double>();
		for (int i = 0; i < Vtmp.rows(); i++)
		{
			if (SVI[i] < 0)
			{
				SVI[i] = id;
				SVJ.push_back(i);
				if (potentially_duplicate_mask(i))
				{
					Eigen::VectorXd diffs = (candidates.rowwise() - Vtmp.row(i)).rowwise().norm();
					for (int j = 0; j < diffs.size(); j++)
						if (diffs(j) < eps)
							SVI[indices[j]] = id;
				}
				id++;
			}
		}

		Vnew = Vtmp(SVJ, Eigen::all);

		Enew.resizeLike(Etmp);
		for (int d = 0; d < Etmp.cols(); d++)
			Enew.col(d) = SVI(Etmp.col(d));

		std::vector<bool> is_on_surface = ipc::CollisionMesh::construct_is_on_surface(Vnew.rows(), Enew);

		Eigen::MatrixXi boundary_triangles;
		Eigen::SparseMatrix<double> displacement_map;
		periodic_collision_mesh = ipc::CollisionMesh(is_on_surface,
													 std::vector<bool>(Vnew.rows(), false),
													 Vnew,
													 Enew,
													 boundary_triangles,
													 displacement_map);

		periodic_collision_mesh.init_area_jacobians();

		periodic_collision_mesh_to_basis.setConstant(Vnew.rows(), -1);
		for (int i = 0; i < V.rows(); i++)
			for (int j = 0; j < n_tiles * n_tiles; j++)
				periodic_collision_mesh_to_basis(SVI[j * V.rows() + i]) = i;

		if (periodic_collision_mesh_to_basis.maxCoeff() + 1 != V.rows())
			log_and_throw_error("Failed to tile mesh!");
	}

	void VariationalForm::build_collision_mesh()
	{
		build_collision_mesh(
			*mesh, n_bases, bases, geom_bases(), total_local_boundary, obstacle,
			args, [this](const std::string &p) { return resolve_input_path(p); },
			in_node_to_node, collision_mesh);
	}

	void VariationalForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const mesh::Obstacle &obstacle,
		const json &args,
		const std::function<std::string(const std::string &)> &resolve_input_path,
		const Eigen::VectorXi &in_node_to_node,
		ipc::CollisionMesh &collision_mesh)
	{
		Eigen::MatrixXd collision_vertices;
		Eigen::VectorXi collision_codim_vids;
		Eigen::MatrixXi collision_edges, collision_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;

		if (args.contains("/contact/collision_mesh"_json_pointer)
			&& args.at("/contact/collision_mesh/enabled"_json_pointer).get<bool>())
		{
			const json collision_mesh_args = args.at("/contact/collision_mesh"_json_pointer);
			if (collision_mesh_args.contains("linear_map"))
			{
				assert(displacement_map_entries.empty());
				assert(collision_mesh_args.contains("mesh"));
				const std::string root_path = utils::json_value<std::string>(args, "root_path", "");
				// TODO: handle transformation per geometry
				const json transformation = json_as_array(args["geometry"])[0]["transformation"];
				mesh::load_collision_proxy(
					utils::resolve_path(collision_mesh_args["mesh"], root_path),
					utils::resolve_path(collision_mesh_args["linear_map"], root_path),
					in_node_to_node, transformation, collision_vertices, collision_codim_vids,
					collision_edges, collision_triangles, displacement_map_entries);
			}
			else if (collision_mesh_args.contains("max_edge_length"))
			{
				logger().debug(
					"Building collision proxy with max edge length={} ...",
					collision_mesh_args["max_edge_length"].get<double>());
				igl::Timer timer;
				timer.start();
				build_collision_proxy(
					bases, geom_bases, total_local_boundary, n_bases, mesh.dimension(),
					collision_mesh_args["max_edge_length"], collision_vertices,
					collision_triangles, displacement_map_entries,
					collision_mesh_args["tessellation_type"]);
				if (collision_triangles.size())
					igl::edges(collision_triangles, collision_edges);
				timer.stop();
				logger().debug(fmt::format(
					std::locale("en_US.UTF-8"),
					"Done (took {:g}s, {:L} vertices, {:L} triangles)",
					timer.getElapsedTime(),
					collision_vertices.rows(), collision_triangles.rows()));
			}
			else
			{
				io::OutGeometryData::extract_boundary_mesh(
					mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
					collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
			}
		}
		else
		{
			io::OutGeometryData::extract_boundary_mesh(
				mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
				collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
		}

		std::vector<bool> is_orientable_vertex(collision_vertices.rows(), true);

		// n_bases already contains the obstacle vertices
		const int num_fe_nodes = n_bases - obstacle.n_vertices();
		const int num_fe_collision_vertices = collision_vertices.rows();
		assert(collision_edges.size() == 0 || collision_edges.maxCoeff() < num_fe_collision_vertices);
		assert(collision_triangles.size() == 0 || collision_triangles.maxCoeff() < num_fe_collision_vertices);

		// Append the obstacles to the collision mesh
		if (obstacle.n_vertices() > 0)
		{
			append_rows(collision_vertices, obstacle.v());
			append_rows(collision_codim_vids, obstacle.codim_v().array() + num_fe_collision_vertices);
			append_rows(collision_edges, obstacle.e().array() + num_fe_collision_vertices);
			append_rows(collision_triangles, obstacle.f().array() + num_fe_collision_vertices);

			for (int i = 0; i < obstacle.n_vertices(); i++)
			{
				is_orientable_vertex.push_back(false);
			}

			if (!displacement_map_entries.empty())
			{
				displacement_map_entries.reserve(displacement_map_entries.size() + obstacle.n_vertices());
				for (int i = 0; i < obstacle.n_vertices(); i++)
				{
					displacement_map_entries.emplace_back(num_fe_collision_vertices + i, num_fe_nodes + i, 1.0);
				}
			}
		}

		std::vector<bool> is_on_surface = ipc::CollisionMesh::construct_is_on_surface(
			collision_vertices.rows(), collision_edges);
		for (const int vid : collision_codim_vids)
		{
			is_on_surface[vid] = true;
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(collision_vertices.rows(), n_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		collision_mesh = ipc::CollisionMesh(
			is_on_surface, is_orientable_vertex, collision_vertices, collision_edges, collision_triangles,
			displacement_map);

		collision_mesh.can_collide = [&collision_mesh, num_fe_collision_vertices](size_t vi, size_t vj) {
			// obstacles do not collide with other obstacles
			return collision_mesh.to_full_vertex_id(vi) < num_fe_collision_vertices
				   || collision_mesh.to_full_vertex_id(vj) < num_fe_collision_vertices;
		};

		collision_mesh.init_area_jacobians();
	}

	void VariationalForm::assemble_mass_mat()
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
		if (assembler->name() == "OperatorSplitting")
		{
			timings.assembling_stiffness_mat_time = 0;
			avg_mass = 1;
			return;
		}

		if (!problem->is_time_dependent())
		{
			avg_mass = 1;
			timings.assembling_mass_mat_time = 0;
			return;
		}

		mass.resize(0, 0);

		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");

		if (mixed_assembler != nullptr)
		{
			StiffnessMatrix velocity_mass;
			mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, geom_bases(), mass_ass_vals_cache, 0, velocity_mass, true);

			std::vector<Eigen::Triplet<double>> mass_blocks;
			mass_blocks.reserve(velocity_mass.nonZeros());

			for (int k = 0; k < velocity_mass.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
				{
					mass_blocks.emplace_back(it.row(), it.col(), it.value());
				}
			}

			mass.resize(n_bases * assembler->size(), n_bases * assembler->size());
			mass.setFromTriplets(mass_blocks.begin(), mass_blocks.end());
			mass.makeCompressed();
		}
		else
		{
			mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, geom_bases(), mass_ass_vals_cache, 0, mass, true);
		}

		assert(mass.size() > 0);

		avg_mass = 0;
		for (int k = 0; k < mass.outerSize(); ++k)
		{

			for (StiffnessMatrix::InnerIterator it(mass, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass += it.value();
			}
		}

		avg_mass /= mass.rows();
		logger().info("average mass {}", avg_mass);

		if (args["solver"]["advanced"]["lump_mass_matrix"])
		{
			mass = lump_matrix(mass);
		}

		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);

		stats.nn_zero = mass.nonZeros();
		stats.num_dofs = mass.rows();
		stats.mat_size = (long long)mass.rows() * (long long)mass.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);
	}

	std::shared_ptr<RhsAssembler> VariationalForm::build_rhs_assembler(
		const int n_bases_,
		const std::vector<basis::ElementBases> &bases_,
		const assembler::AssemblyValsCache &ass_vals_cache_) const
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		const int size = problem->is_scalar() ? 1 : mesh->dimension();

		return std::make_shared<RhsAssembler>(
			*assembler, *mesh, obstacle,
			dirichlet_nodes, neumann_nodes,
			dirichlet_nodes_position, neumann_nodes_position,
			n_bases_, size, bases_, geom_bases(), ass_vals_cache_, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	std::shared_ptr<PressureAssembler> VariationalForm::build_pressure_assembler(
		const int n_bases_,
		const std::vector<basis::ElementBases> &bases_) const
	{
		const int size = problem->is_scalar() ? 1 : mesh->dimension();

		return std::make_shared<PressureAssembler>(
			*assembler, *mesh, obstacle,
			local_pressure_boundary,
			local_pressure_cavity,
			boundary_nodes,
			primitive_to_node(), node_to_primitive(),
			n_bases_, size, bases_, geom_bases(), *problem);
	}

	void VariationalForm::assemble_rhs()
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

		json p_params = {};
		p_params["formulation"] = assembler->name();
		p_params["root_path"] = root_path();
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

		rhs.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		solve_data.rhs_assembler = build_rhs_assembler();
		solve_data.rhs_assembler->assemble(mass_matrix_assembler->density(), rhs);
		rhs *= -1;

		// if(problem->is_mixed())
		if (mixed_assembler != nullptr)
		{
			const int prev_size = rhs.size();
			const int n_larger = n_pressure_bases + (use_avg_pressure ? (assembler->is_fluid() ? 1 : 0) : 0);
			rhs.conservativeResize(prev_size + n_larger, rhs.cols());
			if (assembler->name() == "OperatorSplitting")
			{
				timings.assigning_rhs_time = 0;
				return;
			}
			// Divergence free rhs
			if (assembler->name() != "Bilaplacian" || local_neumann_boundary.empty())
			{
				rhs.block(prev_size, 0, n_larger, rhs.cols()).setZero();
			}
			else
			{
				Eigen::MatrixXd tmp(n_pressure_bases, 1);
				tmp.setZero();

				std::shared_ptr<RhsAssembler> tmp_rhs_assembler = build_rhs_assembler(
					n_pressure_bases, pressure_bases, pressure_ass_vals_cache);

				tmp_rhs_assembler->set_bc(std::vector<LocalBoundary>(), std::vector<int>(), n_boundary_samples(), local_neumann_boundary, tmp);
				rhs.block(prev_size, 0, n_larger, rhs.cols()) = tmp;
			}
		}

		timer.stop();
		timings.assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assigning_rhs_time);
	}

	void VariationalForm::solve_problem(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
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

		// if (rhs.size() <= 0)
		// {
		// 	logger().error("Assemble the rhs first!");
		// 	return;
		// }

		// sol.resize(0, 0);
		// pressure.resize(0, 0);
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		init_solve(sol, pressure);

		if (problem->is_time_dependent())
		{
			const double t0 = args["time"]["t0"];
			const int time_steps = args["time"]["time_steps"];
			const double dt = args["time"]["dt"];

			// Pre log the output path for easier watching
			if (args["output"]["advanced"]["save_time_sequence"])
			{
				logger().info("Time sequence of simulation will be written to: \"{}\"",
							  resolve_output_path(args["output"]["paraview"]["file_name"]));
			}

			if (assembler->name() == "NavierStokes")
				solve_transient_navier_stokes(time_steps, t0, dt, sol, pressure);
			else if (assembler->name() == "OperatorSplitting")
				solve_transient_navier_stokes_split(time_steps, dt, sol, pressure);
			else if (is_homogenization())
				solve_homogenization(time_steps, t0, dt, sol);
			else if (is_problem_linear())
				solve_transient_linear(time_steps, t0, dt, sol, pressure);
			else if (!assembler->is_linear() && problem->is_scalar())
				throw std::runtime_error("Nonlinear scalar problems are not supported yet!");
			else
				solve_transient_tensor_nonlinear(time_steps, t0, dt, sol);
		}
		else
		{
			if (assembler->name() == "NavierStokes")
				solve_navier_stokes(sol, pressure);
			else if (is_homogenization())
				solve_homogenization(/* time steps */ 0, /* t0 */ 0, /* dt */ 0, sol);
			else if (is_problem_linear())
			{
				init_linear_solve(sol);
				solve_linear(sol, pressure);
				if (optimization_enabled != solver::CacheLevel::None)
					cache_transient_adjoint_quantities(0, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));
			}
			else if (!assembler->is_linear() && problem->is_scalar())
				throw std::runtime_error("Nonlinear scalar problems are not supported yet!");
			else
			{
				init_nonlinear_tensor_solve(sol);
				solve_tensor_nonlinear(sol);
				if (optimization_enabled != solver::CacheLevel::None)
					cache_transient_adjoint_quantities(0, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));

				const std::string state_path = resolve_output_path(args["output"]["data"]["state"]);
				if (!state_path.empty())
					write_matrix(state_path, "u", sol);
			}
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}
} // namespace polyfem
