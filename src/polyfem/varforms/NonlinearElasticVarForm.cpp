#include "NonlinearElasticVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/MacroStrain.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/mesh/collision_proxy/CollisionProxy.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Jacobian.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/SolverCSVWriter.hpp>

#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
#include <polyfem/solver/forms/lagrangian/PeriodicBoundaryLagrangianForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <igl/Timer.h>
#include <igl/edges.h>

#include <ipc/ipc.hpp>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace polyfem::varform
{
	using namespace solver;
	using namespace time_integrator;

	void NonlinearElasticVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		json clean_args = args;
		const bool contact_dhat_was_explicit = clean_args["contact"].value("_dhat_was_explicit", false);
		clean_args["contact"].erase("_dhat_was_explicit");
		ElasticVarForm::init(formulation, units, clean_args, out_path);
		contact_dhat_was_explicit_ = contact_dhat_was_explicit;
	}

	void NonlinearElasticVarForm::reset()
	{
		ElasticVarForm::reset();
		collision_mesh_ = ipc::CollisionMesh();
		periodic_collision_mesh_ = ipc::CollisionMesh();
		periodic_collision_mesh_to_basis_.resize(0);
		obstacle.clear();
		solve_data_ = solver::SolveData();
		forms.clear();
		elasticity_pressure_assembler = nullptr;
		damping_assembler_ = nullptr;
		damping_prev_assembler_ = nullptr;
		contact_dhat_was_explicit_ = false;
	}

	void NonlinearElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		ElasticVarForm::load_mesh(mesh, args);
	}

	void NonlinearElasticVarForm::set_obstacle(mesh::Obstacle &&loaded_obstacle)
	{
		obstacle = std::move(loaded_obstacle);
	}

	io::OutputSpace NonlinearElasticVarForm::output_space() const
	{
		auto space = ElasticVarForm::output_space();
		space.collision_mesh = is_contact_enabled() ? &collision_mesh_ : nullptr;
		space.obstacle = &obstacle;
		return space;
	}

	std::vector<io::OutputField> NonlinearElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields = elastic_output_fields(
			sample, solution, options, &obstacle, solve_data_.time_integrator.get(),
			solve_data_.named_forms(), solve_data_.elastic_form.get(), solve_data_.contact_form.get());
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;
		if (sample.domain != io::OutputSample::Domain::Contact)
			return fields;

		const int actual_dim = problem->is_scalar() ? 1 : mesh_->dimension();
		const auto &paraview_options = args["output"]["paraview"]["options"];
		const bool explicit_fields = !options.fields.empty();

		const auto has_field = [&](const std::string &name) {
			return std::any_of(fields.begin(), fields.end(), [&](const io::OutputField &field) {
				return field.association == io::OutputField::Association::Point && field.name == name;
			});
		};

		const auto append_collision_dof_field = [&](const std::string &name, const Eigen::MatrixXd &dof_values) {
			if (has_field(name) || dof_values.size() <= 0)
				return;

			Eigen::MatrixXd values = collision_mesh_.map_displacements(utils::unflatten(dof_values, actual_dim));
			if (values.rows() == sample.points.rows())
				fields.push_back({name, values, io::OutputField::Association::Point});
		};

		const auto append_collision_form_force = [&](const std::string &name, const std::shared_ptr<solver::Form> &form) {
			if (!form || !form->enabled() || sample.points.rows() != collision_mesh_.rest_positions().rows())
				return;

			Eigen::VectorXd force;
			form->first_derivative(solution.col(0), force);
			const double acceleration_scaling =
				solve_data_.time_integrator ? solve_data_.time_integrator->acceleration_scaling() : 1;
			force *= -1.0 / acceleration_scaling;
			append_collision_dof_field(name, force);
		};

		if (paraview_options["forces"] && !problem->is_scalar())
		{
			const double s = solve_data_.time_integrator ? solve_data_.time_integrator->acceleration_scaling() : 1;
			for (const auto &[name, form] : solve_data_.named_forms())
			{
				const std::string field_name = name + "_forces";
				if (!options.export_field(field_name))
					continue;

				Eigen::VectorXd force;
				if (form && form->enabled())
				{
					form->first_derivative(solution, force);
					force *= -1.0 / s;
				}
				else
				{
					force.setZero(solution.size());
				}
				append_collision_dof_field(field_name, force);
			}
		}

		if (options.export_field("gradient_of_elastic_potential") && solve_data_.elastic_form)
		{
			Eigen::VectorXd potential_grad;
			solve_data_.elastic_form->first_derivative(solution, potential_grad);
			append_collision_dof_field("gradient_of_elastic_potential", potential_grad);
		}

		if (options.export_field("gradient_of_contact_potential") && solve_data_.contact_form && solve_data_.contact_form->weight() > 0)
		{
			Eigen::VectorXd potential_grad;
			solve_data_.contact_form->first_derivative(solution, potential_grad);
			potential_grad *= -solve_data_.contact_form->barrier_stiffness() / solve_data_.contact_form->weight();
			append_collision_dof_field("gradient_of_contact_potential", potential_grad);
		}

		if (options.export_field("displacement"))
			append_collision_dof_field("displacement", solution);
		if (options.export_field("solution"))
			append_collision_dof_field("solution", solution);

		if ((paraview_options["contact_forces"] || explicit_fields) && options.export_field("contact_forces"))
			append_collision_form_force("contact_forces", solve_data_.contact_form);
		if ((paraview_options["friction_forces"] || explicit_fields) && options.export_field("friction_forces"))
			append_collision_form_force("friction_forces", solve_data_.friction_form);
		if ((paraview_options["normal_adhesion_forces"] || explicit_fields) && options.export_field("normal_adhesion_forces"))
			append_collision_form_force("normal_adhesion_forces", solve_data_.normal_adhesion_form);
		if ((paraview_options["tangential_adhesion_forces"] || explicit_fields) && options.export_field("tangential_adhesion_forces"))
			append_collision_form_force("tangential_adhesion_forces", solve_data_.tangential_adhesion_form);

		if (explicit_fields
			&& options.export_field("adaptive_dhat")
			&& args["contact"]["use_gcp_formulation"]
			&& args["contact"]["use_adaptive_dhat"])
		{
			const auto smooth_contact = std::dynamic_pointer_cast<solver::SmoothContactForm>(solve_data_.contact_form);
			if (smooth_contact)
			{
				const auto &set = smooth_contact->collision_set();
				if (actual_dim == 2)
				{
					Eigen::VectorXd dhats(collision_mesh_.num_edges());
					for (int e = 0; e < dhats.size(); ++e)
						dhats(e) = set.get_edge_dhat(e);
					fields.push_back({"dhat", dhats, io::OutputField::Association::Cell});
				}
				else
				{
					Eigen::VectorXd dhats(collision_mesh_.num_faces());
					for (int f = 0; f < dhats.size(); ++f)
						dhats(f) = set.get_face_dhat(f);
					fields.push_back({"dhat_face", dhats, io::OutputField::Association::Cell});

					Eigen::VectorXd vertex_dhats(collision_mesh_.num_vertices());
					for (int v = 0; v < vertex_dhats.size(); ++v)
						vertex_dhats(v) = set.get_vert_dhat(v);
					fields.push_back({"dhat_vert", vertex_dhats, io::OutputField::Association::Point});
				}
			}
		}

		return fields;
	}

	void NonlinearElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		ElasticVarForm::build_basis(mesh, iso_parametric, args);

		// Legacy nonlinear/contact code assumes the displacement space includes obstacle vertices.
		// The shared build path only counts FE bases, so extend it here
		// before constructing collision/contact state.
		const int n_fe_bases = space_.n_bases;
		space_.n_bases += obstacle.n_vertices();

		if (is_contact_enabled())
		{
			logger().info("Building collision mesh...");
			build_collision_mesh(mesh, args);
			preprocess_contact_parameters();

			if (args["contact"]["periodic"])
				build_periodic_collision_mesh();
		}

		logger().info("Done!");

		for (int i = n_fe_bases; i < space_.n_bases; ++i)
		{
			for (int d = 0; d < mesh.dimension(); ++d)
				boundary_.boundary_nodes.push_back(i * mesh.dimension() + d);
		}

		boundary_.normalize_boundary_nodes();
	}

	void NonlinearElasticVarForm::preprocess_contact_parameters()
	{
		if (!is_contact_enabled())
			return;

		double min_boundary_edge_length = std::numeric_limits<double>::max();
		for (const auto &edge : collision_mesh_.edges().rowwise())
		{
			const VectorNd v0 = collision_mesh_.rest_positions().row(edge(0));
			const VectorNd v1 = collision_mesh_.rest_positions().row(edge(1));
			min_boundary_edge_length = std::min(min_boundary_edge_length, (v1 - v0).norm());
		}

		double dhat = Units::convert(args["contact"]["dhat"], units.length());
		args["contact"]["epsv"] = Units::convert(args["contact"]["epsv"], units.velocity());

		if (!contact_dhat_was_explicit_
			&& std::isfinite(min_boundary_edge_length)
			&& dhat > min_boundary_edge_length)
		{
			dhat = args["contact"]["dhat_percentage"].get<double>() * min_boundary_edge_length;
			logger().info("dhat set to {}", dhat);
		}
		else if (std::isfinite(min_boundary_edge_length) && dhat > min_boundary_edge_length)
		{
			logger().warn("dhat larger than min boundary edge, {} > {}", dhat, min_boundary_edge_length);
		}

		args["contact"]["dhat"] = dhat;
	}

	void NonlinearElasticVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		const int size = problem->is_scalar() ? 1 : mesh_->dimension();

		solve_data_.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, &obstacle,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, size, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, *problem, resources_,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			/*fe_space_id=*/-1);
		rhs_assembler_ = solve_data_.rhs_assembler;
	}

	void NonlinearElasticVarForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const json &args)
	{
		build_collision_mesh(
			mesh, space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), boundary_.total_local_boundary, obstacle,
			args, resources_,
			space_.space_in_node_to_node, collision_mesh_);
	}

	void NonlinearElasticVarForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const mesh::Obstacle &obstacle,
		const json &args,
		const io::ResourceIO &resources,
		const Eigen::VectorXi &in_node_to_node,
		ipc::CollisionMesh &collision_mesh_)
	{
		Eigen::MatrixXd collision_vertices;
		Eigen::VectorXi collision_codim_vids;
		Eigen::MatrixXi collision_edges, collision_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;

		const auto extract_default_collision_mesh = [&]() {
			if (args.at("/space/basis_type"_json_pointer) == "Spline")
			{
				io::OutGeometryData::extract_boundary_mesh_sampled(
					mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
					collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
			}
			else
			{
				io::OutGeometryData::extract_boundary_mesh(
					mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
					collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
			}
		};

		if (args.contains("/contact/collision_mesh"_json_pointer)
			&& args.at("/contact/collision_mesh/enabled"_json_pointer).get<bool>())
		{
			const json collision_mesh_args = args.at("/contact/collision_mesh"_json_pointer);
			if (collision_mesh_args.contains("linear_map"))
			{
				assert(displacement_map_entries.empty());
				assert(collision_mesh_args.contains("mesh"));
				// TODO: handle transformation per geometry
				const json transformation = utils::json_as_array(args["geometry"])[0]["transformation"];
				mesh::load_collision_proxy(
					resources,
					collision_mesh_args["mesh"].get<std::string>(),
					collision_mesh_args["linear_map"].get<std::string>(),
					in_node_to_node, transformation, collision_vertices, collision_codim_vids,
					collision_edges, collision_triangles, displacement_map_entries);
			}
			else if (collision_mesh_args.contains("tessellation_type")
					 && collision_mesh_args["tessellation_type"] == "max_order")
			{
				io::OutGeometryData::extract_boundary_mesh_sampled(
					mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
					collision_vertices, collision_edges, collision_triangles, displacement_map_entries,
					utils::json_value<int>(collision_mesh_args, "sampling_order", 0));
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
				extract_default_collision_mesh();
			}
		}
		else
		{
			extract_default_collision_mesh();
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
			utils::append_rows(collision_vertices, obstacle.v());
			utils::append_rows(collision_codim_vids, obstacle.codim_v().array() + num_fe_collision_vertices);
			utils::append_rows(collision_edges, obstacle.e().array() + num_fe_collision_vertices);
			utils::append_rows(collision_triangles, obstacle.f().array() + num_fe_collision_vertices);

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

		collision_mesh_ = ipc::CollisionMesh(
			is_on_surface, is_orientable_vertex, collision_vertices, collision_edges, collision_triangles,
			displacement_map);

		collision_mesh_.can_collide = [&collision_mesh_, num_fe_collision_vertices](size_t vi, size_t vj) {
			// obstacles do not collide with other obstacles
			return collision_mesh_.to_full_vertex_id(vi) < num_fe_collision_vertices
				   || collision_mesh_.to_full_vertex_id(vj) < num_fe_collision_vertices;
		};

		collision_mesh_.init_area_jacobians();
	}

	void NonlinearElasticVarForm::build_periodic_collision_mesh()
	{
		assert(!mesh_->is_volume());
		const int dim = mesh_->dimension();
		const int n_tiles = 2;

		if (mesh_->dimension() != 2)
			log_and_throw_error("Periodic collision mesh is only implemented in 2D!");
		if (obstacle.n_vertices() != 0)
			log_and_throw_error("Periodic contact does not support obstacles.");

		const int n_bases = space_.n_bases;
		const json &conditions = args["boundary_conditions"]["periodic"];
		Eigen::VectorXi periodic_dof_mask = Eigen::VectorXi::Zero(n_bases);
		Eigen::MatrixXd periodic_tile_offsets(dim, conditions.size());
		for (int i = 0; i < int(conditions.size()); ++i)
		{
			const json &condition = conditions[i];
			const std::array<int, 2> boundary_ids = {{condition["boundary_ids"][0].get<int>(),
													  condition["boundary_ids"][1].get<int>()}};
			const auto mapping = solver::PeriodicBoundaryLagrangianForm::build_mapping(
				n_bases * dim, dim, *mesh_, space_.basis_list(), boundary_.total_local_boundary,
				boundary_ids, condition.value("tolerance", 1e-5));
			periodic_tile_offsets.col(i) = mapping.translation.transpose();
			for (const int dof : mapping.boundary_dofs)
				periodic_dof_mask(dof) = 1;
		}

		Eigen::MatrixXd V(n_bases, dim);
		for (const auto &bs : space_.basis_list())
			for (const auto &b : bs.bases)
				for (const auto &g : b.global())
					V.row(g.index) = g.node;

		Eigen::MatrixXi E = collision_mesh_.edges();
		for (int i = 0; i < E.size(); i++)
		{
			E(i) = collision_mesh_.to_full_vertex_id(E(i));
			if (E(i) < 0 || E(i) >= n_bases)
				log_and_throw_error("Periodic contact requires collision vertices to map to FE basis nodes.");
		}

		Eigen::MatrixXd bbox(V.cols(), 2);
		bbox.col(0) = V.colwise().minCoeff();
		bbox.col(1) = V.colwise().maxCoeff();

		// remove boundary edges on periodic BC, buggy
		{
			std::vector<int> ind;
			for (int i = 0; i < E.rows(); i++)
			{
				if (!periodic_dof_mask(E(i, 0)) || !periodic_dof_mask(E(i, 1)))
					ind.push_back(i);
			}

			E = E(ind, Eigen::all).eval();
		}

		Eigen::MatrixXd Vtmp, Vnew;
		Eigen::MatrixXi Etmp, Enew;
		Vtmp.setZero(V.rows() * n_tiles * n_tiles, V.cols());
		Etmp.setZero(E.rows() * n_tiles * n_tiles, E.cols());

		if (periodic_tile_offsets.rows() != dim || periodic_tile_offsets.cols() != dim
			|| Eigen::FullPivLU<Eigen::MatrixXd>(periodic_tile_offsets).rank() != dim)
			log_and_throw_error("Periodic contact requires {} linearly independent periodic boundary pairs", dim);
		const Eigen::MatrixXd &tile_offset = periodic_tile_offsets;

		for (int i = 0, idx = 0; i < n_tiles; i++)
		{
			for (int j = 0; j < n_tiles; j++)
			{
				Eigen::Vector2d block_id;
				block_id << i, j;

				Vtmp.middleRows(idx * V.rows(), V.rows()) = V;
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
				if (periodic_dof_mask(i))
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
		double relative_tolerance = 1e-5;
		for (const json &condition : conditions)
			relative_tolerance = std::max(relative_tolerance, condition.value("tolerance", 1e-5));
		const double eps = (bbox.col(1) - bbox.col(0)).maxCoeff() * relative_tolerance;
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
		periodic_collision_mesh_ = ipc::CollisionMesh(is_on_surface,
													  std::vector<bool>(Vnew.rows(), false),
													  Vnew,
													  Enew,
													  boundary_triangles,
													  displacement_map);

		periodic_collision_mesh_.init_area_jacobians();

		periodic_collision_mesh_to_basis_.setConstant(Vnew.rows(), -1);
		for (int i = 0; i < V.rows(); i++)
			for (int j = 0; j < n_tiles * n_tiles; j++)
				periodic_collision_mesh_to_basis_(SVI[j * V.rows() + i]) = i;

		if (periodic_collision_mesh_to_basis_.maxCoeff() + 1 != V.rows())
			log_and_throw_error("Failed to tile mesh!");
	}

	std::shared_ptr<assembler::PressureAssembler> NonlinearElasticVarForm::build_pressure_assembler() const
	{
		const int size = problem->is_scalar() ? 1 : mesh_->dimension();

		return std::make_shared<assembler::PressureAssembler>(
			*primary_assembler_, *mesh_, obstacle,
			boundary_.local_pressure_boundary,
			boundary_.local_pressure_cavity,
			boundary_.boundary_nodes,
			elastic_primitive_to_node(), elastic_node_to_primitive(),
			space_.n_bases, size, space_.basis_list(), space_.geometry_basis_list(), *problem);
	}

	void NonlinearElasticStaticVarForm::solve_problem(
		Eigen::MatrixXd &sol,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		assert((!initial_condition_override || (initial_condition_override->velocity.size() == 0 && initial_condition_override->acceleration.size() == 0))
			   && "Static elasticity does not accept initial velocity or acceleration overrides");

		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			// FIXME
			//  read_initial_x_from_file(
			//  resolve_input_path(args["input"]["data"]["state"]), "u",
			//  args["input"]["data"]["reorder"], in_node_to_node,
			//  mesh->dimension(), solution);

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				initial_solution(sol, initial_condition_override);
			else if (sol.size() <= 0)
				initial_solution(sol, initial_condition_override);

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				assert(sol.cols() == 1 && "Static initial solution override must have exactly one column");
			else if (sol.cols() != 1)
				log_and_throw_error("Static elasticity requires exactly one initial solution column.");
		}
		init_solve(sol, 1.0, initial_condition_override);

		solve_tensor_nonlinear(0, sol, true);
		if (post_step)
			post_step(0, sol);

		const std::string state_path = resolve_output_path(args["output"]["data"]["state"]);
		if (!state_path.empty())
			io::write_matrix(state_path, "u", sol);

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void NonlinearElasticTransientVarForm::solve_problem(
		Eigen::MatrixXd &sol,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		const bool save_stats = args["output"]["stats"];
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			// FIXME
			//  read_initial_x_from_file(
			//  resolve_input_path(args["input"]["data"]["state"]), "u",
			//  args["input"]["data"]["reorder"], in_node_to_node,
			//  mesh->dimension(), solution);

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				initial_solution(sol, initial_condition_override);
			else if (sol.size() <= 0)
				initial_solution(sol, initial_condition_override);

			if (sol.cols() > 1) // ignore previous solutions
				sol.conservativeResize(Eigen::NoChange, 1);
		}
		init_solve(sol, t0 + dt, initial_condition_override);
		if (post_step)
			post_step(0, sol);

		// Write the total energy to a CSV file
		int save_i = 0;

		std::unique_ptr<io::EnergyCSVWriter> energy_csv = nullptr;
		std::unique_ptr<io::RuntimeStatsCSVWriter> stats_csv = nullptr;

		if (save_stats)
		{
			logger().debug("Saving nl stats to {} and {}", resolve_output_path("energy.csv"), resolve_output_path("stats.csv"));
			energy_csv = std::make_unique<io::EnergyCSVWriter>(resolve_output_path("energy.csv"), solve_data_);
			const io::OutputSpace space = output_space();
			stats_csv = std::make_unique<io::RuntimeStatsCSVWriter>(
				resolve_output_path("stats.csv"),
				space_.n_bases,
				space.mesh ? space.mesh->n_elements() : 0,
				t0, dt);
		}

		// Save the initial solution
		if (energy_csv)
			energy_csv->write(save_i, sol);
		save_timestep(t0, 0, t0, dt, sol);

		save_i++;

		for (int t = 1; t <= time_steps; ++t)
		{
			double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

			{
				POLYFEM_SCOPED_TIMER(forward_solve_time);
				solve_tensor_nonlinear(t, sol, true);
			}
			if (post_step)
				post_step(t, sol);

			// Always save the solution for consistency
			if (energy_csv)
				energy_csv->write(save_i, sol);
			save_timestep(t0 + dt * t, t, t0, dt, sol);
			save_i++;

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				if (solve_data_.time_integrator)
					solve_data_.time_integrator->update_quantities(sol);

				solve_data_.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data_.update_dt();
				solve_data_.update_barrier_stiffness(sol);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
			notify_time_step(t, time_steps, t0, dt);

			save_elastic_step_state(t0, dt, t, sol, solve_data_.time_integrator.get());
			if (stats_csv)
				stats_csv->write(t, forward_solve_time, remeshing_time, global_relaxation_time);
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void NonlinearElasticVarForm::init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t)
	{
		damping_assembler_ = std::make_shared<assembler::ViscousDamping>();
		set_materials(*damping_assembler_, mesh_->dimension());

		elasticity_pressure_assembler = build_pressure_assembler();

		// for backward solve
		damping_prev_assembler_ = std::make_shared<assembler::ViscousDampingPrev>();
		set_materials(*damping_prev_assembler_, mesh_->dimension());

		const ElementInversionCheck check_inversion = args["solver"]["advanced"]["check_inversion"];

		// NOTE: some stuff are legacy and hardcoded to be off
		forms = solve_data_.init_forms(
			// General
			units,
			resources_,
			dim, t, space_.space_in_node_to_node,
			// Elastic form
			space_.n_bases, *space_.bases, space_.geometry_basis_list(), *primary_assembler_, ass_vals_cache_, mass_ass_vals_cache_, args["solver"]["advanced"]["jacobian_threshold"], check_inversion,
			args["solver"]["advanced"]["conservative_max_iter"],
			// Body form
			0, boundary_.boundary_nodes, boundary_.local_boundary,
			boundary_.local_neumann_boundary,
			elastic_boundary_samples(), rhs_, sol, mass_assembler_->density(),
			// Pressure form
			boundary_.local_pressure_boundary, boundary_.local_pressure_cavity, elasticity_pressure_assembler,
			// Inertia form
			args.value("/time/quasistatic"_json_pointer, true), mass_,
			damping_assembler_->is_valid() ? damping_assembler_ : nullptr,
			// Lagged regularization form
			args["solver"]["advanced"]["lagged_regularization_weight"],
			args["solver"]["advanced"]["lagged_regularization_iterations"],
			// Augmented lagrangian form
			obstacle.ndof(), args["constraints"]["hard"], args["constraints"]["soft"], args["constraints"]["zero_mean"],
			// Contact form
			args["contact"]["enabled"], collision_mesh_, args["contact"]["dhat"],
			avg_mass_, args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_area_weighting"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_improved_max_operator"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_physical_barrier"]) : false,
			args["solver"]["contact"]["barrier_stiffness"],
			args["solver"]["contact"]["initial_barrier_stiffness"],
			args["solver"]["contact"]["CCD"]["broad_phase"],
			args["solver"]["contact"]["CCD"]["tolerance"],
			args["solver"]["contact"]["CCD"]["max_iterations"],
			false,
			// Smooth Contact Form
			args["contact"]["use_gcp_formulation"],
			args["contact"]["alpha_t"],
			args["contact"]["alpha_n"],
			args["contact"]["use_adaptive_dhat"],
			args["contact"]["min_distance_ratio"],
			// Normal Adhesion Form
			args["contact"]["adhesion"]["adhesion_enabled"],
			args["contact"]["adhesion"]["dhat_p"],
			args["contact"]["adhesion"]["dhat_a"],
			args["contact"]["adhesion"]["adhesion_strength"],
			// Tangential Adhesion Form
			args["contact"]["adhesion"]["tangential_adhesion_coefficient"],
			args["contact"]["adhesion"]["epsa"],
			args["solver"]["contact"]["tangential_adhesion_iterations"],
			// Homogenization
			assembler::MacroStrainValue(),
			// Periodic contact
			false, Eigen::VectorXi(),
			// Friction form
			args["contact"]["friction_coefficient"],
			args["contact"]["epsv"],
			args["solver"]["contact"]["friction_iterations"],
			// Rayleigh damping form
			args["solver"]["rayleigh_damping"],

			// BC AL lumping
			args["solver"]["augmented_lagrangian"]["lumping"],

			// Boundary-ID periodic constraints
			mesh_.get(), &boundary_.total_local_boundary,
			args["boundary_conditions"]["periodic"], /*fe_space_id=*/-1);

		for (const auto &form : forms)
			form->set_output_dir(output_path);

		if (solve_data_.contact_form != nullptr)
			solve_data_.contact_form->save_ccd_debug_meshes = args["output"]["advanced"]["save_ccd_debug_meshes"];
	}

	void NonlinearElasticVarForm::init_solve(
		Eigen::MatrixXd &sol,
		const double t,
		const InitialConditionOverride *initial_condition_override)
	{
		init_solve_data(sol, t, "", initial_condition_override);

		double characteristic_length = 0;
		if (args["solver"]["advanced"]["characteristic_length"] > 0)
		{
			characteristic_length = args["solver"]["advanced"]["characteristic_length"];
		}
		else
		{
			RowVectorNd min, max;
			mesh_->bounding_box(min, max);
			characteristic_length = (max - min).norm();
		}

		double characteristic_force_density = 0;
		if (args["solver"]["advanced"]["characteristic_force_density"] <= 0)
		{
			logger().warn("No user-specified force density was provided, defaulting to 10000.");
			characteristic_force_density = 10000;
		}
		else
		{
			characteristic_force_density = args["solver"]["advanced"]["characteristic_force_density"];
		}

		const int ndof = space_.n_bases * mesh_->dimension();
		solve_data_.nl_problem = std::make_shared<solver::NLProblem>(
			ndof, t, forms, solve_data_.al_form,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			characteristic_length, characteristic_force_density, pure_mass_, mesh_->dimension());
		solve_data_.nl_problem->init(sol);
		solve_data_.nl_problem->update_quantities(t, sol);

		stats.solver_info = json::array();
	}

	void NonlinearElasticVarForm::init_solve_data(
		Eigen::MatrixXd &sol,
		const double t,
		const std::string &state_prefix,
		const InitialConditionOverride *initial_condition_override)
	{
		assert(sol.cols() == 1);
		assert(!problem->is_scalar()); // tensor

		// FIXME
		//  if (optimization_enabled != solver::CacheLevel::None)
		//  {
		//  	if (initial_sol_update.size() == ndof())
		//  		sol = initial_sol_update;
		//  	else
		//  		initial_sol_update = sol;
		//  }

		// --------------------------------------------------------------------
		// Check for initial intersections
		if (args["contact"]["enabled"])
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			const Eigen::MatrixXd displaced = collision_mesh_.displace_vertices(
				utils::unflatten(sol, mesh_->dimension()));

			if (ipc::has_intersections(collision_mesh_, displaced, ipc::create_broad_phase(args["solver"]["contact"]["CCD"]["broad_phase"]).get()))
			{
				io::OBJWriter::write(
					resolve_output_path("intersection.obj"), displaced,
					collision_mesh_.edges(), collision_mesh_.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		// --------------------------------------------------------------------

		if (problem->is_time_dependent())
		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");
			solve_data_.time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);

			Eigen::MatrixXd solution, velocity, acceleration;
			initial_solution(solution, initial_condition_override, state_prefix); // Reload this because we need all previous solutions
			solution.col(0) = sol;                                                // Make sure the current solution is the same as `sol`
			assert(solution.rows() == sol.size());
			initial_velocity(velocity, initial_condition_override, state_prefix);
			assert(velocity.rows() == sol.size());
			initial_acceleration(acceleration, initial_condition_override, state_prefix);
			assert(acceleration.rows() == sol.size());
			if (solution.cols() != velocity.cols() || solution.cols() != acceleration.cols())
			{
				log_and_throw_error(
					"Incompatible initial-condition history for transient solve: "
					"solution has {} columns, velocity has {}, acceleration has {}.",
					solution.cols(), velocity.cols(), acceleration.cols());
			}

			solve_data_.time_integrator->init(solution, velocity, acceleration, dt);
			restore_checkpoint_integrator(solve_data_.time_integrator, "/checkpoint/state/primary_integrator", dt);
			assert(solve_data_.time_integrator != nullptr && "Transient nonlinear elasticity requires an initialized time integrator");
		}
		else
		{
			solve_data_.time_integrator = nullptr;
		}

		// --------------------------------------------------------------------
		// Initialize forms

		// --------------------------------------------------------------------
		// Initialize nonlinear problems

		init_forms(args, mesh_->dimension(), sol, t);

		if (pure_mass_.size() == 0)
			pure_mass_assembler_->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), pure_mass_ass_vals_cache_, 0, pure_mass_, true);
	}

	void NonlinearElasticVarForm::prepare_for_embedding()
	{
		prepare();
	}

	void NonlinearElasticVarForm::initial_solution_for_embedding(
		Eigen::MatrixXd &solution, const std::string &state_prefix) const
	{
		initial_solution(solution, nullptr, state_prefix);
		if (solution.cols() > 1)
			solution.conservativeResize(Eigen::NoChange, 1);
	}

	void NonlinearElasticVarForm::init_forms_for_embedding(
		Eigen::MatrixXd &solution, const double t, const std::string &state_prefix)
	{
		prepare();
		init_solve_data(solution, t, state_prefix);
	}

	void NonlinearElasticVarForm::advance_for_embedding(const Eigen::VectorXd &solution)
	{
		assert(solve_data_.time_integrator);
		solve_data_.time_integrator->update_quantities(solution);
		solve_data_.update_dt();
	}

	void NonlinearElasticVarForm::update_barrier_stiffness_for_embedding(
		const Eigen::VectorXd &solution)
	{
		solve_data_.update_barrier_stiffness(solution);
	}

	bool NonlinearElasticVarForm::save_timestep_for_embedding(
		const double time, const int step, const double dt,
		const Eigen::MatrixXd &solution, paraviewo::VTMWriter &vtm,
		const std::string &block_prefix) const
	{
		return save_timestep_to_vtm(time, step, dt, solution, vtm, block_prefix);
	}

	int NonlinearElasticVarForm::embedding_ndof() const
	{
		return mesh_ ? space_.n_bases * mesh_->dimension() : 0;
	}

	void NonlinearElasticVarForm::solve_tensor_nonlinear(
		const int step,
		Eigen::MatrixXd &sol,
		const bool init_lagging)
	{
		assert(solve_data_.nl_problem != nullptr && "Nonlinear forms must initialize the nonlinear problem before solving");
		solver::NLProblem &nl_problem = *(solve_data_.nl_problem);

		assert(sol.size() == rhs_.size());

		if (nl_problem.uses_lagging())
		{
			if (init_lagging)
			{
				POLYFEM_SCOPED_TIMER("Initializing lagging");
				nl_problem.init_lagging(sol);
			}
			logger().info("Lagging iteration 1:");
		}

		save_subsolve(0, step, sol);

		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver =
			polysolve::nonlinear::Solver::create(args["solver"]["augmented_lagrangian"]["nonlinear"], args["solver"]["linear"], units.characteristic_length(), logger());

		ALSolver al_solver(
			solve_data_.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			args["solver"]["augmented_lagrangian"]["eta"],
			[&](const Eigen::VectorXd &x) {
				this->solve_data_.update_barrier_stiffness(sol);
			});

		al_solver.post_subsolve = [&](const double al_weight) {
			stats.solver_info.push_back(
				{{"type", al_weight > 0 ? "al" : "rc"},
				 {"t", step},
				 {"info", nl_solver->info()}});
			if (al_weight > 0)
				stats.solver_info.back()["weight"] = al_weight;
			save_subsolve(stats.solver_info.size(), step, sol);
		};

		Eigen::MatrixXd prev_sol = sol;
		al_solver.solve_al(nl_problem, sol,
						   args["solver"]["augmented_lagrangian"]["nonlinear"], args["solver"]["linear"], units.characteristic_length());

		al_solver.solve_reduced(nl_problem, sol,
								args["solver"]["nonlinear"], args["solver"]["linear"], units.characteristic_length());

		if (args["space"]["advanced"]["count_flipped_els_continuous"])
		{
			const auto invalidList = utils::count_invalid(mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), sol);
			logger().debug("Flipped elements (cnt {}) : {}", invalidList.size(), invalidList);
		}

		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2) * units.characteristic_length();

		bool lagging_converged = !nl_problem.uses_lagging();
		for (int lag_i = 1; !lagging_converged; lag_i++)
		{
			Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);

			nl_problem.update_lagging(tmp_sol, lag_i);

			Eigen::VectorXd grad;
			nl_problem.gradient(tmp_sol, grad);
			const double delta_x_norm = (prev_sol - sol).lpNorm<Eigen::Infinity>();
			logger().debug("Lagging convergence grad_norm={:g} tol={:g} (||Δx||={:g})", grad.norm(), lagging_tol, delta_x_norm);
			if (grad.norm() <= lagging_tol)
			{
				logger().info(
					"Lagging converged in {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = true;
				break;
			}

			if (delta_x_norm <= 1e-12)
			{
				logger().warn(
					"Lagging produced tiny update between iterations {:d} and {:d} (grad_norm={:g} grad_tol={:g} ||Δx||={:g} Δx_tol={:g}); stopping early",
					lag_i - 1, lag_i, grad.norm(), lagging_tol, delta_x_norm, 1e-6);
				lagging_converged = false;
				break;
			}

			if (lag_i >= nl_problem.max_lagging_iterations())
			{
				logger().warn(
					"Lagging failed to converge with {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = false;
				break;
			}

			logger().info("Lagging iteration {:d}:", lag_i + 1);
			nl_problem.init(sol);
			solve_data_.update_barrier_stiffness(sol);
			nl_problem.normalize_forms();
			nl_solver->minimize(nl_problem, tmp_sol);
			nl_problem.finish();
			prev_sol = sol;
			sol = nl_problem.reduced_to_full(tmp_sol);

			stats.solver_info.push_back(
				{{"type", "rc"},
				 {"t", step},
				 {"lag_i", lag_i},
				 {"info", nl_solver->info()}});
			save_subsolve(stats.solver_info.size(), step, sol);
		}
	}

} // namespace polyfem::varform
