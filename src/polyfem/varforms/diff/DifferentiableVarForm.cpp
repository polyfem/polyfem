#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/solver/forms/lagrangian/PeriodicBoundaryLagrangianForm.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::varform
{
	const ipc::CollisionMesh &DifferentiableVarForm::collision_mesh() const
	{
		log_and_throw_error("Variational formulation {} does not expose a collision mesh.", name());
	}

	const mesh::Obstacle &DifferentiableVarForm::get_obstacle() const
	{
		log_and_throw_error("Variational formulation {} does not expose an obstacle.", name());
	}

	void DifferentiableVarForm::initial_solution(Eigen::MatrixXd &, const InitialConditionOverride *) const
	{
		log_and_throw_error("Variational formulation {} does not expose an initial solution.", name());
	}

	void DifferentiableVarForm::initial_velocity(Eigen::MatrixXd &, const InitialConditionOverride *) const
	{
		log_and_throw_error("Variational formulation {} does not expose an initial velocity.", name());
	}

	void DifferentiableVarForm::initial_acceleration(Eigen::MatrixXd &, const InitialConditionOverride *) const
	{
		log_and_throw_error("Variational formulation {} does not expose an initial acceleration.", name());
	}

	Eigen::MatrixXd DifferentiableVarForm::displacement_gradient() const
	{
		return Eigen::MatrixXd::Zero(get_mesh().dimension(), get_mesh().dimension());
	}

	void DifferentiableVarForm::get_vertices(Eigen::MatrixXd &vertices) const
	{
		vertices.resize(get_mesh().n_vertices(), get_mesh().dimension());
		for (int v = 0; v < get_mesh().n_vertices(); ++v)
			vertices.row(v) = get_mesh().point(v);
	}

	std::unordered_map<int, std::array<bool, 3>> DifferentiableVarForm::boundary_conditions_ids(const std::string &bc_type) const
	{
		assert(get_args()["boundary_conditions"].contains(bc_type) && "Requested boundary-condition type must exist");
		const std::vector<json> json_bcs = utils::json_as_array(get_args()["boundary_conditions"][bc_type]);
		std::unordered_map<int, std::array<bool, 3>> bcs;
		for (const json &bc : json_bcs)
		{
			assert(bc["dimension"].size() >= get_mesh().dimension() && "Boundary-condition dimensions must cover the mesh dimension");
			std::array<bool, 3> dimension{{true, true, true}};
			for (int d = 0; d < bc["dimension"].size(); ++d)
				dimension[d] = bc["dimension"][d];
			assert(bc.contains("id") && bc["id"].is_number_integer() && "Boundary conditions must have an integer id");
			bcs[bc["id"].get<int>()] = dimension;
		}
		return bcs;
	}

	bool DifferentiableVarForm::is_homogenization() const
	{
		return get_args().contains("/constraints/macro_displacement_gradient"_json_pointer);
	}

	bool DifferentiableVarForm::has_periodic_boundary() const
	{
		return get_args().contains("/boundary_conditions/periodic"_json_pointer)
			   && !get_args().at("/boundary_conditions/periodic"_json_pointer).empty();
	}

	Eigen::MatrixXd DifferentiableVarForm::periodic_tile_offsets() const
	{
		const int dim = get_mesh().dimension();
		const json &conditions = get_args().at("/boundary_conditions/periodic"_json_pointer);
		Eigen::MatrixXd offsets(dim, conditions.size());
		for (int i = 0; i < int(conditions.size()); ++i)
		{
			const json &condition = conditions[i];
			if (!condition.contains("boundary_ids")
				|| !condition["boundary_ids"].is_array()
				|| condition["boundary_ids"].size() != 2)
			{
				log_and_throw_adjoint_error(
					"Periodic boundary condition {} must contain exactly two boundary_ids.", i);
			}
			const std::array<int, 2> boundary_ids = {{condition["boundary_ids"][0].get<int>(),
													  condition["boundary_ids"][1].get<int>()}};
			const auto mapping = solver::PeriodicBoundaryLagrangianForm::build_mapping(
				primary_space().ndof(), primary_space().value_dim, get_mesh(),
				primary_space().basis_list(), boundary_state().total_local_boundary,
				boundary_ids, condition.value("tolerance", 1e-5));
			offsets.col(i) = mapping.translation.transpose();
		}
		return offsets;
	}

	bool DifferentiableVarForm::is_adhesion_enabled() const
	{
		return get_args()["contact"]["adhesion"]["adhesion_enabled"];
	}

	bool DifferentiableVarForm::is_pressure_enabled() const
	{
		return !get_args()["boundary_conditions"]["pressure_boundary"].empty()
			   || !get_args()["boundary_conditions"]["pressure_cavity"].empty();
	}

	bool DifferentiableVarForm::has_constraints() const
	{
		return !get_args()["constraints"]["hard"].empty()
			   || !get_args()["constraints"]["soft"].empty();
	}

	bool DifferentiableVarForm::is_problem_linear() const
	{
		return primary_assembler().is_linear() && !is_contact_enabled() && !is_pressure_enabled() && !has_constraints();
	}

	void DifferentiableVarForm::build_stiffness_matrix(StiffnessMatrix &stiffness) const
	{
		const FESpace &space = primary_space();
		primary_assembler().assemble(
			get_mesh().is_volume(), space.n_bases, space.basis_list(), space.geometry_basis_list(),
			assembly_cache(), 0, stiffness);
	}

	std::vector<int> DifferentiableVarForm::primitive_to_node() const
	{
		const FESpace &space = primary_space();
		assert(space.geometry && "Node mapping requires an initialized geometry mapping");
		const auto &mesh_nodes = space.geometry->mesh_nodes;
		if (!mesh_nodes)
			log_and_throw_error("Variational formulation {} does not expose a primitive-to-node mapping.", name());
		std::vector<int> indices = mesh_nodes->primitive_to_node();
		assert(indices.size() >= get_mesh().n_vertices() && "Primitive-to-node mapping must contain every mesh vertex");
		indices.resize(get_mesh().n_vertices());
		return indices;
	}

	std::vector<int> DifferentiableVarForm::node_to_primitive() const
	{
		const std::vector<int> p2n = primitive_to_node();
		assert(primary_space().geometry && "Node mapping requires an initialized geometry mapping");
		assert(primary_space().geometry->n_bases == p2n.size() && "Optimization requires first-order geometry bases");
		std::vector<int> indices(p2n.size());
		for (int i = 0; i < p2n.size(); ++i)
		{
			assert(p2n[i] >= 0 && p2n[i] < indices.size() && "Primitive-to-node entries must be valid node indices");
			indices[p2n[i]] = i;
		}
		return indices;
	}

	void DifferentiableVarForm::get_elements(Eigen::MatrixXi &elements) const
	{
		if (!get_mesh().is_simplicial())
			log_and_throw_error("Element extraction requires a simplicial mesh.");
		const std::vector<int> n2p = node_to_primitive();
		const auto &geometry_bases = primary_space().geometry_basis_list();
		elements.resize(geometry_bases.size(), get_mesh().dimension() + 1);
		for (int e = 0; e < geometry_bases.size(); ++e)
		{
			int i = 0;
			for (const auto &basis : geometry_bases[e].bases)
				elements(e, i++) = n2p[basis.global()[0].index];
		}
	}

	QuadratureOrders DifferentiableVarForm::n_boundary_samples() const
	{
		const FESpace &space = primary_space();
		assert(space.disc_orders.size() > 0 && "Boundary quadrature requires initialized FE orders");
		assert(space.geometry && "Boundary quadrature requires an initialized geometry mapping");
		assert(space.geometry->disc_orders.size() > 0 && "Boundary quadrature requires initialized geometry orders");
		const int geometry_discr_order = get_mesh().orders().size() <= 0 ? 1 : get_mesh().orders().maxCoeff();
		return boundary_samples(space.disc_orders.maxCoeff(), space.disc_ordersq.maxCoeff(), geometry_discr_order);
	}

	void DifferentiableVarForm::set_vertex_positions(const Eigen::MatrixXd &vertices)
	{
		mesh::Mesh &mesh = mutable_mesh();
		if (vertices.rows() != mesh.n_vertices() || vertices.cols() != mesh.dimension())
			log_and_throw_error(
				"Invalid vertex matrix shape ({}, {}), expected ({}, {}).",
				vertices.rows(), vertices.cols(), mesh.n_vertices(), mesh.dimension());
		for (int i = 0; i < vertices.rows(); ++i)
			mesh.set_point(i, vertices.row(i));
		invalidate_after_geometry_update();
	}

	void DifferentiableVarForm::set_lame_parameters(const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu)
	{
		if (lambda.size() != get_mesh().n_elements() || mu.size() != get_mesh().n_elements())
			log_and_throw_error("Lamé parameter vectors must contain one value per element.");
		const_cast<assembler::Assembler &>(primary_assembler()).update_lame_params(lambda, mu);
		invalidate_after_parameter_update();
	}

	void DifferentiableVarForm::set_friction_coefficient(const double coefficient)
	{
		get_args()["contact"]["friction_coefficient"] = coefficient;
		invalidate_after_parameter_update();
	}

	void DifferentiableVarForm::set_damping_coefficients(const double psi, const double phi)
	{
		auto update_material = [psi, phi](json &material) {
			material["psi"] = psi;
			material["phi"] = phi;
		};
		if (get_args()["materials"].is_array())
		{
			for (auto &material : get_args()["materials"])
				update_material(material);
		}
		else
			update_material(get_args()["materials"]);
		invalidate_after_parameter_update();
	}

	void DifferentiableVarForm::set_dirichlet_boundary(const int boundary_id, const int time_step, const Eigen::VectorXd &value)
	{
		auto *tensor_problem = dynamic_cast<assembler::GenericTensorProblem *>(&get_problem());
		if (!tensor_problem)
			log_and_throw_error("Dirichlet boundary updates require a generic tensor problem.");
		tensor_problem->update_dirichlet_boundary(boundary_id, time_step, value);
		invalidate_after_parameter_update();
	}

	void DifferentiableVarForm::set_dirichlet_nodes(const Eigen::VectorXi &input_nodes, const Eigen::MatrixXd &values)
	{
		auto *tensor_problem = dynamic_cast<assembler::GenericTensorProblem *>(&get_problem());
		if (!tensor_problem)
			log_and_throw_error("Nodal Dirichlet updates require a generic tensor problem.");
		tensor_problem->update_dirichlet_nodes(primary_space().space_in_node_to_node, input_nodes, values);
		invalidate_after_parameter_update();
	}

	void DifferentiableVarForm::set_pressure_boundary(const int boundary_id, const int time_step, const double value)
	{
		auto *tensor_problem = dynamic_cast<assembler::GenericTensorProblem *>(&get_problem());
		if (!tensor_problem)
			log_and_throw_error("Pressure updates require a generic tensor problem.");
		tensor_problem->update_pressure_boundary(boundary_id, time_step, value);
		invalidate_after_parameter_update();
	}
} // namespace polyfem::varform
