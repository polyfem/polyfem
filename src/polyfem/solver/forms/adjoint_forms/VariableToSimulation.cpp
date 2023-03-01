#include "VariableToSimulation.hpp"
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/mesh/GeometryReader.hpp>

namespace polyfem::solver
{
	namespace
	{
		template <typename T>
		std::string to_string_with_precision(const T a_value, const int n = 6)
		{
			std::ostringstream out;
			out.precision(n);
			out << std::fixed << a_value;
			return out.str();
		}

		RowVectorNd get_barycenter(const mesh::Mesh &mesh, int e)
		{
			RowVectorNd barycenter;
			if (!mesh.is_volume())
			{
				const auto &mesh2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
				barycenter = mesh2d.face_barycenter(e);
			}
			else
			{
				const auto &mesh3d = dynamic_cast<const mesh::Mesh3D &>(mesh);
				barycenter = mesh3d.cell_barycenter(e);
			}
			return barycenter;
		}
	} // namespace

	void ShapeVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		const int dim = state_ptr_->mesh->dimension();

		// If indices include one vertex entry, we assume it include all entries of this vertex.
		for (int i = 0; i < indices.size(); i += dim)
			for (int j = 0; j < dim; j++)
				assert(indices(i + j) == indices(i) + j);

		for (int i = 0; i < indices.size(); i += dim)
			state_ptr_->set_mesh_vertex(indices(i) / dim, state_variable(Eigen::seqN(i, dim)));

		// TODO: move this to the end of all variable to simulation
		// state_ptr_->build_basis();
	}
	Eigen::VectorXd ShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_shape_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			AdjointTools::dJ_shape_static_adjoint_term(get_state(), get_state().diff_cached[0].u, get_state().get_adjoint_mat(0), term);
		}
		return parametrization_.apply_jacobian(term, x);
	}

	SDFShapeVariableToSimulation::SDFShapeVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization, const json &args) : ShapeVariableToSimulation(state_ptr, parametrization), mesh_id_(args["mesh_id"]), mesh_path_(args["mesh"])
	{
	}
	void SDFShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		parametrization_.eval(x);

		state_ptr_->in_args["geometry"][mesh_id_]["mesh"] = mesh_path_;
		state_ptr_->in_args["geometry"][mesh_id_].erase("transformation");

		state_ptr_->mesh = nullptr;
		state_ptr_->assembler.update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

		state_ptr_->init(state_ptr_->in_args, false);

		int start = 0, end = 0; // start vertex index of the mesh
		{
			assert(state_ptr_->args["geometry"].is_array());
			auto geometries = state_ptr_->args["geometry"].get<std::vector<json>>();

			int i = 0;
			for (const json &geometry : geometries)
			{
				if (!geometry["enabled"].get<bool>() || geometry["is_obstacle"].get<bool>())
					continue;

				if (geometry["type"] != "mesh")
					log_and_throw_error(
						fmt::format("Invalid geometry type \"{}\" for FEM mesh!", geometry["type"]));

				if (i == mesh_id_)
					start = state_ptr_->mesh ? state_ptr_->mesh->n_vertices() : 0;

				if (state_ptr_->mesh == nullptr)
					state_ptr_->mesh = mesh::read_fem_mesh(geometry, state_ptr_->args["root_path"], false);
				else
					state_ptr_->mesh->append(mesh::read_fem_mesh(geometry, state_ptr_->args["root_path"], false));

				if (i == mesh_id_)
					end = state_ptr_->mesh->n_vertices();

				i++;
			}
		}

		state_ptr_->load_mesh();
		state_ptr_->stats.compute_mesh_stats(*state_ptr_->mesh);
		state_ptr_->build_basis();

		const int dim = state_ptr_->mesh->dimension();
		parametrization_.set_output_indexing(Eigen::VectorXi::LinSpaced((end - start) * dim, start * dim, end * dim - 1));
	}

	void ElasticVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		const int n_elem = state_ptr_->bases.size();
		assert(n_elem * 2 == state_variable.size());
		state_ptr_->assembler.update_lame_params(state_variable.segment(0, n_elem), state_variable.segment(n_elem, n_elem));
	}
	Eigen::VectorXd ElasticVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_material_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			AdjointTools::dJ_material_static_adjoint_term(get_state(), get_state().diff_cached[0].u, get_state().get_adjoint_mat(0), term);
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void FrictionCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 1);
		assert(state_variable(0) >= 0);
		state_ptr_->args["contact"]["friction_coefficient"] = state_variable(0);
	}
	Eigen::VectorXd FrictionCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_friction_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Friction coefficient grad in static simulations not implemented!");
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void DampingCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 2);
		json damping_param = {
			{"psi", state_variable(0)},
			{"phi", state_variable(1)},
		};
		state_ptr_->assembler.add_multimaterial(0, damping_param);
		logger().info("Current damping params: {}, {}", state_variable(0), state_variable(1));
	}
	Eigen::VectorXd DampingCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_damping_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Static damping not supported!");
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void InitialConditionVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == state_ptr_->ndof() * 2);
		state_ptr_->initial_sol_update = state_variable.head(state_ptr_->ndof());
		state_ptr_->initial_vel_update = state_variable.tail(state_ptr_->ndof());
	}
	Eigen::VectorXd InitialConditionVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_initial_condition_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Static initial condition not supported!");
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void DirichletVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		log_and_throw_error("Dirichlet variable to simulation not implemented!");
		// auto &problem = *dynamic_cast<assembler::GenericTensorProblem *>(state_ptr_->problem.get());
		// // This should eventually update dirichlet boundaries per boundary element, using the shape constraint.
		// auto constraint_string = control_constraints_->constraint_to_string(state_variable);
		// for (const auto &kv : boundary_id_to_reduced_param)
		// {
		// 	json dirichlet_bc = constraint_string[kv.first];
		// 	// Need time_steps + 1 entry, though unused.
		// 	for (int k = 0; k < states_ptr_[0]->mesh->dimension(); ++k)
		// 		dirichlet_bc[k].push_back(dirichlet_bc[k][time_steps - 1]);
		// 	logger().trace("Updating boundary id {} to dirichlet bc {}", kv.first, dirichlet_bc);
		// 	problem.update_dirichlet_boundary(kv.first, dirichlet_bc, true, true, true, "");
		// }
	}
	Eigen::VectorXd DirichletVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_dirichlet_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Static dirichlet boundary optimization not supported!");
		}
		return parametrization_.apply_jacobian(term, x);
	}
	std::string DirichletVariableToSimulation::variable_to_string(const Eigen::VectorXd &variable)
	{
		return "";
	}

	void MacroStrainVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == state_ptr_->mesh->dimension() * state_ptr_->mesh->dimension());
		state_ptr_->disp_grad = utils::unflatten(state_variable, state_ptr_->mesh->dimension());
	}
	Eigen::VectorXd MacroStrainVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			log_and_throw_error("Transient macro strain optimization not supported!");
		}
		else
		{
			AdjointTools::dJ_macro_strain_adjoint_term(get_state(), get_state().diff_cached[0].u, get_state().get_adjoint_mat(0), term);
		}
		return parametrization_.apply_jacobian(term, x);
	}
} // namespace polyfem::solver