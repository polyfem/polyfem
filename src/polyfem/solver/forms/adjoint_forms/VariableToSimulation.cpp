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
		for (auto state : states_)
		{
			const int dim = state->mesh->dimension();

			// If indices include one vertex entry, we assume it include all entries of this vertex.
			for (int i = 0; i < indices.size(); i += dim)
				for (int j = 0; j < dim; j++)
					assert(indices(i + j) == indices(i) + j);

			for (int i = 0; i < indices.size(); i += dim)
				state->set_mesh_vertex(indices(i) / dim, state_variable(Eigen::seqN(i, dim)));
		}
	}
	Eigen::VectorXd ShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_shape_transient_adjoint_term(*state, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				if (state->disp_grad_.size() == 0)
					AdjointTools::dJ_shape_static_adjoint_term(*state, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);
				else
					AdjointTools::dJ_shape_homogenization_adjoint_term(*state, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	Eigen::VectorXd ShapeVariableToSimulation::inverse_eval()
	{
		const int dim = states_[0]->mesh->dimension();
		const int npts = states_[0]->mesh->n_vertices();

		Eigen::VectorXd x;
		Eigen::VectorXi indices = parametrization_.get_output_indexing(x);

		if (indices.size() == 0)
			indices.setLinSpaced(npts * dim, 0, npts * dim - 1);

		Eigen::MatrixXd V, V_flat;
		states_[0]->get_vertices(V);
		V_flat = utils::flatten(V);

		x.setZero(indices.size());
		for (int i = 0; i < indices.size(); i++)
			x(i) = V_flat(indices(i));

		return parametrization_.inverse_eval(x);
	}

	SDFShapeVariableToSimulation::SDFShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args) : ShapeVariableToSimulation(states, parametrization), mesh_id_(args["mesh_id"]), mesh_path_(args["mesh"])
	{
	}
	void SDFShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		parametrization_.eval(x);

		int start = 0, end = 0; // start vertex index of the mesh
		for (auto state : states_)
		{
			state->in_args["geometry"][mesh_id_]["mesh"] = mesh_path_;
			state->in_args["geometry"][mesh_id_].erase("transformation");

			state->mesh = nullptr;
			state->assembler.update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

			state->init(state->in_args, false);

			{
				assert(state->args["geometry"].is_array());
				auto geometries = state->args["geometry"].get<std::vector<json>>();

				int i = 0;
				for (const json &geometry : geometries)
				{
					if (!geometry["enabled"].get<bool>() || geometry["is_obstacle"].get<bool>())
						continue;

					if (geometry["type"] != "mesh")
						log_and_throw_error(
							fmt::format("Invalid geometry type \"{}\" for FEM mesh!", geometry["type"]));

					if (i == mesh_id_)
						start = state->mesh ? state->mesh->n_vertices() : 0;

					if (state->mesh == nullptr)
						state->mesh = mesh::read_fem_mesh(geometry, state->args["root_path"], false);
					else
						state->mesh->append(mesh::read_fem_mesh(geometry, state->args["root_path"], false));

					if (i == mesh_id_)
						end = state->mesh->n_vertices();

					i++;
				}
			}

			state->load_mesh();
			state->stats.compute_mesh_stats(*state->mesh);
			// state->build_basis();

			// state->set_log_level(static_cast<spdlog::level::level_enum>(cur_log));
		}

		const int dim = states_[0]->mesh->dimension();
		parametrization_.set_output_indexing(Eigen::VectorXi::LinSpaced((end - start) * dim, start * dim, end * dim - 1));
	}
	Eigen::VectorXd SDFShapeVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("SDF shape doesn't support inverse evaluation!");
		return Eigen::VectorXd();
	}

	void ElasticVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		for (auto state : states_)
		{
			const int n_elem = state->bases.size();
			assert(n_elem * 2 == state_variable.size());
			state->assembler.update_lame_params(state_variable.segment(0, n_elem), state_variable.segment(n_elem, n_elem));
		}
	}
	Eigen::VectorXd ElasticVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_material_transient_adjoint_term(*state, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				AdjointTools::dJ_material_static_adjoint_term(*state, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	Eigen::VectorXd ElasticVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("Not implemented!");
		return Eigen::VectorXd();
	}

	void FrictionCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 1);
		assert(state_variable(0) >= 0);
		for (auto state : states_)
			state->args["contact"]["friction_coefficient"] = state_variable(0);
	}
	Eigen::VectorXd FrictionCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_friction_transient_adjoint_term(*state, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				log_and_throw_error("Friction coefficient grad in static simulations not implemented!");
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	Eigen::VectorXd FrictionCoeffientVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("Not implemented!");
		return Eigen::VectorXd();
	}

	void DampingCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 2);
		json damping_param = {
			{"psi", state_variable(0)},
			{"phi", state_variable(1)},
		};
		for (auto state : states_)
			state->assembler.add_multimaterial(0, damping_param);
		logger().info("Current damping params: {}, {}", state_variable(0), state_variable(1));
	}
	Eigen::VectorXd DampingCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_damping_transient_adjoint_term(*state, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				log_and_throw_error("Static damping not supported!");
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	Eigen::VectorXd DampingCoeffientVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("Not implemented!");
		return Eigen::VectorXd();
	}

	void InitialConditionVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		for (auto state : states_)
		{
			assert(state_variable.size() == state->ndof() * 2);
			state->initial_sol_update = state_variable.head(state->ndof());
			state->initial_vel_update = state_variable.tail(state->ndof());
		}
	}
	Eigen::VectorXd InitialConditionVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_initial_condition_adjoint_term(*state, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				log_and_throw_error("Static initial condition not supported!");
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	Eigen::VectorXd InitialConditionVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("Not implemented!");
		return Eigen::VectorXd();
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
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_dirichlet_transient_adjoint_term(*state, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				log_and_throw_error("Static dirichlet boundary optimization not supported!");
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	std::string DirichletVariableToSimulation::variable_to_string(const Eigen::VectorXd &variable)
	{
		return "";
	}
	Eigen::VectorXd DirichletVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("Not implemented!");
		return Eigen::VectorXd();
	}

	void MacroStrainVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		for (auto state : states_)
		{
			assert(state_variable.size() == state->mesh->dimension() * state->mesh->dimension());
			state->disp_grad_ = utils::unflatten(state_variable, state->mesh->dimension());
		}
	}
	Eigen::VectorXd MacroStrainVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				log_and_throw_error("Transient macro strain optimization not supported!");
			}
			else
			{
				AdjointTools::dJ_macro_strain_adjoint_term(*state, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return parametrization_.apply_jacobian(term, x);
	}
	Eigen::VectorXd MacroStrainVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("Not implemented!");
		return Eigen::VectorXd();
	}
} // namespace polyfem::solver