#include "VariableToSimulation.hpp"
#include <polyfem/State.hpp>
#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

namespace polyfem::solver
{
	SDFShapeVariableToSimulation::SDFShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args) : ShapeVariableToSimulation(states, parametrization), mesh_id_(args["mesh_id"]), mesh_path_(args["mesh"])
	{
	}
	void SDFShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		parametrization_.eval(x);

		int start = 0, end = 0; // start vertex index of the mesh
		for (auto state : states_)
		{
			state->args["geometry"][mesh_id_]["mesh"] = mesh_path_;
			state->args["geometry"][mesh_id_].erase("transformation");

			state->mesh = nullptr;
			state->assembler->update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

			json in_args = state->args;
			state->init(in_args, false);

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
		set_output_indexing(Eigen::VectorXi::LinSpaced((end - start) * dim, start * dim, end * dim - 1));
	}

	SDFPeriodicShapeVariableToSimulation::SDFPeriodicShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args) : PeriodicShapeVariableToSimulation(states, parametrization), mesh_path_(args["mesh"])
	{
        logger().warn("SDFPeriodicShapeVariableToSimulation only supports mesh with periodic nodes on the bounding box and unit size!");
	}
	void SDFPeriodicShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		auto y = parametrization_.eval(x);

		for (auto state : states_)
		{
			if (state->args["geometry"].is_array())
			{
				assert(state->args["geometry"].size() == 1);
				state->args["geometry"][0]["mesh"] = mesh_path_;
				state->args["geometry"][0].erase("transformation");
			}
			else
			{
				state->args["geometry"]["mesh"] = mesh_path_;
				state->args["geometry"].erase("transformation");
			}

			state->mesh = nullptr;
			state->assembler->update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

			json in_args = state->args;
			state->init(state->args, false);

			state->load_mesh();
			state->stats.compute_mesh_stats(*state->mesh);
		}
	}
    Eigen::VectorXd SDFPeriodicShapeVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
    {
        auto &periodic_mesh_map = *(states_[0]->periodic_mesh_map);
        const int dim = periodic_mesh_map.dim();
        const int n_full_verts = periodic_mesh_map.n_full_dof();
        const int n_periodic_verts = periodic_mesh_map.n_periodic_dof();

        assert(term.size() == periodic_mesh_map.input_size());
        Eigen::VectorXd full_term;
        full_term.setZero(n_full_verts * dim);
        Eigen::Matrix<bool, -1, 1> visited_mask;
        visited_mask.setZero(n_periodic_verts);
        for (int i = 0; i < n_full_verts; i++)
        {
            int i_periodic = periodic_mesh_map.full_to_periodic(i);
            if (!visited_mask(i_periodic))
                full_term.segment(i * dim, dim) = term.segment(i_periodic * dim, dim);
            visited_mask(i_periodic) = true;
        }
        
        return parametrization_.apply_jacobian(full_term, x);
    }

	PeriodicShapeScaleVariableToSimulation::PeriodicShapeScaleVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args) : PeriodicShapeVariableToSimulation(states, parametrization)
	{
		for (const auto &state : states)
		{
			if (!state->args["space"]["advanced"]["periodic_mesh"].get<bool>())
				log_and_throw_error("PeriodicShapeScaleVariableToSimulation is designed for periodic mesh representation!");
			dim = state->mesh->dimension();
		}
	}
	void PeriodicShapeScaleVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == dim);

		logger().info("mesh scale: {}", y.transpose());
		
		for (auto state : states_)
		{
			assert(state->periodic_mesh_representation.size() > dim);
			state->periodic_mesh_representation.tail(dim) = y;
		
			auto V = utils::unflatten(state->periodic_mesh_map->eval(state->periodic_mesh_representation), dim);
			for (int i = 0; i < V.rows(); i++)
				state->set_mesh_vertex(i, V.row(i));
		}
	}
	Eigen::VectorXd PeriodicShapeScaleVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
	{
		assert(x.size() == dim);
		return parametrization_.apply_jacobian(term.tail(dim), x);
	}
}