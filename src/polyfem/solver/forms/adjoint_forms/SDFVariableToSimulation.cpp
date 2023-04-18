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
		set_output_indexing(Eigen::VectorXi::LinSpaced((end - start) * dim, start * dim, end * dim - 1));
	}
	Eigen::VectorXd SDFShapeVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("SDF shape doesn't support inverse evaluation!");
		return Eigen::VectorXd();
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
			if (state->in_args["geometry"].is_array())
			{
				assert(state->in_args["geometry"].size() == 1);
				state->in_args["geometry"][0]["mesh"] = mesh_path_;
				state->in_args["geometry"][0].erase("transformation");
			}
			else
			{
				state->in_args["geometry"]["mesh"] = mesh_path_;
				state->in_args["geometry"].erase("transformation");
			}

			state->mesh = nullptr;
			state->assembler.update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

			state->init(state->in_args, false);

			state->load_mesh();
			state->stats.compute_mesh_stats(*state->mesh);
		}
	}
	Eigen::VectorXd SDFPeriodicShapeVariableToSimulation::inverse_eval()
	{
		log_and_throw_error("SDF shape doesn't support inverse evaluation!");
		return Eigen::VectorXd();
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
}