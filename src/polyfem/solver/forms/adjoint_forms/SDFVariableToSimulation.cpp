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
						state->mesh = mesh::read_fem_mesh(state->units, geometry, state->args["root_path"], false);
					else
						state->mesh->append(mesh::read_fem_mesh(state->units, geometry, state->args["root_path"], false));

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
} // namespace polyfem::solver