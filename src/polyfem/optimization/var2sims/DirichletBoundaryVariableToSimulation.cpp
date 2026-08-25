#include <polyfem/optimization/var2sims/DirichletBoundaryVariableToSimulation.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/optimization/var2sims/ActiveSelectionUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <string>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>

namespace polyfem::solver
{

	DirichletBoundaryVariableToSimulation::DirichletBoundaryVariableToSimulation(
		StatePtrs states,
		DiffCachePtrs diff_caches,
		CompositeParametrization parametrizations,
		Eigen::VectorXi active_boundary_ids,
		Eigen::VectorXi active_time_slices)
		: dim_(states[0]->mesh->dimension()),
		  time_steps_(0),
		  states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations)),
		  active_boundary_ids_(std::move(active_boundary_ids)),
		  active_time_slices_(std::move(active_time_slices))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		// Static problem support is not implemented.
		for (auto &s : states_)
		{
			if (!s->problem->is_time_dependent())
			{
				log_and_throw_adjoint_error("Fail to construct dirichlet boundary variable to simulation. Reason: only transient simulations supported.");
			}
		}

		time_steps_ = states_[0]->args["time"]["time_steps"].get<int>();

		// Expand implicit all-active boundary id selection.
		if (active_boundary_ids_.size() == 0)
		{
			// PolyFEM core lacks dedicate API to query all boundary ids. Use API for boundary
			// active dimension to collect boundary ids.

			// boundary_dims is a map [boundary id, active dim].
			auto boundary_dims = states_[0]->boundary_conditions_ids("dirichlet_boundary");
			active_boundary_ids_.resize(boundary_dims.size());
			int i = 0;
			for (auto [id, _] : boundary_dims)
			{
				active_boundary_ids_[i] = id;
				++i;
			}
			std::sort(active_boundary_ids_.begin(), active_boundary_ids_.end());
		}
		// Expand implicit all-active time slice selection.
		if (active_time_slices_.size() == 0)
		{
			active_time_slices_ = Eigen::VectorXi::LinSpaced(time_steps_, 0, time_steps_ - 1);
		}

		// Validate expanded active selections against every state.
		std::string reason;
		if (!is_active_dirichlet_boundary_ids_valid(active_boundary_ids_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct dirichlet boundary variable to simulation. Reason: {}", reason);
		}
		if (!is_active_time_slices_valid(active_time_slices_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct dirichlet boundary variable to simulation. Reason: {}", reason);
		}

		build_boundary_node_maps();
	}

	std::string DirichletBoundaryVariableToSimulation::name() const
	{
		return "dirichlet-boundary";
	}

	ParameterType DirichletBoundaryVariableToSimulation::parameter_type() const
	{
		return ParameterType::DirichletBC;
	}

	bool DirichletBoundaryVariableToSimulation::affect_state(const legacy::State &target) const
	{
		for (auto &s : states_)
		{
			if (s.get() == &target)
			{
				return true;
			}
		}
		return false;
	}

	void DirichletBoundaryVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		int boundary_num = active_boundary_ids_.size();
		for (auto &s : states_)
		{
			auto tensor_problem = std::dynamic_pointer_cast<polyfem::assembler::GenericTensorProblem>(s->problem);
			if (!tensor_problem)
			{
				log_and_throw_adjoint_error("Only tensor problems are supported.");
			}

			for (int ti = 0; ti < active_time_slices_.size(); ++ti)
			{
				int t = active_time_slices_(ti) + 1;

				for (int bi = 0; bi < boundary_num; ++bi)
				{
					int boundary_id = active_boundary_ids_(bi);
					int offset = (ti * boundary_num + bi) * dim_;
					tensor_problem->update_dirichlet_boundary(boundary_id, t, y.segment(offset, dim_));
				}
			}
		}
	}

	void DirichletBoundaryVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		// This is not implemented because in practice, we only call this method for ShapeVariableToSimulation.
		log_and_throw_adjoint_error("update_state_variables not implemented in DirichletBoundaryVariableToSimulation.");
	}

	Eigen::VectorXd DirichletBoundaryVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term = Eigen::VectorXd::Zero(para_out_dof());

		for (int si = 0; si < states_.size(); ++si)
		{
			auto &state = states_[si];
			auto &diff_cache = diff_caches_[si];

			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
			Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*state, *diff_cache, 1);

			Eigen::VectorXd node_term;
			AdjointTools::dJ_dirichlet_transient_adjoint_term(*state, adjoint_nu, adjoint_p, node_term);

			int boundary_node_num = state->boundary_nodes.size();
			assert(node_term.size() == time_steps_ * boundary_node_num);

			// dJ_dirichlet_transient_adjoint_term compute adjoint terms per boundary node.
			// Gather FE node value into boundary selection value.
			const BoundaryNodeMap &map = boundary_node_maps_[si];
			for (int ti = 0; ti < active_time_slices_.size(); ++ti)
			{
				auto seg = node_term.segment(active_time_slices_(ti) * boundary_node_num, boundary_node_num);
				for (int bi = 0; bi < active_boundary_ids_.size(); ++bi)
				{
					for (int d = 0; d < dim_; ++d)
					{
						double sum = 0;
						for (int offset : map[bi][d])
						{
							sum += seg(offset);
						}
						int boundary_count = ti * active_boundary_ids_.size() + bi;
						term(boundary_count * dim_ + d) += sum;
					}
				}
			}
		}

		assert(term.size() == para_out_dof());
		return parametrization_.apply_jacobian(term, x);
	}

	int DirichletBoundaryVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd DirichletBoundaryVariableToSimulation::inverse_eval() const
	{
		Eigen::VectorXd y = Eigen::VectorXd::Zero(para_out_dof());
		std::vector<json> boundary_jsons =
			utils::json_as_array(states_[0]->args["boundary_conditions"]["dirichlet_boundary"]);

		for (int bi = 0; bi < active_boundary_ids_.size(); ++bi)
		{
			int boundary_id = active_boundary_ids_(bi);
			auto pred = [boundary_id](const json &bc) { return bc["id"].get<int>() == boundary_id; };
			auto iter = std::find_if(boundary_jsons.begin(), boundary_jsons.end(), pred);
			if (iter == boundary_jsons.end())
			{
				log_and_throw_adjoint_error("Cannot find boundary id {} in JSON.", boundary_id);
			}

			// User can specify dirichlet boundary value via list of value, const value, expression, or file.
			// We only support list of value.
			const json &value = (*iter)["value"];
			Eigen::MatrixXd dirichlet_mat;
			try
			{
				dirichlet_mat = value;
			}
			catch (std::exception &err)
			{
			}

			int required_cols = time_steps_ + 1;
			if (dirichlet_mat.rows() != dim_ || dirichlet_mat.cols() != required_cols)
			{
				logger().warn("Unsupported value type for dirichlet boundary id {}; inverse_eval falling back to zero.", boundary_id);
				dirichlet_mat = Eigen::MatrixXd::Zero(dim_, time_steps_ + 1);
			}

			for (int ti = 0; ti < active_time_slices_.size(); ++ti)
			{
				int slice = active_time_slices_(ti);
				for (int c = 0; c < dim_; ++c)
				{
					int boundary_count = ti * active_boundary_ids_.size() + bi;
					y(boundary_count * dim_ + c) = dirichlet_mat(c, slice + 1);
				}
			}
		}

		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd DirichletBoundaryVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &, const Eigen::VectorXd &) const
	{
		// Not implemented because there's no user
		log_and_throw_adjoint_error("apply_parametrization_jacobian is not implemented in {} variable to simulation.", name());
	}

	int DirichletBoundaryVariableToSimulation::para_out_dof() const
	{
		return active_time_slices_.size() * active_boundary_ids_.size() * dim_;
	}

	void DirichletBoundaryVariableToSimulation::build_boundary_node_maps()
	{
		// Map active boundary id to it's offset in active_boundary_ids_ vector.
		std::unordered_map<int, int> active_boundary_id_offset;
		for (int i = 0; i < active_boundary_ids_.size(); ++i)
		{
			active_boundary_id_offset[active_boundary_ids_(i)] = i;
		}

		boundary_node_maps_.clear();
		boundary_node_maps_.resize(states_.size());

		for (int si = 0; si < states_.size(); ++si)
		{
			const legacy::State &state = *states_[si];

			// Map boundary node (FE space dof) to offset in boundary_nodes vector.
			std::unordered_map<int, int> boundary_node_offset;
			for (int p = 0; p < state.boundary_nodes.size(); ++p)
			{
				boundary_node_offset[state.boundary_nodes[p]] = p;
			}

			BoundaryNodeMap map(active_boundary_ids_.size(), std::vector<std::vector<int>>(dim_));

			// - LocalBoundary stores boundary primitives (edge/face) per element.
			// - Each primitive can be tagged with a boundary id (selection id).
			// - Each primitive can be associated with mutiple geometric nodes (vertices).
			// - Basis maps each geometric node to FE dof.
			// - boundary_nodes stores all boundary dof in FE space.
			//
			// So to map boundary id to offsets in boundary_nodes, we have to
			// 1. Find primitives selected by boundary id.
			// 2. Map primitives to geoemtric nodes.
			// 3. Map geometric nodes to FE dof.
			// 4. Map FE dof to offset in boundary_nodes.
			for (auto &lb : state.local_boundary)
			{
				int e = lb.element_id();
				const basis::ElementBases &bs = state.bases[e];

				for (int i = 0; i < lb.size(); ++i)
				{
					int primitive_global_id = lb.global_primitive_id(i);
					int boundary_id = state.mesh->get_boundary_id(primitive_global_id);

					// 1. Find primitives selected by active boundary id.
					auto iter = active_boundary_id_offset.find(boundary_id);
					if (iter == active_boundary_id_offset.end())
					{
						continue;
					}
					int boundary_offset = iter->second;

					// 2. Map primitives to geometric nodes.
					Eigen::VectorXi geom_nodes = bs.local_nodes_for_primitive(primitive_global_id, *state.mesh);
					for (int geom_node : geom_nodes)
					{
						// 3. Map geometric nodes to FE dof.
						auto &local_to_globals = bs.bases[geom_node].global();
						for (auto &lg : local_to_globals)
						{
							for (int c = 0; c < dim_; ++c)
							{
								int fe_dof = lg.index * dim_ + c;

								// 4. Map FE dof to offset in boundary_nodes.
								auto iter = boundary_node_offset.find(fe_dof);
								assert(iter != boundary_node_offset.end() && "Expect boundary dof to exist in boundary_nodes");
								map[boundary_offset][c].push_back(iter->second);
							}
						}
					}
				}
			}

			boundary_node_maps_[si] = std::move(map);
		}
	}

} // namespace polyfem::solver
