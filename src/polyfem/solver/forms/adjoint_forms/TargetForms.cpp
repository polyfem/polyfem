#include "TargetForms.hpp"
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/State.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/assembler/Mass.hpp>

#include <polyfem/utils/IntegrableFunctional.hpp>

using namespace polyfem::utils;

namespace polyfem::solver
{
	namespace
	{
		class LocalThreadScalarStorage
		{
		public:
			double val;
			assembler::ElementAssemblyValues vals;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};
	} // namespace

	IntegrableFunctional TargetForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		if (target_state_)
		{
			assert(target_state_->diff_cached.size() > 0);

			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
				int e_ref = params.elem;
				if (auto search = e_to_ref_e_.find(params.elem); search != e_to_ref_e_.end())
					e_ref = search->second;

				Eigen::MatrixXd pts_ref;
				target_state_->geom_bases()[e_ref].eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::VectorXd &sol_ref = target_state_->diff_cached.u(target_state_->problem->is_time_dependent() ? params.step : 0);
				io::Evaluator::interpolate_at_local_vals(*(target_state_->mesh), target_state_->problem->is_scalar(), target_state_->bases, target_state_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				val = (u_ref + pts_ref - u - pts).rowwise().squaredNorm();
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
				int e_ref = params.elem;
				if (auto search = e_to_ref_e_.find(params.elem); search != e_to_ref_e_.end())
					e_ref = search->second;

				Eigen::MatrixXd pts_ref;
				target_state_->geom_bases()[e_ref].eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::VectorXd &sol_ref = target_state_->diff_cached.u(target_state_->problem->is_time_dependent() ? params.step : 0);
				io::Evaluator::interpolate_at_local_vals(*(target_state_->mesh), target_state_->problem->is_scalar(), target_state_->bases, target_state_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				val = 2 * (u + pts - u_ref - pts_ref);
			};

			j.set_j(j_func);
			j.set_dj_du(djdu_func);
			j.set_dj_dx(djdu_func); // only used for shape derivative
		}
		else if (have_target_func)
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);

				const Eigen::VectorXd X = u + pts;
				for (int q = 0; q < u.rows(); q++)
					val(q) = target_func(X(q, 0), X(q, 1), X.cols() == 2 ? 0 : X(q, 2), 0, params.elem);
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());

				const Eigen::VectorXd X = u + pts;
				for (int q = 0; q < u.rows(); q++)
					for (int d = 0; d < val.cols(); d++)
						val(q, d) = target_func_grad[d](X(q, 0), X(q, 1), X.cols() == 2 ? 0 : X(q, 2), 0, params.elem);
			};

			j.set_j(j_func);
			j.set_dj_du(djdu_func);
			j.set_dj_dx(djdu_func); // only used for shape derivative
		}
		else // error wrt. a constant displacement
		{
			if (target_disp.size() == state_.mesh->dimension())
			{
				auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
					val.setZero(u.rows(), 1);

					for (int q = 0; q < u.rows(); q++)
					{
						Eigen::VectorXd err = u.row(q) - this->target_disp.transpose();
						for (int d = 0; d < active_dimension_mask.size(); d++)
							if (!active_dimension_mask[d])
								err(d) = 0;
						val(q) = err.squaredNorm();
					}
				};
				auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
					val.setZero(u.rows(), u.cols());

					for (int q = 0; q < u.rows(); q++)
					{
						Eigen::VectorXd err = u.row(q) - this->target_disp.transpose();
						for (int d = 0; d < active_dimension_mask.size(); d++)
							if (!active_dimension_mask[d])
								err(d) = 0;
						val.row(q) = 2 * err;
					}
				};

				j.set_j(j_func);
				j.set_dj_du(djdu_func);
			}
			else
				log_and_throw_adjoint_error("[{}] Only constant target displacement is supported!", name());
		}

		return j;
	}

	void TargetForm::set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids)
	{
		target_state_ = target_state;

		std::map<int, std::vector<int>> ref_interested_body_id_to_e;
		int ref_count = 0;
		for (int e = 0; e < target_state_->bases.size(); ++e)
		{
			int body_id = target_state_->mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (ref_interested_body_id_to_e.find(body_id) != ref_interested_body_id_to_e.end())
				ref_interested_body_id_to_e[body_id].push_back(e);
			else
				ref_interested_body_id_to_e[body_id] = {e};
			ref_count++;
		}

		std::map<int, std::vector<int>> interested_body_id_to_e;
		int count = 0;
		for (int e = 0; e < state_.bases.size(); ++e)
		{
			int body_id = state_.mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (interested_body_id_to_e.find(body_id) != interested_body_id_to_e.end())
				interested_body_id_to_e[body_id].push_back(e);
			else
				interested_body_id_to_e[body_id] = {e};
			count++;
		}

		if (count != ref_count)
			adjoint_logger().error("[{}] Number of interested elements in the reference and optimization examples do not match! {} {}", name(), count, ref_count);
		else
			adjoint_logger().trace("[{}] Found {} matching elements.", name(), count);

		for (const auto &kv : interested_body_id_to_e)
		{
			for (int i = 0; i < kv.second.size(); ++i)
			{
				e_to_ref_e_[kv.second[i]] = ref_interested_body_id_to_e[kv.first][i];
			}
		}
	}

	void TargetForm::set_reference(const json &func, const json &grad_func)
	{
		target_func.init(func);
		for (size_t k = 0; k < grad_func.size(); k++)
			target_func_grad[k].init(grad_func[k]);
		have_target_func = true;
	}

	NodeTargetForm::NodeTargetForm(const State &state, const VariableToSimulationGroup &variable_to_simulations, const json &args) : StaticForm(variable_to_simulations), state_(state)
	{
		std::string target_data_path = args["target_data_path"];
		if (!std::filesystem::is_regular_file(target_data_path))
		{
			throw std::runtime_error("Marker path invalid!");
		}
		Eigen::MatrixXd tmp;
		io::read_matrix(target_data_path, tmp);

		// markers to nodes
		Eigen::VectorXi nodes = tmp.col(0).cast<int>();
		target_vertex_positions.setZero(nodes.size(), state_.mesh->dimension());
		active_nodes.reserve(nodes.size());
		for (int s = 0; s < nodes.size(); s++)
		{
			const int node_id = state_.in_node_to_node(nodes(s));
			target_vertex_positions.row(s) = tmp.block(s, 1, 1, tmp.cols() - 1);
			active_nodes.push_back(node_id);
		}
	}

	NodeTargetForm::NodeTargetForm(const State &state, const VariableToSimulationGroup &variable_to_simulations, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_) : StaticForm(variable_to_simulations), state_(state), target_vertex_positions(target_vertex_positions_), active_nodes(active_nodes_)
	{
	}

	Eigen::VectorXd NodeTargetForm::compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::VectorXd rhs;
		rhs.setZero(state.diff_cached.u(0).size());

		const int dim = state_.mesh->dimension();

		if (&state == &state_)
		{
			int i = 0;
			Eigen::VectorXd disp = state_.diff_cached.u(time_step);
			for (int v : active_nodes)
			{
				RowVectorNd cur_pos = state_.mesh_nodes->node_position(v) + disp.segment(v * dim, dim).transpose();

				rhs.segment(v * dim, dim) = 2 * (cur_pos - target_vertex_positions.row(i++));
			}
		}

		return rhs * weight();
	}

	double NodeTargetForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		const int dim = state_.mesh->dimension();
		double val = 0;
		int i = 0;
		Eigen::VectorXd disp = state_.diff_cached.u(time_step);
		for (int v : active_nodes)
		{
			RowVectorNd cur_pos = state_.mesh_nodes->node_position(v) + disp.segment(v * dim, dim).transpose();
			val += (cur_pos - target_vertex_positions.row(i++)).squaredNorm();
		}
		return val;
	}

	void NodeTargetForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this]() {
			log_and_throw_adjoint_error("[{}] Doesn't support derivatives wrt. shape!", name());
			return Eigen::VectorXd::Zero(0).eval();
		});
	}

	BarycenterTargetForm::BarycenterTargetForm(const VariableToSimulationGroup &variable_to_simulations, const json &args, const std::shared_ptr<State> &state1, const std::shared_ptr<State> &state2) : StaticForm(variable_to_simulations)
	{
		dim = state1->mesh->dimension();
		json tmp_args = args;
		for (int d = 0; d < dim; d++)
		{
			tmp_args["dim"] = d;
			center1.push_back(std::make_unique<PositionForm>(variable_to_simulations, *state1, tmp_args));
			center2.push_back(std::make_unique<PositionForm>(variable_to_simulations, *state2, tmp_args));
		}
	}

	Eigen::VectorXd BarycenterTargetForm::compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::VectorXd term;
		term.setZero(state.ndof());
		for (int d = 0; d < dim; d++)
		{
			double value = center1[d]->value_unweighted_step(time_step, x) - center2[d]->value_unweighted_step(time_step, x);
			term += (2 * value) * (center1[d]->compute_adjoint_rhs_step(time_step, x, state) - center2[d]->compute_adjoint_rhs_step(time_step, x, state));
		}
		return term * weight();
	}
	void BarycenterTargetForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		Eigen::VectorXd tmp1, tmp2;
		for (int d = 0; d < dim; d++)
		{
			double value = center1[d]->value_unweighted_step(time_step, x) - center2[d]->value_unweighted_step(time_step, x);
			center1[d]->compute_partial_gradient_step(time_step, x, tmp1);
			center2[d]->compute_partial_gradient_step(time_step, x, tmp2);
			gradv += (2 * value) * (tmp1 - tmp2);
		}
		gradv *= weight();
	}
	double BarycenterTargetForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		double dist = 0;
		for (int d = 0; d < dim; d++)
			dist += std::pow(center1[d]->value_unweighted_step(time_step, x) - center2[d]->value_unweighted_step(time_step, x), 2);

		return dist;
	}
} // namespace polyfem::solver