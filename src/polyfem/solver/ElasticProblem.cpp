#include "ElasticProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/writeOBJ.h>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <filesystem>

namespace polyfem
{
	using namespace utils;

	ElasticProblem::ElasticProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
	{
		optimization_name = "material";

		x_to_param = [](const TVector &x, State &state) {
			auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
			auto cur_mus = state.assembler.lame_params().mu_mat_;

			for (int e = 0; e < cur_mus.size(); e++)
			{
				cur_mus(e) = x(e + cur_mus.size());
				cur_lambdas(e) = x(e);
			}
			state.assembler.update_lame_params(cur_lambdas, cur_mus);
		};

		param_to_x = [](TVector &x, State &state) {
			const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
			const auto &cur_mus = state.assembler.lame_params().mu_mat_;

			x.setZero(cur_lambdas.size() + cur_mus.size());
			for (int e = 0; e < cur_mus.size(); e++)
			{
				x(e + cur_mus.size()) = cur_mus(e);
				x(e) = cur_lambdas(e);
			}
		};

		dparam_to_dx = [](TVector &dx, const Eigen::VectorXd &dparams, State &state) {
			dx = dparams.head(state.bases.size() * 2);
		};

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "material")
			{
				material_params = param;
				break;
			}
		}

		if (material_params["mu_bound"].get<std::vector<double>>().size() == 0)
		{
			min_mu = 0.0;
			max_mu = std::numeric_limits<double>::max();
		}
		else
		{
			min_mu = material_params["mu_bound"][0];
			max_mu = material_params["mu_bound"][1];
		}

		if (material_params["lambda_bound"].get<std::vector<double>>().size() == 0)
		{
			min_lambda = 0.0;
			max_lambda = std::numeric_limits<double>::max();
		}
		else
		{
			min_lambda = material_params["lambda_bound"][0];
			max_lambda = material_params["lambda_bound"][1];
		}

		if (material_params["E_bound"].get<std::vector<double>>().size() == 0)
		{
			min_E = 0.0;
			max_E = std::numeric_limits<double>::max();
		}
		else
		{
			min_E = material_params["E_bound"][0];
			max_E = material_params["E_bound"][1];
		}

		if (material_params["nu_bound"].get<std::vector<double>>().size() == 0)
		{
			min_nu = 0.0;
			max_nu = std::numeric_limits<double>::max();
		}
		else
		{
			min_nu = material_params["nu_bound"][0];
			max_nu = material_params["nu_bound"][1];
		}

		has_material_smoothing = false;
		for (const auto &param : opt_params["functionals"])
		{
			if (param["type"] == "material_smoothing")
			{
				smoothing_params = param;
				has_material_smoothing = true;
				smoothing_weight = smoothing_params.value("weight", 1.0);
				break;
			}
			else
				target_weight = param.value("weight", 1.0);
		}
		
		// Only works for 2d for now
		if (has_material_smoothing && !state.mesh->is_volume())
		{
			std::vector<Eigen::Triplet<bool>> tt_adjacency_list;
			const auto &mesh2d = *dynamic_cast<mesh::Mesh2D *>(state.mesh.get());
			for (int i = 0; i < state.mesh->n_faces(); ++i)
			{
				auto idx = mesh2d.get_index_from_face(i);
				assert(idx.face == i);
				{
					auto adjacent_idx = mesh2d.switch_face(idx);
					if (adjacent_idx.face != -1)
						tt_adjacency_list.emplace_back(idx.face, adjacent_idx.face, true);
				}
				idx = mesh2d.next_around_face(idx);
				assert(idx.face == i);
				{
					auto adjacent_idx = mesh2d.switch_face(idx);
					if (adjacent_idx.face != -1)
						tt_adjacency_list.emplace_back(idx.face, adjacent_idx.face, true);
				}
				idx = mesh2d.next_around_face(idx);
				assert(idx.face == i);
				{
					auto adjacent_idx = mesh2d.switch_face(idx);
					if (adjacent_idx.face != -1)
						tt_adjacency_list.emplace_back(idx.face, adjacent_idx.face, true);
				}
			}
			tt_adjacency.resize(state.mesh->n_faces(), state.mesh->n_faces());
			tt_adjacency.setFromTriplets(tt_adjacency_list.begin(), tt_adjacency_list.end());
		}
	}

	void ElasticProblem::line_search_end()
	{
	}

	double ElasticProblem::target_value(const TVector &x)
	{
		return target_weight * j->energy(state);
	}

	double ElasticProblem::smooth_value(const TVector &x)
	{
		if (!has_material_smoothing || state.mesh->is_volume())
			return 0;

		// no need to use x because x_to_state was called in the solve
		const auto &lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &mus = state.assembler.lame_params().mu_mat_;

		double value = 0;
		for (int k = 0; k < tt_adjacency.outerSize(); ++k)
			for (Eigen::SparseMatrix<bool>::InnerIterator it(tt_adjacency, k); it; ++it)
			{
				value += pow((1 - lambdas(it.row()) / lambdas(it.col())), 2);
				value += pow((1 - mus(it.row()) / mus(it.col())), 2);
			}
		value /= 3 * tt_adjacency.rows();
		return smoothing_weight * value;
	}

	double ElasticProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			double target_val, smooth_val;
			target_val = target_value(x);
			smooth_val = smooth_value(x);
			logger().debug("elastic: target = {}, smooth = {}", target_val, smooth_val);
			cur_val = target_val + smooth_val;
		}
		return cur_val;
	}

	void ElasticProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd dparam = j->gradient(state, "material");

		dparam_to_dx(gradv, dparam, state);
		gradv *= target_weight;
	}

	void ElasticProblem::smooth_gradient(const TVector &x, TVector &gradv)
	{
		if (!has_material_smoothing || state.mesh->is_volume())
		{
			gradv.setZero(x.size());
			return;
		}

		const auto &lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &mus = state.assembler.lame_params().mu_mat_;
		Eigen::MatrixXd dJ_dmu, dJ_dlambda;
		dJ_dmu.setZero(mus.size(), 1);
		dJ_dlambda.setZero(lambdas.size(), 1);

		for (int k = 0; k < tt_adjacency.outerSize(); ++k)
			for (Eigen::SparseMatrix<bool>::InnerIterator it(tt_adjacency, k); it; ++it)
			{
				dJ_dlambda(it.row()) += 2 * (lambdas(it.row()) / lambdas(it.col()) - 1) / lambdas(it.col());
				dJ_dlambda(it.col()) += 2 * (1 - lambdas(it.row()) / lambdas(it.col())) * lambdas(it.row()) / lambdas(it.col()) / lambdas(it.col());
				dJ_dmu(it.row()) += 2 * (mus(it.row()) / mus(it.col()) - 1) / mus(it.col());
				dJ_dmu(it.col()) += 2 * (1 - mus(it.row()) / mus(it.col())) * mus(it.row()) / mus(it.col()) / mus(it.col());
			}

		dJ_dmu /= 3 * tt_adjacency.rows();
		dJ_dlambda /= 3 * tt_adjacency.rows();

		Eigen::VectorXd dparam;
		dparam.setZero(dJ_dmu.size() + dJ_dlambda.size() + 3);
		dparam.head(dJ_dlambda.size()) = dJ_dlambda;
		dparam.segment(dJ_dlambda.size(), dJ_dmu.size()) = dJ_dmu;

		dparam_to_dx(gradv, dparam, state);
		gradv *= smoothing_weight;
	}

	void ElasticProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			Eigen::VectorXd grad_target, grad_smoothing;
			target_gradient(x, grad_target);
			smooth_gradient(x, grad_smoothing);
			logger().debug("elastic: ‖∇ target‖ = {}, ‖∇ smooth‖ = {}", grad_target.norm(), grad_smoothing.norm());
			cur_grad = grad_target + grad_smoothing;
		}

		gradv = cur_grad;
	}

	bool ElasticProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		if ((x1 - x0).cwiseAbs().maxCoeff() > max_change)
			return false;
		solution_changed_pre(x1);

		const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &cur_mus = state.assembler.lame_params().mu_mat_;

		bool flag = true;

		if (cur_lambdas.minCoeff() < min_lambda || cur_mus.minCoeff() < min_mu)
			flag = false;
		if (cur_lambdas.maxCoeff() > max_lambda || cur_mus.maxCoeff() > max_mu)
			flag = false;

		for (int e = 0; e < cur_lambdas.size(); e++)
		{
			const double E = cur_mus(e) * (3 * cur_lambdas(e) + 2 * cur_mus(e)) / (cur_lambdas(e) + cur_mus(e));
			const double nu = cur_lambdas(e) / (2 * (cur_lambdas(e) + cur_mus(e)));

			if (E < min_E || E > max_E || nu < min_nu || E > max_E)
				flag = false;
		}

		solution_changed_pre(x0);

		return flag;
	}

	bool ElasticProblem::solution_changed_pre(const TVector &newX)
	{
		x_to_param(newX, state);
		// state.set_materials();
		return true;
	}

} // namespace polyfem