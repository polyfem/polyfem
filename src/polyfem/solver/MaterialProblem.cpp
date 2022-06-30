#include <polyfem/MaterialProblem.hpp>

#include <polyfem/Types.hpp>
#include <polyfem/Timer.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <igl/writeOBJ.h>

#include <filesystem>

namespace polyfem
{
	MaterialProblem::MaterialProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args) : OptimizationProblem(state_, j_, args)
	{
		optimization_name = "material";
		state.args["export"]["material_params"] = true;

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

		min_mu = 1;
		if (opt_params.contains("min_mu"))
			min_mu = opt_params["min_mu"];
		max_mu = 1e10;
		if (opt_params.contains("max_mu"))
			max_mu = opt_params["max_mu"];

		min_lambda = 1;
		if (opt_params.contains("min_lambda"))
			min_lambda = opt_params["min_lambda"];
		max_lambda = 1e10;
		if (opt_params.contains("max_lambda"))
			max_lambda = opt_params["max_lambda"];

		if (args.contains("target_weight"))
			target_weight = args["target_weight"];
		smoothing_weight = args.contains("smoothing_weight") ? args["smoothing_weight"].get<double>() : 0.;

		// Only works for 2d for now
		if (!state.mesh->is_volume())
		{
			std::vector<Eigen::Triplet<bool>> tt_adjacency_list;
			const Mesh2D &mesh2d = *dynamic_cast<const Mesh2D *>(state.mesh.get());
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

	void MaterialProblem::line_search_end(bool failed)
	{
		if (opt_params.contains("export_energies"))
		{
			std::ofstream outfile;
			outfile.open(opt_params["export_energies"], std::ofstream::out | std::ofstream::app);

			outfile << value(cur_x) << ", " << target_value(cur_x) << ", " << smooth_value(cur_x) << "\n";
			outfile.close();
		}
	}

	double MaterialProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double size = 1;
		const auto lambda0 = state.assembler.lame_params().lambda_mat_;
		const auto mu0 = state.assembler.lame_params().mu_mat_;
		while (size > 0)
		{
			auto newX = x0 + (x1 - x0) * size;
			x_to_param(newX, state);
			const auto lambda1 = state.assembler.lame_params().lambda_mat_;
			const auto mu1 = state.assembler.lame_params().mu_mat_;

			const double max_change = opt_params.contains("max_change") ? opt_params["max_change"].get<double>() : 1e10;
			if ((lambda1 - lambda0).cwiseAbs().maxCoeff() > max_change || (mu1 - mu0).cwiseAbs().maxCoeff() > max_change || !is_step_valid(x0, newX))
				size /= 2.;
			else
				break;
		}
		x_to_param(x0, state);

		return size;
	}

	double MaterialProblem::heuristic_max_step(const TVector &dx)
	{
		return opt_params.contains("max_step") ? opt_params["max_step"].get<double>() : 1;
	}

	double MaterialProblem::target_value(const TVector &x)
	{
		return target_weight * j->energy(state);
	}

	double MaterialProblem::smooth_value(const TVector &x)
	{
		if (state.mesh->is_volume())
			return 0;

		// no need to use x because x_to_state was called in the solve
		const auto &lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &mus = state.assembler.lame_params().mu_mat_;

		double value = 0;
		for (int k = 0; k < tt_adjacency.outerSize(); ++k)
			for (SparseMatrix<bool>::InnerIterator it(tt_adjacency, k); it; ++it)
			{
				value += pow((1 - lambdas(it.row()) / lambdas(it.col())), 2);
				value += pow((1 - mus(it.row()) / mus(it.col())), 2);
			}
		value /= 3 * tt_adjacency.rows();
		return smoothing_weight * value;
	}

	double MaterialProblem::value(const TVector &x)
	{
		double target_val, smooth_val;
		target_val = target_value(x);
		smooth_val = smooth_value(x);
		logger().debug("target = {}, smooth = {}", target_val, smooth_val);
		return target_val + smooth_val;
	}

	void MaterialProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd dparam = j->gradient(state, state.problem->is_time_dependent() ? "material-full" : "material");

		dparam_to_dx(gradv, dparam, state);
		gradv *= target_weight;
	}

	void MaterialProblem::smooth_gradient(const TVector &x, TVector &gradv)
	{
		if (state.mesh->is_volume())
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
			for (SparseMatrix<bool>::InnerIterator it(tt_adjacency, k); it; ++it)
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

	void MaterialProblem::gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd grad_target, grad_smoothing;
		target_gradient(x, grad_target);
		smooth_gradient(x, grad_smoothing);
		logger().debug("‖∇ target‖ = {}, ‖∇ smooth‖ = {}", grad_target.norm(), grad_smoothing.norm());

		gradv = grad_target + grad_smoothing;
	}

	bool MaterialProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &cur_mus = state.assembler.lame_params().mu_mat_;
		const double mu = state.args["mu"];
		const double psi = state.args["params"]["psi"];
		const double phi = state.args["params"]["phi"];

		if (mu < 0 || psi < 0 || phi < 0)
			return false;

		if (cur_lambdas.minCoeff() < min_lambda || cur_mus.minCoeff() < min_mu)
			return false;
		if (cur_lambdas.maxCoeff() > max_lambda || cur_mus.maxCoeff() > max_mu)
			return false;
		if (opt_params.contains("min_phi") && opt_params["min_phi"].get<double>() > phi)
			return false;
		if (opt_params.contains("max_phi") && opt_params["max_phi"].get<double>() < phi)
			return false;
		if (opt_params.contains("min_psi") && opt_params["min_psi"].get<double>() > psi)
			return false;
		if (opt_params.contains("max_psi") && opt_params["max_psi"].get<double>() < psi)
			return false;
		if (opt_params.contains("min_fric") && opt_params["min_fric"].get<double>() > mu)
			return false;
		if (opt_params.contains("max_fric") && opt_params["max_fric"].get<double>() < mu)
			return false;

		return true;
	}

	void MaterialProblem::solution_changed(const TVector &newX)
	{
		if (cur_x.size() == newX.size() && cur_x == newX)
			return;

		x_to_param(newX, state);
		state.build_basis();
		solve_pde(newX);

		cur_x = newX;
	}
} // namespace polyfem