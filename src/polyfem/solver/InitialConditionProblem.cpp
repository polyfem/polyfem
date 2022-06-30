#include <polyfem/InitialConditionProblem.hpp>

#include <polyfem/Types.hpp>
#include <polyfem/Timer.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <igl/writeOBJ.h>

#include <filesystem>

namespace polyfem
{
	InitialConditionProblem::InitialConditionProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args) : OptimizationProblem(state_, j_, args)
	{
		assert(state.problem->is_time_dependent());
		optimization_name = "initial";

		const int dof = state.n_bases;
		const int dim_ = dim;
		x_to_param = [dim_, dof](const TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel) {
			init_sol.setZero(dof * dim_, 1);
			init_vel.setZero(dof * dim_, 1);
			for (int i = 0; i < dof; i++)
				for (int d = 0; d < x.size(); d++)
					init_vel(i * dim_ + d) = x(d);
		};

		param_to_x = [dim_](TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel) {
			x = init_vel.block(0, 0, dim_, 1);
		};

		dparam_to_dx = [dim_, dof](TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel) {
			x.setZero(dim_);
			for (int i = 0; i < dof; i++)
				x += init_vel.block(i * dim_, 0, dim_, 1);
		};
	}

	void InitialConditionProblem::line_search_end(bool failed)
	{
		if (opt_params.contains("export_energies"))
		{
			std::ofstream outfile;
			outfile.open(opt_params["export_energies"], std::ofstream::out | std::ofstream::app);

			outfile << value(cur_x) << "\n";
			outfile.close();
		}

		if (!failed)
			return;
	}

	double InitialConditionProblem::value(const TVector &x)
	{
		return j->energy(state);
	}

	void InitialConditionProblem::gradient(const TVector &x, TVector &gradv)
	{
		TVector tmp = j->gradient(state, "initial-condition");

		Eigen::MatrixXd init_sol(tmp.size() / 2, 1), init_vel(tmp.size() / 2, 1);
		for (int i = 0; i < init_sol.size(); i++)
		{
			init_sol(i) = tmp(i);
			init_vel(i) = tmp(i + init_sol.size());
		}

		dparam_to_dx(gradv, init_sol, init_vel);
	}

	void InitialConditionProblem::solution_changed(const TVector &newX)
	{
		if (cur_x.size() == newX.size() && cur_x == newX)
			return;

		Eigen::MatrixXd init_sol, init_vel;
		x_to_param(newX, init_sol, init_vel);
		state.initial_sol_update = init_sol;
		state.initial_vel_update = init_vel;

		solve_pde(newX);
		cur_x = newX;
	}
} // namespace polyfem