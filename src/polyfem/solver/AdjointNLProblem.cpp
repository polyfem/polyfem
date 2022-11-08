#include "AdjointNLProblem.hpp"

namespace polyfem::solver
{
	double AdjointNLProblem::value(const Eigen::VectorXd &x)
	{
		return obj_->value();
	}

	void AdjointNLProblem::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		int cumulative = 0;
		gradv.setZero(optimization_dim_);

		for (auto &state_ptr : all_states_)
			state_ptr->solve_adjoint(obj_->compute_adjoint_rhs(*state_ptr));

		Eigen::VectorXd gradv_param;
		for (const auto &p : parameters_)
		{
			gradv_param.setZero(p->full_dim());
			for (auto &state_ptr : all_states_)
				gradv_param += obj_->gradient(*state_ptr, *p);
			
			gradv.segment(cumulative, p->optimization_dim()) += p->map_grad(gradv_param);
			cumulative += p->optimization_dim();
		}
	}

	void AdjointNLProblem::smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x)
	{
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			Eigen::VectorXd tmp = new_x.segment(cumulative, p->optimization_dim());
			p->smoothing(x.segment(cumulative, p->optimization_dim()), tmp);
			assert(tmp.size() == p->optimization_dim());
			new_x.segment(cumulative, p->optimization_dim()) = tmp;
			cumulative += p->optimization_dim();
		}
	}

	bool AdjointNLProblem::remesh(Eigen::VectorXd &x)
	{
		bool remesh = false;
		// TODO: remesh changes size of parameters, need to be careful
		// int cumulative = 0;
		// for (const auto &p : parameters_)
		// {
		// 	Eigen::VectorXd tmp = x.segment(cumulative, p->optimization_dim());
		// 	remesh |= p->remesh(tmp);
		// 	assert(tmp.size() == p->optimization_dim());
		// 	x.segment(cumulative, p->optimization_dim()) = tmp;
		// 	cumulative += p->optimization_dim();
		// }
		return remesh;
	}

	void AdjointNLProblem::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		int cumulative = 0;
		for (const auto p : parameters_)
		{
			p->line_search_begin(x0.segment(cumulative, p->optimization_dim()), x1.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
	}

	void AdjointNLProblem::line_search_end()
	{

		int cumulative = 0;
		for (const auto p : parameters_)
		{
			p->line_search_end();
			cumulative += p->optimization_dim();
		}
	}

	void AdjointNLProblem::post_step(const int iter_num, const Eigen::VectorXd &x0)
	{
		int cumulative = 0;
		for (const auto p : parameters_)
		{
			p->post_step(iter_num, x0.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		iter++;
	}

	Eigen::VectorXd AdjointNLProblem::get_lower_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd min(optimization_dim_);
		int cumulative = 0;
		for (const auto p : parameters_)
		{
			min.segment(cumulative, p->optimization_dim()) = p->get_lower_bound(x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return min;
	}
	Eigen::VectorXd AdjointNLProblem::get_upper_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd max(optimization_dim_);
		int cumulative = 0;
		for (const auto p : parameters_)
		{
			max.segment(cumulative, p->optimization_dim()) = p->get_upper_bound(x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return max;
	}

	Eigen::VectorXd AdjointNLProblem::force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx)
	{
		Eigen::VectorXd newX(optimization_dim_);
		int cumulative = 0;
		for (const auto p : parameters_)
		{
			newX.segment(cumulative, p->optimization_dim()) = p->force_inequality_constraint(x0.segment(cumulative, p->optimization_dim()), dx.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return newX;
	}

	int AdjointNLProblem::n_inequality_constraints()
	{
		int num = 0;
		for (const auto p : parameters_)
		{
			num += p->n_inequality_constraints();
		}
		return num;
	}

	double AdjointNLProblem::inequality_constraint_val(const Eigen::VectorXd &x, const int index)
	{
		int num = 0;
		int cumulative = 0;
		for (const auto p : parameters_)
		{
			num += p->n_inequality_constraints();
			if (num > index)
			{
				num -= p->n_inequality_constraints();
				return p->inequality_constraint_val(x.segment(cumulative, p->optimization_dim()), index - num);
			}
			cumulative += p->optimization_dim();
		}
		log_and_throw_error("Exceeding number of inequality constraints!");
		return 0;
	}

	Eigen::VectorXd AdjointNLProblem::inequality_constraint_grad(const Eigen::VectorXd &x, const int index)
	{
		int num = 0;
		int cumulative = 0;
		Eigen::VectorXd grad(optimization_dim_);
		grad.setZero();
		for (const auto p : parameters_)
		{
			num += p->n_inequality_constraints();
			if (num > index)
			{
				num -= p->n_inequality_constraints();
				grad.segment(cumulative, p->optimization_dim()) = p->inequality_constraint_grad(x.segment(cumulative, p->optimization_dim()), index - num);
				break;
			}
			cumulative += p->optimization_dim();
		}
		return grad;
	}

	void AdjointNLProblem::solution_changed(const Eigen::VectorXd &newX)
	{
		if (cur_x.size() == newX.size() && cur_x == newX)
			return;

		bool solve = true;
		for (const auto p : parameters_)
		{
			solve &= p->pre_solve(newX);
		}

		if (solve)
		{
			for (const auto state : all_states_)
			{
				state->assemble_rhs();
				state->assemble_stiffness_mat();
				Eigen::MatrixXd sol, pressure;
				state->solve_problem(sol, pressure);
			}
		}

		for (const auto p : parameters_)
		{
			p->post_solve(newX);
		}
	}

} // namespace polyfem