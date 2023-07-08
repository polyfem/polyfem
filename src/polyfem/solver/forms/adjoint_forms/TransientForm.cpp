#include "TransientForm.hpp"
#include <polyfem/State.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

namespace polyfem::solver
{
	namespace
	{
		class LocalThreadScalarStorage
		{
		public:
			double val;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};

		class LocalThreadMatStorage
		{
		public:
			Eigen::MatrixXd mat;

			LocalThreadMatStorage(const int row, const int col = 1)
			{
				mat.resize(row, col);
				mat.setZero();
			}
		};
	} // namespace
	std::vector<double> TransientForm::get_transient_quadrature_weights() const
	{
		std::vector<double> weights;
		weights.assign(time_steps_ + 1, dt_);
		if (transient_integral_type_ == "uniform")
		{
			weights[0] = 0;
		}
		else if (transient_integral_type_ == "trapezoidal")
		{
			weights[0] = dt_ / 2.;
			weights[weights.size() - 1] = dt_ / 2.;
		}
		else if (transient_integral_type_ == "simpson")
		{
			weights[0] = dt_ / 3.;
			weights[weights.size() - 1] = dt_ / 3.;
			for (int i = 1; i < weights.size() - 1; i++)
			{
				if (i % 2)
					weights[i] = dt_ * 4. / 3.;
				else
					weights[i] = dt_ * 2. / 4.;
			}
		}
		else if (transient_integral_type_ == "final")
		{
			weights.assign(time_steps_ + 1, 0);
			weights[time_steps_] = 1;
		}
		else if (transient_integral_type_ == "steps")
		{
			weights.assign(time_steps_ + 1, 0);
			for (const int step : steps_)
			{
				assert(step > 0 && step < weights.size());
				weights[step] += 1. / steps_.size();
			}
		}
		else
			assert(false);

		return weights;
	}

	double TransientForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		std::vector<double> weights = get_transient_quadrature_weights();
		auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

		utils::maybe_parallel_for(time_steps_ + 1, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int i = start; i < end; i++)
			{
				if (weights[i] == 0)
					continue;
				const double tmp = obj_->value_unweighted_step(i, x);
				local_storage.val += (weights[i] * obj_->weight()) * tmp;
			}
		});

		double value = 0;
		for (const LocalThreadScalarStorage &local_storage : storage)
			value += local_storage.val;

		return value;
	}
	Eigen::MatrixXd TransientForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), time_steps_ + 1);
		std::vector<double> weights = get_transient_quadrature_weights();

		auto storage = utils::create_thread_storage(LocalThreadMatStorage(terms.rows(), terms.cols()));

		utils::maybe_parallel_for(time_steps_ + 1, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int i = start; i < end; i++)
			{
				if (weights[i] == 0)
					continue;
				local_storage.mat.col(i) = (weights[i] * obj_->weight()) * obj_->compute_adjoint_rhs_unweighted_step(i, x, state);
				if (obj_->depends_on_step_prev() && i > 0)
					local_storage.mat.col(i - 1) = (weights[i] * obj_->weight()) * obj_->compute_adjoint_rhs_unweighted_step_prev(i, x, state);
			}
		});

		for (const LocalThreadMatStorage &local_storage : storage)
			terms += local_storage.mat;

		return terms;
	}
	void TransientForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		std::vector<double> weights = get_transient_quadrature_weights();
		auto storage = utils::create_thread_storage(LocalThreadMatStorage(gradv.size()));

		utils::maybe_parallel_for(time_steps_ + 1, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			Eigen::VectorXd tmp;
			for (int i = start; i < end; i++)
			{
				if (weights[i] == 0)
					continue;
				obj_->compute_partial_gradient_unweighted_step(i, x, tmp);
				local_storage.mat += (weights[i] * obj_->weight()) * tmp;
			}
		});

		for (const LocalThreadMatStorage &local_storage : storage)
			gradv += local_storage.mat;
	}

	void TransientForm::init(const Eigen::VectorXd &x)
	{
		obj_->init(x);
	}

	bool TransientForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->is_step_valid(x0, x1);
	}

	double TransientForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->max_step_size(x0, x1);
	}

	void TransientForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		obj_->line_search_begin(x0, x1);
	}

	void TransientForm::line_search_end()
	{
		obj_->line_search_end();
	}

	void TransientForm::post_step(const int iter_num, const Eigen::VectorXd &x)
	{
		obj_->post_step(iter_num, x);
	}

	void TransientForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		AdjointForm::solution_changed(new_x);
		obj_->solution_changed(new_x);
		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->solution_changed_step(i, new_x);
		}
	}

	void TransientForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		obj_->update_quantities(t, x);
	}

	void TransientForm::init_lagging(const Eigen::VectorXd &x)
	{
		obj_->init_lagging(x);
	}

	void TransientForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		obj_->update_lagging(x, iter_num);
	}

	void TransientForm::set_apply_DBC(const Eigen::VectorXd &x, bool apply_DBC)
	{
		obj_->set_apply_DBC(x, apply_DBC);
	}

	bool TransientForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->is_step_collision_free(x0, x1);
	}
} // namespace polyfem::solver