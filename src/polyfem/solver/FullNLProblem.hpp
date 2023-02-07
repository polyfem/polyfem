#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <cppoptlib/problem.h>

#include <memory>
#include <vector>

namespace polyfem::solver
{
	class FullNLProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;
		typedef StiffnessMatrix THessian;

		FullNLProblem(const std::vector<std::shared_ptr<Form>> &forms);
		virtual void init(const TVector &x0);

		virtual double value(const TVector &x) override;
		virtual void gradient(const TVector &x, TVector &gradv) override;
		virtual void hessian(const TVector &x, THessian &hessian);

		TVector component_values(const TVector &x);
		Eigen::MatrixXd component_gradients(const TVector &x);
		bool verify_gradient(const TVector &x, const TVector &gradv) { return true; }

		virtual bool is_step_valid(const TVector &x0, const TVector &x1) const;
		virtual bool is_step_collision_free(const TVector &x0, const TVector &x1) const;
		virtual double max_step_size(const TVector &x0, const TVector &x1) const;

		virtual void line_search_begin(const TVector &x0, const TVector &x1);
		virtual void line_search_end();
		virtual void post_step(const int iter_num, const TVector &x);

		virtual void set_project_to_psd(bool val);

		virtual void solution_changed(const TVector &new_x);

		virtual void init_lagging(const TVector &x);
		virtual void update_lagging(const TVector &x, const int iter_num);
		int max_lagging_iterations() const;
		bool uses_lagging() const;

		int n_inequality_constraints() { return 0; }
		double inequality_constraint_val(const TVector &x, const int index)
		{
			assert(false);
			return std::nan("");
		}
		TVector inequality_constraint_grad(const TVector &x, const int index)
		{
			assert(false);
			return TVector();
		}

		bool remesh(TVector &x) { return false; }
		bool smoothing(const TVector &x, const TVector &new_x, TVector &smoothed_x) { return false; }
		void save_to_file(const TVector &x0) {}
		std::vector<std::shared_ptr<Form>> &forms() { return forms_; }

	protected:
		std::vector<std::shared_ptr<Form>> forms_;
	};
} // namespace polyfem::solver
