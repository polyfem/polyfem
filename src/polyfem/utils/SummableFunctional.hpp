#pragma once

#include <functional>
#include <Eigen/Dense>
#include <polyfem/utils/JSONUtils.hpp>

namespace polyfem
{
	class SummableFunctional
	{
	public:
		typedef std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const json &, Eigen::MatrixXd &)> functionalType;

		SummableFunctional() {}

		void set_type(bool depend_on_x, bool depend_on_u)
		{
			has_x = depend_on_x;
			has_u = depend_on_u;
		}

		void set_j(const functionalType &j_) { j_func = j_; }
		void set_dj_dx(const functionalType &dj_dx_)
		{
			dj_dx_func = dj_dx_;
			has_x = true;
		}
		void set_dj_du(const functionalType &dj_du_)
		{
			dj_du_func = dj_du_;
			has_u = true;
		}

		void evaluate(const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const json &params, Eigen::MatrixXd &val) const
		{
			assert(j_func);
			j_func(pts, u, params, val);
		}

		void dj_dx(const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const json &params, Eigen::MatrixXd &val) const
		{
			assert(has_x);
			if (dj_dx_func)
			{
				dj_dx_func(pts, u, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_du(const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const json &params, Eigen::MatrixXd &val) const
		{
			assert(has_u);
			if (dj_du_func)
			{
				dj_du_func(pts, u, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		bool depend_on_x() const { return has_x; }
		bool depend_on_u() const { return has_u; }

	private:
		bool has_x = false, has_u = false;
		functionalType j_func, dj_dx_func, dj_du_func;
	};
} // namespace polyfem
