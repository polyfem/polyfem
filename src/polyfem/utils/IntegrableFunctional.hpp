#pragma once

#include <functional>
#include <Eigen/Dense>
#include <polyfem/utils/JSONUtils.hpp>

namespace polyfem
{
	class IntegrableFunctional
	{
	public:
		typedef std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const json &, Eigen::MatrixXd &)> functionalType;

		IntegrableFunctional(bool is_surface_integral = false) { is_surface_integral_ = is_surface_integral; }

		void set_type(bool depend_on_x, bool depend_on_u, bool depend_on_gradu)
		{
			has_x = depend_on_x;
			has_u = depend_on_u;
			has_gradu = depend_on_gradu;
		}

		void set_name(const std::string &name_) { name = name_; }
		std::string get_name() const { return name; }

		void set_surface_integral() { is_surface_integral_ = true; }
		void set_volume_integral() { is_surface_integral_ = false; }
		void set_transient_integral_type(const std::string &integral_type) { transient_integral_type = integral_type; }

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
		void set_dj_dgradu(const functionalType &dj_dgradu_)
		{
			dj_dgradu_func = dj_dgradu_;
			has_gradu = true;
		}

		void evaluate(const LameParameters &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(j_func);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			params["density"] = lame_params.density(params["elem"]);
			j_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
		}

		void dj_dx(const LameParameters &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_x);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			params["density"] = lame_params.density(params["elem"]);
			if (dj_dx_func)
			{
				dj_dx_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_du(const LameParameters &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_u);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			params["density"] = lame_params.density(params["elem"]);
			if (dj_du_func)
			{
				dj_du_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_dgradu(const LameParameters &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_gradu);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			params["density"] = lame_params.density(params["elem"]);
			if (dj_dgradu_func)
			{
				dj_dgradu_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		Eigen::MatrixXd grad_j(const LameParameters &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params) const
		{
			Eigen::MatrixXd val;
			if (depend_on_gradu())
				dj_dgradu(lame_params, local_pts, pts, u, grad_u, params, val);
			else if (depend_on_u())
				dj_du(lame_params, local_pts, pts, u, grad_u, params, val);
			else
				assert(false);
			return val;
		}

		void get_transient_quadrature_weights(const int n, const double dt, std::vector<double> &weights) const
		{
			weights.assign(n + 1, dt);
			if (transient_integral_type == "uniform")
			{
				weights[0] = 0;
			}
			else if (transient_integral_type == "trapezoidal")
			{
				weights[0] = dt / 2.;
				weights[weights.size() - 1] = dt / 2.;
			}
			else if (transient_integral_type == "simpson")
			{
				weights[0] = dt / 3.;
				weights[weights.size() - 1] = dt / 3.;
				for (int i = 1; i < weights.size() - 1; i++)
				{
					if (i % 2)
						weights[i] = dt * 4. / 3.;
					else
						weights[i] = dt * 2. / 4.;
				}
			}
			else if (transient_integral_type == "final")
			{
				weights.assign(n + 1, 0);
				weights[n] = 1;
			}
			else
				assert(false);
		}

		bool depend_on_x() const { return has_x; }
		bool depend_on_u() const { return has_u; }
		bool depend_on_gradu() const { return has_gradu; }

		bool is_volume_integral() const { return !is_surface_integral_; }
		bool is_surface_integral() const { return is_surface_integral_; }

		void lambda_mu(const LameParameters &lame_params, const int e, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, Eigen::MatrixXd &lambda, Eigen::MatrixXd &mu) const
		{
			lambda.setZero(local_pts.rows(), 1);
			mu.setZero(local_pts.rows(), 1);
			for (int p = 0; p < local_pts.rows(); p++)
				lame_params.lambda_mu(local_pts.row(p), pts.row(p), e, lambda(p), mu(p));
		}

	private:
		std::string name = "";
		bool is_surface_integral_;
		std::string transient_integral_type = "simpson";
		bool has_x = false, has_u = false, has_gradu = false;
		functionalType j_func, dj_dx_func, dj_du_func, dj_dgradu_func;
	};
} // namespace polyfem
