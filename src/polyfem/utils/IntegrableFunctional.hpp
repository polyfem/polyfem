#pragma once

#include <functional>
#include <Eigen/Dense>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/assembler/Assembler.hpp>

using namespace polyfem::assembler;

namespace polyfem
{
	class IntegrableFunctional
	{
	public:
		typedef std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const json &, Eigen::MatrixXd &)> functionalType;

		IntegrableFunctional() = default;

		void set_name(const std::string &name_) { name = name_; }
		std::string get_name() const { return name; }

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

		void evaluate(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(j_func);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			j_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
		}

		void dj_dx(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_x);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_dx_func)
			{
				dj_dx_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_du(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_u);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_du_func)
			{
				dj_du_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_dgradu(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_gradu);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_dgradu_func)
			{
				dj_dgradu_func(local_pts, pts, u, grad_u, lambda, mu, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		Eigen::MatrixXd grad_j(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params) const
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

		bool depend_on_x() const { return has_x; }
		bool depend_on_u() const { return has_u; }
		bool depend_on_gradu() const { return has_gradu; }

	private:
		void lambda_mu(const std::map<std::string, Assembler::ParamFunc> &lame_params, const int e, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, Eigen::MatrixXd &lambda, Eigen::MatrixXd &mu) const
		{
			const double t = 0;
			lambda.setZero(local_pts.rows(), 1);
			mu.setZero(local_pts.rows(), 1);
			
			auto search_lambda = lame_params.find("lambda");
			auto search_mu = lame_params.find("mu");

			if (search_lambda == lame_params.end() || search_mu == lame_params.end())
				return;

			for (int p = 0; p < local_pts.rows(); p++)
			{
				lambda(p) = search_lambda->second(local_pts.row(p), pts.row(p), t, e);
				mu(p) = search_mu->second(local_pts.row(p), pts.row(p), t, e);
			}
		}

		std::string name = "";
		bool has_x = false, has_u = false, has_gradu = false;
		functionalType j_func, dj_dx_func, dj_du_func, dj_dgradu_func;
	};
} // namespace polyfem
