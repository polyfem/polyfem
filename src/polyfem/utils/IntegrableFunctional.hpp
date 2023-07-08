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
		typedef std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const assembler::ElementAssemblyValues &, const json &, Eigen::MatrixXd &)> functionalType;

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
		void set_dj_dgradu_local(const functionalType &dj_dgradu_local_)
		{
			dj_dgradu_local_func = dj_dgradu_local_;
			has_gradu_local = true;
		}
		void set_dj_dgradx(const functionalType &dj_dgradx_)
		{
			dj_dgradx_func = dj_dgradx_;
			has_gradx = true;
		}

		void evaluate(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, json &params, Eigen::MatrixXd &val) const
		{
			assert(j_func);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			j_func(local_pts, pts, u, grad_u, lambda, mu, reference_normals, vals, params, val);
		}

		void dj_dx(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_x);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_dx_func)
			{
				dj_dx_func(local_pts, pts, u, grad_u, lambda, mu, reference_normals, vals, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_du(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_u);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_du_func)
			{
				dj_du_func(local_pts, pts, u, grad_u, lambda, mu, reference_normals, vals, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_dgradu(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_gradu);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_dgradu_func)
			{
				dj_dgradu_func(local_pts, pts, u, grad_u, lambda, mu, reference_normals, vals, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_dgradu_local(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_gradu_local);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_dgradu_local_func)
			{
				dj_dgradu_local_func(local_pts, pts, u, grad_u, lambda, mu, reference_normals, vals, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		void dj_dgradx(const std::map<std::string, Assembler::ParamFunc> &lame_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, json &params, Eigen::MatrixXd &val) const
		{
			assert(has_gradx);
			Eigen::MatrixXd lambda, mu;
			lambda_mu(lame_params, params["elem"], local_pts, pts, lambda, mu);
			if (dj_dgradx_func)
			{
				dj_dgradx_func(local_pts, pts, u, grad_u, lambda, mu, reference_normals, vals, params, val);
			}
			else
			{
				// TODO: Use AutoDiff
			}
		}

		bool depend_on_x() const { return has_x; }
		bool depend_on_u() const { return has_u; }
		bool depend_on_gradu() const { return has_gradu; }
		bool depend_on_gradu_local() const { return has_gradu_local; }
		bool depend_on_gradx() const { return has_gradx; }

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
		bool has_x = false, has_u = false, has_gradu = false, has_gradu_local = false, has_gradx = false;
		functionalType j_func, dj_dx_func, dj_du_func, dj_dgradu_func, dj_dgradu_local_func, dj_dgradx_func;
	};
} // namespace polyfem
