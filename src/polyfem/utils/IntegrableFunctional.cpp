#include "IntegrableFunctional.hpp"

namespace polyfem
{
	void IntegrableFunctional::set_j(const functionalType &j)
	{
		j_func_ = j;
	}
	void IntegrableFunctional::set_dj_dx(const functionalType &dj_dx)
	{
		dj_dx_func_ = dj_dx;
		has_x = true;
	}
	void IntegrableFunctional::set_dj_du(const functionalType &dj_du)
	{
		dj_du_func_ = dj_du;
		assert(!has_gradu && !has_gradu_local);
		has_u = true;
	}
	void IntegrableFunctional::set_dj_dgradu(const functionalType &dj_dgradu)
	{
		dj_dgradu_func_ = dj_dgradu;
		assert(!has_u && !has_gradu_local);
		has_gradu = true;
	}
	void IntegrableFunctional::set_dj_dgradu_local(const functionalType &dj_dgradu_local)
	{
		dj_dgradu_local_func_ = dj_dgradu_local;
		assert(!has_u && !has_gradu);
		has_gradu_local = true;
	}
	void IntegrableFunctional::set_dj_dgradx(const functionalType &dj_dgradx)
	{
		dj_dgradx_func_ = dj_dgradx;
		has_gradx = true;
	}

	void IntegrableFunctional::evaluate(const Eigen::MatrixXd &elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const
	{
		assert(j_func_);
		j_func_(local_pts, pts, u, grad_u, elastic_params.col(0), elastic_params.col(1), reference_normals, vals, params, val);
	}

	void IntegrableFunctional::dj_dx(const Eigen::MatrixXd &elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const
	{
		assert(has_x);
		dj_dx_func_(local_pts, pts, u, grad_u, elastic_params.col(0), elastic_params.col(1), reference_normals, vals, params, val);
	}

	void IntegrableFunctional::dj_du(const Eigen::MatrixXd &elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const
	{
		assert(has_u);
		dj_du_func_(local_pts, pts, u, grad_u, elastic_params.col(0), elastic_params.col(1), reference_normals, vals, params, val);
	}

	void IntegrableFunctional::dj_dgradu(const Eigen::MatrixXd &elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const
	{
		assert(has_gradu);
		dj_dgradu_func_(local_pts, pts, u, grad_u, elastic_params.col(0), elastic_params.col(1), reference_normals, vals, params, val);
	}

	void IntegrableFunctional::dj_dgradu_local(const Eigen::MatrixXd &elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const
	{
		assert(has_gradu_local);
		dj_dgradu_local_func_(local_pts, pts, u, grad_u, elastic_params.col(0), elastic_params.col(1), reference_normals, vals, params, val);
	}

	void IntegrableFunctional::dj_dgradx(const Eigen::MatrixXd &elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const
	{
		assert(has_gradx);
		dj_dgradx_func_(local_pts, pts, u, grad_u, elastic_params.col(0), elastic_params.col(1), reference_normals, vals, params, val);
	}
} // namespace polyfem
