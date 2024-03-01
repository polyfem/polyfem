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
		/// @brief Parameters for the functional evaluation
		struct ParameterType
		{
			double t = 0.;
			int step = 0;
			int elem = -1;

			int node = -1;
			int body_id = -1;
			int boundary_id = -1;
		};

		typedef std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::MatrixXd &, const assembler::ElementAssemblyValues &, const ParameterType &, Eigen::MatrixXd &)> functionalType;

		IntegrableFunctional() = default;

		void set_j(const functionalType &j_);
		void set_dj_dx(const functionalType &dj_dx_);
		void set_dj_du(const functionalType &dj_du_);
		void set_dj_dgradu(const functionalType &dj_dgradu_);
		void set_dj_dgradu_local(const functionalType &dj_dgradu_local_);
		void set_dj_dgradx(const functionalType &dj_dgradx_);

		void evaluate(const Eigen::MatrixXd& elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const;
		void dj_dx(const Eigen::MatrixXd& elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const;
		void dj_du(const Eigen::MatrixXd& elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const;
		void dj_dgradu(const Eigen::MatrixXd& elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const;
		void dj_dgradu_local(const Eigen::MatrixXd& elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const;
		void dj_dgradx(const Eigen::MatrixXd& elastic_params, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, ParameterType &params, Eigen::MatrixXd &val) const;

		inline bool depend_on_x() const { return has_x; }
		inline bool depend_on_u() const { return has_u; }
		inline bool depend_on_gradu() const { return has_gradu; }
		inline bool depend_on_gradu_local() const { return has_gradu_local; }
		inline bool depend_on_gradx() const { return has_gradx; }

	private:
		bool has_x = false, has_u = false, has_gradu = false, has_gradu_local = false, has_gradx = false;
		functionalType j_func, dj_dx_func, dj_du_func, dj_dgradu_func, dj_dgradu_local_func, dj_dgradx_func;
	};
} // namespace polyfem
