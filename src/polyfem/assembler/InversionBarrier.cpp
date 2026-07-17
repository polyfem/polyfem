#include "InversionBarrier.hpp"

#include <polyfem/utils/ElasticityUtils.hpp>

#include <cmath>

namespace polyfem::assembler
{
	namespace
	{
		// Zero for J >= JBarrierThreshold, smoothly diverges to +inf as J -> 0.
		// Scaled by mu at the call sites.
		template <class T, int p = 3>
		class barrier
		{
			constexpr static double C = 1e2;

		public:
			static_assert(p % 2 == 1);
			static T value(T J, double JBarrierThreshold)
			{
				if (J >= JBarrierThreshold)
					return T(0.);
				const T tmp1 = J / JBarrierThreshold - 1;
				const T tmp2 = pow(tmp1, p);
				return C * (1 / (tmp2 + 1) - 1);
			}

			static T first_derivatives(T J, double JBarrierThreshold)
			{
				if (J >= JBarrierThreshold)
					return T(0);
				const T tmp1 = J / JBarrierThreshold - 1;
				const T tmp2 = pow(tmp1, p);
				return C * (-1 / JBarrierThreshold) * p * tmp2 / tmp1 / pow(1 + tmp2, 2);
			}

			static T second_derivatives(T J, double JBarrierThreshold)
			{
				if (J >= JBarrierThreshold)
					return T(0);
				const T tmp1 = J / JBarrierThreshold - 1;
				const T tmp2 = pow(tmp1, p);
				return C * (1 / (JBarrierThreshold * JBarrierThreshold)) * p * tmp2 / pow(tmp1, 2) * ((1 - p) + (1 + p) * tmp2) / pow(1 + tmp2, 3);
			}
		};

		// Cofactor matrix dJ/dF, and its derivative d2J/dF2 (flattened column-major).
		void jacobian_cofactor_derivatives(const Eigen::MatrixXd &def_grad, const int dim,
										   Eigen::MatrixXd &delJ_delF, Eigen::MatrixXd &del2J_delF2)
		{
			delJ_delF.setZero(dim, dim);
			del2J_delF2.setZero(dim * dim, dim * dim);

			if (dim == 2)
			{
				delJ_delF(0, 0) = def_grad(1, 1);
				delJ_delF(0, 1) = -def_grad(1, 0);
				delJ_delF(1, 0) = -def_grad(0, 1);
				delJ_delF(1, 1) = def_grad(0, 0);

				del2J_delF2(0, 3) = 1;
				del2J_delF2(1, 2) = -1;
				del2J_delF2(2, 1) = -1;
				del2J_delF2(3, 0) = 1;
			}
			else // dim == 3
			{
				const Eigen::Vector3d u = def_grad.col(0), v = def_grad.col(1), w = def_grad.col(2);
				delJ_delF.col(0) = v.cross(w);
				delJ_delF.col(1) = w.cross(u);
				delJ_delF.col(2) = u.cross(v);

				auto hat = [](const Eigen::Vector3d &x) {
					Eigen::Matrix3d prod;
					prod << 0, -x(2), x(1),
						x(2), 0, -x(0),
						-x(1), x(0), 0;
					return prod;
				};

				del2J_delF2.block<3, 3>(0, 6) = hat(v);
				del2J_delF2.block<3, 3>(6, 0) = -hat(v);
				del2J_delF2.block<3, 3>(0, 3) = -hat(w);
				del2J_delF2.block<3, 3>(3, 0) = hat(w);
				del2J_delF2.block<3, 3>(3, 6) = -hat(u);
				del2J_delF2.block<3, 3>(6, 3) = hat(u);
			}
		}
	} // namespace

	InversionBarrier::InversionBarrier()
		: JBarrierThreshold_("JBarrierThreshold")
	{
	}

	void InversionBarrier::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		params_.add_multimaterial(index, params, size() == 3, units.stress(), root_path);

		// Default activation threshold matches the original hardcoded barrier (J >= 0.5 is inactive).
		if (params.count("JBarrierThreshold"))
		{
			JBarrierThreshold_.add_multimaterial(index, params, "", root_path);
		}
		else
		{
			json params_with_default = params;
			params_with_default["JBarrierThreshold"] = 0.5;
			JBarrierThreshold_.add_multimaterial(index, params_with_default, "", root_path);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	InversionBarrier::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;
		assert(false);

		return res;
	}

	void InversionBarrier::assign_stress_tensor(
		const OutputData &data,
		const int all_size,
		const ElasticityTensorType &type,
		Eigen::MatrixXd &all,
		const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		const auto &displacement = data.fun;
		const auto &local_pts = data.local_pts;
		const auto &bs = data.bs;
		const auto &gbs = data.gbs;
		const auto el_id = data.el_id;
		const auto t = data.t;

		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);
		const auto I = Eigen::MatrixXd::Identity(size(), size());

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			const Eigen::MatrixXd def_grad = I + displacement_grad;
			if (type == ElasticityTensorType::F)
			{
				all.row(p) = fun(def_grad);
				continue;
			}

			const double J = polyfem::utils::determinant(def_grad);
			const double JBarrierThreshold = JBarrierThreshold_(local_pts.row(p), t, vals.element_id);

			Eigen::MatrixXd delJ_delF, del2J_delF2;
			jacobian_cofactor_derivatives(def_grad, size(), delJ_delF, del2J_delF2);

			double lambda, mu;
			params_.lambda_mu(local_pts.row(p), vals.val.row(p), t, vals.element_id, lambda, mu);

			// PK1-like stress dW/dF; convert to the requested tensor type via def_grad.
			Eigen::MatrixXd stress_tensor = mu * barrier<double>::first_derivatives(J, JBarrierThreshold) * delJ_delF;
			if (type == ElasticityTensorType::CAUCHY)
				stress_tensor = (stress_tensor * def_grad.transpose()) / J;
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = def_grad.inverse() * stress_tensor;
			// else: PK1, already in the right form.

			all.row(p) = fun(stress_tensor);
		}
	}

	double InversionBarrier::compute_energy(const NonLinearAssemblerData &data) const
	{
		const int dim = size();
		const int n_basis = data.vals.basis_values.size();

		Eigen::VectorXd jacs;
		if (use_robust_jacobian)
			jacs = data.vals.eval_deformed_jacobian_determinant(data.x);

		Eigen::MatrixXd local_disp(n_basis, dim);
		local_disp.setZero();
		for (int i = 0; i < n_basis; ++i)
			for (const auto &g : data.vals.basis_values[i].global)
				for (int d = 0; d < dim; ++d)
					local_disp(i, d) += g.val * data.x(g.index * dim + d);

		double energy = 0.;
		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(n_basis, dim);
			for (int i = 0; i < n_basis; ++i)
				grad.row(i) = data.vals.basis_values[i].grad.row(p);

			const Eigen::MatrixXd jac_it = data.vals.jac_it[p];
			const Eigen::MatrixXd def_grad = (local_disp.transpose() * grad) * jac_it + Eigen::MatrixXd::Identity(dim, dim);

			const double J = use_robust_jacobian ? jacs(p) * jac_it.determinant() : def_grad.determinant();
			const double JBarrierThreshold = JBarrierThreshold_(data.vals.quadrature.points.row(p), data.t, data.vals.element_id);

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.t, data.vals.element_id, lambda, mu);

			energy += mu * barrier<double>::value(J, JBarrierThreshold) * data.da(p);
		}
		return energy;
	}

	Eigen::VectorXd InversionBarrier::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		const int dim = size();
		const int n_basis = data.vals.basis_values.size();

		Eigen::VectorXd jacs;
		if (use_robust_jacobian)
			jacs = data.vals.eval_deformed_jacobian_determinant(data.x);

		Eigen::MatrixXd local_disp(n_basis, dim);
		local_disp.setZero();
		for (int i = 0; i < n_basis; ++i)
			for (const auto &g : data.vals.basis_values[i].global)
				for (int d = 0; d < dim; ++d)
					local_disp(i, d) += g.val * data.x(g.index * dim + d);

		Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n_basis, dim);

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(n_basis, dim);
			for (int i = 0; i < n_basis; ++i)
				grad.row(i) = data.vals.basis_values[i].grad.row(p);

			const Eigen::MatrixXd jac_it = data.vals.jac_it[p];
			const Eigen::MatrixXd delF_delU = grad * jac_it;

			const Eigen::MatrixXd def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(dim, dim);

			const double J = use_robust_jacobian ? jacs(p) * jac_it.determinant() : def_grad.determinant();
			const double JBarrierThreshold = JBarrierThreshold_(data.vals.quadrature.points.row(p), data.t, data.vals.element_id);

			Eigen::MatrixXd delJ_delF, del2J_delF2;
			jacobian_cofactor_derivatives(def_grad, dim, delJ_delF, del2J_delF2);

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.t, data.vals.element_id, lambda, mu);

			const Eigen::MatrixXd gradient_temp = mu * barrier<double>::first_derivatives(J, JBarrierThreshold) * delJ_delF;
			const Eigen::MatrixXd gradient = delF_delU * gradient_temp.transpose();

			G.noalias() += gradient * data.da(p);
		}

		const Eigen::MatrixXd G_T = G.transpose();
		return Eigen::Map<const Eigen::VectorXd>(G_T.data(), G_T.size());
	}

	Eigen::MatrixXd InversionBarrier::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const int dim = size();
		const int n_basis = data.vals.basis_values.size();
		const int N = n_basis * dim;

		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);

		Eigen::VectorXd jacs;
		if (use_robust_jacobian)
			jacs = data.vals.eval_deformed_jacobian_determinant(data.x);

		Eigen::MatrixXd local_disp(n_basis, dim);
		local_disp.setZero();
		for (int i = 0; i < n_basis; ++i)
			for (const auto &g : data.vals.basis_values[i].global)
				for (int d = 0; d < dim; ++d)
					local_disp(i, d) += g.val * data.x(g.index * dim + d);

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(n_basis, dim);
			for (int i = 0; i < n_basis; ++i)
				grad.row(i) = data.vals.basis_values[i].grad.row(p);

			const Eigen::MatrixXd jac_it = data.vals.jac_it[p];
			const Eigen::MatrixXd delF_delU = grad * jac_it;

			const Eigen::MatrixXd def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(dim, dim);

			const double J = use_robust_jacobian ? jacs(p) * jac_it.determinant() : def_grad.determinant();
			const double JBarrierThreshold = JBarrierThreshold_(data.vals.quadrature.points.row(p), data.t, data.vals.element_id);

			Eigen::MatrixXd delJ_delF, del2J_delF2;
			jacobian_cofactor_derivatives(def_grad, dim, delJ_delF, del2J_delF2);

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.t, data.vals.element_id, lambda, mu);

			const Eigen::Map<const Eigen::VectorXd> g_j(delJ_delF.data(), delJ_delF.size());

			const Eigen::MatrixXd hessian_temp = mu * barrier<double>::first_derivatives(J, JBarrierThreshold) * del2J_delF2
												 + mu * barrier<double>::second_derivatives(J, JBarrierThreshold) * (g_j * g_j.transpose());

			Eigen::MatrixXd delF_delU_tensor = Eigen::MatrixXd::Zero(dim * dim, N);
			for (int i = 0; i < n_basis; ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(dim, dim);
					temp.row(j) = delF_delU.row(i);
					delF_delU_tensor.col(i * dim + j) = Eigen::Map<const Eigen::VectorXd>(temp.data(), temp.size());
				}
			}

			H += (delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor) * data.da(p);
		}
		return H;
	}

	std::map<std::string, Assembler::ParamFunc> InversionBarrier::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &params = params_;
		const auto &JBarrierThreshold = JBarrierThreshold_;

		res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);
			return mu;
		};

		res["JBarrierThreshold"] = [&JBarrierThreshold](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			return JBarrierThreshold(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
