#include "NeoHookeanElasticity.hpp"

#include <polyfem/autogen/auto_elasticity_rhs.hpp>

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}
	} // namespace

	NeoHookeanElasticity::NeoHookeanElasticity()
	{
	}

	void NeoHookeanElasticity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		params_.add_multimaterial(index, params, size() == 3, units.stress());
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	NeoHookeanElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;

		double lambda, mu;
		// TODO!
		params_.lambda_mu(0, 0, 0, pt(0).getValue(), pt(1).getValue(), size() == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

		if (size() == 2)
			autogen::neo_hookean_2d_function(pt, lambda, mu, res);
		else if (size() == 3)
			autogen::neo_hookean_3d_function(pt, lambda, mu, res);
		else
			assert(false);

		return res;
	}

	Eigen::VectorXd
	NeoHookeanElasticity::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		Eigen::Matrix<double, Eigen::Dynamic, 1> gradient;

		if (size() == 2)
		{
			switch (data.vals.basis_values.size())
			{
			case 3:
			{
				gradient.resize(6);
				compute_energy_aux_gradient_fast<3, 2>(data, gradient);
				break;
			}
			case 6:
			{
				gradient.resize(12);
				compute_energy_aux_gradient_fast<6, 2>(data, gradient);
				break;
			}
			case 10:
			{
				gradient.resize(20);
				compute_energy_aux_gradient_fast<10, 2>(data, gradient);
				break;
			}
			default:
			{
				gradient.resize(data.vals.basis_values.size() * 2);
				compute_energy_aux_gradient_fast<Eigen::Dynamic, 2>(data, gradient);
				break;
			}
			}
		}
		else // if (size() == 3)
		{
			assert(size() == 3);
			switch (data.vals.basis_values.size())
			{
			case 4:
			{
				gradient.resize(12);
				compute_energy_aux_gradient_fast<4, 3>(data, gradient);
				break;
			}
			case 10:
			{
				gradient.resize(30);
				compute_energy_aux_gradient_fast<10, 3>(data, gradient);
				break;
			}
			case 20:
			{
				gradient.resize(60);
				compute_energy_aux_gradient_fast<20, 3>(data, gradient);
				break;
			}
			default:
			{
				gradient.resize(data.vals.basis_values.size() * 3);
				compute_energy_aux_gradient_fast<Eigen::Dynamic, 3>(data, gradient);
				break;
			}
			}
		}

		return gradient;
	}

	void NeoHookeanElasticity::compute_stiffness_value(const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const
	{
		tensor.resize(local_pts.rows(), size() * size() * size() * size());
		assert(displacement.cols() == 1);

		Eigen::MatrixXd displacement_grad(size(), size());

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			double lambda, mu;
			params_.lambda_mu(local_pts.row(p), vals.val.row(p), vals.element_id, lambda, mu);

			compute_diplacement_grad(size(), vals, local_pts, p, displacement, displacement_grad);
			const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(size(), size()) + displacement_grad;
			const Eigen::MatrixXd FmT = def_grad.inverse().transpose();
			const Eigen::VectorXd FmT_vec = utils::flatten(FmT);
			const double J = def_grad.determinant();
			const double tmp1 = mu - lambda * std::log(J);
			for (int i = 0, idx = 0; i < size(); i++)
				for (int j = 0; j < size(); j++)
					for (int k = 0; k < size(); k++)
						for (int l = 0; l < size(); l++)
						{
							tensor(p, idx) = mu * delta(i, k) * delta(j, l) + tmp1 * FmT(i, l) * FmT(k, j);
							idx++;
						}

			tensor.row(p) += lambda * utils::flatten(FmT_vec * FmT_vec.transpose());

			// {
			// 	Eigen::MatrixXd hess = utils::unflatten(tensor.row(p), size()*size());
			// 	Eigen::MatrixXd fhess;
			// 	Eigen::VectorXd x0 = utils::flatten(def_grad);
			// 	fd::finite_jacobian(
			// 		x0, [this, lambda, mu](const Eigen::VectorXd &x1) -> Eigen::VectorXd
			// 		{
			// 			Eigen::MatrixXd def_grad = utils::unflatten(x1, this->size());
			// 			const Eigen::MatrixXd FmT = def_grad.inverse().transpose();
			// 			const double J = def_grad.determinant();
			// 			Eigen::MatrixXd stress_tensor = mu * (def_grad - FmT) + lambda * std::log(J) * FmT;
			// 			return utils::flatten(stress_tensor);
			// 		}, fhess);

			// 	if (!fd::compare_hessian(hess, fhess))
			// 	{
			// 		std::cout << "Hessian: " << hess << std::endl;
			// 		std::cout << "Finite hessian: " << fhess << std::endl;
			// 		log_and_throw_error("Hessian in Neohookean mismatch");
			// 	}
			// }
		}
	}

	Eigen::MatrixXd
	NeoHookeanElasticity::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		Eigen::MatrixXd hessian;

		if (size() == 2)
		{
			switch (data.vals.basis_values.size())
			{
			case 3:
			{
				hessian.resize(6, 6);
				hessian.setZero();
				compute_energy_hessian_aux_fast<3, 2>(data, hessian);
				break;
			}
			case 6:
			{
				hessian.resize(12, 12);
				hessian.setZero();
				compute_energy_hessian_aux_fast<6, 2>(data, hessian);
				break;
			}
			case 10:
			{
				hessian.resize(20, 20);
				hessian.setZero();
				compute_energy_hessian_aux_fast<10, 2>(data, hessian);
				break;
			}
			default:
			{
				hessian.resize(data.vals.basis_values.size() * 2, data.vals.basis_values.size() * 2);
				hessian.setZero();
				compute_energy_hessian_aux_fast<Eigen::Dynamic, 2>(data, hessian);
				break;
			}
			}
		}
		else // if (size() == 3)
		{
			assert(size() == 3);
			switch (data.vals.basis_values.size())
			{
			case 4:
			{
				hessian.resize(12, 12);
				hessian.setZero();
				compute_energy_hessian_aux_fast<4, 3>(data, hessian);
				break;
			}
			case 10:
			{
				hessian.resize(30, 30);
				hessian.setZero();
				compute_energy_hessian_aux_fast<10, 3>(data, hessian);
				break;
			}
			case 20:
			{
				hessian.resize(60, 60);
				hessian.setZero();
				compute_energy_hessian_aux_fast<20, 3>(data, hessian);
				break;
			}
			default:
			{
				hessian.resize(data.vals.basis_values.size() * 3, data.vals.basis_values.size() * 3);
				hessian.setZero();
				compute_energy_hessian_aux_fast<Eigen::Dynamic, 3>(data, hessian);
				break;
			}
			}
		}

		return hessian;
	}

	void NeoHookeanElasticity::assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
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
			const double J = def_grad.determinant();
			const Eigen::MatrixXd b = def_grad * def_grad.transpose();

			double lambda, mu;
			params_.lambda_mu(local_pts.row(p), vals.val.row(p), vals.element_id, lambda, mu);

			Eigen::MatrixXd stress_tensor = (lambda * std::log(J) * I + mu * (b - I)) / J;
			if (type == ElasticityTensorType::PK1)
				stress_tensor = pk1_from_cauchy(stress_tensor, def_grad);
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = pk2_from_cauchy(stress_tensor, def_grad);

			all.row(p) = fun(stress_tensor);
		}
	}

	double NeoHookeanElasticity::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du
	template <typename T>
	T NeoHookeanElasticity::compute_energy_aux(const NonLinearAssemblerData &data) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		AutoDiffVect local_disp;
		get_local_disp(data, size(), local_disp);

		AutoDiffGradMat def_grad(size(), size());

		T energy = T(0.0);

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			compute_disp_grad_at_quad(data, local_disp, p, size(), def_grad);

			// Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += T(1);

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.vals.element_id, lambda, mu);

			const T log_det_j = log(polyfem::utils::determinant(def_grad));
			const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			energy += val * data.da(p);
		}
		return energy;
	}

	template <int dim>
	Eigen::Matrix<double, dim, dim> hat(const Eigen::Matrix<double, dim, 1> &x)
	{

		Eigen::Matrix<double, dim, dim> prod;
		prod.setZero();

		prod(0, 1) = -x(2);
		prod(0, 2) = x(1);
		prod(1, 0) = x(2);
		prod(1, 2) = -x(0);
		prod(2, 0) = -x(1);
		prod(2, 1) = x(0);

		return prod;
	}

	template <int dim>
	Eigen::Matrix<double, dim, 1> cross(const Eigen::Matrix<double, dim, 1> &x, const Eigen::Matrix<double, dim, 1> &y)
	{

		Eigen::Matrix<double, dim, 1> z;
		z.setZero();

		z(0) = x(1) * y(2) - x(2) * y(1);
		z(1) = x(2) * y(0) - x(0) * y(2);
		z(2) = x(0) * y(1) - x(1) * y(0);

		return z;
	}

	template <int n_basis, int dim>
	void NeoHookeanElasticity::compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::Matrix<double, Eigen::Dynamic, 1> &G_flattened) const
	{
		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();

		Eigen::Matrix<double, n_basis, dim> local_disp(data.vals.basis_values.size(), size());
		local_disp.setZero();
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_disp(i, d) += bs.global[ii].val * data.x(bs.global[ii].index * size() + d);
				}
			}
		}

		Eigen::Matrix<double, dim, dim> def_grad(size(), size());

		Eigen::Matrix<double, n_basis, dim> G(data.vals.basis_values.size(), size());
		G.setZero();

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];

			// Id + grad d
			def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

			const double J = def_grad.determinant();
			const double log_det_j = log(J);

			Eigen::Matrix<double, dim, dim> delJ_delF(size(), size());
			delJ_delF.setZero();

			if (dim == 2)
			{

				delJ_delF(0, 0) = def_grad(1, 1);
				delJ_delF(0, 1) = -def_grad(1, 0);
				delJ_delF(1, 0) = -def_grad(0, 1);
				delJ_delF(1, 1) = def_grad(0, 0);
			}

			else if (dim == 3)
			{

				Eigen::Matrix<double, dim, 1> u(def_grad.rows());
				Eigen::Matrix<double, dim, 1> v(def_grad.rows());
				Eigen::Matrix<double, dim, 1> w(def_grad.rows());

				u = def_grad.col(0);
				v = def_grad.col(1);
				w = def_grad.col(2);

				delJ_delF.col(0) = cross<dim>(v, w);
				delJ_delF.col(1) = cross<dim>(w, u);
				delJ_delF.col(2) = cross<dim>(u, v);
			}

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.vals.element_id, lambda, mu);

			Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

			Eigen::Matrix<double, dim, dim> gradient_temp = mu * def_grad - mu * (1 / J) * delJ_delF + lambda * log_det_j * (1 / J) * delJ_delF;
			Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();

			double val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			G.noalias() += gradient * data.da(p);
		}

		Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
		G_flattened = temp;
	}

	template <int n_basis, int dim>
	void NeoHookeanElasticity::compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const
	{
		assert(data.x.cols() == 1);

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		const int n_pts = data.da.size();

		Eigen::Matrix<double, n_basis, dim> local_disp(data.vals.basis_values.size(), size());
		local_disp.setZero();
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_disp(i, d) += bs.global[ii].val * data.x(bs.global[ii].index * size() + d);
				}
			}
		}

		Eigen::Matrix<double, dim, dim> def_grad(size(), size());

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];

			// Id + grad d
			def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

			const double J = def_grad.determinant();
			double log_det_j = log(J);

			Eigen::Matrix<double, dim, dim> delJ_delF(size(), size());
			delJ_delF.setZero();
			Eigen::Matrix<double, dim * dim, dim * dim> del2J_delF2(size() * size(), size() * size());
			del2J_delF2.setZero();

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
			else if (size() == 3)
			{
				Eigen::Matrix<double, dim, 1> u(def_grad.rows());
				Eigen::Matrix<double, dim, 1> v(def_grad.rows());
				Eigen::Matrix<double, dim, 1> w(def_grad.rows());

				u = def_grad.col(0);
				v = def_grad.col(1);
				w = def_grad.col(2);

				delJ_delF.col(0) = cross<dim>(v, w);
				delJ_delF.col(1) = cross<dim>(w, u);
				delJ_delF.col(2) = cross<dim>(u, v);

				del2J_delF2.template block<dim, dim>(0, 6) = hat<dim>(v);
				del2J_delF2.template block<dim, dim>(6, 0) = -hat<dim>(v);
				del2J_delF2.template block<dim, dim>(0, 3) = -hat<dim>(w);
				del2J_delF2.template block<dim, dim>(3, 0) = hat<dim>(w);
				del2J_delF2.template block<dim, dim>(3, 6) = -hat<dim>(u);
				del2J_delF2.template block<dim, dim>(6, 3) = hat<dim>(u);
			}

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.vals.element_id, lambda, mu);

			Eigen::Matrix<double, dim * dim, dim *dim> id = Eigen::Matrix<double, dim * dim, dim * dim>::Identity(size() * size(), size() * size());

			Eigen::Matrix<double, dim * dim, 1> g_j = Eigen::Map<const Eigen::Matrix<double, dim * dim, 1>>(delJ_delF.data(), delJ_delF.size());

			Eigen::Matrix<double, dim * dim, dim *dim> hessian_temp = (mu * id) + (((mu + lambda * (1 - log_det_j)) / (J * J)) * (g_j * g_j.transpose())) + (((lambda * log_det_j - mu) / (J)) * del2J_delF2);

			Eigen::Matrix<double, dim * dim, N> delF_delU_tensor(jac_it.size(), grad.size());

			for (size_t i = 0; i < local_disp.rows(); ++i)
			{
				for (size_t j = 0; j < local_disp.cols(); ++j)
				{
					Eigen::Matrix<double, dim, dim> temp(size(), size());
					temp.setZero();
					temp.row(j) = grad.row(i);
					temp = temp * jac_it;
					Eigen::Matrix<double, dim * dim, 1> temp_flattened(Eigen::Map<Eigen::Matrix<double, dim * dim, 1>>(temp.data(), temp.size()));
					delF_delU_tensor.col(i * size() + j) = temp_flattened;
				}
			}

			Eigen::Matrix<double, N, N> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

			double val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			H += hessian * data.da(p);
		}
	}

	void NeoHookeanElasticity::compute_stress_grad_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const
	{
		double lambda, mu;
		params_.lambda_mu(local_pts, global_pts, el_id, lambda, mu);

		Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		Eigen::MatrixXd FmT = def_grad.inverse().transpose();

		stress = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;
		result = mu * mat + FmT * mat.transpose() * FmT * (mu - lambda * std::log(def_grad.determinant())) + lambda * (FmT.array() * mat.array()).sum() * FmT;
	}

	void NeoHookeanElasticity::compute_stress_grad_multiply_stress(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const
	{
		double lambda, mu;
		params_.lambda_mu(local_pts, global_pts, el_id, lambda, mu);

		Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		Eigen::MatrixXd FmT = def_grad.inverse().transpose();

		stress = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;
		result = mu * stress + FmT * stress.transpose() * FmT * (mu - lambda * std::log(def_grad.determinant())) + lambda * (FmT.array() * stress.array()).sum() * FmT;
	}

	void NeoHookeanElasticity::compute_stress_grad_multiply_vect(
		const int el_id,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &global_pts,
		const Eigen::MatrixXd &grad_u_i,
		const Eigen::MatrixXd &vect,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		double lambda, mu;
		params_.lambda_mu(local_pts, global_pts, el_id, lambda, mu);

		Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		Eigen::MatrixXd FmT = def_grad.inverse().transpose();

		stress = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;
		result.setZero(size() * size(), size());
		if (vect.rows() == 1)
			for (int i = 0; i < size(); ++i)
				for (int j = 0; j < size(); ++j)
					for (int l = 0; l < size(); ++l)
					{
						result(i * size() + j, l) += mu * vect(i) * ((j == l) ? 1 : 0);
						// For some reason, the second gives lower error. From the formula, though, it should be the following.
						// result(i * size() + j, l) += (mu - lambda * std::log(def_grad.determinant())) * FmT(j, l) * (FmT.col(i).array() * vect.transpose().array()).sum();
						result(i * size() + j, l) += (mu - lambda * std::log(def_grad.determinant())) * FmT(i, l) * (FmT.col(j).array() * vect.transpose().array()).sum();
						result(i * size() + j, l) += lambda * FmT(i, j) * (FmT.col(l).array() * vect.transpose().array()).sum();
					}
		else
			for (int i = 0; i < size(); ++i)
				for (int j = 0; j < size(); ++j)
					for (int k = 0; k < size(); ++k)
					{
						result(i * size() + j, k) += mu * vect(j) * ((i == k) ? 1 : 0);
						// For some reason, the second gives lower error. From the formula, though, it should be the following.
						// result(i * size() + j, k) += (mu - lambda * std::log(def_grad.determinant())) * FmT(k, j) * (FmT.row(i).array() * vect.transpose().array()).sum();
						result(i * size() + j, k) += (mu - lambda * std::log(def_grad.determinant())) * FmT(k, i) * (FmT.row(j).array() * vect.transpose().array()).sum();
						result(i * size() + j, k) += lambda * FmT(i, j) * (FmT.row(k).array() * vect.transpose().array()).sum();
					}
	}

	void NeoHookeanElasticity::compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const
	{
		Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		Eigen::MatrixXd FmT = def_grad.inverse().transpose();
		dstress_dmu = def_grad - FmT;
		dstress_dlambda = std::log(def_grad.determinant()) * FmT;
	}

	std::map<std::string, Assembler::ParamFunc> NeoHookeanElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &params = params_;
		const int size = this->size();

		res["lambda"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, e, lambda, mu);
			return lambda;
		};

		res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, e, lambda, mu);
			return mu;
		};

		res["E"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, e, lambda, mu);

			if (size == 3)
				return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
			else
				return 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
		};

		res["nu"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, e, lambda, mu);

			if (size == 3)
				return lambda / (2.0 * (lambda + mu));
			else
				return lambda / (lambda + 2.0 * mu);
		};

		return res;
	}

} // namespace polyfem::assembler
