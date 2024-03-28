#include "MooneyRivlin3ParamSymbolic.hpp"

#include <polyfem/autogen/auto_mooney_rivlin_gradient_hessian.hpp>

namespace polyfem::assembler
{

	MooneyRivlin3ParamSymbolic::MooneyRivlin3ParamSymbolic()
		: c1_("c1"), c2_("c2"), c3_("c3"), d1_("d1")
	{
	}

	Eigen::VectorXd
	MooneyRivlin3ParamSymbolic::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		Eigen::Matrix<double, Eigen::Dynamic, 1> gradient;

		if (codomain_size() == 2)
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
		else // if (codomain_size() == 3)
		{
			assert(codomain_size() == 3);
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

	Eigen::MatrixXd
	MooneyRivlin3ParamSymbolic::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		Eigen::MatrixXd hessian;

		if (codomain_size() == 2)
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
		else // if (codomain_size() == 3)
		{
			assert(codomain_size() == 3);
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

	void MooneyRivlin3ParamSymbolic::assign_stress_tensor(const OutputData &data,
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

		MatrixNd displacement_grad(domain_size(), codomain_size());
		const auto I = MatrixNd::Identity(codomain_size(), codomain_size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

		ElementAssemblyValues vals;
		vals.compute(el_id, codomain_size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_displacement_grad(domain_size(), codomain_size(), bs, vals, local_pts, p, displacement, displacement_grad);

			const Eigen::MatrixXd def_grad = I + displacement_grad;
			const Eigen::MatrixXd def_grad_T = def_grad.transpose();
			if (type == ElasticityTensorType::F)
			{
				all.row(p) = fun(def_grad);
				continue;
			}

			const double t = 0;
			const double c1 = c1_(local_pts.row(p), t, vals.element_id);
			const double c2 = c2_(local_pts.row(p), t, vals.element_id);
			const double c3 = c3_(local_pts.row(p), t, vals.element_id);
			const double d1 = d1_(local_pts.row(p), t, vals.element_id);

			Eigen::MatrixXd stress_tensor;
			autogen::generate_gradient(c1, c2, c3, d1, def_grad_T, stress_tensor);

			stress_tensor = 1.0 / def_grad.determinant() * stress_tensor * def_grad.transpose();
			if (type == ElasticityTensorType::PK1)
				stress_tensor = pk1_from_cauchy(stress_tensor, def_grad);
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = pk2_from_cauchy(stress_tensor, def_grad);

			all.row(p) = fun(stress_tensor);
		}
	}

	void MooneyRivlin3ParamSymbolic::compute_stress_grad_multiply_mat(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &mat,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < codomain_size(); ++d)
			F(d, d) += 1.;
		Eigen::MatrixXd F_T = F.transpose();

		const double c1 = c1_(global_pts, t, el_id);
		const double c2 = c2_(global_pts, t, el_id);
		const double c3 = c3_(global_pts, t, el_id);
		const double d1 = d1_(global_pts, t, el_id);

		Eigen::MatrixXd grad, hess;
		get_grad_hess_symbolic(c1, c2, c3, d1, F, grad, hess);
		// get_grad_hess_autodiff(c1, c2, c3, d1, global_pts, el_id, F, grad, hess);

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		// Compute ∂S_ij/∂F_kl * M_kl, same as M_ij * ∂S_ij/∂F_kl since the hessian is symmetric
		result = (hess * mat.reshaped(codomain_size() * codomain_size(), 1)).reshaped(codomain_size(), codomain_size());
	}

	void MooneyRivlin3ParamSymbolic::compute_stress_grad_multiply_stress(
		const OptAssemblerData &data,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < codomain_size(); ++d)
			F(d, d) += 1.;

		const double c1 = c1_(global_pts, t, el_id);
		const double c2 = c2_(global_pts, t, el_id);
		const double c3 = c3_(global_pts, t, el_id);
		const double d1 = d1_(global_pts, t, el_id);

		Eigen::MatrixXd grad, hess;
		get_grad_hess_symbolic(c1, c2, c3, d1, F, grad, hess);
		// get_grad_hess_autodiff(c1, c2, c3, d1, global_pts, el_id, F, grad, hess);

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		// Compute ∂S_ij/∂F_kl * S_kl, same as S_ij * ∂S_ij/∂F_kl since the hessian is symmetric
		result = (hess * stress.reshaped(codomain_size() * codomain_size(), 1)).reshaped(codomain_size(), codomain_size());
	}

	void MooneyRivlin3ParamSymbolic::compute_stress_grad_multiply_vect(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &vect,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{

		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < codomain_size(); ++d)
			F(d, d) += 1.;
		Eigen::MatrixXd F_T = F.transpose();

		const double c1 = c1_(global_pts, t, el_id);
		const double c2 = c2_(global_pts, t, el_id);
		const double c3 = c3_(global_pts, t, el_id);
		const double d1 = d1_(global_pts, t, el_id);

		Eigen::MatrixXd grad, hess;
		get_grad_hess_symbolic(c1, c2, c3, d1, F, grad, hess);
		// get_grad_hess_autodiff(c1, c2, c3, d1, global_pts, el_id, F, grad, hess);

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		result.resize(hess.rows(), vect.size());
		for (int i = 0; i < hess.rows(); ++i)
			if (vect.rows() == 1)
				// Compute ∂S_ij/∂F_kl * v_k, same as ∂S_ij/∂F_kl * v_i since the hessian is symmetric
				result.row(i) = vect * hess.row(i).reshaped(codomain_size(), codomain_size());
			else
				// Compute ∂S_ij/∂F_kl * v_l, same as ∂S_ij/∂F_kl * v_j since the hessian is symmetric
				result.row(i) = hess.row(i).reshaped(codomain_size(), codomain_size()) * vect;
	}

	template <typename T>
	T MooneyRivlin3ParamSymbolic::elastic_energy(
		const RowVectorNd &p,
		const int el_id,
		const AutoDiffGradMat<T> &def_grad) const
	{
		const double t = 0; // TODO

		const double c1 = c1_(p, t, el_id);
		const double c2 = c2_(p, t, el_id);
		const double c3 = c3_(p, t, el_id);
		const double d1 = d1_(p, t, el_id);

		const T J = polyfem::utils::determinant(def_grad);
		const T log_J = log(J);
		const auto F_tilde = def_grad;
		const auto right_cauchy_green = F_tilde * F_tilde.transpose();

		const auto I1_tilde = pow(J, -2. / codomain_size()) * first_invariant(right_cauchy_green);
		const auto I2_tilde = pow(J, -4. / codomain_size()) * second_invariant(right_cauchy_green);

		const T val = c1 * (I1_tilde - codomain_size()) + c2 * (I2_tilde - codomain_size()) + c3 * (I1_tilde - codomain_size()) * (I2_tilde - codomain_size()) + d1 * log_J * log_J;

		return val;
	}

	void MooneyRivlin3ParamSymbolic::get_grad_hess_symbolic(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &F, Eigen::MatrixXd &grad, Eigen::MatrixXd &hess) const
	{
		Eigen::MatrixXd F_T = F.transpose();

		// Grad is ∂W(F)/∂F_ij
		autogen::generate_gradient(c1, c2, c3, d1, F_T, grad);
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		autogen::generate_hessian(c1, c2, c3, d1, F, hess);
	}

	void MooneyRivlin3ParamSymbolic::get_grad_hess_autodiff(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &global_pts, const int el_id, const Eigen::MatrixXd &F, Eigen::MatrixXd &grad, Eigen::MatrixXd &hess) const
	{
		typedef DScalar2<double, FlatMatrixNd> Diff;

		DiffScalarBase::setVariableCount(codomain_size() * codomain_size());

		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(codomain_size(), codomain_size());

		for (int i = 0; i < codomain_size(); ++i)
			for (int j = 0; j < codomain_size(); ++j)
				def_grad(i, j) = Diff(i + j * codomain_size(), F(i, j));

		const auto energy = elastic_energy(global_pts, el_id, def_grad);

		// Grad is ∂W(F)/∂F_ij
		grad = energy.getGradient().reshaped(codomain_size(), codomain_size());
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		hess = energy.getHessian();
	}

	double MooneyRivlin3ParamSymbolic::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	template <typename T>
	T MooneyRivlin3ParamSymbolic::compute_energy_aux(const NonLinearAssemblerData &data) const
	{
		AutoDiffVect<T> local_disp;
		get_local_disp(data, codomain_size(), local_disp);

		AutoDiffGradMat<T> def_grad(codomain_size(), codomain_size());

		T energy = T(0.0);

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			compute_disp_grad_at_quad(data, local_disp, p, codomain_size(), def_grad);

			// Id + grad d
			for (int d = 0; d < codomain_size(); ++d)
				def_grad(d, d) += T(1);

			const T val = elastic_energy(data.vals.val.row(p), data.vals.element_id, def_grad);

			energy += val * data.da(p);
		}
		return energy;
	}

	template <int n_basis, int dim>
	void MooneyRivlin3ParamSymbolic::compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::Matrix<double, Eigen::Dynamic, 1> &G_flattened) const
	{
		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();

		Eigen::Matrix<double, n_basis, dim> local_disp(data.vals.basis_values.size(), codomain_size());
		local_disp.setZero();
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < codomain_size(); ++d)
				{
					local_disp(i, d) += bs.global[ii].val * data.x(bs.global[ii].index * codomain_size() + d);
				}
			}
		}

		Eigen::Matrix<double, dim, dim> def_grad(codomain_size(), codomain_size());
		Eigen::Matrix<double, dim, dim> def_grad_T(codomain_size(), codomain_size());

		Eigen::Matrix<double, n_basis, dim> G(data.vals.basis_values.size(), codomain_size());
		G.setZero();

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), codomain_size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];

			// Id + grad d
			def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(codomain_size(), codomain_size());
			def_grad_T = def_grad.transpose();

			const double t = 0;
			const double c1 = c1_(data.vals.val.row(p), t, data.vals.element_id);
			const double c2 = c2_(data.vals.val.row(p), t, data.vals.element_id);
			const double c3 = c3_(data.vals.val.row(p), t, data.vals.element_id);
			const double d1 = d1_(data.vals.val.row(p), t, data.vals.element_id);

			Eigen::Matrix<double, dim, dim> gradient_temp;
			autogen::generate_gradient_templated<dim>(c1, c2, c3, d1, def_grad_T, gradient_temp);

			Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;
			Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();
			G.noalias() += gradient * data.da(p);
		}

		Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
		G_flattened = temp;
	}

	template <int n_basis, int dim>
	void MooneyRivlin3ParamSymbolic::compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const
	{
		assert(data.x.cols() == 1);

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		const int n_pts = data.da.size();

		Eigen::Matrix<double, n_basis, dim> local_disp(data.vals.basis_values.size(), codomain_size());
		local_disp.setZero();
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < codomain_size(); ++d)
				{
					local_disp(i, d) += bs.global[ii].val * data.x(bs.global[ii].index * codomain_size() + d);
				}
			}
		}

		Eigen::Matrix<double, dim, dim> def_grad(codomain_size(), codomain_size());

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), codomain_size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];

			// Id + grad d
			def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(codomain_size(), codomain_size());

			const double t = 0;
			const double c1 = c1_(data.vals.val.row(p), t, data.vals.element_id);
			const double c2 = c2_(data.vals.val.row(p), t, data.vals.element_id);
			const double c3 = c3_(data.vals.val.row(p), t, data.vals.element_id);
			const double d1 = d1_(data.vals.val.row(p), t, data.vals.element_id);

			Eigen::Matrix<double, dim * dim, dim * dim> hessian_temp;
			autogen::generate_hessian_templated<dim>(c1, c2, c3, d1, def_grad, hessian_temp);

			// Check by FD
			/*
			{
				double eps = 1e-7;
				Eigen::MatrixXd gradient_temp, gradient_temp_plus;
				autogen::generate_gradient(c1, c2, c3, d1, def_grad, gradient_temp);
				Eigen::MatrixXd x_ = def_grad;
				for (int j = 0; j < hessian_temp.cols(); ++j)
				{
					x_(j / dim, j % dim) += eps;
					autogen::generate_gradient(c1, c2, c3, d1, x_, gradient_temp_plus);
					Eigen::MatrixXd fd = (gradient_temp_plus - gradient_temp) / eps;

					std::cout << "hess " << fd.transpose() << "\t" << hessian_temp.col(j).transpose() << std::endl;
					// for (int i = 0; i < hessian_temp.rows(); ++i)
					// {
					// 	double fd = (gradient_temp_plus(i) - gradient_temp(i)) / eps;
					// 	if (abs(fd - hessian_temp(i, j)) > 1e-7)
					// 		std::cout << "mismatch " << abs(fd - hessian_temp(i, j)) << std::endl;
					// }
					x_(j / dim, j % dim) -= eps;
				}
			}
			*/

			Eigen::Matrix<double, dim * dim, N> delF_delU_tensor(jac_it.size(), grad.size());

			for (size_t i = 0; i < local_disp.rows(); ++i)
			{
				for (size_t j = 0; j < local_disp.cols(); ++j)
				{
					Eigen::Matrix<double, dim, dim> temp(codomain_size(), codomain_size());
					temp.setZero();
					temp.row(j) = grad.row(i);
					temp = temp * jac_it;
					Eigen::Matrix<double, dim * dim, 1> temp_flattened(Eigen::Map<Eigen::Matrix<double, dim * dim, 1>>(temp.data(), temp.size()));
					delF_delU_tensor.col(i * codomain_size() + j) = temp_flattened;
				}
			}

			Eigen::Matrix<double, N, N> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

			H += hessian * data.da(p);
		}
	}

	void MooneyRivlin3ParamSymbolic::add_multimaterial(const int index, const json &params, const Units &units)
	{
		c1_.add_multimaterial(index, params, units.stress());
		c2_.add_multimaterial(index, params, units.stress());
		c3_.add_multimaterial(index, params, units.stress());
		d1_.add_multimaterial(index, params, units.stress());
	}

	std::map<std::string, Assembler::ParamFunc> MooneyRivlin3ParamSymbolic::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &c1 = this->c1();
		const auto &c2 = this->c2();
		const auto &c3 = this->c3();
		const auto &d1 = this->d1();

		res["c1"] = [&c1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c1(p, t, e);
		};

		res["c2"] = [&c2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c2(p, t, e);
		};

		res["c3"] = [&c3](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c3(p, t, e);
		};

		res["d1"] = [&d1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return d1(p, t, e);
		};

		return res;
	}

} // namespace polyfem::assembler
