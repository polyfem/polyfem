#include "FixedCorotational.hpp"

#include <polyfem/autogen/auto_elasticity_rhs.hpp>
#include <polyfem/utils/svd.hpp>

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
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

		template <int dim>
		Eigen::Matrix<double, dim, dim> computeCofactorMtr(Eigen::Matrix<double, dim, dim> &F)
		{
			Eigen::Matrix<double, dim, dim> A;
			A.setZero();
			if constexpr (dim == 2)
			{
				A(0, 0) = F(1, 1);
				A(0, 1) = -F(1, 0);
				A(1, 0) = -F(0, 1);
				A(1, 1) = F(0, 0);
			}
			else if constexpr (dim == 3)
			{
				A(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
				A(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
				A(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
				A(1, 0) = F(0, 2) * F(2, 1) - F(0, 1) * F(2, 2);
				A(1, 1) = F(0, 0) * F(2, 2) - F(0, 2) * F(2, 0);
				A(1, 2) = F(0, 1) * F(2, 0) - F(0, 0) * F(2, 1);
				A(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
				A(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
				A(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
			}
			return A;
		}

	} // namespace

	FixedCorotational::FixedCorotational()
	{
	}

	void FixedCorotational::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		params_.add_multimaterial(index, params, size() == 3, units.stress());
	}

	Eigen::VectorXd
	FixedCorotational::assemble_gradient(const NonLinearAssemblerData &data) const
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

	Eigen::MatrixXd
	FixedCorotational::assemble_hessian(const NonLinearAssemblerData &data) const
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

	void FixedCorotational::assign_stress_tensor(const OutputData &data,
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

			double lambda, mu;
			params_.lambda_mu(local_pts.row(p), vals.val.row(p), t, vals.element_id, lambda, mu);

			Eigen::MatrixXd stress_tensor;
			if (size() == 2)
				stress_tensor = compute_stress_from_def_grad<2>(def_grad, lambda, mu) * def_grad.transpose() / def_grad.determinant();
			else
				stress_tensor = compute_stress_from_def_grad<3>(def_grad, lambda, mu) * def_grad.transpose() / def_grad.determinant();
			if (type == ElasticityTensorType::PK1)
				stress_tensor = pk1_from_cauchy(stress_tensor, def_grad);
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = pk2_from_cauchy(stress_tensor, def_grad);

			all.row(p) = fun(stress_tensor);
		}
	}

	double FixedCorotational::compute_energy(const NonLinearAssemblerData &data) const
	{
		if (size() == 2)
			return compute_energy_aux<2>(data);
		else
			return compute_energy_aux<3>(data);
	}

	template <int dim>
	double FixedCorotational::compute_energy_aux(const NonLinearAssemblerData &data) const
	{
		Eigen::VectorXd local_disp;
		get_local_disp(data, dim, local_disp);

		Eigen::Matrix<double, dim, dim> def_grad;

		double energy = 0.0;

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			compute_disp_grad_at_quad(data, local_disp, p, dim, def_grad);

			// Id + grad d
			def_grad.diagonal().array() += 1.0;

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.t, data.vals.element_id, lambda, mu);

			const double val = compute_energy_from_def_grad(def_grad, lambda, mu);

			energy += val * data.da(p);
		}
		return energy;
	}

	template <int n_basis, int dim>
	void FixedCorotational::compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::Matrix<double, Eigen::Dynamic, 1> &G_flattened) const
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

			const Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];
			const Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

			// Id + grad d
			def_grad = local_disp.transpose() * delF_delU + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.t, data.vals.element_id, lambda, mu);

			Eigen::Matrix<double, dim, dim> gradient_temp = compute_stress_from_def_grad(def_grad, lambda, mu);

			Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();

			G.noalias() += gradient * data.da(p);
		}

		Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
		G_flattened = temp;
	}

	template <int n_basis, int dim>
	void FixedCorotational::compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const
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
			const Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

			// Id + grad d
			def_grad = local_disp.transpose() * delF_delU + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.t, data.vals.element_id, lambda, mu);

			Eigen::Matrix<double, dim * dim, dim * dim> hessian_temp = compute_stiffness_from_def_grad(def_grad, lambda, mu);

			Eigen::Matrix<double, dim * dim, N> delF_delU_tensor = Eigen::Matrix<double, dim * dim, N>::Zero(jac_it.size(), grad.size());

			for (size_t i = 0; i < local_disp.rows(); ++i)
			{
				for (size_t j = 0; j < dim; ++j)
					for (size_t k = 0; k < dim; ++k)
						delF_delU_tensor(dim * k + j, i * dim + j) = delF_delU(i, k);
			}

			Eigen::Matrix<double, N, N> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

			H += hessian * data.da(p);
		}
	}

	void FixedCorotational::compute_stress_grad_multiply_mat(
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

		double lambda, mu;
		params_.lambda_mu(local_pts, global_pts, t, el_id, lambda, mu);

		const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		if (size() == 2)
		{
			stress = compute_stress_from_def_grad<2>(def_grad, lambda, mu);
			result = (compute_stiffness_from_def_grad<2>(def_grad, lambda, mu) * mat.reshaped(size() * size(), 1)).reshaped(size(), size());
		}
		else
		{
			stress = compute_stress_from_def_grad<3>(def_grad, lambda, mu);
			result = (compute_stiffness_from_def_grad<3>(def_grad, lambda, mu) * mat.reshaped(size() * size(), 1)).reshaped(size(), size());
		}
	}

	void FixedCorotational::compute_stress_grad_multiply_stress(
		const OptAssemblerData &data,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		double lambda, mu;
		params_.lambda_mu(local_pts, global_pts, t, el_id, lambda, mu);

		Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		if (size() == 2)
		{
			stress = compute_stress_from_def_grad<2>(def_grad, lambda, mu);
			result = (compute_stiffness_from_def_grad<2>(def_grad, lambda, mu) * stress.reshaped(size() * size(), 1)).reshaped(size(), size());
		}
		else
		{
			stress = compute_stress_from_def_grad<3>(def_grad, lambda, mu);
			result = (compute_stiffness_from_def_grad<3>(def_grad, lambda, mu) * stress.reshaped(size() * size(), 1)).reshaped(size(), size());
		}
	}

	void FixedCorotational::compute_dstress_dmu_dlambda(
		const OptAssemblerData &data,
		Eigen::MatrixXd &dstress_dmu,
		Eigen::MatrixXd &dstress_dlambda) const
	{
		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(size(), size()) + grad_u_i;

		Eigen::MatrixXd UVT;
		Eigen::VectorXd sigmas;
		{
			if (size() == 2)
			{
				utils::AutoFlipSVD<Eigen::Matrix<double, 2, 2>> svd(def_grad, Eigen::ComputeFullU | Eigen::ComputeFullV);
				sigmas = svd.singularValues();
				UVT = svd.matrixU() * svd.matrixV().transpose();
			}
			else
			{
				utils::AutoFlipSVD<Eigen::Matrix<double, 3, 3>> svd(def_grad, Eigen::ComputeFullU | Eigen::ComputeFullV);
				sigmas = svd.singularValues();
				UVT = svd.matrixU() * svd.matrixV().transpose();
			}
		}

		Eigen::MatrixXd delJ_delF(size(), size());
		delJ_delF.setZero();

		if (size() == 2)
		{
			delJ_delF(0, 0) = def_grad(1, 1);
			delJ_delF(0, 1) = -def_grad(1, 0);
			delJ_delF(1, 0) = -def_grad(0, 1);
			delJ_delF(1, 1) = def_grad(0, 0);
		}
		else if (size() == 3)
		{
			const Eigen::Matrix<double, 3, 1> u = def_grad.col(0);
			const Eigen::Matrix<double, 3, 1> v = def_grad.col(1);
			const Eigen::Matrix<double, 3, 1> w = def_grad.col(2);

			delJ_delF.col(0) = cross<3>(v, w);
			delJ_delF.col(1) = cross<3>(w, u);
			delJ_delF.col(2) = cross<3>(u, v);
		}

		dstress_dmu = 2 * (def_grad - UVT);
		dstress_dlambda = (sigmas.prod() - 1) * delJ_delF;
	}

	std::map<std::string, Assembler::ParamFunc> FixedCorotational::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &params = params_;
		const int size = this->size();

		res["lambda"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);
			return lambda;
		};

		res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);
			return mu;
		};

		res["E"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);

			if (size == 3)
				return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
			else
				return 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
		};

		res["nu"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);

			if (size == 3)
				return lambda / (2.0 * (lambda + mu));
			else
				return lambda / (lambda + 2.0 * mu);
		};

		return res;
	}

	template <int dim>
	double FixedCorotational::compute_energy_from_singular_values(const Eigen::Vector<double, dim> &sigmas, const double lambda, const double mu)
	{
		const double sigmam12Sum = (sigmas - Eigen::Matrix<double, dim, 1>::Ones()).squaredNorm();
		const double sigmaProdm1 = sigmas.prod() - 1.0;

		return mu * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
	}

	template <int dim>
	Eigen::Vector<double, dim> FixedCorotational::compute_stress_from_singular_values(const Eigen::Vector<double, dim> &sigmas, const double lambda, const double mu)
	{
		const double sigmaProdm1lambda = lambda * (sigmas.prod() - 1.0);
		Eigen::Matrix<double, dim, 1> sigmaProd_noI;
		if constexpr (dim == 2)
		{
			sigmaProd_noI[0] = sigmas[1];
			sigmaProd_noI[1] = sigmas[0];
		}
		else
		{
			sigmaProd_noI[0] = sigmas[1] * sigmas[2];
			sigmaProd_noI[1] = sigmas[2] * sigmas[0];
			sigmaProd_noI[2] = sigmas[0] * sigmas[1];
		}

		const double _2mu = mu * 2;
		Eigen::Vector<double, dim> dE_div_dsigma;
		dE_div_dsigma.setZero();
		dE_div_dsigma[0] = (_2mu * (sigmas[0] - 1.0) + sigmaProd_noI[0] * sigmaProdm1lambda);
		dE_div_dsigma[1] = (_2mu * (sigmas[1] - 1.0) + sigmaProd_noI[1] * sigmaProdm1lambda);
		if constexpr (dim == 3)
		{
			dE_div_dsigma[2] = (_2mu * (sigmas[2] - 1.0) + sigmaProd_noI[2] * sigmaProdm1lambda);
		}
		return dE_div_dsigma;
	}

	template <int dim>
	Eigen::Matrix<double, dim, dim> FixedCorotational::compute_stiffness_from_singular_values(const Eigen::Vector<double, dim> &sigmas, const double lambda, const double mu)
	{
		const double sigmaProd = sigmas.prod();
		Eigen::Matrix<double, dim, 1> sigmaProd_noI;
		if constexpr (dim == 2)
		{
			sigmaProd_noI[0] = sigmas[1];
			sigmaProd_noI[1] = sigmas[0];
		}
		else
		{
			sigmaProd_noI[0] = sigmas[1] * sigmas[2];
			sigmaProd_noI[1] = sigmas[2] * sigmas[0];
			sigmaProd_noI[2] = sigmas[0] * sigmas[1];
		}

		double _2mu = mu * 2;
		Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
		d2E_div_dsigma2.setZero();
		d2E_div_dsigma2(0, 0) = _2mu + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
		d2E_div_dsigma2(1, 1) = _2mu + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
		if constexpr (dim == 3)
		{
			d2E_div_dsigma2(2, 2) = _2mu + lambda * sigmaProd_noI[2] * sigmaProd_noI[2];
		}

		if constexpr (dim == 2)
		{
			d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda * ((sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
		}
		else
		{
			d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda * (sigmas[2] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
			d2E_div_dsigma2(0, 2) = d2E_div_dsigma2(2, 0) = lambda * (sigmas[1] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[2]);
			d2E_div_dsigma2(2, 1) = d2E_div_dsigma2(1, 2) = lambda * (sigmas[0] * (sigmaProd - 1.0) + sigmaProd_noI[2] * sigmaProd_noI[1]);
		}

		return d2E_div_dsigma2;
	}

	template <int dim>
	double FixedCorotational::compute_energy_from_def_grad(const Eigen::Matrix<double, dim, dim> &F, const double lambda, const double mu)
	{
		const Eigen::Vector<double, dim> sigmas = utils::singular_values<dim>(F);
		return compute_energy_from_singular_values(sigmas, lambda, mu);
	}

	template <int dim>
	Eigen::Matrix<double, dim, dim> FixedCorotational::compute_stress_from_def_grad(const Eigen::Matrix<double, dim, dim> &F, const double lambda, const double mu)
	{
		utils::AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::Vector<double, dim> sigmas = svd.singularValues();

		Eigen::Matrix<double, dim, dim> delJ_delF(dim, dim);
		delJ_delF.setZero();

		if (dim == 2)
		{
			delJ_delF(0, 0) = F(1, 1);
			delJ_delF(0, 1) = -F(1, 0);
			delJ_delF(1, 0) = -F(0, 1);
			delJ_delF(1, 1) = F(0, 0);
		}
		else if (dim == 3)
		{
			const Eigen::Matrix<double, dim, 1> u = F.col(0);
			const Eigen::Matrix<double, dim, 1> v = F.col(1);
			const Eigen::Matrix<double, dim, 1> w = F.col(2);

			delJ_delF.col(0) = cross<dim>(v, w);
			delJ_delF.col(1) = cross<dim>(w, u);
			delJ_delF.col(2) = cross<dim>(u, v);
		}

		return lambda * (sigmas.prod() - 1) * delJ_delF + mu * 2 * (F - svd.matrixU() * svd.matrixV().transpose());
	}

	template <int dim>
	Eigen::Matrix<double, dim * dim, dim * dim> FixedCorotational::compute_stiffness_from_def_grad(const Eigen::Matrix<double, dim, dim> &F, const double lambda, const double mu)
	{
		utils::AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::Vector<double, dim> sigmas = svd.singularValues();
		Eigen::Matrix<double, dim, 1> dE_div_dsigma = compute_stress_from_singular_values(sigmas, lambda, mu);
		Eigen::Matrix<double, dim, dim> d2E_div_dsigma2 = compute_stiffness_from_singular_values(sigmas, lambda, mu);

		constexpr int Cdim2 = dim * (dim - 1) / 2;
		Eigen::Matrix<double, Cdim2, 1> BLeftCoef;
		BLeftCoef.setZero();
		{
			const double tmp = lambda / 2.0 * (sigmas.prod() - 1);
			if constexpr (dim == 2)
			{
				BLeftCoef[0] = mu - tmp;
			}
			else
			{
				BLeftCoef[0] = mu - tmp * sigmas[2];
				BLeftCoef[1] = mu - tmp * sigmas[0];
				BLeftCoef[2] = mu - tmp * sigmas[1];
			}
		}
		Eigen::Matrix2d B[Cdim2];
		for (int cI = 0; cI < Cdim2; cI++)
		{
			B[cI].setZero();
			int cI_post = (cI + 1) % dim;

			double rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
			double sum_sigma = sigmas[cI] + sigmas[cI_post];
			rightCoef /= 2.0 * std::max(sum_sigma, 1.0e-12);

			const double &leftCoef = BLeftCoef[cI];
			B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
			B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
		}

		// compute M using A(d2E_div_dsigma2) and B
		Eigen::Matrix<double, dim * dim, dim * dim> M;
		M.setZero();
		if constexpr (dim == 2)
		{
			M(0, 0) = d2E_div_dsigma2(0, 0);
			M(0, 3) = d2E_div_dsigma2(0, 1);
			M.block(1, 1, 2, 2) = B[0];
			M(3, 0) = d2E_div_dsigma2(1, 0);
			M(3, 3) = d2E_div_dsigma2(1, 1);
		}
		else
		{
			// A
			M(0, 0) = d2E_div_dsigma2(0, 0);
			M(0, 4) = d2E_div_dsigma2(0, 1);
			M(0, 8) = d2E_div_dsigma2(0, 2);
			M(4, 0) = d2E_div_dsigma2(1, 0);
			M(4, 4) = d2E_div_dsigma2(1, 1);
			M(4, 8) = d2E_div_dsigma2(1, 2);
			M(8, 0) = d2E_div_dsigma2(2, 0);
			M(8, 4) = d2E_div_dsigma2(2, 1);
			M(8, 8) = d2E_div_dsigma2(2, 2);
			// B01
			M(1, 1) = B[0](0, 0);
			M(1, 3) = B[0](0, 1);
			M(3, 1) = B[0](1, 0);
			M(3, 3) = B[0](1, 1);
			// B12
			M(5, 5) = B[1](0, 0);
			M(5, 7) = B[1](0, 1);
			M(7, 5) = B[1](1, 0);
			M(7, 7) = B[1](1, 1);
			// B20
			M(2, 2) = B[2](1, 1);
			M(2, 6) = B[2](1, 0);
			M(6, 2) = B[2](0, 1);
			M(6, 6) = B[2](0, 0);
		}

		// compute hessian
		Eigen::Matrix<double, dim * dim, dim * dim> hessian;
		hessian.setZero();
		const Eigen::Matrix<double, dim, dim> &U = svd.matrixU();
		const Eigen::Matrix<double, dim, dim> &V = svd.matrixV();
		for (int i = 0; i < dim; i++)
		{
			// int _dim_i = i * dim;
			for (int j = 0; j < dim; j++)
			{
				int ij = i + j * dim;
				for (int r = 0; r < dim; r++)
				{
					// int _dim_r = r * dim;
					for (int s = 0; s < dim; s++)
					{
						int rs = r + s * dim;
						if (ij > rs)
						{
							// bottom left, same as upper right
							continue;
						}

						if constexpr (dim == 2)
						{
							hessian(ij, rs) = M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) + M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) + M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) + M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) + M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) + M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) + M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) + M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
						}
						else
						{
							hessian(ij, rs) = M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) + M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) + M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) + M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) + M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) + M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) + M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) + M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) + M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) + M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) + M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) + M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) + M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) + M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) + M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) + M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) + M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) + M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) + M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) + M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) + M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
						}

						if (ij < rs)
							hessian(rs, ij) = hessian(ij, rs);
					}
				}
			}
		}

		return hessian;
	}
} // namespace polyfem::assembler
