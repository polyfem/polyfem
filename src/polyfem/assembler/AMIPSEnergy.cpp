#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	namespace
	{

		template <class T, int p = 3>
		class barrier
		{
			constexpr static double C = 1e6;

		public:
			static_assert(p % 2 == 1);
			static T value(T J)
			{
				if (J >= 0.5)
					return T(0.);
				const T tmp1 = 2 * J - 1;
				const T tmp2 = pow(tmp1, p);
				return C * (1 / (tmp2 + 1) - 1);
			}

			static T first_derivatives(T J)
			{
				if (J >= 0.5)
					return T(0);
				const T tmp1 = 2 * J - 1;
				const T tmp2 = pow(tmp1, p);
				return C * -2 * p * tmp2 / tmp1 / pow(1 + tmp2, 2);
			}

			static T second_derivatives(T J)
			{
				if (J >= 0.5)
					return T(0);
				const T tmp1 = 2 * J - 1;
				const T tmp2 = pow(tmp1, p);
				return C * 4 * p * tmp2 / pow(tmp1, 2) * ((1 - p) + (1 + p) * tmp2) / pow(1 + tmp2, 3);
			}
		};

		template <int dimt, class T>
		Eigen::Matrix<T, dimt, dimt> get_standard(const int dim, const bool use_rest_pose)
		{
			Eigen::Matrix<double, dimt, dimt> standard(dim, dim);
			if (use_rest_pose)
			{
				standard.setIdentity();
			}
			else
			{
				if (dim == 2)
					standard << 1, 0,
						0.5, std::sqrt(3) / 2;
				else
					standard << 1, 0, 0,
						0.5, std::sqrt(3) / 2., 0,
						0.5, 0.5 / std::sqrt(3), std::sqrt(3) / 2.;
				standard = standard.inverse().transpose().eval();
			}

			Eigen::Matrix<T, dimt, dimt> res(dim, dim);
			for (int i = 0; i < dim; ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					res(i, j) = T(standard(i, j));
				}
			}

			return res;
		}
	} // namespace
	double AMIPSEnergy::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	Eigen::VectorXd AMIPSEnergy::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		const int n_bases = data.vals.basis_values.size();
		auto gradad = polyfem::gradient_from_energy(
			size(), n_bases, data,
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 60, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(data); });
		Eigen::VectorXd gradient;
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

		const double asd = (gradient - gradad).norm();
		if (asd > 1e-10)
			std::cout << asd << std::endl;
		return gradad;
	}

	// Compute ∫ tr(FᵀF) / J^(1+2/dim) dxdydz
	template <typename T>
	T AMIPSEnergy::compute_energy_aux(const NonLinearAssemblerData &data) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> DoubleGradMat;

		double power = -1;
		if (use_rest_pose_)
			power = size() == 2 ? 1. : (2. / 3.);
		else
			power = size() == 2 ? 2. : 5. / 3.;

		AutoDiffVect local_disp;
		get_local_disp(data, size(), local_disp);

		AutoDiffGradMat standard;

		if (size() == 2)
			standard = get_standard<2, T>(size(), use_rest_pose_);
		else
			standard = get_standard<3, T>(size(), use_rest_pose_);

		AutoDiffGradMat def_grad(size(), size());

		T energy = T(0.0);

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<T, Eigen::Dynamic, 3> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				for (int d = 0; d < size(); ++d)
				{
					grad(i, d) = T(data.vals.basis_values[i].grad(p, d));
				}
			}

			compute_disp_grad_at_quad(data, local_disp, p, size(), def_grad);

			// Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += T(1);

			if (!use_rest_pose_)
			{
				DoubleGradMat tmp_jac_it = data.vals.jac_it[p];
				tmp_jac_it = tmp_jac_it.inverse();

				AutoDiffGradMat jac_it(size(), size());
				for (long k = 0; k < jac_it.size(); ++k)
					jac_it(k) = T(tmp_jac_it(k));

				def_grad *= jac_it;
				def_grad = def_grad * standard;
			}

			const T det = polyfem::utils::determinant(def_grad);
			if (det <= 0)
			{
				energy = std::nan("");
				break;
			}

			const T powJ = pow(det, power);
			const T val = (def_grad.transpose() * def_grad).trace() / powJ; //+ barrier<T>::value(det);

			energy += val * data.da(p);
		}
		return energy;
	}

	template <int n_basis, int dim>
	void AMIPSEnergy::compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::Matrix<double, Eigen::Dynamic, 1> &G_flattened) const
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

		const Eigen::Matrix<double, dim, dim> standard = get_standard<dim, double>(size(), use_rest_pose_);
		Eigen::Matrix<double, dim, dim> def_grad(size(), size());

		Eigen::Matrix<double, dim, dim> chain_rule;

		Eigen::Matrix<double, n_basis, dim> G(data.vals.basis_values.size(), size());
		G.setZero();

		double power = -1;
		if (use_rest_pose_)
			power = size() == 2 ? 1. : (2. / 3.);
		else
			power = size() == 2 ? 2. : 5. / 3.;

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			def_grad = local_disp.transpose() * grad;
			def_grad = def_grad * data.vals.jac_it[p];
			chain_rule = data.vals.jac_it[p];

			for (int d = 0; d < dim; ++d)
				def_grad(d, d) += 1;

			if (!use_rest_pose_)
			{
				Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p].inverse();

				def_grad *= jac_it;
				def_grad = def_grad * standard;

				chain_rule = standard;
			}

			double J = polyfem::utils::determinant(def_grad);
			if (J <= 0)
				J = std::nan("");

			const double powJ = pow(J, power);
			Eigen::Matrix<double, dim, dim> gradient_temp = (2 * def_grad - power * (def_grad.transpose() * def_grad).trace() * def_grad.inverse().transpose()) / powJ;
			Eigen::Matrix<double, n_basis, dim> gradient = grad * chain_rule * gradient_temp.transpose();

			G.noalias() += gradient * data.da(p);
		}

		Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
		G_flattened = temp;
	}

	Eigen::MatrixXd AMIPSEnergy::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const int n_bases = data.vals.basis_values.size();
		Eigen::MatrixXd hessianad = polyfem::hessian_from_energy(
			size(), n_bases, data,
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(data); });

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

		const double asd = (hessianad - hessian).norm();
		if (asd > 1e-10)
		{
			logger().error("Hessian mismatch: {}", asd);
			exit(0);
		}

		return hessian;
	}

	template <int n_basis, int dim>
	void AMIPSEnergy::compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const
	{
		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();
		const int nb = data.vals.basis_values.size();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;

		assert(H.rows() == N);
		assert(H.cols() == N);

		H.setZero();

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

		const Eigen::Matrix<double, dim, dim> standard = get_standard<dim, double>(size(), use_rest_pose_);
		Eigen::Matrix<double, dim, dim> def_grad(size(), size());

		Eigen::Matrix<double, dim, dim> chain_rule;

		Eigen::Matrix<double, n_basis, dim> G(data.vals.basis_values.size(), size());
		G.setZero();

		double power = -1;
		if (use_rest_pose_)
			power = size() == 2 ? 1. : (2. / 3.);
		else
			power = size() == 2 ? 2. : 5. / 3.;

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			def_grad = local_disp.transpose() * grad;
			def_grad = def_grad * data.vals.jac_it[p];
			chain_rule = data.vals.jac_it[p];

			for (int d = 0; d < dim; ++d)
				def_grad(d, d) += 1;

			if (!use_rest_pose_)
			{
				Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p].inverse();

				def_grad *= jac_it;
				def_grad = def_grad * standard;

				chain_rule = standard;
			}

			double J = polyfem::utils::determinant(def_grad);
			if (J <= 0)
				J = std::nan("");

			const double powJ = pow(J, power);
			const Eigen::Matrix<double, dim, dim> &F = def_grad;
			const Eigen::Matrix<double, dim, dim> Fit = def_grad.inverse().transpose();
			const double tr = (F.transpose() * F).trace();
			const double m = power;

			const Eigen::Matrix<double, N, dim> temp = grad * chain_rule;

			Eigen::Matrix<double, N, dim> hessian =
				(2 * temp - 4 * m * temp * F * Fit
				 + m * tr * Fit * temp * Fit
				 + m * m * tr * temp * Fit * Fit)
				/ powJ;
			H += temp * hessian.transpose() * data.da(p);
		}
	}

	void AMIPSEnergy::assign_stress_tensor(const OutputData &data,
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

		all.setZero(local_pts.rows(), all_size);
	}

	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		if (params.contains("use_rest_pose"))
		{
			use_rest_pose_ = params["use_rest_pose"].get<bool>();
		}
	}

} // namespace polyfem::assembler