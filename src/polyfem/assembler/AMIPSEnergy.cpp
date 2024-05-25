#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	namespace {
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
	}
	double AMIPSEnergy::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	Eigen::VectorXd AMIPSEnergy::assemble_gradient(const NonLinearAssemblerData &data) const
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

	// Compute ∫ tr(FᵀF) / J^(1+2/dim) dxdydz
	template <typename T>
	T AMIPSEnergy::compute_energy_aux(const NonLinearAssemblerData &data) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		AutoDiffVect local_disp;
		get_local_disp(data, size(), local_disp);

		AutoDiffGradMat def_grad(size(), size());

		T energy = T(0.0);

		Eigen::MatrixXd standard(size(), size());
		if (size() == 2)
			standard << 1 ,                 0,
						0.5, std::sqrt(3) / 2;
		else
			standard << 1,                    0,                 0,
						0.5,  std::sqrt(3) / 2.,                 0,
						0.5, 0.5 / std::sqrt(3), std::sqrt(3) / 2.;
		standard = standard.inverse().transpose().eval();

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			for (int i = 0; i < size(); ++i)
				for (int j = 0; j < size(); ++j)
					def_grad(i, j) = T(0);

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
				def_grad += local_disp.segment(i * size(), size()) * data.vals.basis_values[i].grad.row(p);

			AutoDiffGradMat jac_it(size(), size());
			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = T(data.vals.jac_it[p](k));
			def_grad += jac_it.inverse();
			def_grad = def_grad * standard;

			const T powJ = pow(polyfem::utils::determinant(def_grad), size() == 2 ? 2. : 5. / 3.);
			const T val = (def_grad.transpose() * def_grad).trace() / powJ;

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

		Eigen::Matrix<double, dim, dim> def_grad(size(), size());

		Eigen::Matrix<double, n_basis, dim> G(data.vals.basis_values.size(), size());
		G.setZero();

		Eigen::MatrixXd standard(size(), size());
		if (size() == 2)
			standard << 1 ,                 0,
						0.5, std::sqrt(3) / 2;
		else
			standard << 1,                    0,                 0,
						0.5,  std::sqrt(3) / 2.,                 0,
						0.5, 0.5 / std::sqrt(3), std::sqrt(3) / 2.;
		standard = standard.inverse().transpose().eval();

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];

			// Id + grad d
			def_grad = (local_disp.transpose() * grad + jac_it.inverse()) * standard;

			double J = def_grad.determinant();
			if (J <= 0)
				J = std::nan("");

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

			const double m = (dim == 2) ? 2. : 5. / 3.;
			const double powJ = pow(J, m);
			Eigen::Matrix<double, dim, dim> gradient_temp = (2 * def_grad - (def_grad.squaredNorm() * m) / J * delJ_delF) / powJ;
			Eigen::Matrix<double, n_basis, dim> gradient = grad * standard * gradient_temp.transpose();

			G.noalias() += gradient * data.da(p);
		}

		Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
		G_flattened = temp;
	}

	Eigen::MatrixXd AMIPSEnergy::assemble_hessian(const NonLinearAssemblerData &data) const
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

	template <int n_basis, int dim>
	void AMIPSEnergy::compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const
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

		Eigen::MatrixXd standard(size(), size());
		if (size() == 2)
			standard << 1 ,                 0,
						0.5, std::sqrt(3) / 2;
		else
			standard << 1,                    0,                 0,
						0.5,  std::sqrt(3) / 2.,                 0,
						0.5, 0.5 / std::sqrt(3), std::sqrt(3) / 2.;
		standard = standard.inverse().transpose().eval();

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::Matrix<double, dim, dim> jac_it = data.vals.jac_it[p];

			// Id + grad d
			def_grad = (local_disp.transpose() * grad + jac_it.inverse()) * standard;

			Eigen::Matrix<double, dim * dim, dim * dim> hessian_temp;
			{
				typedef DScalar2<double, Eigen::Matrix<double, dim * dim, 1>, Eigen::Matrix<double, dim * dim, dim * dim>> Diff;
				DiffScalarBase::setVariableCount(dim * dim);

				Eigen::Matrix<Diff, dim, dim> def_grad_ad(dim, dim);
				for (int i = 0; i < def_grad.size(); i++)
					def_grad_ad(i) = Diff(i, def_grad(i));

				Diff val = pow(utils::determinant(def_grad_ad), -1. - 2. / dim) * def_grad_ad.squaredNorm();

				hessian_temp = val.getHessian();
			}

			// const double J = def_grad.determinant();
			// const double TrFFt = def_grad.squaredNorm();

			// Eigen::Matrix<double, dim, dim> delJ_delF(size(), size());
			// delJ_delF.setZero();
			// Eigen::Matrix<double, dim * dim, dim * dim> del2J_delF2(size() * size(), size() * size());
			// del2J_delF2.setZero();

			// if (dim == 2)
			// {
			// 	delJ_delF(0, 0) = def_grad(1, 1);
			// 	delJ_delF(0, 1) = -def_grad(1, 0);
			// 	delJ_delF(1, 0) = -def_grad(0, 1);
			// 	delJ_delF(1, 1) = def_grad(0, 0);

			// 	del2J_delF2(0, 3) = 1;
			// 	del2J_delF2(1, 2) = -1;
			// 	del2J_delF2(2, 1) = -1;
			// 	del2J_delF2(3, 0) = 1;
			// }
			// else if (size() == 3)
			// {
			// 	Eigen::Matrix<double, dim, 1> u(def_grad.rows());
			// 	Eigen::Matrix<double, dim, 1> v(def_grad.rows());
			// 	Eigen::Matrix<double, dim, 1> w(def_grad.rows());

			// 	u = def_grad.col(0);
			// 	v = def_grad.col(1);
			// 	w = def_grad.col(2);

			// 	delJ_delF.col(0) = cross<dim>(v, w);
			// 	delJ_delF.col(1) = cross<dim>(w, u);
			// 	delJ_delF.col(2) = cross<dim>(u, v);

			// 	del2J_delF2.template block<dim, dim>(0, 6) = hat<dim>(v);
			// 	del2J_delF2.template block<dim, dim>(6, 0) = -hat<dim>(v);
			// 	del2J_delF2.template block<dim, dim>(0, 3) = -hat<dim>(w);
			// 	del2J_delF2.template block<dim, dim>(3, 0) = hat<dim>(w);
			// 	del2J_delF2.template block<dim, dim>(3, 6) = -hat<dim>(u);
			// 	del2J_delF2.template block<dim, dim>(6, 3) = hat<dim>(u);
			// }

			// Eigen::Matrix<double, dim * dim, dim * dim> id = Eigen::Matrix<double, dim * dim, dim * dim>::Identity(size() * size(), size() * size());

			// Eigen::Matrix<double, dim * dim, 1> g_j = Eigen::Map<const Eigen::Matrix<double, dim * dim, 1>>(delJ_delF.data(), delJ_delF.size());
			// Eigen::Matrix<double, dim * dim, 1> F_flattened = Eigen::Map<const Eigen::Matrix<double, dim * dim, 1>>(def_grad.data(), def_grad.size());

			// const double tmp = TrFFt / J / dim;
			// const double Jpow = pow(J, 2. / dim);
			// Eigen::Matrix<double, dim * dim, dim * dim> hessian_temp = -4. / dim / (Jpow * J) * (F_flattened - tmp * g_j) * g_j.transpose() + 
			// 															2. / Jpow * (id - ((2 / dim * F_flattened - tmp * g_j) / J * g_j.transpose() + tmp * del2J_delF2));

			Eigen::Matrix<double, dim * dim, N> delF_delU_tensor(jac_it.size(), grad.size());

			for (size_t i = 0; i < local_disp.rows(); ++i)
			{
				for (size_t j = 0; j < local_disp.cols(); ++j)
				{
					Eigen::Matrix<double, dim, dim> temp(size(), size());
					temp.setZero();
					temp.row(j) = grad.row(i);
					temp = temp * standard;
					Eigen::Matrix<double, dim * dim, 1> temp_flattened(Eigen::Map<Eigen::Matrix<double, dim * dim, 1>>(temp.data(), temp.size()));
					delF_delU_tensor.col(i * size() + j) = temp_flattened;
				}
			}

			Eigen::Matrix<double, N, N> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

			H += hessian * data.da(p);
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

	AMIPSEnergyAutodiff::AMIPSEnergyAutodiff()
	{
		canonical_transformation_.resize(0);
	}

	std::map<std::string, Assembler::ParamFunc> AMIPSEnergyAutodiff::parameters() const
	{
		return std::map<std::string, ParamFunc>();
	}

	void AMIPSEnergyAutodiff::add_multimaterial(const int index, const json &params, const Units &)
	{
		if (params.contains("canonical_transformation"))
		{
			canonical_transformation_.reserve(params["canonical_transformation"].size());
			for (int i = 0; i < params["canonical_transformation"].size(); ++i)
			{
				Eigen::MatrixXd transform_matrix(size(), size());
				for (int j = 0; j < size(); ++j)
					for (int k = 0; k < size(); ++k)
						transform_matrix(j, k) = params["canonical_transformation"][i][j][k];
				canonical_transformation_.push_back(transform_matrix);
			}
		}
	}

} // namespace polyfem::assembler