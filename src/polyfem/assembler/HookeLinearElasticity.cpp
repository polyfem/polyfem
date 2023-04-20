#include "HookeLinearElasticity.hpp"

#include <polyfem/autogen/auto_elasticity_rhs.hpp>

namespace polyfem::assembler
{
	using namespace basis;
	namespace
	{
		template <class Matrix>
		Matrix strain_from_disp_grad(const Matrix &disp_grad)
		{
			Matrix mat = (disp_grad + disp_grad.transpose());

			for (int i = 0; i < mat.size(); ++i)
				mat(i) *= 0.5;

			return mat;
		}

		template <int dim>
		Eigen::Matrix<double, dim, dim> strain(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, int k, int coo)
		{
			Eigen::Matrix<double, dim, dim> jac;
			jac.setZero();
			jac.row(coo) = grad.row(k);
			jac = jac * jac_it;

			return strain_from_disp_grad(jac);
		}

		template <typename T, unsigned long N>
		T stress(const ElasticityTensor &elasticity_tensor, const std::array<T, N> &strain, const int j)
		{
			T res = elasticity_tensor(j, 0) * strain[0];

			for (unsigned long k = 1; k < N; ++k)
				res += elasticity_tensor(j, k) * strain[k];

			return res;
		}
	} // namespace

	HookeLinearElasticity::HookeLinearElasticity()
	{
	}

	void HookeLinearElasticity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		if (!params.contains("elasticity_tensor") || params["elasticity_tensor"].empty())
		{
			if (params.count("young"))
			{
				if (params["young"].is_number() && params["nu"].is_number())
					elasticity_tensor_.set_from_young_poisson(params["young"], params["nu"], units.stress());
			}
			else if (params.count("E"))
			{
				if (params["E"].is_number() && params["nu"].is_number())
					elasticity_tensor_.set_from_young_poisson(params["E"], params["nu"], units.stress());
			}
			else if (params.count("lambda"))
			{
				if (params["lambda"].is_number() && params["mu"].is_number())
					elasticity_tensor_.set_from_lambda_mu(params["lambda"], params["mu"], units.stress());
			}
		}
		else
		{
			std::vector<double> entries = params["elasticity_tensor"];
			elasticity_tensor_.set_from_entries(entries, units.stress());
		}
	}

	void HookeLinearElasticity::set_size(const int size)
	{
		Assembler::set_size(size);
		elasticity_tensor_.resize(size);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	HookeLinearElasticity::assemble(const LinearAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad;

		// (C : gradi) : gradj
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size());
		res.setZero();
		assert(gradi.cols() == size());
		assert(gradj.cols() == size());
		assert(size_t(gradi.rows()) == data.vals.jac_it.size());

		for (long k = 0; k < gradi.rows(); ++k)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size() * size());

			if (size() == 2)
			{
				const Eigen::Matrix2d eps_x_i = strain<2>(gradi, data.vals.jac_it[k], k, 0);
				const Eigen::Matrix2d eps_y_i = strain<2>(gradi, data.vals.jac_it[k], k, 1);

				const Eigen::Matrix2d eps_x_j = strain<2>(gradj, data.vals.jac_it[k], k, 0);
				const Eigen::Matrix2d eps_y_j = strain<2>(gradj, data.vals.jac_it[k], k, 1);

				std::array<double, 3> e_x, e_y;
				e_x[0] = eps_x_i(0, 0);
				e_x[1] = eps_x_i(1, 1);
				e_x[2] = 2 * eps_x_i(0, 1);

				e_y[0] = eps_y_i(0, 0);
				e_y[1] = eps_y_i(1, 1);
				e_y[2] = 2 * eps_y_i(0, 1);

				Eigen::Matrix2d sigma_x;
				sigma_x << elasticity_tensor_.compute_stress<3>(e_x, 0), elasticity_tensor_.compute_stress<3>(e_x, 2),
					elasticity_tensor_.compute_stress<3>(e_x, 2), elasticity_tensor_.compute_stress<3>(e_x, 1);

				Eigen::Matrix2d sigma_y;
				sigma_y << elasticity_tensor_.compute_stress<3>(e_y, 0), elasticity_tensor_.compute_stress<3>(e_y, 2),
					elasticity_tensor_.compute_stress<3>(e_y, 2), elasticity_tensor_.compute_stress<3>(e_y, 1);

				res_k(0) = (sigma_x * eps_x_j).trace();
				res_k(2) = (sigma_x * eps_y_j).trace();

				res_k(1) = (sigma_y * eps_x_j).trace();
				res_k(3) = (sigma_y * eps_y_j).trace();
			}
			else
			{
				const Eigen::Matrix3d eps_x_i = strain<3>(gradi, data.vals.jac_it[k], k, 0);
				const Eigen::Matrix3d eps_y_i = strain<3>(gradi, data.vals.jac_it[k], k, 1);
				const Eigen::Matrix3d eps_z_i = strain<3>(gradi, data.vals.jac_it[k], k, 2);

				const Eigen::Matrix3d eps_x_j = strain<3>(gradj, data.vals.jac_it[k], k, 0);
				const Eigen::Matrix3d eps_y_j = strain<3>(gradj, data.vals.jac_it[k], k, 1);
				const Eigen::Matrix3d eps_z_j = strain<3>(gradj, data.vals.jac_it[k], k, 2);

				std::array<double, 6> e_x, e_y, e_z;
				e_x[0] = eps_x_i(0, 0);
				e_x[1] = eps_x_i(1, 1);
				e_x[2] = eps_x_i(2, 2);
				e_x[3] = 2 * eps_x_i(1, 2);
				e_x[4] = 2 * eps_x_i(0, 2);
				e_x[5] = 2 * eps_x_i(0, 1);

				e_y[0] = eps_y_i(0, 0);
				e_y[1] = eps_y_i(1, 1);
				e_y[2] = eps_y_i(2, 2);
				e_y[3] = 2 * eps_y_i(1, 2);
				e_y[4] = 2 * eps_y_i(0, 2);
				e_y[5] = 2 * eps_y_i(0, 1);

				e_z[0] = eps_z_i(0, 0);
				e_z[1] = eps_z_i(1, 1);
				e_z[2] = eps_z_i(2, 2);
				e_z[3] = 2 * eps_z_i(1, 2);
				e_z[4] = 2 * eps_z_i(0, 2);
				e_z[5] = 2 * eps_z_i(0, 1);

				Eigen::Matrix3d sigma_x;
				sigma_x << elasticity_tensor_.compute_stress<6>(e_x, 0), elasticity_tensor_.compute_stress<6>(e_x, 5), elasticity_tensor_.compute_stress<6>(e_x, 4),
					elasticity_tensor_.compute_stress<6>(e_x, 5), elasticity_tensor_.compute_stress<6>(e_x, 1), elasticity_tensor_.compute_stress<6>(e_x, 3),
					elasticity_tensor_.compute_stress<6>(e_x, 4), elasticity_tensor_.compute_stress<6>(e_x, 3), elasticity_tensor_.compute_stress<6>(e_x, 2);

				Eigen::Matrix3d sigma_y;
				sigma_y << elasticity_tensor_.compute_stress<6>(e_y, 0), elasticity_tensor_.compute_stress<6>(e_y, 5), elasticity_tensor_.compute_stress<6>(e_y, 4),
					elasticity_tensor_.compute_stress<6>(e_y, 5), elasticity_tensor_.compute_stress<6>(e_y, 1), elasticity_tensor_.compute_stress<6>(e_y, 3),
					elasticity_tensor_.compute_stress<6>(e_y, 4), elasticity_tensor_.compute_stress<6>(e_y, 3), elasticity_tensor_.compute_stress<6>(e_y, 2);

				Eigen::Matrix3d sigma_z;
				sigma_z << elasticity_tensor_.compute_stress<6>(e_z, 0), elasticity_tensor_.compute_stress<6>(e_z, 5), elasticity_tensor_.compute_stress<6>(e_z, 4),
					elasticity_tensor_.compute_stress<6>(e_z, 5), elasticity_tensor_.compute_stress<6>(e_z, 1), elasticity_tensor_.compute_stress<6>(e_z, 3),
					elasticity_tensor_.compute_stress<6>(e_z, 4), elasticity_tensor_.compute_stress<6>(e_z, 3), elasticity_tensor_.compute_stress<6>(e_z, 2);

				res_k(0) = (sigma_x * eps_x_j).trace();
				res_k(3) = (sigma_x * eps_y_j).trace();
				res_k(6) = (sigma_x * eps_z_j).trace();

				res_k(1) = (sigma_y * eps_x_j).trace();
				res_k(4) = (sigma_y * eps_y_j).trace();
				res_k(7) = (sigma_y * eps_z_j).trace();

				res_k(2) = (sigma_z * eps_x_j).trace();
				res_k(5) = (sigma_z * eps_y_j).trace();
				res_k(8) = (sigma_z * eps_z_j).trace();
			}

			res += res_k * data.da(k);
		}

		// std::cout<<"res\n"<<res<<"\n"<<std::endl;

		return res;
	}

	void HookeLinearElasticity::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			if (type == ElasticityTensorType::F)
			{
				all.row(p) = fun(displacement_grad + Eigen::MatrixXd::Identity(size(), size()));
				continue;
			}

			Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose()) / 2;
			Eigen::MatrixXd sigma(size(), size());

			if (size() == 2)
			{
				std::array<double, 3> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = 2 * strain(0, 1);

				sigma << elasticity_tensor_.compute_stress<3>(eps, 0), elasticity_tensor_.compute_stress<3>(eps, 2),
					elasticity_tensor_.compute_stress<3>(eps, 2), elasticity_tensor_.compute_stress<3>(eps, 1);
			}
			else
			{
				std::array<double, 6> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = strain(2, 2);
				eps[3] = 2 * strain(1, 2);
				eps[4] = 2 * strain(0, 2);
				eps[5] = 2 * strain(0, 1);

				sigma << elasticity_tensor_.compute_stress<6>(eps, 0), elasticity_tensor_.compute_stress<6>(eps, 5), elasticity_tensor_.compute_stress<6>(eps, 4),
					elasticity_tensor_.compute_stress<6>(eps, 5), elasticity_tensor_.compute_stress<6>(eps, 1), elasticity_tensor_.compute_stress<6>(eps, 3),
					elasticity_tensor_.compute_stress<6>(eps, 4), elasticity_tensor_.compute_stress<6>(eps, 3), elasticity_tensor_.compute_stress<6>(eps, 2);
			}

			if (type == ElasticityTensorType::PK1)
				sigma = pk1_from_cauchy(sigma, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));
			else if (type == ElasticityTensorType::PK2)
				sigma = pk2_from_cauchy(sigma, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));

			all.row(p) = fun(sigma);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	HookeLinearElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;

		if (size() == 2)
			autogen::hooke_2d_function(pt, elasticity_tensor_, res);
		else if (size() == 3)
			autogen::hooke_3d_function(pt, elasticity_tensor_, res);
		else
			assert(false);

		return res;
	}

	std::map<std::string, Assembler::ParamFunc> HookeLinearElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &elast_tensor = elasticity_tensor();
		const int size = this->size() == 2 ? 3 : 6;

		for (int i = 0; i < size; ++i)
		{
			for (int j = i; j < size; ++j)
			{
				res[fmt::format("C_{}{}", i, j)] = [&elast_tensor, i, j](const RowVectorNd &, const RowVectorNd &, double, int) {
					return elast_tensor(i, j);
				};
			}
		}
		return res;
	}

	double HookeLinearElasticity::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	Eigen::VectorXd HookeLinearElasticity::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		const int n_bases = data.vals.basis_values.size();
		return polyfem::gradient_from_energy(
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
	}

	Eigen::MatrixXd HookeLinearElasticity::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const int n_bases = data.vals.basis_values.size();
		return polyfem::hessian_from_energy(
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
	}

	template <typename T>
	T HookeLinearElasticity::compute_energy_aux(const NonLinearAssemblerData &data) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		AutoDiffVect local_disp;
		get_local_disp(data, size(), local_disp);

		AutoDiffGradMat disp_grad(size(), size());

		T energy = T(0.0);

		const int n_pts = data.da.size();
		for (long p = 0; p < n_pts; ++p)
		{
			compute_disp_grad_at_quad(data, local_disp, p, size(), disp_grad);

			AutoDiffGradMat strain = strain_from_disp_grad(disp_grad);
			AutoDiffGradMat stress_tensor(size(), size());

			if (size() == 2)
			{
				std::array<T, 3> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = 2 * strain(0, 1);

				stress_tensor << stress(elasticity_tensor_, eps, 0), stress(elasticity_tensor_, eps, 2),
					stress(elasticity_tensor_, eps, 2), stress(elasticity_tensor_, eps, 1);
			}
			else
			{
				std::array<T, 6> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = strain(2, 2);
				eps[3] = 2 * strain(1, 2);
				eps[4] = 2 * strain(0, 2);
				eps[5] = 2 * strain(0, 1);

				stress_tensor << stress(elasticity_tensor_, eps, 0), stress(elasticity_tensor_, eps, 5), stress(elasticity_tensor_, eps, 4),
					stress(elasticity_tensor_, eps, 5), stress(elasticity_tensor_, eps, 1), stress(elasticity_tensor_, eps, 3),
					stress(elasticity_tensor_, eps, 4), stress(elasticity_tensor_, eps, 3), stress(elasticity_tensor_, eps, 2);
			}

			energy += (stress_tensor * strain).trace() * data.da(p);
		}

		return energy * 0.5;
	}
} // namespace polyfem::assembler
