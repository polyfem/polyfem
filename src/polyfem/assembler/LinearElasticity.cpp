#include "LinearElasticity.hpp"

#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
// #include <finitediff.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem
{
	using namespace basis;

	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}
	} // namespace

	namespace assembler
	{
		void LinearElasticity::add_multimaterial(const int index, const json &params, const Units &units)
		{
			assert(size() == 2 || size() == 3);

			params_.add_multimaterial(index, params, size() == 3, units.stress());
		}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		LinearElasticity::assemble(const LinearAssemblerData &data) const
		{
			// mu ((gradi' gradj) Id + ((gradi gradj')') + lambda gradi *gradj';
			const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
			const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;

			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size());
			res.setZero();

			for (long k = 0; k < gradi.rows(); ++k)
			{
				Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size() * size());
				//            res_k.setZero();
				const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> outer = gradi.row(k).transpose() * gradj.row(k);
				const double dot = gradi.row(k).dot(gradj.row(k));

				double lambda, mu;
				params_.lambda_mu(data.vals.quadrature.points.row(k), data.vals.val.row(k), data.vals.element_id, lambda, mu);

				for (int ii = 0; ii < size(); ++ii)
				{
					for (int jj = 0; jj < size(); ++jj)
					{
						res_k(jj * size() + ii) = outer(ii * size() + jj) * mu + outer(jj * size() + ii) * lambda;
						if (ii == jj)
							res_k(jj * size() + ii) += mu * dot;
					}
				}
				res += res_k * data.da(k);
			}

			return res;
		}

		double LinearElasticity::compute_energy(const NonLinearAssemblerData &data) const
		{
			return compute_energy_aux<double>(data);
		}

		Eigen::VectorXd LinearElasticity::assemble_gradient(const NonLinearAssemblerData &data) const
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

		Eigen::MatrixXd LinearElasticity::assemble_hessian(const NonLinearAssemblerData &data) const
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

		// Compute \int mu eps : eps + lambda/2 tr(eps)^2 = \int mu tr(eps^2) + lambda/2 tr(eps)^2
		template <typename T>
		T LinearElasticity::compute_energy_aux(const NonLinearAssemblerData &data) const
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

				const AutoDiffGradMat strain = (disp_grad + disp_grad.transpose()) / T(2);

				double lambda, mu;
				params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.vals.element_id, lambda, mu);

				const T val = mu * (strain.transpose() * strain).trace() + lambda / 2 * strain.trace() * strain.trace();

				energy += val * data.da(p);
			}
			return energy;
		}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		LinearElasticity::compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(pt.size() == size());
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

			double lambda, mu;
			// TODO!
			params_.lambda_mu(0, 0, 0, pt(0).getValue(), pt(1).getValue(), size() == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

			if (size() == 2)
				autogen::linear_elasticity_2d_function(pt, lambda, mu, res);
			else if (size() == 3)
				autogen::linear_elasticity_3d_function(pt, lambda, mu, res);
			else
				assert(false);

			return res;
		}

		void LinearElasticity::compute_stiffness_value(const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const
		{
			tensor.resize(local_pts.rows(), size() * size() * size() * size());
			assert(displacement.cols() == 1);

			for (long p = 0; p < local_pts.rows(); ++p)
			{
				double lambda, mu;
				params_.lambda_mu(local_pts.row(p), vals.val.row(p), vals.element_id, lambda, mu);

				for (int i = 0, idx = 0; i < size(); i++)
					for (int j = 0; j < size(); j++)
						for (int k = 0; k < size(); k++)
							for (int l = 0; l < size(); l++)
								tensor(p, idx++) = mu * delta(i, k) * delta(j, l) + mu * delta(i, l) * delta(j, k) + lambda * delta(i, j) * delta(k, l);
			}
		}

		void LinearElasticity::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
		{
			all.resize(local_pts.rows(), all_size);
			assert(displacement.cols() == 1);

			Eigen::MatrixXd displacement_grad(size(), size());

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

				double lambda, mu;
				params_.lambda_mu(local_pts.row(p), vals.val.row(p), vals.element_id, lambda, mu);

				const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose()) / 2;
				Eigen::MatrixXd stress = 2 * mu * strain + lambda * strain.trace() * Eigen::MatrixXd::Identity(size(), size());
				if (type == ElasticityTensorType::PK1)
					stress = pk1_from_cauchy(stress, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));
				else if (type == ElasticityTensorType::PK2)
					stress = pk2_from_cauchy(stress, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));

				all.row(p) = fun(stress);
			}
		}

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> LinearElasticity::kernel(const int dim, const AutodiffGradPt &r, const AutodiffScalarGrad &) const
		{
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(dim);
			assert(r.size() == dim);

			double mu, nu, lambda;
			// per body lame parameter dont work here!
			params_.lambda_mu(0, 0, 0, 0, 0, 0, 0, lambda, mu);

			// convert to nu!
			nu = lambda / (lambda * (dim - 1) + 2 * mu);

			if (dim == 2)
			{
				res(0) = 1. / (8 * M_PI * mu * (1 - nu)) * ((3 - 4 * nu) * log(1. / r.norm()) + r(0) * r(0) / r.squaredNorm());
				res(1) = 1. / (8 * M_PI * mu * (1 - nu)) * r(0) * r(1) / r.squaredNorm();
			}
			else if (dim == 3)
			{
				res(0) = 1. / (16 * M_PI * mu * (1 - nu)) * ((3 - 4 * nu) / r.norm() + r(0) * r(0) / r.norm() / r.squaredNorm());
				res(1) = 1. / (16 * M_PI * mu * (1 - nu)) * r(0) * r(1) / r.norm() / r.squaredNorm();
				res(2) = 1. / (16 * M_PI * mu * (1 - nu)) * r(0) * r(2) / r.norm() / r.squaredNorm();
			}
			else
				assert(false);

			return res;
		}

		void LinearElasticity::compute_stress_grad_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const
		{
			double lambda, mu;
			params_.lambda_mu(local_pts, global_pts, el_id, lambda, mu);

			stress = mu * (grad_u_i + grad_u_i.transpose()) + lambda * grad_u_i.trace() * Eigen::MatrixXd::Identity(size(), size());
			result = mu * (mat + mat.transpose()) + lambda * mat.trace() * Eigen::MatrixXd::Identity(size(), size());
		}

		void LinearElasticity::compute_stress_grad_multiply_stress(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const
		{
			double lambda, mu;
			params_.lambda_mu(local_pts, global_pts, el_id, lambda, mu);

			stress = mu * (grad_u_i + grad_u_i.transpose()) + lambda * grad_u_i.trace() * Eigen::MatrixXd::Identity(size(), size());
			result = mu * (stress + stress.transpose()) + lambda * stress.trace() * Eigen::MatrixXd::Identity(size(), size());
		}

		void LinearElasticity::compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const
		{
			dstress_dmu = grad_u_i.transpose() + grad_u_i;
			dstress_dlambda = grad_u_i.trace() * Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols());
		}

		std::map<std::string, Assembler::ParamFunc> LinearElasticity::parameters() const
		{
			std::map<std::string, ParamFunc> res;
			const auto &params = lame_params();
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
	} // namespace assembler
} // namespace polyfem
