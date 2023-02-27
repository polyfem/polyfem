#include "IncompressibleLinElast.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::assembler
{
	using namespace basis;

	void IncompressibleLinearElasticityDispacement::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		params_.add_multimaterial(index, params, size_ == 3);

		// std::cout<<mu_<<std::endl;
		// std::cout<<lambda_<<std::endl;
	}

	void IncompressibleLinearElasticityDispacement::set_size(const int size)
	{
		size_ = size;
		assert(size_ == 2 || size_ == 3);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	IncompressibleLinearElasticityDispacement::assemble(const LinearAssemblerData &data) const
	{
		// 2mu (epsi : epsj)
		assert(size_ == 2 || size_ == 3);

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size());
		res.setZero();

		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> epsi(size(), size());
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> epsj(size(), size());

		for (long p = 0; p < gradi.rows(); ++p)
		{

			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.vals.element_id, lambda, mu);

			for (int di = 0; di < size(); ++di)
			{
				epsi.setZero();
				epsi.row(di) = gradi.row(p);
				epsi = ((epsi + epsi.transpose()) / 2).eval();
				for (int dj = 0; dj < size(); ++dj)
				{
					epsj.setZero();
					epsj.row(dj) = gradj.row(p);
					epsj = ((epsj + epsj.transpose()) / 2.0).eval();

					res(dj * size() + di) += 2 * mu * (epsi.array() * epsj.array()).sum() * data.da(p);
				}
			}
		}

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityDispacement::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());
		assert(false);
		return res;
	}

	void IncompressibleLinearElasticityDispacement::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const ElasticityTensorType &type, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), type, stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
			return Eigen::MatrixXd(a);
		});
	}

	void IncompressibleLinearElasticityDispacement::compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, ElasticityTensorType::CAUCHY, stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::Matrix<double, 1, 1> res;
			res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void IncompressibleLinearElasticityDispacement::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		assert(size_ == 2 || size_ == 3);
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

	void IncompressibleLinearElasticityMixed::set_size(const int size)
	{
		size_ = size;
		assert(size_ == 2 || size_ == 3);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityMixed::assemble(const MixedAssemblerData &data) const
	{
		// (psii : div phij)  = -psii * gradphij
		assert(size_ == 2 || size_ == 3);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows() * cols());
		res.setZero();

		const Eigen::MatrixXd &psii = data.psi_vals.basis_values[data.i].val;
		const Eigen::MatrixXd &gradphij = data.phi_vals.basis_values[data.j].grad_t_m;
		assert(psii.size() == gradphij.rows());
		assert(gradphij.cols() == rows());

		for (int k = 0; k < gradphij.rows(); ++k)
		{
			res -= psii(k) * gradphij.row(k) * data.da(k);
		}

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityMixed::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == rows());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows());
		assert(false);
		return res;
	}

	void IncompressibleLinearElasticityPressure::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		params_.add_multimaterial(index, params, size_ == 3);
	}

	Eigen::Matrix<double, 1, 1>
	IncompressibleLinearElasticityPressure::assemble(const LinearAssemblerData &data) const
	{
		// -1/lambda phi_ * phi_j

		const Eigen::MatrixXd &phii = data.vals.basis_values[data.i].val;
		const Eigen::MatrixXd &phij = data.vals.basis_values[data.j].val;

		double res = 0;

		for (long p = 0; p < data.da.size(); ++p)
		{
			double lambda, mu;
			params_.lambda_mu(data.vals.quadrature.points.row(p), data.vals.val.row(p), data.vals.element_id, lambda, mu);

			res += -phii(p) * phij(p) * data.da(p) / lambda;
		}

		// double res = (phii.array() * phij.array() * da.array()).sum();
		// res *= -1. / lambda;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}
} // namespace polyfem::assembler
