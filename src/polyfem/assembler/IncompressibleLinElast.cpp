#include "IncompressibleLinElast.hpp"

namespace polyfem::assembler
{
	using namespace basis;

	void IncompressibleLinearElasticityDispacement::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		params_.add_multimaterial(index, params, size() == 3, units.stress());

		// std::cout<<mu_<<std::endl;
		// std::cout<<lambda_<<std::endl;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	IncompressibleLinearElasticityDispacement::assemble(const LinearAssemblerData &data) const
	{
		// 2mu (epsi : epsj)
		assert(size() == 2 || size() == 3);

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

	void IncompressibleLinearElasticityDispacement::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		assert(size() == 2 || size() == 3);
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

	std::map<std::string, Assembler::ParamFunc> IncompressibleLinearElasticityDispacement::parameters() const
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

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityMixed::assemble(const MixedAssemblerData &data) const
	{
		// (psii : div phij)  = -psii * gradphij
		assert(size() == 2 || size() == 3);
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

	void IncompressibleLinearElasticityPressure::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		params_.add_multimaterial(index, params, size() == 3, units.stress());
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
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
