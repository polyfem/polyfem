#include <polyfem/IncompressibleLinElast.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{
	void IncompressibleLinearElasticityDispacement::set_parameters(const json &params)
	{
		set_size(params["size"]);
		assert(size_ == 2 || size_ == 3);

		if(params.count("young")) {
			lambda() = convert_to_lambda(size_ == 3, params["young"], params["nu"]);
			mu() = convert_to_mu(params["young"], params["nu"]);
		} else if(params.count("E")) {
			lambda() = convert_to_lambda(size_ == 3, params["E"], params["nu"]);
			mu() = convert_to_mu(params["E"], params["nu"]);
		}
		else
		{
			lambda() = params["lambda"];
			mu() = params["mu"];
		}

		// std::cout<<mu_<<std::endl;
		// std::cout<<lambda_<<std::endl;
	}

	void IncompressibleLinearElasticityDispacement::set_size(const int size)
	{
		size_ = size;
		assert(size_ == 2 || size_ == 3);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	IncompressibleLinearElasticityDispacement::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// 2mu (epsi : epsj)
		assert(size_ == 2 || size_ == 3);

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();

		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> epsi(size(), size());
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> epsj(size(), size());


		for(long p = 0; p < gradi.rows(); ++p)
		{
			for(int di = 0; di < size(); ++di)
			{
				epsi.setZero();
				epsi.row(di) = gradi.row(p);
				epsi = ((epsi + epsi.transpose()) / 2).eval();
				for(int dj = 0; dj < size(); ++dj)
				{
					epsj.setZero();
					epsj.row(dj) = gradj.row(p);
					epsj = ((epsj + epsj.transpose()) / 2.0).eval();

					res(dj*size() + di) += 2 * mu_ * (epsi.array() * epsj.array()).sum() * da(p);
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


	void IncompressibleLinearElasticityDispacement::compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(bs, gbs, local_pts, displacement, size()*size(), stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::MatrixXd tmp = stress;
			return Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size()*size());
		});
	}

	void IncompressibleLinearElasticityDispacement::compute_von_mises_stresses(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::Matrix<double, 1,1> res; res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void IncompressibleLinearElasticityDispacement::assign_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		assert(size_ == 2 || size_ == 3);
		all.resize(local_pts.rows(), all_size);
		assert(displacement.cols() == 1);

		Eigen::MatrixXd displacement_grad(size(), size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for(long p = 0; p < local_pts.rows(); ++p)
		{
			displacement_grad.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.grad.rows() == local_pts.rows());
				assert(loc_val.grad.cols() == size());

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index*size() + d);
					}
				}
			}

			displacement_grad = (displacement_grad * vals.jac_it[p]).eval();

			const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose())/2;
			const Eigen::MatrixXd stress = 2 * mu_ * strain + lambda_ * strain.trace() * Eigen::MatrixXd::Identity(size(), size());

			all.row(p) = fun(stress);
		}
	}




	void IncompressibleLinearElasticityMixed::set_parameters(const json &params)
	{
		set_size(params["size"]);
		assert(size_ == 2 || size_ == 3);

		if(params.count("young")) {
			lambda() = convert_to_lambda(size_ == 3, params["young"], params["nu"]);
			mu() = convert_to_mu(params["young"], params["nu"]);
		} else if(params.count("E")) {
			lambda() = convert_to_lambda(size_ == 3, params["E"], params["nu"]);
			mu() = convert_to_mu(params["E"], params["nu"]);
		}
		else
		{
			lambda() = params["lambda"];
			mu() = params["mu"];
		}

		// std::cout<<mu_<<std::endl;
		// std::cout<<lambda_<<std::endl;
	}

	void IncompressibleLinearElasticityMixed::set_size(const int size)
	{
		size_ = size;
		assert(size_ == 2 || size_ == 3);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityMixed::assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const
	{
		// (psii : div phij)  = -psii * gradphij
		assert(size_ == 2 || size_ == 3);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows()*cols());
		res.setZero();

		const Eigen::MatrixXd &psii 	= psi_vals.basis_values[i].val;
		const Eigen::MatrixXd &gradphij = phi_vals.basis_values[j].grad_t_m;
		assert(psii.size() == gradphij.rows());
		assert(gradphij.cols() == rows());

		for (int k = 0; k < gradphij.rows(); ++k) {
			res -= psii(k) * gradphij.row(k) * da(k);
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



	void IncompressibleLinearElasticityPressure::set_parameters(const json &params)
	{
		const int size = params["size"];
		assert(size == 2 || size == 3);

		if(params.count("young")) {
			lambda() = convert_to_lambda(size == 3, params["young"], params["nu"]);
			mu() = convert_to_mu(params["young"], params["nu"]);
		} else if(params.count("E")) {
			lambda() = convert_to_lambda(size == 3, params["E"], params["nu"]);
			mu() = convert_to_mu(params["E"], params["nu"]);
		}
		else
		{
			lambda() = params["lambda"];
			mu() = params["mu"];
		}

		// std::cout<<mu_<<std::endl;
		// std::cout<<lambda_<<std::endl;
	}

	Eigen::Matrix<double, 1, 1>
	IncompressibleLinearElasticityPressure::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// -1/lambda phi_ * phi_j

		const Eigen::MatrixXd &phii = vals.basis_values[i].val;
		const Eigen::MatrixXd &phij = vals.basis_values[j].val;

		double res = (phii.array() * phij.array() * da.array()).sum();
		res *= -1./lambda_;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}
}
