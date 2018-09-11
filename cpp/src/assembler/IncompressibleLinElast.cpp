#include <polyfem/IncompressibleLinElast.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{
	void IncompressibleLinearElasticityVelocity::set_parameters(const json &params)
	{
		set_size(params["size"]);

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

	}

	void IncompressibleLinearElasticityVelocity::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	IncompressibleLinearElasticityVelocity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// 2mu (gradi : gradj)  = 2mu <gradi, gradj> * Id

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();

		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;
		double dot = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			dot += gradi.row(k).dot(gradj.row(k)) * da(k);
		}

		dot *= 2 * mu_;

		for(int d = 0; d < size(); ++d)
			res(d*size() + d) = dot;

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());
		assert(false);
		return res;
	}




	void IncompressibleLinearElasticityMixed::set_parameters(const json &params)
	{
		set_size(params["size"]);

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
	}

	void IncompressibleLinearElasticityMixed::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityMixed::assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const
	{
		// (psii : div phij)  = -psii * gradphij

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
	}

	Eigen::Matrix<double, 1, 1>
	IncompressibleLinearElasticityPressure::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// -1/lambad phi_ * phi_j

		const Eigen::MatrixXd &phii = vals.basis_values[i].val;
		const Eigen::MatrixXd &phij = vals.basis_values[j].val;
		double res = 0;
		for (int k = 0; k < phii.size(); ++k) {
			res += phii(k)* phij(k) * da(k);
		}

		res *= -1./lambda_;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}
}
