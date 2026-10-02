#include "Helmholtz.hpp"
#include <polyfem/utils/Bessel.hpp>

namespace polyfem::assembler
{
	namespace
	{
		inline RowVectorNd zero_point(const int dim)
		{
			assert(dim == 2 || dim == 3);
			return RowVectorNd::Zero(dim);
		}
	} // namespace

	Helmholtz::Helmholtz()
		: k_("k")
	{
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	Helmholtz::assemble(const LinearAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;

		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k)
		{
			res += gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}

		for (int k = 0; k < gradi.rows(); ++k)
		{
			const double tmp = k_(data.vals.val.row(k), data.t, data.vals.element_id);
			res -= data.vals.basis_values[data.i].val(k) * data.vals.basis_values[data.j].val(k) * data.da[k] * tmp * tmp;
		}

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	VectorNd Helmholtz::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		const RowVectorNd val = zero_point(pt(0).getHessian().rows());

		const double tmp = k_(val, 0, 0);
		result(0) = pt(0).getHessian().trace() + tmp * tmp * pt(0).getValue();
		return result;
	}

	void Helmholtz::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		k_.add_multimaterial(index, params, "", root_path);
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Helmholtz::kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

		const RowVectorNd val = zero_point(dim);
		const double tmp = k_(val, 0, 0);

		if (dim == 2)
			res(0) = -0.25 * utils::bessy0(tmp * r);
		else if (dim == 3)
			res(0) = 0.25 * cos(tmp * r) / (M_PI * r);
		else
			assert(false);

		return res;
	}

	std::map<std::string, Assembler::ParamFunc> Helmholtz::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["k"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return this->k_(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
