#include "Mass.hpp"

namespace polyfem::assembler
{
	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> Mass::assemble(const LinearAssemblerData &data) const
	{
		double tmp = 0;
		for (int q = 0; q < data.da.size(); ++q)
		{
			const double rho = density_(data.vals.quadrature.points.row(q), data.vals.val.row(q), data.vals.element_id);
			tmp += rho * data.vals.basis_values[data.i].val(q) * data.vals.basis_values[data.j].val(q) * data.da(q);
		}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size(), 1);
		res.setZero();
		for (int i = 0; i < size(); ++i)
			res(i * size() + i) = tmp;

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> Mass::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(false);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> result;

		return result;
	}

	void Mass::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size_ == 1 || size_ == 2 || size_ == 3);

		density_.add_multimaterial(index, params, units.density());
	}

	std::map<std::string, Assembler::ParamFunc> Mass::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["rho"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return this->density_(uv, p, e); };

		return res;
	}

} // namespace polyfem::assembler
