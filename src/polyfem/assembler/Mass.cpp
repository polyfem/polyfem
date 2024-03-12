#include "Mass.hpp"

namespace polyfem::assembler
{
	FlatMatrixNd Mass::assemble(const LinearAssemblerData &data) const
	{
		double tmp = 0;

		// loop over quadrature points
		for (int q = 0; q < data.da.size(); ++q)
		{
			const double rho = density_(data.vals.quadrature.points.row(q), data.vals.val.row(q), data.t, data.vals.element_id);
			// phi_i * phi_j weighted by quadrature weights
			tmp += rho * data.vals.basis_values[data.i].val(q) * data.vals.basis_values[data.j].val(q) * data.da(q);
		}

		FlatMatrixNd res(codomain_size() * codomain_size(), 1);
		res.setZero();
		for (int i = 0; i < codomain_size(); ++i)
			res(i * codomain_size() + i) = tmp;

		return res;
	}

	VectorNd Mass::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(false);
		VectorNd result;

		return result;
	}

	void Mass::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(1 <= codomain_size() && codomain_size() <= 3);
		density_.add_multimaterial(index, params, units.density());
	}

	std::map<std::string, Assembler::ParamFunc> Mass::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["rho"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return this->density_(uv, p, t, e); };

		return res;
	}

} // namespace polyfem::assembler
