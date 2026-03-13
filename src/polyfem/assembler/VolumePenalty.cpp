#include "VolumePenalty.hpp"

#include <polyfem/autogen/elastic_energies/VolumePenalty2d.hpp>
#include <polyfem/autogen/elastic_energies/VolumePenalty3d.hpp>

namespace polyfem::assembler
{
	VolumePenalty::VolumePenalty()
		: k_("k")
	{
		autodiff_type_ = AutodiffType::NONE;
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> VolumePenalty::gradient(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		const double k = k_(p, t, el_id);
		if (size() == 2)
			return autogen::VolumePenalty2d_gradient(p, t, el_id, F, k);
		else
			return autogen::VolumePenalty3d_gradient(p, t, el_id, F, k);
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> VolumePenalty::hessian(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		const double k = k_(p, t, el_id);

		if (size() == 2)
			return autogen::VolumePenalty2d_hessian(p, t, el_id, F, k);
		else
			return autogen::VolumePenalty3d_hessian(p, t, el_id, F, k);
	}

	void VolumePenalty::add_multimaterial(const int index, const json &params, const Units &units)
	{
		k_.add_multimaterial(index, params, units.stress());
	}

	std::map<std::string, Assembler::ParamFunc> VolumePenalty::parameters() const
	{
		std::map<std::string, ParamFunc> res;

		res["k"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k_(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler