#include "VolumePenalty.hpp"

namespace polyfem::assembler
{
	VolumePenalty::VolumePenalty()
		: k_("k")
	{
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