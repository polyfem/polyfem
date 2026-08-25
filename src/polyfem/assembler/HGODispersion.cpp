#include "HGODispersion.hpp"

namespace polyfem::assembler
{
	HGODispersion::HGODispersion()
		: k1_("k1"), k2_("k2"), kappa_("kappa")
	{
	}

	void HGODispersion::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		GenericFiber<HGODispersion>::add_multimaterial(index, params, units, root_path);

		k1_.add_multimaterial(index, params, units.stress(), root_path);
		k2_.add_multimaterial(index, params, "", root_path);
		kappa_.add_multimaterial(index, params, "", root_path); // dimensionless; absent => 0
		k_chi_ = params.value("k_chi", 100.0);                  // absent => 100 (manuscript)
	}

	std::map<std::string, Assembler::ParamFunc> HGODispersion::parameters() const
	{
		std::map<std::string, ParamFunc> res = GenericFiber<HGODispersion>::parameters();

		const auto &k1 = this->k1_;
		const auto &k2 = this->k2_;
		const auto &kappa = this->kappa_;

		res["k1"] = [&k1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k1(p, t, e);
		};

		res["k2"] = [&k2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k2(p, t, e);
		};

		res["kappa"] = [&kappa](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return kappa(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
