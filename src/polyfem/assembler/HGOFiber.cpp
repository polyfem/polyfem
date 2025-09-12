#include "HGOFiber.hpp"

namespace polyfem::assembler
{
	HGOFiber::HGOFiber()
		: k1_("k1"), k2_("k2")
	{
	}

	void HGOFiber::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		GenericFiber<HGOFiber>::add_multimaterial(index, params, units);

		k1_.add_multimaterial(index, params, units.stress());
		k2_.add_multimaterial(index, params, "");
	}

	std::map<std::string, Assembler::ParamFunc> HGOFiber::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &k1 = this->k1_;
		const auto &k2 = this->k2_;

		res["k1"] = [&k1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k1(p, t, e);
		};

		res["k2"] = [&k2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k2(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler