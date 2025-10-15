#include "MooneyRivlinElasticity.hpp"

namespace polyfem::assembler
{
	MooneyRivlinElasticity::MooneyRivlinElasticity()
		: c1_("c1"), c2_("c2"), k_("k")
	{
	}

	void MooneyRivlinElasticity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		c1_.add_multimaterial(index, params, units.stress());
		c2_.add_multimaterial(index, params, units.stress());
		k_.add_multimaterial(index, params, units.stress());
	}

	std::map<std::string, Assembler::ParamFunc> MooneyRivlinElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &c1 = this->c1();
		const auto &c2 = this->c2();
		const auto &k = this->k();

		res["c1"] = [&c1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c1(p, t, e);
		};

		res["c2"] = [&c2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c2(p, t, e);
		};

		res["k"] = [&k](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler