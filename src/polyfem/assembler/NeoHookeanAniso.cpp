#include "NeoHookeanAniso.hpp"

namespace polyfem::assembler
{
	NeoHookeanAniso::NeoHookeanAniso()
		: c_("c"), k1_("k1"), k2_("k2"), d1_("d1")
	{
	}

	void NeoHookeanAniso::add_multimaterial(const int index, const json &params, const Units &units)
	{
		c_.add_multimaterial(index, params, units.stress());
		k1_.add_multimaterial(index, params, units.stress());
		k2_.add_multimaterial(index, params, "");
		d1_.add_multimaterial(index, params, units.stress());
		a1_.add_multimaterial(index, params["a1"], "");
		a2_.add_multimaterial(index, params["a2"], "");
	}

	std::map<std::string, Assembler::ParamFunc> NeoHookeanAniso::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &c = this->c();
		const auto &k1 = this->k1();
		const auto &k2 = this->k2();
		const auto &d1 = this->d1();

		res["c"] = [&c](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c(p, t, e);
		};

		res["k1"] = [&k1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k1(p, t, e);
		};

		res["k2"] = [&k2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k2(p, t, e);
		};

		res["d1"] = [&d1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return d1(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler