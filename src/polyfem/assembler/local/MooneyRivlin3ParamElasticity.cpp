#include "MooneyRivlin3ParamElasticity.hpp"

namespace polyfem::assembler
{
	MooneyRivlin3ParamElasticity::MooneyRivlin3ParamElasticity()
		: c1_("c1"), c2_("c2"), c3_("c3"), d1_("d1")
	{
	}

	void MooneyRivlin3ParamElasticity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		c1_.add_multimaterial(index, params, units.stress());
		c2_.add_multimaterial(index, params, units.stress());
		c3_.add_multimaterial(index, params, units.stress());
		d1_.add_multimaterial(index, params, units.stress());
	}

	std::map<std::string, Assembler::ParamFunc> MooneyRivlin3ParamElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &c1 = this->c1();
		const auto &c2 = this->c2();
		const auto &c3 = this->c3();
		const auto &d1 = this->d1();

		res["c1"] = [&c1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c1(p, t, e);
		};

		res["c2"] = [&c2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c2(p, t, e);
		};

		res["c3"] = [&c3](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c3(p, t, e);
		};

		res["d1"] = [&d1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return d1(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler