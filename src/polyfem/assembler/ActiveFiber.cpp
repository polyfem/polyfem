#include "ActiveFiber.hpp"

namespace polyfem::assembler
{

	ActiveFiber::ActiveFiber()
		: Tmax_("Tmax"), activation_("activation")
	{
	}

	void ActiveFiber::add_multimaterial(const int index, const json &params, const Units &units)
	{
		GenericFiber::add_multimaterial(index, params, units);

		Tmax_.add_multimaterial(index, params, units.stress());
		activation_.add_multimaterial(index, params, "");
	}

	std::map<std::string, Assembler::ParamFunc> ActiveFiber::parameters() const
	{
		std::map<std::string, ParamFunc> res = GenericFiber<ActiveFiber>::parameters();

		const auto &Tmax = this->Tmax_;
		const auto &activation = this->activation_;

		res["Tmax"] = [&Tmax](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return Tmax(p, t, e);
		};

		res["activation"] = [&activation](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return activation(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
