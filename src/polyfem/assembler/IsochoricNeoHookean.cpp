#include "IsochoricNeoHookean.hpp"

namespace polyfem::assembler
{
	IsochoricNeoHookean::IsochoricNeoHookean()
	{
	}

	void IsochoricNeoHookean::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		params_.add_multimaterial(index, params, false, units.stress());
	}

	std::map<std::string, Assembler::ParamFunc> IsochoricNeoHookean::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &params = params_;
		const int size = this->size();

		res["lambda"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);
			return lambda;
		};

		res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);
			return mu;
		};

		res["E"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);

			if (size == 3)
				return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
			else
				return 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
		};

		res["nu"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);

			if (size == 3)
				return lambda / (2.0 * (lambda + mu));
			else
				return lambda / (lambda + 2.0 * mu);
		};

		return res;
	}
} // namespace polyfem::assembler