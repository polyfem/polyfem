#include "Interpolation.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::utils
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		PiecewiseInterpolation::Extend,
		{{PiecewiseInterpolation::Extend::CONSTANT, "constant"}, // also default
		 {PiecewiseInterpolation::Extend::EXTRAPOLATE, "extrapolate"},
		 {PiecewiseInterpolation::Extend::REPEAT, "repeat"},
		 {PiecewiseInterpolation::Extend::REPEAT_OFFSET, "repeat_offset"}});

	std::shared_ptr<Interpolation> Interpolation::build(const json &params)
	{
		const std::string type = params["type"];
		std::shared_ptr<Interpolation> res = nullptr;

		if (type == "none")
			res = std::make_shared<NoInterpolation>();
		else if (type == "linear")
			res = std::make_shared<LinearInterpolation>();
		else if (type == "linear_ramp")
			res = std::make_shared<LinearRamp>();
		else if (type == "piecewise_constant")
			res = std::make_shared<PiecewiseConstantInterpolation>();
		else if (type == "piecewise_linear")
			res = std::make_shared<PiecewiseLinearInterpolation>();
		// else if (type == "piecewise_cubic")
		// 	res = std::make_shared<PiecewiseCubicInterpolation>();
		else
			logger().error("Usupported interpolation type {}", type);

		if (res)
			res->init(params);

		return res;
	}

	void LinearRamp::init(const json &params)
	{
		to_ = params["to"];
		from_ = params.value("from", 0.0);
	}

	double LinearRamp::eval(const double t) const
	{
		if (t >= to_)
			return to_;

		if (t <= from_)
			return 0;

		return t - from_;
	}

	void PiecewiseInterpolation::init(const json &params)
	{
		if (!params["points"].is_array())
			log_and_throw_error(fmt::format("PiecewiseInterpolation points must be an array"));
		if (!params["values"].is_array())
			log_and_throw_error(fmt::format("PiecewiseInterpolation values must be an array"));

		points_ = params["points"].get<std::vector<double>>();
		assert(std::is_sorted(points_.begin(), points_.end()));
		values_ = params["values"].get<std::vector<double>>();

		assert(params.contains("extend"));
		extend_ = params["extend"];
	}

	double PiecewiseConstantInterpolation::eval(const double t) const
	{
		if (t < points_.front() || t >= points_.back())
			return extend(t);

		for (size_t i = 0; i < points_.size() - 1; ++i)
		{
			if (t >= points_[i] && t < points_[i + 1])
			{
				return values_[i];
			}
		}

		assert(points_.size() == 1);
		return values_[0];
	}

	double PiecewiseLinearInterpolation::eval(const double t) const
	{
		if (t < points_.front() || t >= points_.back())
			return extend(t);

		for (size_t i = 0; i < points_.size() - 1; ++i)
		{
			if (t >= points_[i] && t < points_[i + 1])
			{
				const double alpha = ((t - points_[i]) / (points_[i + 1] - points_[i]));
				return (values_[i + 1] - values_[i]) * alpha + values_[i];
			}
		}

		assert(points_.size() == 1);
		return values_[0];
	}

	double PiecewiseLinearInterpolation::dy_dt(const double t) const
	{
		assert(t >= points_.front() && t <= points_.back());

		for (size_t i = 0; i < points_.size() - 1; ++i)
		{
			// NOTE: technically at t ∈ points dy/dt is undefined, but we just use the previous intervals value.
			if (t >= points_[i] && t <= points_[i + 1])
			{
				return (values_[i + 1] - values_[i]) / (points_[i + 1] - points_[i]);
			}
		}

		assert(points_.size() == 1);
		return 0;
	}

	double PiecewiseInterpolation::extend(const double t) const
	{
		assert(points_.size() == values_.size());
		const double t0 = points_.front(), tn = points_.back();
		assert(t < t0 || t >= tn);
		const double y0 = values_.front(), yn = values_.back();

		double offset = 0;
		switch (extend_)
		{
		case Extend::CONSTANT:
			return (t < t0) ? y0 : yn;

		case Extend::EXTRAPOLATE:
		{
			if (t < t0)
				return dy_dt(t0) * (t - t0) + y0;
			else
				return dy_dt(tn) * (t - tn) + yn;
		}

		case Extend::REPEAT_OFFSET:
			offset = floor((t - t0) / (tn - t0)) * (yn - y0);
			// NOTE: fallthrough
			[[fallthrough]]; // suppress warning

		case Extend::REPEAT:
		{
			double t_mod = std::fmod(t - t0, tn - t0);
			t_mod += (t_mod < 0) ? tn : t0;
			return eval(t_mod) + offset;
		}

		default:
			assert(false);
			return 0;
		}
	}
} // namespace polyfem::utils