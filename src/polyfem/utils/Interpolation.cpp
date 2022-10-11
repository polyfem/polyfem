#include "Interpolation.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
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
		else if (type == "piecewise_cubic")
			res = std::make_shared<PiecewiseCubicInterpolation>();
		else
			log_and_throw_error("Usupported interpolation type {}", type);

		assert(res != nullptr);
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
			log_and_throw_error("PiecewiseInterpolation points must be an array");
		if (!params["values"].is_array())
			log_and_throw_error("PiecewiseInterpolation values must be an array");

		points_ = params["points"].get<std::vector<double>>();
		assert(std::is_sorted(points_.begin(), points_.end()));
		values_ = params["values"].get<std::vector<double>>();

		assert(params.contains("extend"));
		extend_ = params["extend"];
	}

	double PiecewiseInterpolation::eval(const double t) const
	{
		if (t < points_.front() || t >= points_.back())
			return extend(t);

		for (size_t i = 0; i < points_.size() - 1; ++i)
		{
			if (t >= points_[i] && t < points_[i + 1])
			{
				return eval_piece(t, i);
			}
		}

		assert(points_.size() == 1);
		return values_[0];
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

	double PiecewiseInterpolation::dy_dt(const double t) const
	{
		assert(t >= points_.front() && t <= points_.back());

		for (size_t i = 0; i < points_.size() - 1; ++i)
		{
			if (t >= points_[i] && t <= points_[i + 1])
			{
				return dy_dt_piece(t, i);
			}
		}

		assert(points_.size() == 1);
		return 0;
	}

	double PiecewiseLinearInterpolation::eval_piece(const double t, const int i) const
	{
		const double alpha = (t - points_[i]) / (points_[i + 1] - points_[i]);
		return (values_[i + 1] - values_[i]) * alpha + values_[i];
	}

	double PiecewiseLinearInterpolation::dy_dt_piece(const double t, const int i) const
	{
		return (values_[i + 1] - values_[i]) / (points_[i + 1] - points_[i]);
	}

	void PiecewiseCubicInterpolation::init(const json &params)
	{
		PiecewiseInterpolation::init(params);

		// N+1 points ⟹ N cubic functions of the form fᵢ(x) = aᵢt³ + bᵢt² + cᵢt + dᵢ
		//            ⟹ 4N unknowns
		const int N = points_.size() - 1;
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4 * N, 4 * N);
		Eigen::VectorXd b = Eigen::VectorXd::Zero(4 * N);

		// 2N equations: fᵢ(tᵢ) = yᵢ and fᵢ(tᵢ₊₁) = yᵢ₊₁
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				// aᵢt³ + bᵢt² + cᵢt + dᵢ = yᵢ (yᵢ₊₁) at tᵢ (tᵢ₊₁)
				const double t = points_[i + j];

				A(2 * i + j, 4 * i + 0) = pow(t, 3);
				A(2 * i + j, 4 * i + 1) = pow(t, 2);
				A(2 * i + j, 4 * i + 2) = t;
				A(2 * i + j, 4 * i + 3) = 1;

				b(2 * i + j) = values_[i + j];
			}
		}
		int offset = 2 * N;

		// N - 1 equations: fᵢ'(tᵢ) = fᵢ₊₁'(tᵢ₊₁) for i = 1, ... , N - 1
		// fᵢ'(t) = 3aᵢt² + 2bᵢt + cᵢ
		for (int i = 0; i < N - 1; i++)
		{
			// 3aᵢt² + 2bᵢt + cᵢ - 3aᵢ₊₁t² - 2bᵢ₊₁t - cᵢ₊₁ = 0 at tᵢ₊₁
			const double t = points_[i + 1];

			A(offset + i, 4 * i + 0) = 3 * pow(t, 2);
			A(offset + i, 4 * i + 1) = 2 * t;
			A(offset + i, 4 * i + 2) = 1;

			A.block<1, 3>(offset + i, 4 * (i + 1)) = -A.block<1, 3>(offset + i, 4 * i);
		}
		offset += N - 1;

		// N - 1 equations: fᵢ"(tᵢ) = fᵢ₊₁"(tᵢ₊₁) for i = 1, ... , N - 1
		// fᵢ"(t) = 6aᵢt + 2bᵢ
		for (int i = 0; i < N - 1; i++)
		{
			// 6aᵢt + 2bᵢ - 6aᵢ₊₁t - 2bᵢ₊₁ = 0 at tᵢ₊₁
			const double t = points_[i + 1];

			A(offset + i, 4 * i + 0) = 6 * t;
			A(offset + i, 4 * i + 1) = 2;

			A.block<1, 2>(offset + i, 4 * (i + 1)) = -A.block<1, 2>(offset + i, 4 * i);
		}
		offset += N - 1;

		// Boundary Conditions:
		if (extend_ == Extend::CONSTANT)
		{
			// Custom Spline
			// f₀'(t₀) = 3a₀t₀² + 2b₀t + c₀ = 0
			A(offset, 0) = 3 * pow(points_[0], 2);
			A(offset, 1) = 2 * points_[0];
			A(offset, 2) = 1;
			offset++;

			// fₙ₋₁"(tₙ) = 6aₙ₋₁tₙ + 2bₙ₋₁ = 0
			A(offset, 4 * (N - 1) + 0) = 3 * pow(points_[N], 2);
			A(offset, 4 * (N - 1) + 1) = 2 * points_[N];
			A(offset, 4 * (N - 1) + 2) = 1;
			offset++;
		}
		else if (extend_ == Extend::EXTRAPOLATE)
		{
			// Natural Spline
			// f₀"(t₀) = 6a₀t₀ + 2b₀ = 0
			A(offset, 0) = 6 * points_[0];
			A(offset, 1) = 2;
			offset++;

			// fₙ₋₁"(tₙ) = 6aₙ₋₁tₙ + 2bₙ₋₁ = 0
			A(offset, 4 * (N - 1) + 0) = 6 * points_[N];
			A(offset, 4 * (N - 1) + 1) = 2;
			offset++;
		}
		else
		{
			// Periodic Spline
			assert(extend_ == Extend::REPEAT || extend_ == Extend::REPEAT_OFFSET);

			// f₀'(t₀) = fₙ₋₁'(tₙ) and f₀"(t₀) = fₙ₋₁"(tₙ)
			A(offset, 0) = 3 * pow(points_[0], 2);
			A(offset, 1) = 2 * points_[0];
			A(offset, 2) = 1;
			A(offset, 4 * (N - 1) + 0) = -3 * pow(points_[N], 2);
			A(offset, 4 * (N - 1) + 1) = -2 * points_[N];
			A(offset, 4 * (N - 1) + 2) = -1;
			offset++;

			A(offset, 0) = 6 * points_[0];
			A(offset, 1) = 2;
			A(offset, 4 * (N - 1) + 0) = -6 * points_[N];
			A(offset, 4 * (N - 1) + 1) = -2;
			offset++;
		}

		assert(offset == 4 * N);

		// Solve the system of linear equations
		coeffs_ = A.lu().solve(b);
		// unflatten coeffs so each row corresponds to a cubic function
		coeffs_ = utils::unflatten(coeffs_, 4);
		assert(coeffs_.rows() == N);
	}

	double PiecewiseCubicInterpolation::eval_piece(const double t, const int i) const
	{
		// fᵢ(t) = aᵢt³ + bᵢt² + cᵢt + dᵢ
		return ((coeffs_(i, 0) * t + coeffs_(i, 1)) * t + coeffs_(i, 2)) * t + coeffs_(i, 3);
	}

	double PiecewiseCubicInterpolation::dy_dt_piece(const double t, const int i) const
	{
		// fᵢ'(t) = 3aᵢt² + 2bᵢt + cᵢ
		return (3 * coeffs_(i, 0) * t + 2 * coeffs_(i, 1)) * t + coeffs_(i, 2);
	}

} // namespace polyfem::utils