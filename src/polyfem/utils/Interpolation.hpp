#pragma once

#include <polyfem/Common.hpp>

#include <memory>
#include <vector>

namespace polyfem::utils
{
	class Interpolation
	{
	public:
		virtual ~Interpolation() {}
		virtual double eval(const double t) const = 0;
		virtual void init(const json &params) {}

		static std::shared_ptr<Interpolation> build(const json &params);
	};

	class NoInterpolation : public Interpolation
	{
	public:
		double eval(const double t) const override { return 1; };
	};

	class LinearInterpolation : public Interpolation
	{
	public:
		double eval(const double t) const override { return t; }
	};

	class LinearRamp : public Interpolation
	{
	public:
		double eval(const double t) const override;
		void init(const json &params) override;

	private:
		double to_;
		double from_;
	};

	class PiecewiseInterpolation : public Interpolation
	{
	public:
		void init(const json &params) override;
		double extend(const double t) const;

	public:
		std::vector<double> points_;
		std::vector<double> values_;

		enum class Extend
		{
			CONSTANT,
			EXTRAPOLATE,
			REPEAT,
			REPEAT_OFFSET
		};
		Extend extend_;

	protected:
		virtual double dy_dt(const double t) const = 0;
	};

	class PiecewiseConstantInterpolation : public PiecewiseInterpolation
	{
	public:
		double eval(const double t) const override;

	protected:
		double dy_dt(const double t) const override { return 0; }
	};

	class PiecewiseLinearInterpolation : public PiecewiseInterpolation
	{
	public:
		double eval(const double t) const override;

	protected:
		double dy_dt(const double t) const override;
	};

	// class PiecewiseCubicInterpolation : public PiecewiseInterpolation
	// {
	// public:
	// 	double eval(const double t) const override;
	// };

} // namespace polyfem::utils