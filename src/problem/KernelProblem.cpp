#include <polyfem/State.hpp>
#include <polyfem/KernelProblem.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <iostream>

namespace polyfem
{

	KernelProblem::KernelProblem(const std::string &name)
		: ProblemWithSolution(name)
	{
	}

	VectorNd KernelProblem::eval_fun(const VectorNd &pt, const double t) const
	{
		AutodiffGradPt a_pt(pt.size());

		DiffScalarBase::setVariableCount(pt.size());
		for (long i = 0; i < pt.size(); ++i)
			a_pt(i) = AutodiffScalarGrad(i, pt(i));

		const auto eval = eval_fun(a_pt, t);

		VectorNd res(eval.size());

		for (long i = 0; i < eval.size(); ++i)
			res(i) = eval(i).getValue();

		return res;
	}

	AutodiffGradPt KernelProblem::eval_fun(const AutodiffGradPt &pt, const double tt) const
	{
		AutodiffGradPt res(is_scalar() ? 1 : pt.size());
		for (long i = 0; i < res.size(); ++i)
			res(i) = AutodiffScalarGrad(0);

		const auto &assembler = state->assembler;

		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_kernels_, 0, 1);
		if (pt.size() == 2)
		{
			for (long i = 0; i < t.size(); ++i)
			{
				const auto dx = pt(0) - (0 - kernel_distance_);
				const auto dy = pt(1) - t(i);
				AutodiffGradPt rvect(2);
				rvect << dx, dy;
				res += assembler.kernel(formulation_, 2, rvect, sqrt(dx * dx + dy * dy));
			}

			for (long i = 0; i < t.size(); ++i)
			{
				const auto dx = pt(0) - (1 + kernel_distance_);
				const auto dy = pt(1) - t(i);
				AutodiffGradPt rvect(2);
				rvect << dx, dy;
				res += assembler.kernel(formulation_, 2, rvect, sqrt(dx * dx + dy * dy));
			}

			for (long i = 0; i < t.size(); ++i)
			{
				const auto dx = pt(0) - t(i);
				const auto dy = pt(1) - (0 - kernel_distance_);
				AutodiffGradPt rvect(2);
				rvect << dx, dy;
				res += assembler.kernel(formulation_, 2, rvect, sqrt(dx * dx + dy * dy));
			}

			for (long i = 0; i < t.size(); ++i)
			{
				const auto dx = pt(0) - t(i);
				const auto dy = pt(1) - (1 + kernel_distance_);
				AutodiffGradPt rvect(2);
				rvect << dx, dy;
				res += assembler.kernel(formulation_, 2, rvect, sqrt(dx * dx + dy * dy));
			}
		}
		else if (pt.size() == 3)
		{
			///////////////////X
			for (long i = 0; i < t.size(); ++i)
			{
				for (long j = 0; j < t.size(); ++j)
				{
					const auto dx = pt(0) - (0 - kernel_distance_);
					const auto dy = pt(1) - t(i);
					const auto dz = pt(2) - t(j);
					AutodiffGradPt rvect(3);
					rvect << dx, dy, dz;
					res += assembler.kernel(formulation_, 3, rvect, sqrt(dx * dx + dy * dy + dz * dz));
				}
			}

			for (long i = 0; i < t.size(); ++i)
			{
				for (long j = 0; j < t.size(); ++j)
				{
					const auto dx = pt(0) - (1 + kernel_distance_);
					const auto dy = pt(1) - t(i);
					const auto dz = pt(2) - t(j);
					AutodiffGradPt rvect(3);
					rvect << dx, dy, dz;
					res += assembler.kernel(formulation_, 3, rvect, sqrt(dx * dx + dy * dy + dz * dz));
				}
			}

			///////////////////Y
			for (long i = 0; i < t.size(); ++i)
			{
				for (long j = 0; j < t.size(); ++j)
				{
					const auto dx = pt(0) - t(i);
					const auto dy = pt(1) - (0 - kernel_distance_);
					const auto dz = pt(2) - t(j);
					AutodiffGradPt rvect(3);
					rvect << dx, dy, dz;
					res += assembler.kernel(formulation_, 3, rvect, sqrt(dx * dx + dy * dy + dz * dz));
				}
			}

			for (long i = 0; i < t.size(); ++i)
			{
				for (long j = 0; j < t.size(); ++j)
				{
					const auto dx = pt(0) - t(i);
					const auto dy = pt(1) - (1 + kernel_distance_);
					const auto dz = pt(2) - t(j);
					AutodiffGradPt rvect(3);
					rvect << dx, dy, dz;
					res += assembler.kernel(formulation_, 3, rvect, sqrt(dx * dx + dy * dy + dz * dz));
				}
			}

			///////////////////Z
			for (long i = 0; i < t.size(); ++i)
			{
				for (long j = 0; j < t.size(); ++j)
				{
					const auto dx = pt(0) - t(i);
					const auto dy = pt(1) - t(j);
					const auto dz = pt(2) - (0 - kernel_distance_);
					AutodiffGradPt rvect(3);
					rvect << dx, dy, dz;
					res += assembler.kernel(formulation_, 3, rvect, sqrt(dx * dx + dy * dy + dz * dz));
				}
			}

			for (long i = 0; i < t.size(); ++i)
			{
				for (long j = 0; j < t.size(); ++j)
				{
					const auto dx = pt(0) - t(i);
					const auto dy = pt(1) - t(j);
					const auto dz = pt(2) - (1 + kernel_distance_);
					AutodiffGradPt rvect(3);
					rvect << dx, dy, dz;
					res += assembler.kernel(formulation_, 3, rvect, sqrt(dx * dx + dy * dy + dz * dz));
				}
			}
		}
		else
		{
			assert(false);
		}

		res = res * AutodiffScalarGrad(tt);

		return res;
	}

	void KernelProblem::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		const int size = size_for(pts);
		val.resize(pts.rows(), size);
		val.setZero();
	}

	void KernelProblem::set_parameters(const json &params)
	{
		if (params.count("formulation"))
			formulation_ = params["formulation"].get<std::string>();

		if (params.count("n_kernels"))
			n_kernels_ = params["n_kernels"];

		if (params.count("kernel_distance"))
			kernel_distance_ = params["kernel_distance"];
	}

	bool KernelProblem::is_scalar() const
	{
		return state->assembler.is_scalar(formulation_);
	}
} // namespace polyfem
