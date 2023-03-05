#include "KernelProblem.hpp"
#include <polyfem/assembler/Assembler.hpp>

#include <iostream>
#include <fstream>

namespace polyfem
{
	namespace problem
	{
		KernelProblem::KernelProblem(const std::string &name, const assembler::Assembler &assembler)
			: ProblemWithSolution(name), assembler_(assembler)
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

			auto kw = kernel_weights_;

			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_kernels_, 0, 1);

			if (pt.size() == 2)
			{
				if (kw.size() == 0)
				{
					kw.resize(t.size() * 4);
					kw.setOnes();
				}

				assert(kw.size() == t.size() * 4);

				for (long i = 0; i < t.size(); ++i)
				{
					const auto dx = pt(0) - (0 - kernel_distance_);
					const auto dy = pt(1) - t(i);
					AutodiffGradPt rvect(2);
					rvect << dx, dy;
					if (kw(i) > 0)
						res += assembler_.kernel(2, rvect, sqrt(dx * dx + dy * dy)) * polyfem::AutodiffScalarGrad(kw(i));
				}

				for (long i = 0; i < t.size(); ++i)
				{
					const auto dx = pt(0) - (1 + kernel_distance_);
					const auto dy = pt(1) - t(i);
					AutodiffGradPt rvect(2);
					rvect << dx, dy;
					if (kw(t.size() + i) > 0)
						res += assembler_.kernel(2, rvect, sqrt(dx * dx + dy * dy)) * polyfem::AutodiffScalarGrad(kw(t.size() + i));
				}

				for (long i = 0; i < t.size(); ++i)
				{
					const auto dx = pt(0) - t(i);
					const auto dy = pt(1) - (0 - kernel_distance_);
					AutodiffGradPt rvect(2);
					rvect << dx, dy;
					if (kw(t.size() * 2 + i) > 0)
						res += assembler_.kernel(2, rvect, sqrt(dx * dx + dy * dy)) * polyfem::AutodiffScalarGrad(kw(t.size() * 2 + i));
				}

				for (long i = 0; i < t.size(); ++i)
				{
					const auto dx = pt(0) - t(i);
					const auto dy = pt(1) - (1 + kernel_distance_);
					AutodiffGradPt rvect(2);
					rvect << dx, dy;
					if (kw(t.size() * 3 + i) > 0)
						res += assembler_.kernel(2, rvect, sqrt(dx * dx + dy * dy)) * polyfem::AutodiffScalarGrad(kw(t.size() * 3 + i));
				}
			}
			else if (pt.size() == 3)
			{
				if (kw.size() == 0)
				{
					kw.resize(t.size() * t.size() * 6);
					kw.setOnes();
				}

				assert(kw.size() == t.size() * t.size() * 6);

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
						if (kw(i * t.size() + j) > 0)
							res += assembler_.kernel(3, rvect, sqrt(dx * dx + dy * dy + dz * dz)) * polyfem::AutodiffScalarGrad(kw(i * t.size() + j));
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
						if (kw(t.size() * t.size() + i * t.size() + j) > 0)
							res += assembler_.kernel(3, rvect, sqrt(dx * dx + dy * dy + dz * dz)) * polyfem::AutodiffScalarGrad(kw(t.size() * t.size() + i * t.size() + j));
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
						if (kw(t.size() * t.size() * 2 + i * t.size() + j) > 0)
							res += assembler_.kernel(3, rvect, sqrt(dx * dx + dy * dy + dz * dz)) * polyfem::AutodiffScalarGrad(kw(t.size() * t.size() * 2 + i * t.size() + j));
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
						if (kw(t.size() * t.size() * 3 + i * t.size() + j) > 0)
							res += assembler_.kernel(3, rvect, sqrt(dx * dx + dy * dy + dz * dz)) * polyfem::AutodiffScalarGrad(kw(t.size() * t.size() * 3 + i * t.size() + j));
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
						if (kw(t.size() * t.size() * 4 + i * t.size() + j) > 0)
							res += assembler_.kernel(3, rvect, sqrt(dx * dx + dy * dy + dz * dz)) * polyfem::AutodiffScalarGrad(kw(t.size() * t.size() * 4 + i * t.size() + j));
						;
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
						if (kw(t.size() * t.size() * 5 + i * t.size() + j) > 0)
							res += assembler_.kernel(3, rvect, sqrt(dx * dx + dy * dy + dz * dz)) * polyfem::AutodiffScalarGrad(kw(t.size() * t.size() * 5 + i * t.size() + j));
						;
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

		void KernelProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			const int size = size_for(pts);
			val.resize(pts.rows(), size);
			val.setZero();
		}

		void KernelProblem::set_parameters(const json &params)
		{
			if (params.count("n_kernels") && !params["n_kernels"] > 0)
				n_kernels_ = params["n_kernels"];

			if (params.count("kernel_distance") && !params["kernel_distance"] > 0)
				kernel_distance_ = params["kernel_distance"];

			if (params.count("kernel_weights") && !params["kernel_weights"].empty())
			{
				std::ifstream in(params["kernel_weights"].get<std::string>());
				std::string token;
				in >> token;
				int n_weights;
				in >> n_weights;
				kernel_weights_.resize(n_weights);
				for (int i = 0; i < n_weights; ++i)
					in >> kernel_weights_(i);
			}
		}

		bool KernelProblem::is_scalar() const
		{
			return !assembler_.is_tensor();
		}
	} // namespace problem
} // namespace polyfem
