#include "RBFInterpolation.hpp"

#include <polyfem/utils/Logger.hpp>
#include <cmath>
#include <iostream>

namespace polyfem
{
	namespace utils
	{
		RBFInterpolation::RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps)
		{
			init(fun, pts, rbf, eps);
		}

		void RBFInterpolation::init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps)
		{
			assert(pts.rows() >= 0);
#ifdef POLYFEM_OPENCL
			std::vector<double> pointscl(pts.size());
			std::vector<double> functioncl(fun.rows());

			int index = 0;
			for (int i = 0; i < pts.rows(); ++i)
			{
				for (int j = 0; j < pts.cols(); ++j)
				{
					pointscl[index++] = pts(i, j);
				}
			}

			data_.resize(fun.cols());
			for (int i = 0; i < fun.cols(); ++i)
			{
				index = 0;

				for (int j = 0; j < fun.rows(); ++j)
					functioncl[index] = fun(j, i);

				rbf_pum::init(pointscl, functioncl, data_[i], verbose_, rbfcl_, opt_, unit_cube_, num_threads_);
			}
#else
			std::function<double(double)> tmp;

			if (rbf == "multiquadric")
			{
				tmp = [eps](const double r) { return sqrt((r / eps) * (r / eps) + 1); };
			}
			else if (rbf == "inverse" || rbf == "inverse_multiquadric" || rbf == "inverse multiquadric")
			{
				tmp = [eps](const double r) { return 1.0 / sqrt((r / eps) * (r / eps) + 1); };
			}
			else if (rbf == "gaussian")
			{
				tmp = [eps](const double r) { return exp(-(r / eps) * (r / eps)); };
			}
			else if (rbf == "linear")
			{
				tmp = [](const double r) { return r; };
			}
			else if (rbf == "cubic")
			{
				tmp = [](const double r) { return r * r * r; };
			}
			else if (rbf == "quintic")
			{
				tmp = [](const double r) { return r * r * r * r * r; };
			}
			else if (rbf == "thin_plate" || rbf == "thin-plate")
			{
				tmp = [](const double r) { return abs(r) < 1e-10 ? 0 : (r * r * log(r)); };
			}
			else
			{
				logger().warn("Unable to match {} rbf, falling back to multiquadric", rbf);
				assert(false);

				tmp = [eps](const double r) { return sqrt((r / eps) * (r / eps) + 1); };
			}

			init(fun, pts, tmp);
#endif
		}

		RBFInterpolation::RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &rbf)
		{
#ifdef POLYFEM_OPENCL
			assert(false);
#else
			init(fun, pts, rbf);
#endif
		}

		void RBFInterpolation::init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &rbf)
		{
#ifdef POLYFEM_OPENCL
			assert(false);
#else
			assert(pts.rows() >= 0);
			assert(pts.rows() == fun.rows());

			rbf_ = rbf;
			centers_ = pts;

			const int n = centers_.rows();

			Eigen::MatrixXd A(n, n);

			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					A(i, j) = rbf((centers_.row(i) - centers_.row(j)).norm());
				}
			}

			Eigen::FullPivLU<Eigen::MatrixXd> lu(A);

			weights_.resize(n, fun.cols());
			for (long i = 0; i < fun.cols(); ++i)
			{
				weights_.col(i) = lu.solve(fun.col(i));
			}
#endif
		}

		Eigen::MatrixXd RBFInterpolation::interpolate(const Eigen::MatrixXd &pts) const
		{
#ifdef POLYFEM_OPENCL
			Eigen::MatrixXd res(pts.rows(), data_.size());

			std::vector<double> pointscl(pts.size());
			int index = 0;
			for (int i = 0; i < pts.rows(); ++i)
			{
				for (int j = 0; j < pts.cols(); ++j)
				{
					pointscl[index++] = pts(i, j);
				}
			}

			std::vector<double> tmp;
			for (size_t i = 0; i < data_.size(); ++i)
			{
				rbf_pum::interpolate(data_[i], pointscl, tmp, verbose_, rbfcl_, opt_, unit_cube_, num_threads_);

				for (size_t j = 0; j < tmp.size(); ++j)
					res(j, i) = tmp[j];
			}
#else
			assert(pts.cols() == centers_.cols());
			const int n = centers_.rows();
			const int m = pts.rows();

			Eigen::MatrixXd mat(m, n);
			for (int i = 0; i < m; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					mat(i, j) = rbf_((centers_.row(j) - pts.row(i)).norm());
				}
			}

			const Eigen::MatrixXd res = mat * weights_;
#endif
			return res;
		}
	} // namespace utils
} // namespace polyfem
