#pragma once

#include <Eigen/Dense>

#include <functional>
#include <string>

#ifdef POLYFEM_OPENCL
#include <rbf_interpolate.hpp>
#endif

namespace polyfem
{
	namespace utils
	{
		class RBFInterpolation
		{
		public:
			RBFInterpolation() {}
			RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &rbf);
			void init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &rbf);

			RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps);
			void init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps);

			Eigen::MatrixXd interpolate(const Eigen::MatrixXd &pts) const;

		private:
#ifdef POLYFEM_OPENCL
			int verbose_ = 0;
			const std::string rbfcl_ = "GA";
			bool opt_ = false;
			bool unit_cube_ = false;
			int num_threads_ = -1;

			std::vector<rbf_pum::RBFData> data_;
#else
			Eigen::MatrixXd centers_;
			Eigen::MatrixXd weights_;

			std::function<double(double)> rbf_;
#endif
		};
	} // namespace utils
} // namespace polyfem
