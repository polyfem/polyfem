#pragma once


#include <Eigen/Dense>

#include <functional>
#include <string>


namespace polyfem
{
	class RBFInterpolation
	{
	public:
		RBFInterpolation() { }
		RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &function);
		void init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &function);

		RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &function, const double eps);
		void init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &function, const double eps);

		Eigen::MatrixXd interpolate(const Eigen::MatrixXd &pts) const;

	private:
		Eigen::MatrixXd centers_;
		Eigen::MatrixXd weights_;

		std::function<double(double)> function_;
	};
}
