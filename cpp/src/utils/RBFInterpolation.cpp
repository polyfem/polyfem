#include "RBFInterpolation.hpp"

#include <cmath>
#include <iostream>

namespace polyfem
{

	RBFInterpolation::RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps)
	{
		init(fun, pts, rbf, eps);
	}

	void RBFInterpolation::init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps)
	{
		std::function<double(double)> tmp;

		if(rbf == "multiquadric"){
			tmp = [eps](const double r){ return sqrt((r/eps)*(r/eps) + 1); };
		}
		else if(rbf == "inverse" || rbf == "inverse_multiquadric" || rbf == "inverse multiquadric"){
			tmp = [eps](const double r){ return 1.0/sqrt((r/eps)*(r/eps) + 1); };
		}
		else if(rbf == "gaussian"){
			tmp = [eps](const double r){ return exp(-(r/eps)*(r/eps)); };
		}
		else if(rbf == "linear"){
			tmp = [](const double r){ return r; };
		}
		else if(rbf == "cubic"){
			tmp = [](const double r){ return r*r*r; };
		}
		else if(rbf == "quintic"){
			tmp = [](const double r){ return r*r*r*r*r; };
		}
		else if(rbf == "thin_plate" || rbf == "thin-plate"){
			tmp = [](const double r){ return abs(r) < 1e-10 ? 0 : (r*r * log(r)); };
		}
		else
		{
			std::cerr<<"Unable to match "<<rbf<<" rbf, falling back to multiquadric"<<std::endl;
			assert(false);

			tmp = [eps](const double r){ return sqrt((r/eps)*(r/eps) + 1); };
		}



		init(fun, pts, tmp);
	}


	RBFInterpolation::RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &rbf)
	{
		init(fun, pts, rbf);
	}

	void RBFInterpolation::init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &rbf)
	{
		assert(pts.rows() == fun.rows());

		rbf_ = rbf;
		centers_ = pts;


		const int n = centers_.rows();

		Eigen::MatrixXd A(n, n);

		for(int i = 0; i < n; ++i){
			for(int j = 0; j < n; ++j){
				A(i,j) = rbf((centers_.row(i)-centers_.row(j)).norm());
			}
		}

		Eigen::FullPivLU<Eigen::MatrixXd> lu(A);

		weights_.resize(n, fun.cols());
		for(long i = 0; i < fun.cols(); ++i){
			weights_.col(i) = lu.solve(fun.col(i));
		}
	}

	Eigen::MatrixXd RBFInterpolation::interpolate(const Eigen::MatrixXd &pts) const
	{
		assert(pts.cols() == centers_.cols());
		const int n = centers_.rows();
		const int m = pts.rows();

		Eigen::MatrixXd mat(m, n);
		for(int i = 0; i < m; ++i){
			for(int j = 0; j < n; ++j){
        		mat(i,j) = rbf_((centers_.row(j)-pts.row(i)).norm());
        	}
        }

		const Eigen::MatrixXd res = mat * weights_;
		return res;
	}
}
