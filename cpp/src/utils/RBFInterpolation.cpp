#include "RBFInterpolation.hpp"


namespace polyfem
{

	RBFInterpolation::RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &function, const double eps)
	{
		init(fun, pts, function, eps);
	}

	void RBFInterpolation::init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::string &function, const double eps)
	{
		std::function<double(double)> tmp;

		if(function == "multiquadric"){
			tmp = [eps](const double r){ return sqrt((r/eps)*(r/eps) + 1); };
		}
		else if(function == "inverse" || function == "inverse_multiquadric" || function == "inverse multiquadric"){
			tmp = [eps](const double r){ return 1.0/sqrt((r/eps)*(r/eps) + 1); };
		}
		else if(function == "gaussian"){
			tmp = [eps](const double r){ return exp(-(r/eps)*(r/eps)); };
		}
		else if(function == "linear"){
			tmp = [](const double r){ return r; };
		}
		else if(function == "cubic"){
			tmp = [](const double r){ return r*r*r; };
		}
		else if(function == "quintic"){
			tmp = [](const double r){ return r*r*r*r*r; };
		}
		else if(function == "thin_plate" || function == "thin-plate"){
			tmp = [](const double r){ return r*r * log(r); };
		}



		init(fun, pts, tmp);
	}


	RBFInterpolation::RBFInterpolation(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &function)
	{
		init(fun, pts, function);
	}

	void RBFInterpolation::init(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const std::function<double(double)> &function)
	{
		assert(pts.rows() == fun.rows());

		function_ = function;
		centers_ = pts;


		const int n = centers_.rows();

		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);

		for(int i = 0; i < n; ++i){
			for(int j = 0; j < n; ++j){
				A(i,j) = function((centers_.row(i)-centers_.row(j)).norm());
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

		Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(m, n);
		for(int i = 0; i < m; ++i){
			for(int j = 0; j < n; ++j){
        		mat(i,j) = function_((centers_.row(j)-pts.row(i)).norm());
        	}
        }

		const Eigen::MatrixXd res = mat * weights_;
		return res;
	}
}
