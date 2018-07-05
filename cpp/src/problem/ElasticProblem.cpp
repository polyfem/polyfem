#include <polyfem/ElasticProblem.hpp>
#include <polyfem/State.hpp>

#include <iostream>

namespace polyfem
{
	ElasticProblem::ElasticProblem(const std::string &name)
	: Problem(name)
	{
		boundary_ids_ = {1, 3, 5, 6};
	}

	void ElasticProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		// val *= t;
	}

	void ElasticProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for(long i = 0; i < pts.rows(); ++i)
		{
			if(mesh.get_boundary_id(global_ids(i))== 1)
				val(i, 0)=-0.25;
			else if(mesh.get_boundary_id(global_ids(i))== 3)
				val(i, 0)=0.25;
			if(mesh.get_boundary_id(global_ids(i))== 5)
				val(i, 1)=-0.25;
			else if(mesh.get_boundary_id(global_ids(i))== 6)
				val(i, 1)=0.25;
		}

		val *= t;
	}


	TorsionElasticProblem::TorsionElasticProblem(const std::string &name)
	: Problem(name)
	{
		boundary_ids_ = {5, 6};

		trans_.resize(2);
		trans_.setConstant(0.5);
	}

	void TorsionElasticProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		// val *= t;
	}

	void TorsionElasticProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		double alpha = n_turns_ * t * 2 * M_PI;
		Eigen::Matrix2d rot; rot<<
		cos(alpha), -sin(alpha),
		sin(alpha),  cos(alpha);

		for(long i = 0; i < pts.rows(); ++i)
		{
			if(mesh.get_boundary_id(global_ids(i)) == boundary_ids_[1]){
				const Eigen::RowVector3d pt = pts.row(i);
				Eigen::RowVector2d pt2; pt2 << pt(coordiante_0_), pt(coordiante_1_);

				pt2 -= trans_;
				pt2 = pt2 * rot;
				pt2 += trans_;

				val(i, coordiante_0_) = pt2(0)-pt(coordiante_0_);
				val(i, coordiante_1_) = pt2(1)-pt(coordiante_1_);
			}
		}
	}

	void TorsionElasticProblem::set_parameters(const json &params)
	{
		if(params.find("axis_coordiante") != params.end())
		{
			const int coord = params["axis_coordiante"];
			coordiante_0_ = (coord + 1) % 3;
			coordiante_1_ = (coord + 2) % 3;
		}

		if(params.find("n_turns") != params.end())
		{
			n_turns_ = params["n_turns"];
		}

		if(params.find("fixed_boundary") != params.end())
		{
			boundary_ids_[0] = params["fixed_boundary"];
		}

		if(params.find("turning_boundary") != params.end())
		{
			boundary_ids_[1] = params["turning_boundary"];
		}

		if(params.find("bbox_extend") != params.end())
		{
			auto bbox_extend = params["bbox_extend"];
			if(bbox_extend.is_array() && bbox_extend.size() >= 3)
			{
				RowVectorNd tmp(3);
				tmp(0) = bbox_extend[0];
				tmp(1) = bbox_extend[1];
				tmp(2) = bbox_extend[2];

				tmp /= 2;

				trans_(0) = tmp(coordiante_0_);
				trans_(1) = tmp(coordiante_1_);
			}
		}
	}


	ElasticProblemZeroBC::ElasticProblemZeroBC(const std::string &name)
	: Problem(name)
	{
		boundary_ids_ = {1, 2, 3, 4, 5, 6};
	}

	void ElasticProblemZeroBC::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		val.col(1).setConstant(0.5);
		val *= t;
	}

	void ElasticProblemZeroBC::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for(long i = 0; i < pts.rows(); ++i)
		{
			if(mesh.get_boundary_id(global_ids(i)) > 0)
				val.row(i).setZero();
		}
		// val *= t;
	}



	namespace
	{
		template<typename T>
		Eigen::Matrix<T, 2, 1> function(T x, T y)
		{
			Eigen::Matrix<T, 2, 1> res;

			res(0) = (y*y*y + x*x + x*y)/50.;
			res(1) = (3*x*x*x*x + x*y*y + x)/50.;

			return res;
		}

		template<typename T>
		Eigen::Matrix<T, 3, 1> function(T x, T y, T z)
		{
			Eigen::Matrix<T, 3, 1> res;

			res(0) = (x*y + x*x + y*y*y + 6*z)/80.;
			res(1) = (z*x - z*z*z + x*y*y + 3*x*x*x*x)/80.;
			res(2) = (x*y*z + z*z*y*y - 2*x)/80.;

			return res;
		}


		template<typename T>
		Eigen::Matrix<T, 2, 1> function_compression(T x, T y)
		{
			Eigen::Matrix<T, 2, 1> res;

			res(0) = -(y*y*y + x*x + x*y)/20.;
			res(1) = -(3*x*x*x*x + x*y*y + x)/20.;

			return res;
		}

		template<typename T>
		Eigen::Matrix<T, 3, 1> function_compression(T x, T y, T z)
		{
			Eigen::Matrix<T, 3, 1> res;

			res(0) = -(x*y + x*x + y*y*y + 6*z)/14.;
			res(1) = -(z*x - z*z*z + x*y*y + 3*x*x*x*x)/14.;
			res(2) = -(x*y*z + z*z*y*y - 2*x)/14.;

			return res;
		}


		template<typename T>
		Eigen::Matrix<T, 2, 1> function_quadratic(T x, T y)
		{
			Eigen::Matrix<T, 2, 1> res;

			res(0) = -(y*y + x*x + x*y)/50.;
			res(1) = -(3*x*x + y)/50.;

			return res;
		}

		template<typename T>
		Eigen::Matrix<T, 3, 1> function_quadratic(T x, T y, T z)
		{
			Eigen::Matrix<T, 3, 1> res;

			res(0) = -(y*y + x*x + x*y + z*y)/50.;
			res(1) = -(3*x*x + y + z*z)/50.;
			res(2) = -(x*z + y*y - 2*z)/50.;

			return res;
		}


		template<typename T>
		Eigen::Matrix<T, 2, 1> function_linear(T x, T y)
		{
			Eigen::Matrix<T, 2, 1> res;

			res(0) = -(y + x)/50.;
			res(1) = -(3*x + y)/50.;

			return res;
		}

		template<typename T>
		Eigen::Matrix<T, 3, 1> function_linear(T x, T y, T z)
		{
			Eigen::Matrix<T, 3, 1> res;

			res(0) = -(y + x + z)/50.;
			res(1) = -(3*x + y - z)/50.;
			res(2) = -(x + y - 2*z)/50.;

			return res;
		}
	}



	ElasticProblemExact::ElasticProblemExact(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd ElasticProblemExact::eval_fun(const VectorNd &pt) const
	{
		if(pt.size() == 2)
			return function(pt(0), pt(1));
		else if(pt.size() == 3)
			return function(pt(0), pt(1), pt(2));

		assert(false);
		return VectorNd(pt.size());
	}

	AutodiffGradPt ElasticProblemExact::eval_fun(const AutodiffGradPt &pt) const
	{
		if(pt.size() == 2)
			return function(pt(0), pt(1));
		else if(pt.size() == 3)
			return function(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffGradPt(pt.size());
	}

	AutodiffHessianPt ElasticProblemExact::eval_fun(const AutodiffHessianPt &pt) const
	{
		if(pt.size() == 2)
			return function(pt(0), pt(1));
		else if(pt.size() == 3)
			return function(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffHessianPt(pt.size());
	}




	CompressionElasticProblemExact::CompressionElasticProblemExact(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd CompressionElasticProblemExact::eval_fun(const VectorNd &pt) const
	{
		if(pt.size() == 2)
			return function_compression(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_compression(pt(0), pt(1), pt(2));

		assert(false);
		return VectorNd(pt.size());
	}

	AutodiffGradPt CompressionElasticProblemExact::eval_fun(const AutodiffGradPt &pt) const
	{
		if(pt.size() == 2)
			return function_compression(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_compression(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffGradPt(pt.size());
	}

	AutodiffHessianPt CompressionElasticProblemExact::eval_fun(const AutodiffHessianPt &pt) const
	{
		if(pt.size() == 2)
			return function_compression(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_compression(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffHessianPt(pt.size());
	}






	QuadraticElasticProblemExact::QuadraticElasticProblemExact(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd QuadraticElasticProblemExact::eval_fun(const VectorNd &pt) const
	{
		if(pt.size() == 2)
			return function_quadratic(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_quadratic(pt(0), pt(1), pt(2));

		assert(false);
		return VectorNd(pt.size());
	}

	AutodiffGradPt QuadraticElasticProblemExact::eval_fun(const AutodiffGradPt &pt) const
	{
		if(pt.size() == 2)
			return function_quadratic(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_quadratic(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffGradPt(pt.size());
	}

	AutodiffHessianPt QuadraticElasticProblemExact::eval_fun(const AutodiffHessianPt &pt) const
	{
		if(pt.size() == 2)
			return function_quadratic(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_quadratic(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffHessianPt(pt.size());
	}






	LinearElasticProblemExact::LinearElasticProblemExact(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd LinearElasticProblemExact::eval_fun(const VectorNd &pt) const
	{
		if(pt.size() == 2)
			return function_linear(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_linear(pt(0), pt(1), pt(2));

		assert(false);
		return VectorNd(pt.size());
	}

	AutodiffGradPt LinearElasticProblemExact::eval_fun(const AutodiffGradPt &pt) const
	{
		if(pt.size() == 2)
			return function_linear(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_linear(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffGradPt(pt.size());
	}

	AutodiffHessianPt LinearElasticProblemExact::eval_fun(const AutodiffHessianPt &pt) const
	{
		if(pt.size() == 2)
			return function_linear(pt(0), pt(1));
		else if(pt.size() == 3)
			return function_linear(pt(0), pt(1), pt(2));

		assert(false);
		return AutodiffHessianPt(pt.size());
	}


	GravityProblem::GravityProblem(const std::string &name)
	: Problem(name)
	{
		boundary_ids_ = {4};
	}

	void GravityProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		val.col(1).setConstant(0.1);
		// val *= t;
	}

	void GravityProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void GravityProblem::velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void GravityProblem::acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void GravityProblem::initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void GravityProblem::initial_velocity(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}
	
	void GravityProblem::initial_acceleration(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

}
