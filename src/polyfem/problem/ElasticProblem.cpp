#include "ElasticProblem.hpp"
#include <polyfem/State.hpp>

#include <iostream>

namespace polyfem
{
	namespace problem
	{
		ElasticProblem::ElasticProblem(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {1, 3, 5, 6};
		}

		void ElasticProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void ElasticProblem::bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				if (mesh.get_boundary_id(global_ids(i)) == 1)
					val(i, 0) = -0.25;
				else if (mesh.get_boundary_id(global_ids(i)) == 3)
					val(i, 0) = 0.25;
				if (mesh.get_boundary_id(global_ids(i)) == 5)
					val(i, 1) = -0.25;
				else if (mesh.get_boundary_id(global_ids(i)) == 6)
					val(i, 1) = 0.25;
			}
		}

		TorsionElasticProblem::TorsionElasticProblem(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {5, 6};

			trans_.resize(2);
			trans_.setConstant(0.5);
		}

		void TorsionElasticProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void TorsionElasticProblem::bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			double alpha = n_turns_ * t * 2 * M_PI;
			Eigen::Matrix2d rot;
			rot << cos(alpha), -sin(alpha),
				sin(alpha), cos(alpha);

			for (long i = 0; i < pts.rows(); ++i)
			{
				if (mesh.get_boundary_id(global_ids(i)) == boundary_ids_[1])
				{
					const Eigen::RowVector3d pt = pts.row(i);
					Eigen::RowVector2d pt2;
					pt2 << pt(coordiante_0_), pt(coordiante_1_);

					pt2 -= trans_;
					pt2 = pt2 * rot;
					pt2 += trans_;

					val(i, coordiante_0_) = pt2(0) - pt(coordiante_0_);
					val(i, coordiante_1_) = pt2(1) - pt(coordiante_1_);
				}
			}
		}

		void TorsionElasticProblem::set_parameters(const json &params)
		{
			if (params.contains("axis_coordiante"))
			{
				const int coord = params["axis_coordiante"];
				coordiante_0_ = (coord + 1) % 3;
				coordiante_1_ = (coord + 2) % 3;
			}

			if (params.contains("n_turns"))
			{
				n_turns_ = params["n_turns"];
			}

			if (params.contains("fixed_boundary"))
			{
				boundary_ids_[0] = params["fixed_boundary"];
			}

			if (params.contains("turning_boundary"))
			{
				boundary_ids_[1] = params["turning_boundary"];
			}

			if (params.contains("bbox_center"))
			{
				auto bbox_center = params["bbox_center"];
				if (bbox_center.is_array() && bbox_center.size() >= 3)
				{
					trans_(0) = bbox_center[coordiante_0_];
					trans_(1) = bbox_center[coordiante_1_];
				}
			}
		}

		DoubleTorsionElasticProblem::DoubleTorsionElasticProblem(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {5, 6};

			trans_0_.resize(2);
			trans_0_.setConstant(0.5);

			trans_1_.resize(2);
			trans_1_.setConstant(0.5);
		}

		void DoubleTorsionElasticProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void DoubleTorsionElasticProblem::velocity_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}
		void DoubleTorsionElasticProblem::acceleration_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void DoubleTorsionElasticProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}
		void DoubleTorsionElasticProblem::initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}
		void DoubleTorsionElasticProblem::initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void DoubleTorsionElasticProblem::bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			double alpha0 = angular_v0_ * t;
			double alpha1 = angular_v1_ * t;
			Eigen::Matrix2d rot0;
			rot0 << cos(alpha0), -sin(alpha0), sin(alpha0), cos(alpha0);

			Eigen::Matrix2d rot1;
			rot1 << cos(alpha1), -sin(alpha1), sin(alpha1), cos(alpha1);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const Eigen::RowVector3d pt = pts.row(i);

				if (mesh.get_boundary_id(global_ids(i)) == boundary_ids_[0])
				{
					Eigen::RowVector2d pt2;
					pt2 << pt(coordiante_0_[0]), pt(coordiante_0_[1]);

					pt2 -= trans_0_;
					pt2 = pt2 * rot0;
					pt2 += trans_0_;

					val(i, coordiante_0_[0]) = pt2(0) - pt(coordiante_0_[0]);
					val(i, coordiante_0_[1]) = pt2(1) - pt(coordiante_0_[1]);
				}
				else if (mesh.get_boundary_id(global_ids(i)) == boundary_ids_[1])
				{
					Eigen::RowVector2d pt2;
					pt2 << pt(coordiante_1_[0]), pt(coordiante_1_[1]);

					pt2 -= trans_1_;
					pt2 = pt2 * rot1;
					pt2 += trans_1_;

					val(i, coordiante_1_[0]) = pt2(0) - pt(coordiante_1_[0]);
					val(i, coordiante_1_[1]) = pt2(1) - pt(coordiante_1_[1]);
				}
			}
		}

		void DoubleTorsionElasticProblem::set_parameters(const json &params)
		{
			if (params.contains("axis_coordiante0"))
			{
				const int coord = params["axis_coordiante0"];
				coordiante_0_[0] = (coord + 1) % 3;
				coordiante_0_[1] = (coord + 2) % 3;
			}

			if (params.contains("axis_coordiante1"))
			{
				const int coord = params["axis_coordiante1"];
				coordiante_1_[0] = (coord + 1) % 3;
				coordiante_1_[1] = (coord + 2) % 3;
			}

			if (params.contains("angular_v0"))
			{
				angular_v0_ = params["angular_v0"];
			}

			if (params.contains("angular_v1"))
			{
				angular_v1_ = params["angular_v1"];
			}

			if (params.contains("turning_boundary0"))
			{
				boundary_ids_[0] = params["turning_boundary0"];
			}

			if (params.contains("turning_boundary1"))
			{
				boundary_ids_[1] = params["turning_boundary1"];
			}

			if (params.contains("bbox_center"))
			{
				auto bbox_center = params["bbox_center"];
				if (bbox_center.is_array() && bbox_center.size() >= 3)
				{
					trans_0_(0) = bbox_center[coordiante_0_[0]];
					trans_0_(1) = bbox_center[coordiante_0_[1]];

					trans_1_(0) = bbox_center[coordiante_1_[0]];
					trans_1_(1) = bbox_center[coordiante_1_[1]];
				}
			}
		}

		ElasticProblemZeroBC::ElasticProblemZeroBC(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {1, 2, 3, 4, 5, 6};
		}

		void ElasticProblemZeroBC::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
			val.col(1).setConstant(0.5);
		}

		void ElasticProblemZeroBC::bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				if (mesh.get_boundary_id(global_ids(i)) > 0)
					val.row(i).setZero();
			}
		}

		namespace
		{
			template <typename T>
			Eigen::Matrix<T, 2, 1> function(T x, T y, const double t)
			{
				Eigen::Matrix<T, 2, 1> res;

				res(0) = t * (y * y * y + x * x + x * y) / 50.;
				res(1) = t * (3 * x * x * x * x + x * y * y + x) / 50.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 3, 1> function(T x, T y, T z, const double t)
			{
				Eigen::Matrix<T, 3, 1> res;

				res(0) = t * (x * y + x * x + y * y * y + 6 * z) / 80.;
				res(1) = t * (z * x - z * z * z + x * y * y + 3 * x * x * x * x) / 80.;
				res(2) = t * (x * y * z + z * z * y * y - 2 * x) / 80.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 2, 1> function_compression(T x, T y, const double t)
			{
				Eigen::Matrix<T, 2, 1> res;

				res(0) = -t * (y * y * y + x * x + x * y) / 20.;
				res(1) = -t * (3 * x * x * x * x + x * y * y + x) / 20.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 3, 1> function_compression(T x, T y, T z, const double t)
			{
				Eigen::Matrix<T, 3, 1> res;

				res(0) = -t * (x * y + x * x + y * y * y + 6 * z) / 14.;
				res(1) = -t * (z * x - z * z * z + x * y * y + 3 * x * x * x * x) / 14.;
				res(2) = -t * (x * y * z + z * z * y * y - 2 * x) / 14.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 2, 1> function_quadratic(T x, T y, const double t)
			{
				Eigen::Matrix<T, 2, 1> res;

				res(0) = -t * (y * y + x * x + x * y) / 50.;
				res(1) = -t * (3 * x * x + y) / 50.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 3, 1> function_quadratic(T x, T y, T z, const double t)
			{
				Eigen::Matrix<T, 3, 1> res;

				res(0) = -t * (y * y + x * x + x * y + z * y) / 50.;
				res(1) = -t * (3 * x * x + y + z * z) / 50.;
				res(2) = -t * (x * z + y * y - 2 * z) / 50.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 2, 1> function_linear(T x, T y, const double t)
			{
				Eigen::Matrix<T, 2, 1> res;

				res(0) = -t * (y + x) / 50.;
				res(1) = -t * (3 * x + y) / 50.;

				return res;
			}

			template <typename T>
			Eigen::Matrix<T, 3, 1> function_linear(T x, T y, T z, const double t)
			{
				Eigen::Matrix<T, 3, 1> res;

				res(0) = -t * (y + x + z) / 50.;
				res(1) = -t * (3 * x + y - z) / 50.;
				res(2) = -t * (x + y - 2 * z) / 50.;

				return res;
			}
		} // namespace

		ElasticProblemExact::ElasticProblemExact(const std::string &name)
			: ProblemWithSolution(name)
		{
		}

		VectorNd ElasticProblemExact::eval_fun(const VectorNd &pt, const double t) const
		{
			if (pt.size() == 2)
				return function(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function(pt(0), pt(1), pt(2), t);

			assert(false);
			return VectorNd(pt.size());
		}

		AutodiffGradPt ElasticProblemExact::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffGradPt(pt.size());
		}

		AutodiffHessianPt ElasticProblemExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffHessianPt(pt.size());
		}

		CompressionElasticProblemExact::CompressionElasticProblemExact(const std::string &name)
			: ProblemWithSolution(name)
		{
		}

		VectorNd CompressionElasticProblemExact::eval_fun(const VectorNd &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_compression(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_compression(pt(0), pt(1), pt(2), t);

			assert(false);
			return VectorNd(pt.size());
		}

		AutodiffGradPt CompressionElasticProblemExact::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_compression(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_compression(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffGradPt(pt.size());
		}

		AutodiffHessianPt CompressionElasticProblemExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_compression(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_compression(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffHessianPt(pt.size());
		}

		QuadraticElasticProblemExact::QuadraticElasticProblemExact(const std::string &name)
			: ProblemWithSolution(name)
		{
		}

		VectorNd QuadraticElasticProblemExact::eval_fun(const VectorNd &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_quadratic(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_quadratic(pt(0), pt(1), pt(2), t);

			assert(false);
			return VectorNd(pt.size());
		}

		AutodiffGradPt QuadraticElasticProblemExact::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_quadratic(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_quadratic(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffGradPt(pt.size());
		}

		AutodiffHessianPt QuadraticElasticProblemExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_quadratic(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_quadratic(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffHessianPt(pt.size());
		}

		LinearElasticProblemExact::LinearElasticProblemExact(const std::string &name)
			: ProblemWithSolution(name)
		{
		}

		VectorNd LinearElasticProblemExact::eval_fun(const VectorNd &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_linear(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_linear(pt(0), pt(1), pt(2), t);

			assert(false);
			return VectorNd(pt.size());
		}

		AutodiffGradPt LinearElasticProblemExact::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_linear(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_linear(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffGradPt(pt.size());
		}

		AutodiffHessianPt LinearElasticProblemExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			if (pt.size() == 2)
				return function_linear(pt(0), pt(1), t);
			else if (pt.size() == 3)
				return function_linear(pt(0), pt(1), pt(2), t);

			assert(false);
			return AutodiffHessianPt(pt.size());
		}

		GravityProblem::GravityProblem(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {4};
		}

		void GravityProblem::set_parameters(const json &params)
		{
			if (params.contains("force"))
			{
				force_ = params["force"];
			}
		}

		void GravityProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
			val.col(1).setConstant(force_);
		}

		void GravityProblem::bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void GravityProblem::velocity_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void GravityProblem::acceleration_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void GravityProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void GravityProblem::initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void GravityProblem::initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		WalkProblem::WalkProblem(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {1, 2};
		}

		void WalkProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void WalkProblem::bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				if (mesh.get_boundary_id(global_ids(i)) == 1)
					val(i, 2) = 0.2 * sin(t);
				else if (mesh.get_boundary_id(global_ids(i)) == 2)
					val(i, 2) = -0.2 * sin(t);
			}
		}

		void WalkProblem::velocity_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void WalkProblem::acceleration_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void WalkProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void WalkProblem::initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void WalkProblem::initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}
	} // namespace problem
} // namespace polyfem
