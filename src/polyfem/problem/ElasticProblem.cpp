#include "ElasticProblem.hpp"

#include <iostream>

using namespace Eigen;

namespace polyfem
{
	namespace problem
	{
		ElasticProblem::ElasticProblem(const std::string &name)
			: Problem(name)
		{
			boundary_ids_ = {1, 3, 5, 6};
		}

		void ElasticProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void ElasticProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void TorsionElasticProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void TorsionElasticProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void DoubleTorsionElasticProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void DoubleTorsionElasticProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void ElasticProblemZeroBC::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
			val.col(1).setConstant(0.5);
		}

		void ElasticProblemZeroBC::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

			template <typename T>
			void compute_singularity(const T &x, const T &y, Eigen::Matrix<T, -1, 1> &res, const double a, const double mu, const double nu, const double lambda, const double t)
			{
				double b1, b2, b3, b4;
				T r = sqrt(x * x + y * y);
				T phi = atan2(y, x);
				T ur, ut;

				b1 = sin(M_PI * a / 2) * ((a - 1) * mu + (1 - 2 * nu) * lambda) * (a - 4 * nu + 3) / cos(M_PI * a / 2) / ((a - 4 * nu + 2) * mu + (1 - 2 * nu) * lambda) / (a + 1);
				b2 = (a + 4 * nu - 3) / (a + 1);
				b3 = -sin(M_PI * a / 2) * ((a - 1) * mu + (1 - 2 * nu) * lambda) / cos(M_PI * a / 2) / ((a - 4 * nu + 2) * mu + (1 - 2 * nu) * lambda);
				b4 = -1;

				ur = -1 / mu * pow(r, a) * (b4 * (a + (4 * nu) - 3) * cos((a - 1) * phi) + b3 * (a + (4 * nu) - 3) * sin((a - 1) * phi) + (b2 * cos((a + 1) * phi) + b1 * sin((a + 1) * phi)) * (a + 1)) / 2;
				ut = -1 / mu * pow(r, a) * (b3 * (a - (4 * nu) + 3) * cos((a - 1) * phi) - b4 * (a - (4 * nu) + 3) * sin((a - 1) * phi) + (b1 * cos((a + 1) * phi) - b2 * sin((a + 1) * phi)) * (a + 1)) / 2;

				res(0) = 79.17 * t * ur * cos(ut);
				res(1) = 79.17 * t * ur * sin(ut);
			}

			template <typename T>
			Eigen::Matrix<T, -1, 1> function_cantilever(T x, T y, const double t, const double delta, const double E, const double nu, const int dim, const double L, const double D)
			{
				Eigen::Matrix<T, -1, 1> res, res1, res2;
				res.setZero(dim, 1);
				res1.setZero(dim, 1);
				res2.setZero(dim, 1);
				const double lambda = (E * nu) / (1 + nu) / (1 - (dim - 1) * nu);
				const double mu = E / (2 * (1 + nu));

				// Boundary condition related formulas in Rossel's paper
				double a = 0.71117293;
				double P = 100;            // force
				double I = D * D * D / 12; // second moment of area of the cross-section

				// Add 2 singular solutions for an infinite wedge with  Dirichlet/Neumann sides
				compute_singularity(y + delta, x + delta, res1, a, mu, nu, lambda, t);
				compute_singularity(D - y + delta, x + delta, res2, a, mu, nu, lambda, t);
				res = res1 + res2;

				// Formulas in Charles's paper
				res(0) += t * P * (y - D / 2) / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * ((y - D / 2) * (y - D / 2) - D * D / 4)) / 3;
				res(1) += -t * P / (6 * E * I) * (3 * nu * (y - D / 2) * (y - D / 2) * (L - x) + (4 + 5 * nu) * D * D * x / 4 + (3 * L - x) * x * x) / 3;

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

		void GravityProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
			val.col(1).setConstant(force_);
		}

		void GravityProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void WalkProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		}

		void WalkProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		ElasticCantileverExact::ElasticCantileverExact(const std::string &name)
			: Problem(name)
		{
			boundary_ids_.clear();
			neumann_boundary_ids_.clear();

			boundary_ids_ = {1, 5, 6};
			neumann_boundary_ids_ = {2, 3, 4};

			singular_point_displacement = 0;
			E = 210000;
			nu = 0.3;
			formulation = "None";
			length = 1;
			width = 1 / 3.;
		}

		void ElasticCantileverExact::set_parameters(const json &params)
		{
			if (params.contains("displacement"))
			{
				singular_point_displacement = params["displacement"];
			}
			if (params.contains("E"))
			{
				E = params["E"];
			}
			if (params.contains("nu"))
			{
				nu = params["nu"];
			}
			if (params.contains("formulation"))
			{
				formulation = params["formulation"];
			}
			if (params.contains("mesh_size"))
			{
				auto size = params["mesh_size"];
				if (size.is_array())
				{
					length = size[0];
					width = size[1];
				}
				else
				{
					throw std::invalid_argument("Mesh_size needs to be an array!");
				}
			}
		}

		void ElasticCantileverExact::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			const int size = size_for(pts);
			val.resize(pts.rows(), size);

			const double lambda = (E * nu) / (1.0 + nu) / (1.0 - (size - 1.0) * nu);
			const double mu = E / (2.0 * (1.0 + nu));

			for (long i = 0; i < pts.rows(); ++i)
			{
				Matrix<double, Dynamic, 1, 0, 3, 1> point(pts.cols()), result;
				point = pts.row(i);

				DiffScalarBase::setVariableCount(pts.cols());
				AutodiffHessianPt pt(pts.cols());

				for (long d = 0; d < pts.cols(); ++d)
					pt(d) = AutodiffScalarHessian(d, pts(i, d));

				const auto res = eval_fun(pt, t);
				val.row(i) = assembler.compute_rhs(res).transpose();
			}
		}

		void ElasticCantileverExact::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			exact(pts, t, val);
		}

		void ElasticCantileverExact::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val.resize(pts.rows(), size_for(pts));

			for (long i = 0; i < pts.rows(); ++i)
			{
				val.row(i) = eval_fun(VectorNd(pts.row(i)), t);
			}
		}

		void ElasticCantileverExact::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			const int size = size_for(pts);
			val.resize(pts.rows(), pts.cols() * size);

			for (long i = 0; i < pts.rows(); ++i)
			{
				DiffScalarBase::setVariableCount(pts.cols());
				AutodiffGradPt pt(pts.cols());

				for (long d = 0; d < pts.cols(); ++d)
					pt(d) = AutodiffScalarGrad(d, pts(i, d));

				const auto res = eval_fun(pt, t);

				for (int m = 0; m < size; ++m)
				{
					const auto &tmp = res(m);
					val.block(i, m * pts.cols(), 1, pts.cols()) = tmp.getGradient().transpose();
				}
			}
		}

		void ElasticCantileverExact::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			const double lambda = (E * nu) / (1.0 + nu) / (1.0 - (mesh.dimension() - 1.0) * nu);
			const double mu = E / 2.0 / (1.0 + nu);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));

				DiffScalarBase::setVariableCount(pts.cols());
				AutodiffHessianPt pt(pts.cols());
				Eigen::VectorXd neumann_bc_normal = normals.row(i);
				Eigen::MatrixXd sigma;

				for (long d = 0; d < pts.cols(); ++d)
					pt(d) = AutodiffScalarHessian(d, pts(i, d));

				AutodiffHessianPt res = eval_fun(pt, t);
				Eigen::MatrixXd grad_u(mesh.dimension(), mesh.dimension());
				for (int d1 = 0; d1 < mesh.dimension(); d1++)
				{
					for (int d2 = 0; d2 < mesh.dimension(); d2++)
					{
						grad_u(d1, d2) = res(d1).getGradient()(d2);
					}
				}

				if (formulation == "LinearElasticity")
				{
					sigma = mu * (grad_u + grad_u.transpose()) + lambda * grad_u.trace() * Eigen::MatrixXd::Identity(mesh.dimension(), mesh.dimension());
				}
				else if (formulation == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u.rows(), grad_u.cols()) + grad_u;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					sigma = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;
				}
				else
				{
					throw std::invalid_argument("No specified formulation in params!");
					assert(false);
				}

				val.row(i) = sigma * neumann_bc_normal;
			}
		}

		VectorNd ElasticCantileverExact::eval_fun(const VectorNd &pt, const double t) const
		{
			return function_cantilever(pt(0), pt(1), t, singular_point_displacement, E, nu, pt.size(), length, width);
		}

		AutodiffGradPt ElasticCantileverExact::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			return function_cantilever(pt(0), pt(1), t, singular_point_displacement, E, nu, pt.size(), length, width);
		}

		AutodiffHessianPt ElasticCantileverExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			return function_cantilever(pt(0), pt(1), t, singular_point_displacement, E, nu, pt.size(), length, width);
		}
	} // namespace problem
} // namespace polyfem
