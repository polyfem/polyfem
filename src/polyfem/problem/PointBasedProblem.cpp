#include "PointBasedProblem.hpp"
#include <polyfem/io/MatrixIO.hpp>

#include <iostream>
#include <string>

namespace polyfem
{
	using namespace io;
	using namespace utils;

	namespace problem
	{
		bool PointBasedTensorProblem::BCValue::init(const json &data)
		{
			Eigen::Matrix<bool, 3, 1> dd;
			dd.setConstant(true);
			bool all_dimensions_dirichlet = true;

			if (data.is_array())
			{
				assert(data.size() == 3);
				init((double)data[0], (double)data[1], (double)data[2], dd);
				// TODO add dimension
			}
			else if (data.is_object())
			{
				Eigen::MatrixXd fun, pts;
				Eigen::MatrixXi tri;

				std::string rbf = "multiquadric";
				double eps = 0.1;

				read_matrix(data["function"], fun);
				read_matrix(data["points"], pts);

				int coord = -1;
				if (data.contains("coordinate"))
					coord = data["coordinate"];

				if (coord >= 0)
					pts = pts.block(0, 0, pts.rows(), 2).eval();

				is_tri = data.contains("triangles");
				if (is_tri)
					read_matrix(data["triangles"], tri);
				else
				{
					if (data.contains("rbf"))
					{
						const std::string tmp = data["rbf"];
						rbf = tmp;
					}
					if (data.contains("epsilon"))
						eps = data["epsilon"];
				}

				if (data.contains("dimension"))
				{
					all_dimensions_dirichlet = false;
					auto &tmp = data["dimension"];
					assert(tmp.is_array());
					for (size_t k = 0; k < tmp.size(); ++k)
						dd(k) = tmp[k];
				}

				if (is_tri)
					init(pts, tri, fun, coord, dd);
				else
					init(pts, fun, rbf, eps, coord, dd);
			}
			else
			{
				init(0, 0, 0, dd);
			}

			return all_dimensions_dirichlet;
		}

		Eigen::RowVector3d PointBasedTensorProblem::BCValue::operator()(const Eigen::RowVector3d &pt) const
		{
			if (is_val)
			{
				return val.transpose();
			}
			Eigen::RowVector3d res;

			if (coordiante_0 >= 0)
			{
				Eigen::RowVector2d pt2;
				pt2 << pt(coordiante_0), pt(coordiante_1);

				if (is_tri)
					res = tri_func.interpolate(pt2);
				else
					res = rbf_func.interpolate(pt2);
			}
			else
			{
				if (is_tri)
					res = tri_func.interpolate(pt);
				else
					res = rbf_func.interpolate(pt);
			}

			return res;
		}

		PointBasedTensorProblem::PointBasedTensorProblem(const std::string &name)
			: Problem(name), rhs_(0), scaling_(1)
		{
			translation_.setZero();
		}

		void PointBasedTensorProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Constant(pts.rows(), pts.cols(), rhs_);
		}

		void PointBasedTensorProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));
				const auto &pt3d = (pts.row(i) + translation_.transpose()) / scaling_;

				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if (id == boundary_ids_[b])
					{
						val.row(i) = bc_[b](pt3d) * scaling_;
					}
				}
			}
		}

		void PointBasedTensorProblem::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));
				const auto &pt3d = (pts.row(i) + translation_.transpose()) / scaling_;

				for (size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
				{
					if (id == neumann_boundary_ids_[b])
					{
						val.row(i) = neumann_bc_[b](pt3d) * scaling_;
					}
				}
			}
		}

		void PointBasedTensorProblem::add_constant(const int bc_tag, const Eigen::Vector3d &value, const Eigen::Matrix<bool, 3, 1> &dd, const bool is_neumann)
		{
			if (is_neumann)
			{
				neumann_boundary_ids_.push_back(bc_tag);
				neumann_bc_.emplace_back();
				neumann_bc_.back().init(value, dd);
			}
			else
			{
				all_dimensions_dirichlet_ = all_dimensions_dirichlet_ && dd(0) && dd(1) && dd(2);
				boundary_ids_.push_back(bc_tag);
				bc_.emplace_back();
				bc_.back().init(value, dd);
			}
		}

		void PointBasedTensorProblem::add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri, const int coord, const Eigen::Matrix<bool, 3, 1> &dd, const bool is_neumann)
		{
			if (is_neumann)
			{
				neumann_boundary_ids_.push_back(bc_tag);
				neumann_bc_.emplace_back();
				neumann_bc_.back().init(pts.block(0, 0, pts.rows(), 2), tri, func, coord, dd);
			}
			else
			{
				all_dimensions_dirichlet_ = all_dimensions_dirichlet_ && dd(0) && dd(1) && dd(2);

				boundary_ids_.push_back(bc_tag);
				bc_.emplace_back();
				bc_.back().init(pts.block(0, 0, pts.rows(), 2), tri, func, coord, dd);
			}
		}

		void PointBasedTensorProblem::add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps, const int coord, const Eigen::Matrix<bool, 3, 1> &dd, const bool is_neumann)
		{
			if (is_neumann)
			{
				neumann_boundary_ids_.push_back(bc_tag);
				neumann_bc_.emplace_back();
				if (coord >= 0)
					neumann_bc_.back().init(pts.block(0, 0, pts.rows(), 2), func, rbf, eps, coord, dd);
				else
					neumann_bc_.back().init(pts, func, rbf, eps, coord, dd);
			}
			else
			{
				all_dimensions_dirichlet_ = all_dimensions_dirichlet_ && dd(0) && dd(1) && dd(2);

				boundary_ids_.push_back(bc_tag);
				bc_.emplace_back();
				if (coord >= 0)
					bc_.back().init(pts.block(0, 0, pts.rows(), 2), func, rbf, eps, coord, dd);
				else
					bc_.back().init(pts, func, rbf, eps, coord, dd);
			}
		}

		bool PointBasedTensorProblem::is_dimension_dirichet(const int tag, const int dim) const
		{
			if (all_dimensions_dirichlet())
				return true;

			for (size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if (tag == boundary_ids_[b])
				{
					return bc_[b].is_dirichet_dim(dim);
				}
			}

			assert(false);
			return true;
		}

		void PointBasedTensorProblem::set_parameters(const json &params)
		{
			if (initialized_)
				return;

			if (params.contains("scaling"))
			{
				scaling_ = params["scaling"];
			}

			if (params.contains("rhs"))
			{
				rhs_ = params["rhs"];
			}

			if (params.contains("translation"))
			{
				const auto &j_translation = params["translation"];

				assert(j_translation.is_array());
				assert(j_translation.size() == 3);

				for (int k = 0; k < 3; ++k)
					translation_(k) = j_translation[k];
			}

			if (params.contains("boundary_ids"))
			{
				boundary_ids_.clear();
				auto j_boundary = params["boundary_ids"];

				boundary_ids_.resize(j_boundary.size());
				bc_.resize(boundary_ids_.size());

				for (size_t i = 0; i < boundary_ids_.size(); ++i)
				{
					boundary_ids_[i] = j_boundary[i]["id"];
					const auto ff = j_boundary[i]["value"];
					const bool all_d = bc_[i].init(ff);
					all_dimensions_dirichlet_ = all_dimensions_dirichlet_ && all_d;
				}
			}

			if (params.contains("neumann_boundary_ids"))
			{
				neumann_boundary_ids_.clear();
				auto j_boundary = params["neumann_boundary_ids"];

				neumann_boundary_ids_.resize(j_boundary.size());
				neumann_bc_.resize(neumann_boundary_ids_.size());

				for (size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
				{
					neumann_boundary_ids_[i] = j_boundary[i]["id"];
					const auto ff = j_boundary[i]["value"];
					neumann_bc_[i].init(ff);
				}
			}
		}
	} // namespace problem
} // namespace polyfem
