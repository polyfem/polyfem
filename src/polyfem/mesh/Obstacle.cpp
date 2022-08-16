#include <polyfem/mesh/Obstacle.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <igl/edges.h>
#include <ipc/utils/eigen_ext.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace utils;

	namespace mesh
	{
		Obstacle::Obstacle()
		{
			clear();
		}

		void Obstacle::clear()
		{
			dim_ = 0;
			v_.resize(0, 0);
			f_.resize(0, 0);
			e_.resize(0, 0);

			in_f_.resize(0, 0);
			in_e_.resize(0, 0);
			in_v_.resize(0);

			displacements_.clear();
			displacements_interpolation_.clear();

			endings_.clear();

			planes_.clear();
		}

		void Obstacle::append_mesh(
			const Eigen::MatrixXd &vertices,
			const Eigen::VectorXi &codim_vertices,
			const Eigen::MatrixXi &codim_edges,
			const Eigen::MatrixXi &faces,
			const json &displacement)
		{
			if (vertices.size() == 0)
				return;

			if (dim_ == 0)
				dim_ = vertices.cols();
			assert(dim_ == vertices.cols());

			if (codim_vertices.size())
			{
				codim_v_.conservativeResize(codim_v_.size() + codim_vertices.size());
				codim_v_.tail(codim_vertices.size()) = codim_vertices.array() + v_.rows();

				in_v_.conservativeResize(in_v_.size() + codim_vertices.size());
				in_v_.tail(codim_vertices.size()) = codim_vertices.array() + v_.rows();
			}

			if (codim_edges.size())
			{
				e_.conservativeResize(e_.rows() + codim_edges.rows(), 2);
				e_.bottomRows(codim_edges.rows()) = codim_edges.array() + v_.rows();

				in_e_.conservativeResize(in_e_.rows() + codim_edges.rows(), 2);
				in_e_.bottomRows(codim_edges.rows()) = codim_edges.array() + v_.rows();
			}

			if (faces.size() && faces.cols() == 3)
			{
				f_.conservativeResize(f_.rows() + faces.rows(), 3);
				f_.bottomRows(faces.rows()) = faces.array() + v_.rows();

				in_f_.conservativeResize(in_f_.rows() + faces.rows(), 3);
				in_f_.bottomRows(faces.rows()) = faces.array() + v_.rows();

				Eigen::MatrixXi edges;
				igl::edges(faces, edges);

				e_.conservativeResize(e_.rows() + edges.rows(), 2);
				e_.bottomRows(edges.rows()) = edges.array() + v_.rows();
			}
			else if (faces.size())
			{
				logger().error("Obstacle supports only segments and triangles!");
				return;
			}

			v_.conservativeResize(v_.rows() + vertices.rows(), dim_);
			v_.bottomRows(vertices.rows()) = vertices;

			displacements_.emplace_back();
			for (size_t d = 0; d < dim_; ++d)
			{
				assert(displacement["value"].is_array());
				displacements_.back()[d].init(displacement["value"][d]);
			}

			if (displacement.contains("interpolation"))
				displacements_interpolation_.emplace_back(Interpolation::build(displacement["interpolation"]));
			else
				displacements_interpolation_.emplace_back(std::make_shared<NoInterpolation>());

			endings_.push_back(v_.rows());
		}

		void Obstacle::append_plane(const VectorNd &origin, const VectorNd &normal)
		{
			if (dim_ == 0)
				dim_ = origin.size();
			assert(normal.size() >= dim_);
			assert(origin.size() >= dim_);
			planes_.emplace_back(origin.head(dim_), normal.head(dim_).normalized());
		}

		void Obstacle::change_displacement(const int oid, const Eigen::RowVector3d &val, const std::string &interp)
		{
			change_displacement(oid, val, interp.empty() ? std::make_shared<NoInterpolation>() : Interpolation::build(interp));
		}
		void Obstacle::change_displacement(const int oid, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::string &interp)
		{
			change_displacement(oid, func, interp.empty() ? std::make_shared<NoInterpolation>() : Interpolation::build(interp));
		}
		void Obstacle::change_displacement(const int oid, const json &val, const std::string &interp)
		{
			change_displacement(oid, val, interp.empty() ? std::make_shared<NoInterpolation>() : Interpolation::build(interp));
		}

		void Obstacle::change_displacement(const int oid, const Eigen::RowVector3d &val, const std::shared_ptr<Interpolation> &interp)
		{
			for (size_t k = 0; k < val.size(); ++k)
				displacements_[oid][k].init(val[k]);
			displacements_interpolation_[oid] = interp;
		}

		void Obstacle::change_displacement(const int oid, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			for (size_t k = 0; k < displacements_.back().size(); ++k)
				displacements_[oid][k].init(func, k);
			displacements_interpolation_[oid] = interp;
		}

		void Obstacle::change_displacement(const int oid, const json &val, const std::shared_ptr<Interpolation> &interp)
		{
			for (size_t k = 0; k < val.size(); ++k)
				displacements_[oid][k].init(val[k]);
			displacements_interpolation_[oid] = interp;
		}

		void Obstacle::update_displacement(const double t, Eigen::MatrixXd &sol) const
		{
			// NOTE: assumes obstacle displacements is stored at the bottom of sol
			const int offset = sol.rows() - v_.rows() * (sol.cols() == 1 ? dim_ : 1);

			int start = 0;

			for (int k = 0; k < endings_.size(); ++k)
			{
				const int to = endings_[k];
				const auto &disp = displacements_[k];
				const auto &interp = displacements_interpolation_[k];

				for (int i = start; i < to; ++i)
				{
					double x = v_(i, 0), y = v_(i, 1), z = dim_ == 2 ? 0 : v_(i, 2);
					const double interp_val = interp->eval(t);

					for (int d = 0; d < dim_; ++d)
					{
						if (sol.cols() == 1)
						{
							sol(offset + i * dim_ + d) = disp[d](x, y, z, t) * interp_val;
						}
						else
						{
							sol(offset + i, d) = disp[d](x, y, z, t) * interp_val;
						}
					}
				}

				start = to;
			}
			assert(endings_.empty() || (start * (sol.cols() == 1 ? dim_ : 1) + offset) == sol.rows());
		}

		void Obstacle::set_zero(Eigen::MatrixXd &sol) const
		{
			// NOTE: assumes obstacle displacements is stored at the bottom of sol
			sol.bottomRows(v_.rows() * (sol.cols() == 1 ? dim_ : 1)).setZero();
		}

		void Obstacle::Plane::construct_vis_mesh()
		{
			constexpr int size_x = 10;
			constexpr int size_y = 10;

			if (dim_ == 2)
			{
				Eigen::Vector2d tangent(normal().y(), -normal().x());

				vis_v_.resize(size_x + 1, 2);
				for (int x = 0; x <= size_x; ++x)
				{
					vis_v_.row(x) = (x - size_x / 2.0) * tangent;
				}

				vis_e_.resize(size_x, 2);
				for (int x = 0; x < size_x; ++x)
				{
					vis_e_.row(x) << x, x + 1;
				}
			}
			else
			{
				assert(dim_ == 3);

				Eigen::Vector3d cross_x = ipc::cross(Eigen::Vector3d::UnitX(), normal());
				Eigen::Vector3d cross_y = ipc::cross(Eigen::Vector3d::UnitY(), normal());

				Eigen::Vector3d tangent_x, tangent_y;
				if (cross_x.squaredNorm() > cross_y.squaredNorm())
				{
					tangent_x = cross_x.normalized();
					tangent_y = ipc::cross(normal(), cross_x).normalized();
				}
				else
				{
					tangent_x = cross_y.normalized();
					tangent_y = ipc::cross(normal(), cross_y).normalized();
				}

				vis_v_.resize((size_x + 1) * (size_y + 1), 3);
				for (int x = 0; x <= size_x; ++x)
				{
					for (int y = 0; y <= size_y; ++y)
					{
						vis_v_.row(x * size_y + y) = (x - size_x / 2.0) * tangent_x + (y - size_y / 2.0) * tangent_y;
					}
				}

				vis_f_.resize(2 * size_x * size_y, 3);
				for (int x = 0; x < size_x; ++x)
				{
					for (int y = 0; y < size_y; ++y)
					{
						vis_f_.row(2 * (x * size_x + y)) << x * size_x + y, x * size_x + y + 1, (x + 1) * size_x + y;
						vis_f_.row(2 * (x * size_x + y) + 1) << x * size_x + y + 1, (x + 1) * size_x + (y + 1), (x + 1) * size_x + y;
					}
				}

				igl::edges(vis_f_, vis_e_);
			}
		}
	} // namespace mesh
} // namespace polyfem