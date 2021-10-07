#include <polyfem/Obstacle.hpp>

#include <polyfem/MeshUtils.hpp>
#include <polyfem/Logger.hpp>

#include <igl/edges.h>
#include <ipc/utils/faces_to_edges.hpp>

namespace polyfem
{
	void Obstacle::clear()
	{
		dim_ = 0;
		v_.resize(0, 0);
		f_.resize(0, 0);
		e_.resize(0, 0);
		f_2_e_.resize(0, 0);

		in_f_.resize(0, 0);
		in_e_.resize(0, 0);

		displacements_.clear();
		displacements_interpolation_.clear();

		endings_.clear();
	}

	void Obstacle::init(const json &meshes, const std::string &root_path)
	{
		clear();

		for (int i = 0; i < meshes.size(); i++)
		{
			json jmesh;
			Eigen::MatrixXd tmp_vertices;
			Eigen::MatrixXi tmp_cells;
			std::vector<std::vector<int>> tmp_elements;
			std::vector<std::vector<double>> tmp_weights;

			read_mesh_from_json(meshes[i], root_path, tmp_vertices, tmp_cells, tmp_elements, tmp_weights, jmesh);

			if (tmp_vertices.size() == 0 || tmp_cells.size() == 0)
			{
				continue;
			}

			if (dim_ == 0)
			{
				dim_ = tmp_vertices.cols();
			}
			else if (dim_ != tmp_vertices.cols())
			{
				logger().error("Mixed dimension meshes is not implemented!");
				continue;
			}

			if (tmp_cells.cols() == 2)
			{
				e_.conservativeResize(e_.rows() + tmp_cells.rows(), 2);
				e_.bottomRows(tmp_cells.rows()) = tmp_cells.array() + v_.rows();
			}
			else if (tmp_cells.cols() == 3)
			{
				Eigen::MatrixXi tmp_edges, tmp_f_2_e;
				igl::edges(tmp_cells, tmp_edges);
				tmp_f_2_e = ipc::faces_to_edges(tmp_cells, tmp_edges);

				f_.conservativeResize(f_.rows() + tmp_cells.rows(), 3);
				f_.bottomRows(tmp_cells.rows()) = tmp_cells.array() + v_.rows();

				f_2_e_.conservativeResize(f_2_e_.rows() + tmp_f_2_e.rows(), tmp_f_2_e.cols());
				f_2_e_.bottomRows(tmp_f_2_e.rows()) = tmp_f_2_e.array() + e_.rows();

				e_.conservativeResize(e_.rows() + tmp_edges.rows(), 2);
				e_.bottomRows(tmp_edges.rows()) = tmp_edges.array() + v_.rows();
			}
			else
			{
				logger().error("Obstacle supports only segments and triangles!");
				continue;
			}

			v_.conservativeResize(v_.rows() + tmp_vertices.rows(), dim_);
			v_.bottomRows(tmp_vertices.rows()) = tmp_vertices;

			displacements_.emplace_back();
			for (size_t k = 0; k < dim_; ++k)
				displacements_.back()[k].init(jmesh["displacement"][k]);

			const std::string interpolation = jmesh.contains("interpolation") ? jmesh["interpolation"].get<std::string>() : "";
			if (interpolation.empty())
				displacements_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
			else
				displacements_interpolation_.emplace_back(Interpolation::build(interpolation));

			endings_.push_back(v_.rows());
		}
	}

	void Obstacle::update_displacement(const double t, Eigen::MatrixXd &sol) const
	{
		const int offset = sol.rows() - v_.rows() * dim_;

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
					sol(offset + i * dim_ + d) = disp[d](x, y, z, t) * interp_val;
				}
			}

			start = to;
		}
	}

	void Obstacle::set_zero(Eigen::MatrixXd &sol) const
	{
		const int offset = sol.rows() - v_.rows() * dim_;

		int start = 0;

		for (int k = 0; k < endings_.size(); ++k)
		{
			const int to = endings_[k];

			for (int i = start; i < to; ++i)
			{
				for (int d = 0; d < dim_; ++d)
				{
					sol(offset + i * dim_ + d) = 0;
				}
			}

			start = to;
		}
	}
} // namespace polyfem