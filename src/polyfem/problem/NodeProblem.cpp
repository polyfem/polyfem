#include "NodeProblem.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <iostream>
#include <fstream>

namespace polyfem
{
	using namespace utils;

	namespace problem
	{
		NodeValues::NodeValues()
		{
		}

		void NodeValues::load(const std::string &path)
		{
			std::fstream file;
			file.open(path.c_str());

			if (!file.good())
			{
				logger().error("Failed to open file: {}", path);
				file.close();
			}

			std::string s;

			while (getline(file, s))
			{
				std::stringstream input(s);

				int id;
				input >> id;
				raw_ids_.push_back(id);

				bool flag;
				input >> flag;
				raw_dirichlet_.push_back(flag);

				double temp;
				raw_data_.emplace_back();
				std::vector<double> &currentLine = raw_data_.back();

				while (input >> temp)
					currentLine.push_back(temp);
			}

			file.close();
		}

		void NodeValues::init(const mesh::Mesh &mesh)
		{
			const int n_primitive = mesh.is_volume() ? mesh.n_faces() : mesh.n_edges();
			if (!data_.empty())
			{
				assert(data_.size() == n_primitive);
				assert(dirichlet_.size() == n_primitive);
				return;
			}

			data_.resize(n_primitive);
			dirichlet_.resize(n_primitive);

			for (size_t i = 0; i < raw_ids_.size(); ++i)
			{
				const int index = raw_ids_[i];

				dirichlet_[index] = raw_dirichlet_[i];

				data_[index].resize(raw_data_[i].size());
				for (size_t j = 0; j < raw_data_[i].size(); ++j)
					data_[index](j) = raw_data_[i][j];
			}
		}

		double NodeValues::interpolate(const int p_id, const Eigen::MatrixXd &uv, bool is_dirichlet) const
		{
			assert(dirichlet_[p_id] == is_dirichlet);

			const auto &vals = data_[p_id];
			assert(uv.size() == vals.size());

			return (uv.array() * vals.transpose().array()).sum();
		}

		NodeProblem::NodeProblem(const std::string &name)
			: Problem(name), rhs_(0), is_all_(false)
		{
		}

		void NodeProblem::init(const mesh::Mesh &mesh)
		{
			values_.init(mesh);
		}

		void NodeProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Constant(pts.rows(), pts.cols(), rhs_);
		}

		void NodeProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), 1);

			for (long i = 0; i < uv.rows(); ++i)
			{
				const int p_id = global_ids(i);
				if (is_all_)
				{
					val(i) = values_.dirichlet_interpolate(p_id, uv.row(i));
				}
				else
				{
					const int id = mesh.get_boundary_id(p_id);
					for (size_t b = 0; b < boundary_ids_.size(); ++b)
					{
						if (id == boundary_ids_[b])
						{
							val(i) = values_.dirichlet_interpolate(p_id, uv.row(i));
							break;
						}
					}
				}
			}
		}

		void NodeProblem::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), 1);

			for (long i = 0; i < uv.rows(); ++i)
			{
				const int p_id = global_ids(i);
				const int id = mesh.get_boundary_id(p_id);
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if (id == boundary_ids_[b])
					{
						val(i) = values_.neumann_interpolate(p_id, uv.row(i));
						break;
					}
				}
			}
		}

		bool NodeProblem::is_dimension_dirichet(const int tag, const int dim) const
		{
			if (all_dimensions_dirichlet())
				return true;

			if (is_all_)
			{
				return dirichlet_dimensions_[0][dim];
			}

			for (size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if (tag == boundary_ids_[b])
				{
					auto &tmp = dirichlet_dimensions_[b];
					return tmp[dim];
				}
			}

			assert(false);
			return true;
		}

		void NodeProblem::set_parameters(const json &params)
		{
			if (params.contains("rhs"))
			{
				rhs_ = params["rhs"];
			}

			if (params.contains("values"))
			{
				const std::string path = params["values"];
				values_.load(path);
			}

			if (params.contains("dirichlet_boundary"))
			{
				boundary_ids_.clear();
				std::vector<json> j_boundary = utils::json_as_array(params["dirichlet_boundary"]);

				boundary_ids_.resize(j_boundary.size());
				dirichlet_dimensions_.resize(j_boundary.size());

				for (size_t i = 0; i < boundary_ids_.size(); ++i)
				{
					if (j_boundary[i]["id"] == "all")
					{
						assert(boundary_ids_.size() == 1);
						is_all_ = true;
						boundary_ids_.clear();
					}
					else
						boundary_ids_[i] = j_boundary[i]["id"];

					dirichlet_dimensions_[i].setConstant(false);

					if (j_boundary[i].contains("dimension"))
					{
						all_dimensions_dirichlet_ = false;
						auto &tmp = j_boundary[i]["dimension"];
						assert(tmp.is_array());
						for (size_t k = 0; k < tmp.size(); ++k)
							dirichlet_dimensions_[i](k) = tmp[k];
					}
				}
			}

			if (params.contains("neumann_boundary"))
			{
				neumann_boundary_ids_.clear();
				auto j_boundary = params["neumann_boundary"];

				neumann_boundary_ids_.resize(j_boundary.size());

				for (size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
				{
					neumann_boundary_ids_[i] = j_boundary[i]["id"];
				}
			}
		}
	} // namespace problem
} // namespace polyfem
