#include <polyfem/NodeProblem.hpp>
#include <polyfem/Logger.hpp>

#include <iostream>
#include <fstream>

namespace polyfem
{

	NodeValues::NodeValues()
	{ }

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
			raw_dirichelt_.push_back(flag);


			double temp;
			raw_data_.emplace_back();
			std::vector<double> &currentLine = raw_data_.back();

			while (input >> temp)
				currentLine.push_back(temp);
		}

		file.close();
	}

	void NodeValues::init(const Mesh &mesh)
	{
		const int n_primitive = mesh.is_volume() ? mesh.n_faces() : mesh.n_edges();
		if(!data_.empty())
		{
			assert(data_.size() == n_primitive);
			assert(dirichelt_.size() == n_primitive);
			return;
		}

		data_.resize(n_primitive);
		dirichelt_.resize(n_primitive);

		for(size_t i = 0; i < raw_ids_.size(); ++i)
		{
			const int index = raw_ids_[i];

			dirichelt_[index] = raw_dirichelt_[i];

			data_[index].resize(raw_data_[i].size());
			for(size_t j = 0; j < raw_data_[i].size(); ++j)
				data_[index](j) = raw_data_[i][j];
		}
	}

	double NodeValues::interpolate(const int p_id, const Eigen::MatrixXd &uv, bool is_dirichelt) const
	{
		assert(dirichelt_[p_id] == is_dirichelt);

		const auto &vals = data_[p_id];
		assert(uv.size() == vals.size());

		return (uv.array()*vals.transpose().array()).sum();
	}

	NodeProblem::NodeProblem(const std::string &name)
	: Problem(name), rhs_(0), is_all_(false)
	{
	}

	void NodeProblem::init(const Mesh &mesh)
	{
		values_.init(mesh);
	}

	void NodeProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Constant(pts.rows(), pts.cols(), rhs_);
		val *= t;
	}

	void NodeProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);

		for(long i = 0; i < uv.rows(); ++i)
		{
			const int p_id = global_ids(i);
			if(is_all_)
			{
				val(i) = values_.dirichlet_interpolate(p_id, uv.row(i));
			}
			else
			{
				const int id = mesh.get_boundary_id(p_id);
				for(size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if(id == boundary_ids_[b])
					{
						val(i) = values_.dirichlet_interpolate(p_id, uv.row(i));
						break;
					}
				}
			}
		}

		val *= t;
	}

	void NodeProblem::neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);

		for(long i = 0; i < uv.rows(); ++i)
		{
			const int p_id = global_ids(i);
			const int id = mesh.get_boundary_id(p_id);
			for(size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if(id == boundary_ids_[b])
				{
					val(i) = values_.neumann_interpolate(p_id, uv.row(i));
					break;
				}
			}
		}

		val *= t;
	}

	bool NodeProblem::is_dimention_dirichet(const int tag, const int dim) const
	{
		if(all_dimentions_dirichelt())
			return true;

		if(is_all_)
		{
			return dirichelt_dimentions_[0][dim];
		}

		for(size_t b = 0; b < boundary_ids_.size(); ++b)
		{
			if(tag == boundary_ids_[b])
			{
				auto &tmp = dirichelt_dimentions_[b];
				return tmp[dim];
			}
		}

		assert(false);
		return true;
	}

	void NodeProblem::set_parameters(const json &params)
	{
		if(params.find("rhs") != params.end())
		{
			rhs_ = params["rhs"];
		}

		if(params.find("values") != params.end())
		{
			const std::string path = params["values"];
			values_.load(path);
		}



		if(params.find("dirichlet_boundary") != params.end())
		{
			boundary_ids_.clear();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(j_boundary.size());
			dirichelt_dimentions_.resize(j_boundary.size());

			for(size_t i = 0; i < boundary_ids_.size(); ++i)
			{
				if(j_boundary[i]["id"] == "all")
				{
					assert(boundary_ids_.size() == 1);
					is_all_ = true;
					boundary_ids_.clear();
				}
				else
					boundary_ids_[i] = j_boundary[i]["id"];

				dirichelt_dimentions_[i].setConstant(false);

				if(j_boundary[i].find("dimension") != j_boundary[i].end())
				{
					all_dimentions_dirichelt_ = false;
					auto &tmp = j_boundary[i]["dimension"];
					assert(tmp.is_array());
					for(size_t k = 0; k < tmp.size(); ++k)
						dirichelt_dimentions_[i](k) = tmp[k];
				}

			}
		}

		if(params.find("neumann_boundary") != params.end())
		{
			neumann_boundary_ids_.clear();
			auto j_boundary = params["neumann_boundary"];

			neumann_boundary_ids_.resize(j_boundary.size());

			for(size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				neumann_boundary_ids_[i] = j_boundary[i]["id"];
			}
		}
	}
}
