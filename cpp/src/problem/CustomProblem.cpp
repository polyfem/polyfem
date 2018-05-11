#include "CustomProblem.hpp"
#include "State.hpp"
#include "MatrixUtils.hpp"

#include <iostream>

namespace poly_fem
{
	CustomProblem::CustomProblem(const std::string &name)
	: Problem(name), rhs_(0), scaling_(1)
	{
		translation_.setZero();
	}

	void CustomProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Constant(pts.rows(), pts.cols(), rhs_);
	}

	void CustomProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for(long i = 0; i < pts.rows(); ++i)
		{
			const auto id = mesh.get_boundary_id(global_ids(i));
			const auto it = std::find(boundary_ids_.begin(), boundary_ids_.end(), id);

			if(it == boundary_ids_.end())
				continue;

			const auto index = std::distance(boundary_ids_.begin(), it);
			const bool is_val = val_bc_[index];

			if(is_val)
			{
				const auto &bc_val = bc_[index];
				val.row(i) = bc_val*scaling_;
			}
			else
			{
				const auto &pt3d = pts.row(i) + translation_.transpose();
				Eigen::Matrix<double, 1, 2> pt;
				if(id == 1 || id == 3)
				    pt << pt3d(0)/scaling_, pt3d(1)/scaling_;

				const auto &bc_fun = funcs_[index];
				const Eigen::MatrixXd value = bc_fun.interpolate(pt) * scaling_;
				val.row(i) = value;
				// std::cout<<pt<<"->"<<value<<std::endl;
			}
		}
	}

	void CustomProblem::init(const std::vector<int> &b_id)
	{
		scaling_  = 1;
		translation_.setZero();

		boundary_ids_ = b_id;
		bc_.resize(boundary_ids_.size());
		val_bc_.resize(boundary_ids_.size());
		funcs_.resize(boundary_ids_.size());

		initialized_ = true;
	}

	void CustomProblem::set_constant(const int index, const Eigen::Vector3d &value)
	{
		bc_[index] = value;
		val_bc_[index] = true;
	}

	void CustomProblem::set_function(const int index, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri)
	{
		funcs_[index] = InterpolatedFunction2d(func, pts.block(0, 0, pts.rows(), 2), tri);
	}

	void CustomProblem::set_parameters(const json &params)
	{
		if(initialized_)
			return;
		
		if(params.find("scaling") != params.end())
		{
			scaling_ = params["scaling"];
		}

		if(params.find("rhs") != params.end())
		{
			rhs_ = params["rhs"];
		}

		if(params.find("translation") != params.end())
		{
			const auto &j_translation = params["translation"];

			assert(j_translation.is_array());
			assert(j_translation.size() == 3);

			for(int k = 0; k < 3; ++k)
				translation_(k) = j_translation[k];
		}

		if(params.find("boundary_ids") != params.end())
		{
			boundary_ids_.clear();
			auto j_boundary_ids = params["boundary_ids"];

			boundary_ids_.resize(j_boundary_ids.size());
			bc_.resize(boundary_ids_.size());
			val_bc_.resize(boundary_ids_.size());
			funcs_.resize(boundary_ids_.size());

			for(size_t i = 0; i < boundary_ids_.size(); ++i)
			{
				boundary_ids_[i] = j_boundary_ids[i];
				const auto id = std::to_string(boundary_ids_[i]);

				if(params.find(id) != params.end())
				{
					auto val = params[id];
					if(val.is_array()){
						assert(val.size() == 3);
						for(int k = 0; k < 3; ++k)
							bc_[i](k) = val[k];
						val_bc_[i] = true;
					}
					else if(val.is_object())
					{
					    Eigen::MatrixXd f, pts;
					    Eigen::MatrixXi tri;
					    read_matrix(val["function"], f);
					    read_matrix(val["points"], pts);
					    pts = pts.block(0, 0, pts.rows(), 2).eval();
					    read_matrix(val["triangles"], tri);

                        funcs_[i] = InterpolatedFunction2d(f, pts, tri);
					}
					else
					{
						bc_[i] = Eigen::Vector3d::Zero();
						val_bc_[i] = true;
					}
				}
				else{
					bc_[i] = Eigen::Vector3d::Zero();
					val_bc_[i] = true;
				}
			}
		}
	}
}
