#include <polyfem/GenericProblem.hpp>
#include <polyfem/State.hpp>

#include <iostream>

namespace polyfem
{
	GenericTensorProblem::GenericTensorProblem(const std::string &name)
	: Problem(name)
	{
		// boundary_ids_ = {2};
		// neumann_boundary_ids_ = {4};

		// forces_.resize(1);
		// forces_.front()(0).init(0.1);
		// forces_.front()(1).init(0);
		// forces_.front()(2).init(0);

		// displacements_.resize(1);
		// displacements_.front()(0).init(0);
		// displacements_.front()(1).init(0);
		// displacements_.front()(2).init(0);

		// dirichelt_dimentions_.resize(1);
		// dirichelt_dimentions_.front().setConstant(true);
	}

	void GenericTensorProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		// val *= t;
	}

	bool GenericTensorProblem::is_dimention_dirichet(const int tag, const int dim) const
	{
		if(all_dimentions_dirichelt())
			return true;

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

	void GenericTensorProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for(long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));
			for(size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if(id == boundary_ids_[b])
				{
					for(int d = 0; d < val.cols(); ++d)
						val(i, d) = pts.cols() == 2 ? displacements_[b](d)(pts(i, 0), pts(i, 1)) : displacements_[b](d)(pts(i, 0), pts(i, 1), pts(i, 2));
					break;
				}
			}
		}

		val *= t;
	}

	void GenericTensorProblem::neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for(long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));

			for(size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
			{
				if(id == neumann_boundary_ids_[b])
				{
					for(int d = 0; d < val.cols(); ++d)
						val(i, d) = pts.cols() == 2 ? forces_[b](d)(pts(i, 0), pts(i, 1)) : forces_[b](d)(pts(i, 0), pts(i, 1), pts(i, 2));
					break;
				}
			}
		}

		val *= t;
	}

	void GenericTensorProblem::set_parameters(const json &params)
	{
		if(params.find("use_mixed_formulation") != params.end())
		{
			is_mixed_ = params["use_mixed_formulation"];
		}

		if(params.find("dirichlet_boundary") != params.end())
		{
			boundary_ids_.clear();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(j_boundary.size());
			displacements_.resize(j_boundary.size());
			dirichelt_dimentions_.resize(j_boundary.size());


			for(size_t i = 0; i < boundary_ids_.size(); ++i)
			{
				boundary_ids_[i] = j_boundary[i]["id"];

				auto ff = j_boundary[i]["value"];
				if(ff.is_array())
				{
					for(size_t k = 0; k < ff.size(); ++k)
						displacements_[i](k).init(ff[k]);
				}
				else
				{
					assert(false);
					displacements_[i][0].init(0);
					displacements_[i][1].init(0);
					displacements_[i][2].init(0);
				}

				dirichelt_dimentions_[i].setConstant(true);
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
			forces_.resize(j_boundary.size());


			for(size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				neumann_boundary_ids_[i] = j_boundary[i]["id"];

				auto ff = j_boundary[i]["value"];
				if(ff.is_array())
				{
					for(size_t k = 0; k < ff.size(); ++k)
						forces_[i](k).init(ff[k]);
				}
				else
				{
					assert(false);
					forces_[i][0].init(0);
					forces_[i][1].init(0);
					forces_[i][2].init(0);
				}
			}
		}
	}












	GenericScalarProblem::GenericScalarProblem(const std::string &name)
	: Problem(name)
	{	}

	void GenericScalarProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);
		// val *= t;
	}

	void GenericScalarProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);

		for(long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));
			for(size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if(id == boundary_ids_[b])
				{
					val(i) = pts.cols() == 2 ? dirichlet_[b](0)(pts(i, 0), pts(i, 1)) : dirichlet_[b](0)(pts(i, 0), pts(i, 1), pts(i, 2));
					break;
				}
			}
		}

		val *= t;
	}

	void GenericScalarProblem::neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);

		for(long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));

			for(size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
			{
				if(id == neumann_boundary_ids_[b])
				{
					val(i) = pts.cols() == 2 ? neumann_[b](0)(pts(i, 0), pts(i, 1)) : neumann_[b](0)(pts(i, 0), pts(i, 1), pts(i, 2));
					break;
				}
			}
		}

		val *= t;
	}

	void GenericScalarProblem::set_parameters(const json &params)
	{
		if(params.find("dirichlet_boundary") != params.end())
		{
			boundary_ids_.clear();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(j_boundary.size());
			dirichlet_.resize(j_boundary.size());


			for(size_t i = 0; i < boundary_ids_.size(); ++i)
			{
				boundary_ids_[i] = j_boundary[i]["id"];

				auto ff = j_boundary[i]["value"];
				dirichlet_[i](0).init(ff);
			}
		}

		if(params.find("neumann_boundary") != params.end())
		{
			neumann_boundary_ids_.clear();
			auto j_boundary = params["neumann_boundary"];

			neumann_boundary_ids_.resize(j_boundary.size());
			neumann_.resize(j_boundary.size());


			for(size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				neumann_boundary_ids_[i] = j_boundary[i]["id"];

				auto ff = j_boundary[i]["value"];
				neumann_[i](0).init(ff);
			}
		}
	}




}
