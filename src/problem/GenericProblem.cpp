#include <polyfem/GenericProblem.hpp>
#include <polyfem/State.hpp>

#include <iostream>

namespace polyfem
{
	GenericTensorProblem::GenericTensorProblem(const std::string &name)
		: Problem(name), is_all_(false)
	{
	}

	void GenericTensorProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());

		if (is_rhs_zero())
		{
			val.setZero();
			return;
		}

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? rhs_[j](pts(i, 0), pts(i, 1)) : rhs_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}

		// val.col(i).setConstant(rhs_(i));
		// val *= t;
	}

	bool GenericTensorProblem::is_dimention_dirichet(const int tag, const int dim) const
	{
		if (all_dimentions_dirichelt())
			return true;

		if (is_all_)
		{
			assert(dirichelt_dimentions_.size() == 1);
			return dirichelt_dimentions_[0][dim];
		}

		for (size_t b = 0; b < boundary_ids_.size(); ++b)
		{
			if (tag == boundary_ids_[b])
			{
				auto &tmp = dirichelt_dimentions_[b];
				return tmp[dim];
			}
		}

		assert(false);
		return true;
	}

	void GenericTensorProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for (long i = 0; i < pts.rows(); ++i)
		{
			if (is_all_)
			{
				assert(displacements_.size() == 1);
				for (int d = 0; d < val.cols(); ++d)
					val(i, d) = pts.cols() == 2 ? displacements_[0][d](pts(i, 0), pts(i, 1)) : displacements_[0][d](pts(i, 0), pts(i, 1), pts(i, 2));
			}
			else
			{
				const int id = mesh.get_boundary_id(global_ids(i));
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if (id == boundary_ids_[b])
					{
						for (int d = 0; d < val.cols(); ++d)
							val(i, d) = pts.cols() == 2 ? displacements_[b][d](pts(i, 0), pts(i, 1)) : displacements_[b][d](pts(i, 0), pts(i, 1), pts(i, 2));
						break;
					}
				}
			}
		}

		val *= t;
	}

	void GenericTensorProblem::neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for (long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));

			for (size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
			{
				if (id == neumann_boundary_ids_[b])
				{
					for (int d = 0; d < val.cols(); ++d)
						val(i, d) = pts.cols() == 2 ? forces_[b][d](pts(i, 0), pts(i, 1)) : forces_[b][d](pts(i, 0), pts(i, 1), pts(i, 2));
					break;
				}
			}

			for (size_t b = 0; b < pressure_boundary_ids_.size(); ++b)
			{
				if (id == pressure_boundary_ids_[b])
				{
					for (int d = 0; d < val.cols(); ++d)
						val(i, d) = (pts.cols() == 2 ? pressures_[b](pts(i, 0), pts(i, 1)) : pressures_[b](pts(i, 0), pts(i, 1), pts(i, 2))) * normals(i, d);
					break;
				}
			}
		}

		val *= t;
	}

	void GenericTensorProblem::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		assert(has_exact_sol());
		const bool planar = pts.cols() == 2;
		val.resize(pts.rows(), pts.cols());

		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? exact_[j](pts(i, 0), pts(i, 1)) : exact_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		const int size = pts.cols();
		val.resize(pts.rows(), pts.cols() * size);
		if (!has_exact_grad_)
			return;

		const bool planar = size == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols() * size; ++j)
				val(i, j) = planar ? exact_grad_[j](pts(i, 0), pts(i, 1)) : exact_grad_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::add_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz)
	{
		boundary_ids_.push_back(id);
		displacements_.emplace_back();
		for (size_t k = 0; k < val.size(); ++k)
			displacements_.back()[k].init(val[k]);

		dirichelt_dimentions_.emplace_back(isx, isy, isz);
	}

	void GenericTensorProblem::add_neumann_boundary(const int id, const Eigen::RowVector3d &val)
	{
		neumann_boundary_ids_.push_back(id);
		forces_.emplace_back();
		for (size_t k = 0; k < val.size(); ++k)
			forces_.back()[k].init(val[k]);
	}

	void GenericTensorProblem::add_pressure_boundary(const int id, const double val)
	{
		pressure_boundary_ids_.push_back(id);
		pressures_.emplace_back();
		pressures_.back().init(val);
	}

	void GenericTensorProblem::add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const bool isx, const bool isy, const bool isz)
	{
		boundary_ids_.push_back(id);
		displacements_.emplace_back();
		for (size_t k = 0; k < displacements_.back().size(); ++k)
			displacements_.back()[k].init(func, k);

		dirichelt_dimentions_.emplace_back(isx, isy, isz);
	}

	void GenericTensorProblem::add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z)> &func)
	{
		neumann_boundary_ids_.push_back(id);
		forces_.emplace_back();
		for (size_t k = 0; k < forces_.back().size(); ++k)
			forces_.back()[k].init(func, k);
	}

	void GenericTensorProblem::add_pressure_boundary(const int id, const std::function<double(double x, double y, double z)> &func)
	{
		pressure_boundary_ids_.push_back(id);
		pressures_.emplace_back();
		pressures_.back().init(func);
	}

	void GenericTensorProblem::set_rhs(double x, double y, double z)
	{
		rhs_[0].init(x);
		rhs_[1].init(y);
		rhs_[2].init(z);
	}

	void GenericTensorProblem::set_parameters(const json &params)
	{
		if (params.find("is_time_dependent") != params.end())
		{
			is_time_dept_ = params["is_time_dependent"];
		}

		if (params.find("rhs") != params.end())
		{
			auto rr = params["rhs"];
			if (rr.is_array())
			{
				for (size_t k = 0; k < rr.size(); ++k)
					rhs_[k].init(rr[k]);
			}
			else
			{
				assert(false);
			}
		}

		if (params.find("exact") != params.end())
		{
			auto ex = params["exact"];
			has_exact_ = !ex.is_null();
			if (has_exact_)
			{
				if (ex.is_array())
				{
					for (size_t k = 0; k < ex.size(); ++k)
						exact_[k].init(ex[k]);
				}
				else
				{
					assert(false);
				}
			}
		}
		else
		{
			has_exact_ = false;
		}

		if (params.find("exact_grad") != params.end())
		{
			auto ex = params["exact_grad"];
			has_exact_grad_ = !ex.is_null();
			if (has_exact_grad_)
			{
				if (ex.is_array())
				{
					for (size_t k = 0; k < ex.size(); ++k)
						exact_grad_[k].init(ex[k]);
				}
				else
				{
					assert(false);
				}
			}
		}
		else
		{
			has_exact_grad_ = false;
		}

		if (params.find("dirichlet_boundary") != params.end())
		{
			// boundary_ids_.clear();
			int offset = boundary_ids_.size();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(offset + j_boundary.size());
			displacements_.resize(offset + j_boundary.size());
			dirichelt_dimentions_.resize(offset + j_boundary.size());

			for (size_t i = offset; i < boundary_ids_.size(); ++i)
			{
				if (j_boundary[i - offset]["id"] == "all")
				{
					assert(boundary_ids_.size() == 1);
					boundary_ids_.clear();
					is_all_ = true;
				}
				else
					boundary_ids_[i] = j_boundary[i - offset]["id"];

				auto ff = j_boundary[i - offset]["value"];
				if (ff.is_array())
				{
					for (size_t k = 0; k < ff.size(); ++k)
						displacements_[i][k].init(ff[k]);
				}
				else
				{
					assert(false);
					displacements_[i][0].init(0);
					displacements_[i][1].init(0);
					displacements_[i][2].init(0);
				}

				dirichelt_dimentions_[i].setConstant(true);
				if (j_boundary[i - offset].find("dimension") != j_boundary[i - offset].end())
				{
					all_dimentions_dirichelt_ = false;
					auto &tmp = j_boundary[i - offset]["dimension"];
					assert(tmp.is_array());
					for (size_t k = 0; k < tmp.size(); ++k)
						dirichelt_dimentions_[i](k) = tmp[k];
				}
			}
		}

		if (params.find("neumann_boundary") != params.end())
		{
			// neumann_boundary_ids_.clear();
			const int offset = neumann_boundary_ids_.size();

			auto j_boundary = params["neumann_boundary"];

			neumann_boundary_ids_.resize(offset + j_boundary.size());
			forces_.resize(offset + j_boundary.size());

			for (size_t i = offset; i < neumann_boundary_ids_.size(); ++i)
			{
				neumann_boundary_ids_[i] = j_boundary[i - offset]["id"];

				auto ff = j_boundary[i - offset]["value"];
				if (ff.is_array())
				{
					for (size_t k = 0; k < ff.size(); ++k)
						forces_[i][k].init(ff[k]);
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

		if (params.find("pressure_boundary") != params.end())
		{
			// pressure_boundary_ids_.clear();
			const int offset = pressure_boundary_ids_.size();

			auto j_boundary = params["pressure_boundary"];

			pressure_boundary_ids_.resize(offset + j_boundary.size());
			pressures_.resize(offset + j_boundary.size());

			for (size_t i = offset; i < pressure_boundary_ids_.size(); ++i)
			{
				pressure_boundary_ids_[i] = j_boundary[i - offset]["id"];

				auto ff = j_boundary[i - offset]["value"];
				pressures_[i].init(ff);
			}
		}

		if (params.find("initial_solution") != params.end())
		{
			auto rr = params["initial_solution"];
			if (rr.is_array())
			{
				for (size_t k = 0; k < rr.size(); ++k)
					initial_position_[k].init(rr[k]);
			}
			else
			{
				assert(false);
			}
		}

		if (params.find("initial_velocity") != params.end())
		{
			auto rr = params["initial_velocity"];
			if (rr.is_array())
			{
				for (size_t k = 0; k < rr.size(); ++k)
					initial_velocity_[k].init(rr[k]);
			}
			else
			{
				assert(false);
			}
		}

		if (params.find("initial_acceleration") != params.end())
		{
			auto rr = params["initial_acceleration"];
			if (rr.is_array())
			{
				for (size_t k = 0; k < rr.size(); ++k)
					initial_acceleration_[k].init(rr[k]);
			}
			else
			{
				assert(false);
			}
		}
	}

	int GenericTensorProblem::n_incremental_load_steps(const double diag) const
	{
		double max;
		Eigen::Matrix<double, 1, 3, Eigen::RowMajor> tmp;

		for (const auto &vec : forces_)
		{
			const int dim = vec.size();
			for (int i = 0; i < dim; ++i)
			{
				tmp[i] = vec[i](0, 0, 0);
			}

			max = std::max(max, tmp.norm());
		}

		for (const auto &vec : displacements_)
		{
			const int dim = vec.size();
			for (int i = 0; i < dim; ++i)
			{
				tmp[i] = vec[i](0, 0, 0);
			}

			max = std::max(max, tmp.norm());
		}

		const int dim = rhs_.size();
		for (int i = 0; i < dim; ++i)
		{
			tmp[i] = rhs_[i](0, 0, 0);
		}

		max = std::max(max, tmp.norm());

		return 4 * max / diag;
	}

	void GenericTensorProblem::velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		//TODO
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void GenericTensorProblem::acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		//TODO
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void GenericTensorProblem::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? initial_position_[j](pts(i, 0), pts(i, 1)) : initial_position_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::initial_velocity(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? initial_velocity_[j](pts(i, 0), pts(i, 1)) : initial_velocity_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::initial_acceleration(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? initial_acceleration_[j](pts(i, 0), pts(i, 1)) : initial_acceleration_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::clear()
	{
		all_dimentions_dirichelt_ = true;
		has_exact_ = false;
		has_exact_grad_ = false;
		is_time_dept_ = false;

		forces_.clear();
		displacements_.clear();
		pressures_.clear();

		dirichelt_dimentions_.clear();

		for (int i = 0; i < rhs_.size(); ++i)
			rhs_[i].clear();
		for (int i = 0; i < exact_.size(); ++i)
			exact_[i].clear();
		for (int i = 0; i < exact_grad_.size(); ++i)
			exact_grad_[i].clear();
		is_all_ = false;
	}

	GenericScalarProblem::GenericScalarProblem(const std::string &name)
		: Problem(name), is_all_(false)
	{
	}

	void GenericScalarProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), 1);
		if (is_rhs_zero())
		{
			val.setZero();
			return;
		}
		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			val(i) = planar ? rhs_(pts(i, 0), pts(i, 1)) : rhs_(pts(i, 0), pts(i, 1), pts(i, 2));
		}
		// val = Eigen::MatrixXd::Constant(pts.rows(), 1, rhs_);
		// val *= t;
	}

	void GenericScalarProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);

		for (long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));
			if (is_all_)
			{
				assert(dirichlet_.size() == 1);
				val(i) = pts.cols() == 2 ? dirichlet_[0](pts(i, 0), pts(i, 1)) : dirichlet_[0](pts(i, 0), pts(i, 1), pts(i, 2));
			}
			else
			{
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if (id == boundary_ids_[b])
					{
						val(i) = pts.cols() == 2 ? dirichlet_[b](pts(i, 0), pts(i, 1)) : dirichlet_[b](pts(i, 0), pts(i, 1), pts(i, 2));
						break;
					}
				}
			}
		}

		val *= t;
	}

	void GenericScalarProblem::neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);

		for (long i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_boundary_id(global_ids(i));

			for (size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
			{
				if (id == neumann_boundary_ids_[b])
				{
					val(i) = pts.cols() == 2 ? neumann_[b](pts(i, 0), pts(i, 1)) : neumann_[b](pts(i, 0), pts(i, 1), pts(i, 2));
					break;
				}
			}
		}

		val *= t;
	}

	void GenericScalarProblem::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		assert(has_exact_sol());
		const bool planar = pts.cols() == 2;
		val.resize(pts.rows(), 1);

		for (int i = 0; i < pts.rows(); ++i)
		{
			val(i) = planar ? exact_(pts(i, 0), pts(i, 1)) : exact_(pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericScalarProblem::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());
		if (!has_exact_grad_)
			return;

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? exact_grad_[j](pts(i, 0), pts(i, 1)) : exact_grad_[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericScalarProblem::set_parameters(const json &params)
	{
		if (params.find("is_time_dependent") != params.end())
		{
			is_time_dept_ = params["is_time_dependent"];
		}

		if (params.find("rhs") != params.end())
		{
			// rhs_ = params["rhs"];
			rhs_.init(params["rhs"]);
		}

		if (params.find("exact") != params.end())
		{
			has_exact_ = !params["exact"].is_null();
			if (has_exact_)
				exact_.init(params["exact"]);
		}
		else
		{
			has_exact_ = false;
		}

		if (params.find("exact_grad") != params.end())
		{
			auto ex = params["exact_grad"];
			has_exact_grad_ = !ex.is_null();
			if (has_exact_grad_)
			{
				if (ex.is_array())
				{
					for (size_t k = 0; k < ex.size(); ++k)
						exact_grad_[k].init(ex[k]);
				}
				else
				{
					assert(false);
				}
			}
		}
		else
		{
			has_exact_grad_ = false;
		}

		if (params.find("dirichlet_boundary") != params.end())
		{
			// boundary_ids_.clear();
			const int offset = boundary_ids_.size();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(offset + j_boundary.size());
			dirichlet_.resize(offset + j_boundary.size());

			for (size_t i = offset; i < boundary_ids_.size(); ++i)
			{
				if (j_boundary[i - offset]["id"] == "all")
				{
					assert(boundary_ids_.size() == 1);

					is_all_ = true;
					boundary_ids_.clear();
				}
				else
					boundary_ids_[i] = j_boundary[i - offset]["id"];

				auto ff = j_boundary[i - offset]["value"];
				dirichlet_[i].init(ff);
			}
		}

		if (params.find("neumann_boundary") != params.end())
		{
			// neumann_boundary_ids_.clear();
			const int offset = neumann_boundary_ids_.size();
			auto j_boundary = params["neumann_boundary"];

			neumann_boundary_ids_.resize(offset + j_boundary.size());
			neumann_.resize(offset + j_boundary.size());

			for (size_t i = offset; i < neumann_boundary_ids_.size(); ++i)
			{
				neumann_boundary_ids_[i] = j_boundary[i - offset]["id"];

				auto ff = j_boundary[i - offset]["value"];
				neumann_[i].init(ff);
			}
		}
	}

	void GenericScalarProblem::add_dirichlet_boundary(const int id, const double val)
	{
		boundary_ids_.push_back(id);
		dirichlet_.emplace_back();
		dirichlet_.back().init(val);
	}

	void GenericScalarProblem::add_neumann_boundary(const int id, const double val)
	{
		neumann_boundary_ids_.push_back(id);
		neumann_.emplace_back();
		neumann_.back().init(val);
	}

	void GenericScalarProblem::add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z)> &func)
	{
		boundary_ids_.push_back(id);
		dirichlet_.emplace_back();
		dirichlet_.back().init(func);
	}

	void GenericScalarProblem::add_neumann_boundary(const int id, const std::function<double(double x, double y, double z)> &func)
	{
		neumann_boundary_ids_.push_back(id);
		neumann_.emplace_back();
		neumann_.back().init(func);
	}

	void GenericScalarProblem::clear()
	{
		neumann_.clear();
		dirichlet_.clear();

		rhs_.clear();
		exact_.clear();
		for (int i = 0; i < exact_grad_.size(); ++i)
			exact_grad_[i].clear();
		is_all_ = false;
		has_exact_ = false;
		has_exact_grad_ = false;
		is_time_dept_ = false;
	}
} // namespace polyfem
