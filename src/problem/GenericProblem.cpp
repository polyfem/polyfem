#include <polyfem/GenericProblem.hpp>
#include <polyfem/State.hpp>

#include <iostream>

namespace polyfem
{

	std::shared_ptr<Interpolation> Interpolation::build(const json &params)
	{
		const std::string type = params["type"];
		std::shared_ptr<Interpolation> res = nullptr;

		if (type == "none")
			res = std::make_shared<NoInterpolation>();
		else if (type == "linear")
			res = std::make_shared<LinearInterpolation>();
		else if (type == "linear_ramp")
			res = std::make_shared<LinearRamp>();
		else
			logger().error("Usupported interpolation type {}", type);

		if (res)
			res->init(params);

		return res;
	}

	double LinearRamp::eval(const double t) const
	{
		if (t >= to_)
			return to_;

		return t;
	}

	void LinearRamp::init(const json &params)
	{
		to_ = params["to"];
	}

	GenericTensorProblem::GenericTensorProblem(const std::string &name)
		: Problem(name), is_all_(false)
	{
	}

	void GenericTensorProblem::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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
			{
				double x = pts(i, 0), y = pts(i, 1), z = planar ? 0 : pts(i, 2);
				val(i, j) = rhs_[j](x, y, z, t);
			}
		}
	}

	bool GenericTensorProblem::is_dimension_dirichet(const int tag, const int dim) const
	{
		if (all_dimensions_dirichlet())
			return true;

		if (is_all_)
		{
			assert(dirichlet_dimensions_.size() == 1);
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

	void GenericTensorProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

		for (long i = 0; i < pts.rows(); ++i)
		{
			if (is_all_)
			{
				assert(displacements_.size() == 1);
				for (int d = 0; d < val.cols(); ++d)
				{
					double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
					val(i, d) = displacements_[0][d](x, y, z, t);
				}
				val.row(i) *= displacements_interpolation_[0]->eval(t);
			}
			else
			{
				const int id = mesh.get_boundary_id(global_ids(i));
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if (id == boundary_ids_[b])
					{
						for (int d = 0; d < val.cols(); ++d)
						{
							double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
							val(i, d) = displacements_[b][d](x, y, z, t);
						}
						val.row(i) *= displacements_interpolation_[b]->eval(t);
						break;
					}
				}
			}
		}
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
					{
						double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
						val(i, d) = forces_[b][d](x, y, z, t);
					}
					val.row(i) *= forces_interpolation_[b]->eval(t);
					break;
				}
			}

			for (size_t b = 0; b < pressure_boundary_ids_.size(); ++b)
			{
				if (id == pressure_boundary_ids_[b])
				{
					for (int d = 0; d < val.cols(); ++d)
					{
						double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
						val(i, d) = pressures_[b](x, y, z, t) * normals(i, d);
					}
					val.row(i) *= pressure_interpolation_[b]->eval(t);
					break;
				}
			}
		}
	}

	void GenericTensorProblem::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		assert(has_exact_sol());
		const bool planar = pts.cols() == 2;
		val.resize(pts.rows(), pts.cols());

		for (int i = 0; i < pts.rows(); ++i)
		{
			for (int j = 0; j < pts.cols(); ++j)
			{
				double x = pts(i, 0), y = pts(i, 1), z = planar ? 0 : pts(i, 2);
				val(i, j) = exact_[j](x, y, z, t);
			}
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
			{
				double x = pts(i, 0), y = pts(i, 1), z = planar ? 0 : pts(i, 2);
				val(i, j) = exact_grad_[j](x, y, z, t);
			}
		}
	}

	void GenericTensorProblem::add_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz)
	{
		boundary_ids_.push_back(id);
		displacements_.emplace_back();
		displacements_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
		for (size_t k = 0; k < val.size(); ++k)
			displacements_.back()[k].init(val[k]);

		dirichlet_dimensions_.emplace_back(isx, isy, isz);

		if (!isx || !isy || !isz)
			all_dimensions_dirichlet_ = false;
	}

	void GenericTensorProblem::add_neumann_boundary(const int id, const Eigen::RowVector3d &val)
	{
		neumann_boundary_ids_.push_back(id);
		forces_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
		forces_.emplace_back();
		for (size_t k = 0; k < val.size(); ++k)
			forces_.back()[k].init(val[k]);
	}

	void GenericTensorProblem::add_pressure_boundary(const int id, const double val)
	{
		pressure_boundary_ids_.push_back(id);
		pressure_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
		pressures_.emplace_back();
		pressures_.back().init(val);
	}

	void GenericTensorProblem::add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz)
	{
		boundary_ids_.push_back(id);
		displacements_.emplace_back();
		displacements_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
		for (size_t k = 0; k < displacements_.back().size(); ++k)
			displacements_.back()[k].init(func, k);

		dirichlet_dimensions_.emplace_back(isx, isy, isz);

		if (!isx || !isy || !isz)
			all_dimensions_dirichlet_ = false;
	}

	void GenericTensorProblem::add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func)
	{
		neumann_boundary_ids_.push_back(id);
		forces_.emplace_back();
		forces_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
		for (size_t k = 0; k < forces_.back().size(); ++k)
			forces_.back()[k].init(func, k);
	}

	void GenericTensorProblem::add_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func)
	{
		pressure_boundary_ids_.push_back(id);
		pressure_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
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
		if (params.contains("is_time_dependent"))
		{
			is_time_dept_ = params["is_time_dependent"];
		}

		if (params.contains("rhs"))
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

		if (params.contains("exact"))
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

		if (params.contains("exact_grad"))
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

		if (params.contains("dirichlet_boundary"))
		{
			// boundary_ids_.clear();
			int offset = boundary_ids_.size();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(offset + j_boundary.size());
			displacements_.resize(offset + j_boundary.size());
			displacements_interpolation_.resize(offset + j_boundary.size());
			dirichlet_dimensions_.resize(offset + j_boundary.size());

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

				dirichlet_dimensions_[i].setConstant(true);
				if (j_boundary[i - offset].contains("dimension"))
				{
					all_dimensions_dirichlet_ = false;
					auto &tmp = j_boundary[i - offset]["dimension"];
					assert(tmp.is_array());
					for (size_t k = 0; k < tmp.size(); ++k)
						dirichlet_dimensions_[i](k) = tmp[k];
				}

				if (j_boundary[i - offset].contains("interpolation"))
					displacements_interpolation_[i] = Interpolation::build(j_boundary[i - offset]["interpolation"]);
				else
					displacements_interpolation_[i] = std::make_shared<NoInterpolation>();
			}
		}

		if (params.contains("neumann_boundary"))
		{
			// neumann_boundary_ids_.clear();
			const int offset = neumann_boundary_ids_.size();

			auto j_boundary = params["neumann_boundary"];

			neumann_boundary_ids_.resize(offset + j_boundary.size());
			forces_.resize(offset + j_boundary.size());
			forces_interpolation_.resize(offset + j_boundary.size());

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

				if (j_boundary[i - offset].contains("interpolation"))
					forces_interpolation_[i] = Interpolation::build(j_boundary[i - offset]["interpolation"]);
				else
					forces_interpolation_[i] = std::make_shared<NoInterpolation>();
			}
		}

		if (params.contains("pressure_boundary"))
		{
			// pressure_boundary_ids_.clear();
			const int offset = pressure_boundary_ids_.size();

			auto j_boundary = params["pressure_boundary"];

			pressure_boundary_ids_.resize(offset + j_boundary.size());
			pressures_.resize(offset + j_boundary.size());
			pressure_interpolation_.resize(offset + j_boundary.size());

			for (size_t i = offset; i < pressure_boundary_ids_.size(); ++i)
			{
				pressure_boundary_ids_[i] = j_boundary[i - offset]["id"];

				auto ff = j_boundary[i - offset]["value"];
				pressures_[i].init(ff);

				if (j_boundary[i - offset].contains("interpolation"))
					pressure_interpolation_[i] = Interpolation::build(j_boundary[i - offset]["interpolation"]);
				else
					pressure_interpolation_[i] = std::make_shared<NoInterpolation>();
			}
		}

		if (params.contains("initial_solution"))
		{
			auto rr = params["initial_solution"];
			initial_position_.resize(rr.size());
			assert(rr.is_array());

			for (size_t k = 0; k < rr.size(); ++k)
			{
				initial_position_[k].first = rr[k]["id"];
				const auto v = rr[k]["value"];
				for (size_t d = 0; d < v.size(); ++d)
					initial_position_[k].second[d].init(v[d]);
			}
		}

		if (params.contains("initial_velocity"))
		{
			auto rr = params["initial_velocity"];
			initial_velocity_.resize(rr.size());
			assert(rr.is_array());

			for (size_t k = 0; k < rr.size(); ++k)
			{
				initial_velocity_[k].first = rr[k]["id"];
				const auto v = rr[k]["value"];
				for (size_t d = 0; d < v.size(); ++d)
					initial_velocity_[k].second[d].init(v[d]);
			}
		}

		if (params.contains("initial_acceleration"))
		{
			auto rr = params["initial_acceleration"];
			initial_acceleration_.resize(rr.size());
			assert(rr.is_array());

			for (size_t k = 0; k < rr.size(); ++k)
			{
				initial_acceleration_[k].first = rr[k]["id"];
				const auto v = rr[k]["value"];
				for (size_t d = 0; d < v.size(); ++d)
					initial_acceleration_[k].second[d].init(v[d]);
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
		if (initial_position_.empty())
		{
			val.setZero();
			return;
		}

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_body_id(global_ids(i));
			int index = -1;
			for (int j = 0; j < initial_position_.size(); ++j)
			{
				if (initial_position_[j].first == id)
				{
					index = j;
					break;
				}
			}
			if (index < 0)
			{
				val.row(i).setZero();
				continue;
			}

			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? initial_position_[index].second[j](pts(i, 0), pts(i, 1)) : initial_position_[index].second[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::initial_velocity(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());
		if (initial_velocity_.empty())
		{
			val.setZero();
			return;
		}

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_body_id(global_ids(i));
			int index = -1;
			for (int j = 0; j < initial_velocity_.size(); ++j)
			{
				if (initial_velocity_[j].first == id)
				{
					index = j;
					break;
				}
			}
			if (index < 0)
			{
				val.row(i).setZero();
				continue;
			}

			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? initial_velocity_[index].second[j](pts(i, 0), pts(i, 1)) : initial_velocity_[index].second[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::initial_acceleration(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), pts.cols());
		if (initial_acceleration_.empty())
		{
			val.setZero();
			return;
		}

		const bool planar = pts.cols() == 2;
		for (int i = 0; i < pts.rows(); ++i)
		{
			const int id = mesh.get_body_id(global_ids(i));
			int index = -1;
			for (int j = 0; j < initial_acceleration_.size(); ++j)
			{
				if (initial_acceleration_[j].first == id)
				{
					index = j;
					break;
				}
			}
			if (index < 0)
			{
				val.row(i).setZero();
				continue;
			}

			for (int j = 0; j < pts.cols(); ++j)
				val(i, j) = planar ? initial_acceleration_[index].second[j](pts(i, 0), pts(i, 1)) : initial_acceleration_[index].second[j](pts(i, 0), pts(i, 1), pts(i, 2));
		}
	}

	void GenericTensorProblem::clear()
	{
		all_dimensions_dirichlet_ = true;
		has_exact_ = false;
		has_exact_grad_ = false;
		is_time_dept_ = false;

		forces_.clear();
		forces_interpolation_.clear();
		displacements_.clear();
		displacements_interpolation_.clear();
		pressures_.clear();
		pressure_interpolation_.clear();

		initial_position_.clear();
		initial_velocity_.clear();
		initial_acceleration_.clear();

		dirichlet_dimensions_.clear();

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

	void GenericScalarProblem::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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
			double x = pts(i, 0), y = pts(i, 1), z = planar ? 0 : pts(i, 2);
			val(i) = rhs_(x, y, z, t);
		}
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
				double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
				val(i) = dirichlet_[0](x, y, z, t);
				val(i) *= dirichlet_interpolation_[0]->eval(t);
			}
			else
			{
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if (id == boundary_ids_[b])
					{
						double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
						val(i) = dirichlet_[b](x, y, z, t);
						val(i) *= dirichlet_interpolation_[b]->eval(t);
						break;
					}
				}
			}
		}
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
					double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
					val(i) = neumann_[b](x, y, z, t);
					val(i) *= neumann_interpolation_[b]->eval(t);
					break;
				}
			}
		}
	}

	void GenericScalarProblem::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		assert(has_exact_sol());
		const bool planar = pts.cols() == 2;
		val.resize(pts.rows(), 1);

		for (int i = 0; i < pts.rows(); ++i)
		{
			double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
			val(i) = exact_(x, y, z, t);
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
			{
				double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
				val(i, j) = exact_grad_[j](x, y, z, t);
			}
		}
	}

	void GenericScalarProblem::set_parameters(const json &params)
	{
		if (params.contains("is_time_dependent"))
		{
			is_time_dept_ = params["is_time_dependent"];
		}

		if (params.contains("rhs"))
		{
			// rhs_ = params["rhs"];
			rhs_.init(params["rhs"]);
		}

		if (params.contains("exact"))
		{
			has_exact_ = !params["exact"].is_null();
			if (has_exact_)
				exact_.init(params["exact"]);
		}

		if (params.contains("exact_grad"))
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

		if (params.contains("dirichlet_boundary"))
		{
			// boundary_ids_.clear();
			const int offset = boundary_ids_.size();
			auto j_boundary = params["dirichlet_boundary"];

			boundary_ids_.resize(offset + j_boundary.size());
			dirichlet_.resize(offset + j_boundary.size());
			dirichlet_interpolation_.resize(offset + j_boundary.size());

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

				if (j_boundary[i - offset].contains("interpolation"))
					dirichlet_interpolation_[i] = Interpolation::build(j_boundary[i - offset]["interpolation"]);
				else
					dirichlet_interpolation_[i] = std::make_shared<NoInterpolation>();
			}
		}

		if (params.contains("neumann_boundary"))
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

				if (j_boundary[i - offset].contains("interpolation"))
					neumann_interpolation_[i] = Interpolation::build(j_boundary[i - offset]["interpolation"]);
				else
					neumann_interpolation_[i] = std::make_shared<NoInterpolation>();
			}
		}
	}

	void GenericScalarProblem::add_dirichlet_boundary(const int id, const double val)
	{
		boundary_ids_.push_back(id);
		dirichlet_.emplace_back();
		dirichlet_.back().init(val);
		dirichlet_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
	}

	void GenericScalarProblem::add_neumann_boundary(const int id, const double val)
	{
		neumann_boundary_ids_.push_back(id);
		neumann_.emplace_back();
		neumann_.back().init(val);
		neumann_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
	}

	void GenericScalarProblem::add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func)
	{
		boundary_ids_.push_back(id);
		dirichlet_.emplace_back();
		dirichlet_.back().init(func);
		dirichlet_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
	}

	void GenericScalarProblem::add_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func)
	{
		neumann_boundary_ids_.push_back(id);
		neumann_.emplace_back();
		neumann_.back().init(func);
		neumann_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
	}

	void GenericScalarProblem::clear()
	{
		neumann_.clear();
		dirichlet_.clear();

		dirichlet_interpolation_.clear();
		neumann_interpolation_.clear();

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
