#include "GenericProblem.hpp"
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <iostream>

namespace polyfem
{
	using namespace utils;

	namespace assembler
	{
		namespace
		{
			std::vector<json> flatten_ids(const json &j_boundary_tmp)
			{
				std::vector<json> j_boundary;

				for (size_t i = 0; i < j_boundary_tmp.size(); ++i)
				{
					const auto &tmp = j_boundary_tmp[i];
					if (!tmp.contains("id"))
						continue;

					if (tmp["id"].is_array())
					{
						for (size_t j = 0; j < tmp["id"].size(); ++j)
						{
							json newj = tmp;
							newj["id"] = tmp["id"][j].get<int>();
							j_boundary.push_back(newj);
						}
					}
					else
						j_boundary.push_back(tmp);
				}

				return j_boundary;
			}
		} // namespace

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
			else if (type == "piecewise_constant")
				res = std::make_shared<PiecewiseConstantInterpolation>();
			else if (type == "piecewise_linear")
				res = std::make_shared<PiecewiseLinearInterpolation>();
			// else if (type == "piecewise_cubic")
			// 	res = std::make_shared<PiecewiseCubicInterpolation>();
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

			if (t <= form_)
				return 0;

			return t - form_;
		}

		void LinearRamp::init(const json &params)
		{
			to_ = params["to"];

			if (params.contains("from"))
				form_ = params["from"];
			else
				form_ = 0;
		}

		double PiecewiseConstantInterpolation::eval(const double t) const
		{
			for (size_t i = 0; i < points_.size()-1; ++i) {
				if (t >= points_[i] && t < points_[i+1]) {
					return values_[i];
				}
			}

			return extend(t);
		}

		double PiecewiseLinearInterpolation::eval(const double t) const
		{
			for (size_t i = 0; i < points_.size()-1; ++i) {
				if (t >= points_[i] && t < points_[i+1]) {
					double val = (values_[i+1]-values_[i]) * ((t-points_[i])/(points_[i+1]-points_[i])) + values_[i];
					return val;
				}
			}
			
			return extend(t);
		}


		void PiecewiseInterpolation::init(const json &params)
		{
			if (!params["points"].is_array())
				log_and_throw_error(fmt::format("Points must be an array"));
			if (!params["values"].is_array())
				log_and_throw_error(fmt::format("Values must be an array"));
			
			points_.reserve(params["points"].size());
			values_.reserve(params["values"].size());

			for (int i = 0; i < params["points"].size(); ++i) {
				points_[i] = params["points"][i];
				values_[i] = params["values"][i];
			}

			if (params.contains("extend")) {
				if (params["extend"] == "constant")
					ext_ = constant;
				else if (params["extend"] == "extrapolate")
					ext_ = extrapolate;
				else if (params["extend"] == "repeat")
					ext_ = repeat;
				else if (params["extend"] == "repeat_offset")
					ext_ = repeat_offset;
				else
					log_and_throw_error(fmt::format("Extend Method not recognized. Should be one of {constant, extrapolate, repeat, repeat_offset}"));
			}
			else
				ext_ = constant;
		}

		double PiecewiseInterpolation::extend(const double t) const
		{
			if (ext_ == constant) {
				if (t < points_[0])
					return values_[0];
				else
					return values_[values_.size()-1];
			}
			else if (ext_ == extrapolate) {
				if (t < points_[0])
					return values_[0]*t;
				else
					return values_[values_.size()-1]*t;
			}
			else if (ext_ == repeat) {
				if (t < points_[0]) 
					return eval(t + points_[points_.size()-1]);
				else
					return eval(std::fmod(t, points_[points_.size()-1]) + points_[0]);
			}
			else if (ext_ == repeat_offset) {
				if (t < points_[0]) 
					return eval(t + points_[points_.size()-1]) + (values_[values_.size()-1] - values_[0]);
				else
					return eval(std::fmod(t, points_[points_.size()-1]) + points_[0]) + (values_[values_.size()-1] - values_[0]);
			}	

			return 0;
		}

		GenericTensorProblem::GenericTensorProblem(const std::string &name)
			: Problem(name), is_all_(false)
		{
		}

		void GenericTensorProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void GenericTensorProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				if (is_all_)
				{
					assert(displacements_.size() == 1);
					double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
					for (int d = 0; d < val.cols(); ++d)
					{
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
							double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
							for (int d = 0; d < val.cols(); ++d)
							{
								val(i, d) = displacements_[b][d](x, y, z, t);
							}
							val.row(i) *= displacements_interpolation_[b]->eval(t);
							break;
						}
					}
				}
			}
		}

		void GenericTensorProblem::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
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

		void GenericTensorProblem::add_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp)
		{
			boundary_ids_.push_back(id);
			displacements_.emplace_back();
			displacements_interpolation_.emplace_back(interp);
			for (size_t k = 0; k < val.size(); ++k)
				displacements_.back()[k].init(val[k]);

			dirichlet_dimensions_.emplace_back(isx, isy, isz);

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::update_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < boundary_ids_.size(); ++i)
			{
				if (boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			displacements_interpolation_[index] = interp;
			for (size_t k = 0; k < val.size(); ++k)
				displacements_[index][k].init(val[k]);

			dirichlet_dimensions_[index] << isx, isy, isz;

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::add_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			forces_interpolation_.emplace_back(interp);
			forces_.emplace_back();
			for (size_t k = 0; k < val.size(); ++k)
				forces_.back()[k].init(val[k]);
		}

		void GenericTensorProblem::update_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				if (neumann_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			forces_interpolation_[index] = interp;
			for (size_t k = 0; k < val.size(); ++k)
				forces_[index][k].init(val[k]);
		}

		void GenericTensorProblem::add_pressure_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			pressure_boundary_ids_.push_back(id);
			pressure_interpolation_.emplace_back(interp);
			pressures_.emplace_back();
			pressures_.back().init(val);
		}

		void GenericTensorProblem::update_pressure_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < pressure_boundary_ids_.size(); ++i)
			{
				if (pressure_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}
			pressure_interpolation_[index] = interp;
			pressures_[index].init(val);
		}

		void GenericTensorProblem::add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp)
		{
			boundary_ids_.push_back(id);
			displacements_.emplace_back();
			displacements_interpolation_.emplace_back(interp);
			for (size_t k = 0; k < displacements_.back().size(); ++k)
				displacements_.back()[k].init(func, k);

			dirichlet_dimensions_.emplace_back(isx, isy, isz);

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::update_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < boundary_ids_.size(); ++i)
			{
				if (boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}
			displacements_interpolation_[index] = interp;
			for (size_t k = 0; k < displacements_.back().size(); ++k)
				displacements_[index][k].init(func, k);

			dirichlet_dimensions_[index] << isx, isy, isz;

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			forces_.emplace_back();
			forces_interpolation_.emplace_back(interp);
			for (size_t k = 0; k < forces_.back().size(); ++k)
				forces_.back()[k].init(func, k);
		}

		void GenericTensorProblem::update_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				if (neumann_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			forces_interpolation_[index] = interp;
			for (size_t k = 0; k < forces_.back().size(); ++k)
				forces_[index][k].init(func, k);
		}

		void GenericTensorProblem::add_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			pressure_boundary_ids_.push_back(id);
			pressure_interpolation_.emplace_back(interp);
			pressures_.emplace_back();
			pressures_.back().init(func);
		}

		void GenericTensorProblem::update_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < pressure_boundary_ids_.size(); ++i)
			{
				if (pressure_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			pressure_interpolation_[index] = interp;
			pressures_[index].init(func);
		}

		void GenericTensorProblem::add_dirichlet_boundary(const int id, const json &val, const bool isx, const bool isy, const bool isz, const std::string &interpolation)
		{
			if (!val.is_array())
				throw "Val must be an array";

			boundary_ids_.push_back(id);
			displacements_.emplace_back();
			if (interpolation.empty())
				displacements_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
			else
				displacements_interpolation_.emplace_back(Interpolation::build(interpolation));

			for (size_t k = 0; k < val.size(); ++k)
				displacements_.back()[k].init(val[k]);

			dirichlet_dimensions_.emplace_back(isx, isy, isz);

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::add_neumann_boundary(const int id, const json &val, const std::string &interpolation)
		{
			if (!val.is_array())
				throw "Val must be an array";

			neumann_boundary_ids_.push_back(id);

			if (interpolation.empty())
				forces_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
			else
				forces_interpolation_.emplace_back(Interpolation::build(interpolation));

			forces_.emplace_back();
			for (size_t k = 0; k < val.size(); ++k)
				forces_.back()[k].init(val[k]);
		}

		void GenericTensorProblem::add_pressure_boundary(const int id, json val, const std::string &interpolation)
		{
			pressure_boundary_ids_.push_back(id);
			if (interpolation.empty())
				pressure_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
			else
				pressure_interpolation_.emplace_back(Interpolation::build(interpolation));
			pressures_.emplace_back();
			pressures_.back().init(val);
		}

		void GenericTensorProblem::update_dirichlet_boundary(const int id, const json &val, const bool isx, const bool isy, const bool isz, const std::string &interpolation)
		{
			if (!val.is_array())
				throw "Val must be an array";
			int index = -1;
			for (int i = 0; i < boundary_ids_.size(); ++i)
			{
				if (boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			if (interpolation.empty())
				displacements_interpolation_[index] = std::make_shared<NoInterpolation>();
			else
				displacements_interpolation_[index] = Interpolation::build(interpolation);

			for (size_t k = 0; k < val.size(); ++k)
				displacements_[index][k].init(val[k]);

			dirichlet_dimensions_[index] << isx, isy, isz;

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::update_neumann_boundary(const int id, const json &val, const std::string &interpolation)
		{
			if (!val.is_array())
				throw "Val must be an array";

			int index = -1;
			for (int i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				if (neumann_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			if (interpolation.empty())
				forces_interpolation_[index] = std::make_shared<NoInterpolation>();
			else
				forces_interpolation_[index] = Interpolation::build(interpolation);

			for (size_t k = 0; k < val.size(); ++k)
				forces_[index][k].init(val[k]);
		}

		void GenericTensorProblem::update_pressure_boundary(const int id, json val, const std::string &interpolation)
		{
			int index = -1;
			for (int i = 0; i < pressure_boundary_ids_.size(); ++i)
			{
				if (pressure_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			if (interpolation.empty())
				pressure_interpolation_[index] = std::make_shared<NoInterpolation>();
			else
				pressure_interpolation_[index] = Interpolation::build(interpolation);
			pressures_[index].init(val);
		}

		void GenericTensorProblem::set_rhs(double x, double y, double z)
		{
			rhs_[0].init(x);
			rhs_[1].init(y);
			rhs_[2].init(z);
		}

		void GenericTensorProblem::set_parameters(const json &params)
		{
			if (is_param_valid(params, "is_time_dependent"))
			{
				is_time_dept_ = params["is_time_dependent"];
			}

			if (is_param_valid(params, "rhs"))
			{
				auto rr = params["rhs"];
				if (rr.is_array())
				{
					for (size_t k = 0; k < rr.size(); ++k)
						rhs_[k].init(rr[k]);
				}
				else
				{
					logger().warn("Invalid problem rhs: should be an array.");
					assert(false);
				}
			}

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "solution"))
			{
				auto ex = params["reference"]["solution"];
				has_exact_ = ex.size() > 0;
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

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "gradient"))
			{
				auto ex = params["reference"]["gradient"];
				has_exact_grad_ = ex.size() > 0;
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

			if (is_param_valid(params, "dirichlet_boundary"))
			{
				// boundary_ids_.clear();
				int offset = boundary_ids_.size();
				auto j_boundary_tmp = params["dirichlet_boundary"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

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

			if (is_param_valid(params, "neumann_boundary"))
			{
				// neumann_boundary_ids_.clear();
				const int offset = neumann_boundary_ids_.size();

				auto j_boundary_tmp = params["neumann_boundary"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

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

			if (is_param_valid(params, "pressure_boundary"))
			{
				// pressure_boundary_ids_.clear();
				const int offset = pressure_boundary_ids_.size();

				auto j_boundary_tmp = params["pressure_boundary"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

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

			if (is_param_valid(params, "solution"))
			{
				auto rr = params["solution"];
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

			if (is_param_valid(params, "velocity"))
			{
				auto rr = params["velocity"];
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

			if (is_param_valid(params, "acceleration"))
			{
				auto rr = params["acceleration"];
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

		void GenericTensorProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
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

		void GenericTensorProblem::initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
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

		void GenericTensorProblem::initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
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

		void GenericScalarProblem::rhs(const assembler::AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void GenericScalarProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

		void GenericScalarProblem::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
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

		void GenericScalarProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
		{
			val.resize(pts.rows(), 1);
			if (initial_solution_.empty())
			{
				val.setZero();
				return;
			}

			const bool planar = pts.cols() == 2;
			for (int i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_body_id(global_ids(i));
				int index = -1;
				for (int j = 0; j < initial_solution_.size(); ++j)
				{
					if (initial_solution_[j].first == id)
					{
						index = j;
						break;
					}
				}
				if (index < 0)
				{
					val(i) = 0;
					continue;
				}

				val(i) = planar ? initial_solution_[index].second(pts(i, 0), pts(i, 1)) : initial_solution_[index].second(pts(i, 0), pts(i, 1), pts(i, 2));
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
			if (is_param_valid(params, "is_time_dependent"))
			{
				is_time_dept_ = params["is_time_dependent"];
			}

			if (is_param_valid(params, "rhs"))
			{
				rhs_.init(params["rhs"]);
			}

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "solution"))
			{
				has_exact_ = !params["reference"]["solution"].empty();
				exact_.init(params["reference"]["solution"]);
			}

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "gradient"))
			{
				auto ex = params["reference"]["gradient"];
				has_exact_grad_ = ex.size() > 0;
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

			if (is_param_valid(params, "dirichlet_boundary"))
			{
				// boundary_ids_.clear();
				const int offset = boundary_ids_.size();
				auto j_boundary_tmp = params["dirichlet_boundary"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

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

			if (is_param_valid(params, "neumann_boundary"))
			{
				// neumann_boundary_ids_.clear();
				const int offset = neumann_boundary_ids_.size();
				auto j_boundary_tmp = params["neumann_boundary"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

				neumann_boundary_ids_.resize(offset + j_boundary.size());
				neumann_.resize(offset + j_boundary.size());
				neumann_interpolation_.resize(offset + j_boundary.size());

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

			if (is_param_valid(params, "solution"))
			{
				auto rr = params["solution"];
				initial_solution_.resize(rr.size());
				assert(rr.is_array());

				for (size_t k = 0; k < rr.size(); ++k)
				{
					initial_solution_[k].first = rr[k]["id"];
					initial_solution_[k].second.init(rr[k]["value"]);
				}
			}
		}

		void GenericScalarProblem::add_dirichlet_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			boundary_ids_.push_back(id);
			dirichlet_.emplace_back();
			dirichlet_.back().init(val);
			dirichlet_interpolation_.emplace_back(interp);
		}

		void GenericScalarProblem::update_dirichlet_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < boundary_ids_.size(); ++i)
			{
				if (boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			dirichlet_[index].init(val);
			dirichlet_interpolation_[index] = interp;
		}

		void GenericScalarProblem::add_neumann_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			neumann_.emplace_back();
			neumann_.back().init(val);
			neumann_interpolation_.emplace_back(interp);
		}

		void GenericScalarProblem::update_neumann_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				if (neumann_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			neumann_[index].init(val);
			neumann_interpolation_[index] = interp;
		}

		void GenericScalarProblem::add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			boundary_ids_.push_back(id);
			dirichlet_.emplace_back();
			dirichlet_.back().init(func);
			dirichlet_interpolation_.emplace_back(interp);
		}

		void GenericScalarProblem::update_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < boundary_ids_.size(); ++i)
			{
				if (boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}
			dirichlet_[index].init(func);
			dirichlet_interpolation_[index] = interp;
		}

		void GenericScalarProblem::add_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			neumann_.emplace_back();
			neumann_.back().init(func);
			neumann_interpolation_.emplace_back(interp);
		}

		void GenericScalarProblem::update_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			int index = -1;
			for (int i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				if (neumann_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}
			neumann_[index].init(func);
			neumann_interpolation_[index] = interp;
		}

		void GenericScalarProblem::add_dirichlet_boundary(const int id, const json &val, const std::string &interp)
		{
			boundary_ids_.push_back(id);
			dirichlet_.emplace_back();
			dirichlet_.back().init(val);
			if (interp.empty())
				dirichlet_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
			else
				dirichlet_interpolation_.emplace_back(Interpolation::build(interp));
		}

		void GenericScalarProblem::add_neumann_boundary(const int id, const json &val, const std::string &interp)
		{
			neumann_boundary_ids_.push_back(id);
			neumann_.emplace_back();
			neumann_.back().init(val);
			if (interp.empty())
				neumann_interpolation_.emplace_back(std::make_shared<NoInterpolation>());
			else
				neumann_interpolation_.emplace_back(Interpolation::build(interp));
		}

		void GenericScalarProblem::update_dirichlet_boundary(const int id, const json &val, const std::string &interp)
		{
			int index = -1;
			for (int i = 0; i < boundary_ids_.size(); ++i)
			{
				if (boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}

			dirichlet_[index].init(val);
			if (interp.empty())
				dirichlet_interpolation_[index] = std::make_shared<NoInterpolation>();
			else
				dirichlet_interpolation_[index] = Interpolation::build(interp);
		}

		void GenericScalarProblem::update_neumann_boundary(const int id, const json &val, const std::string &interp)
		{
			int index = -1;
			for (int i = 0; i < neumann_boundary_ids_.size(); ++i)
			{
				if (neumann_boundary_ids_[i] == id)
				{
					index = i;
					break;
				}
			}
			if (index == -1)
			{
				throw "Invalid boundary id";
			}
			neumann_[index].init(val);

			if (interp.empty())
				neumann_interpolation_[index] = std::make_shared<NoInterpolation>();
			else
				neumann_interpolation_[index] = Interpolation::build(interp);
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
	} // namespace assembler
} // namespace polyfem
