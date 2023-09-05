#include "GenericProblem.hpp"

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/MatrixIO.hpp>

namespace polyfem
{
	using namespace utils;

	namespace assembler
	{
		namespace
		{
			std::vector<json> flatten_ids(const json &p_j_boundary_tmp)
			{
				const std::vector<json> j_boundary_tmp = utils::json_as_array(p_j_boundary_tmp);

				std::vector<json> j_boundary;

				for (size_t i = 0; i < j_boundary_tmp.size(); ++i)
				{
					const auto &tmp = j_boundary_tmp[i];

					if (tmp.is_string())
					{
						j_boundary.push_back(tmp);
						continue;
					}
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

		double TensorBCValue::eval(const RowVectorNd &pts, const int dim, const double t, const int el_id) const
		{
			double x = pts(0);
			double y = pts(1);
			double z = pts.size() == 2 ? 0 : pts(2);

			double val = value[dim](x, y, z, t, el_id);

			if (interpolation.empty())
			{
			}
			else if (interpolation.size() == 1)
				val *= interpolation[0]->eval(t);
			else
			{
				assert(dim < interpolation.size());
				val *= interpolation[dim]->eval(t);
			}

			return val;
		}

		double ScalarBCValue::eval(const RowVectorNd &pts, const double t) const
		{
			assert(pts.size() == 2 || pts.size() == 3);
			double x = pts(0), y = pts(1), z = pts.size() == 3 ? pts(2) : 0.0;
			return value(x, y, z, t) * interpolation->eval(t);
		}

		GenericTensorProblem::GenericTensorProblem(const std::string &name)
			: Problem(name), is_all_(false)
		{
		}

		void GenericTensorProblem::set_units(const assembler::Assembler &assembler, const Units &units)
		{
			if (assembler.is_fluid())
			{
				// TODO
				assert(false);
			}
			else
			{
				for (int i = 0; i < 3; ++i)
				{
					rhs_[i].set_unit_type(units.acceleration());
					exact_[i].set_unit_type(units.length());
				}
				for (int i = 0; i < 3; ++i)
					exact_grad_[i].set_unit_type("");

				for (auto &v : displacements_)
					v.set_unit_type(units.length());

				for (auto &v : forces_)
					v.set_unit_type(units.force());

				for (auto &v : pressures_)
					v.set_unit_type(units.pressure());

				for (auto &v : initial_position_)
					for (int i = 0; i < 3; ++i)
						v.second[i].set_unit_type(units.length());

				for (auto &v : initial_velocity_)
					for (int i = 0; i < 3; ++i)
						v.second[i].set_unit_type(units.velocity());

				for (auto &v : initial_acceleration_)
					for (int i = 0; i < 3; ++i)
						v.second[i].set_unit_type(units.acceleration());

				for (auto &v : nodal_dirichlet_)
					v.second.set_unit_type(units.length());

				for (auto &v : nodal_neumann_)
					v.second.set_unit_type(units.force());
			}
		}

		void GenericTensorProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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
				assert(displacements_.size() == 1);
				return displacements_[0].dirichlet_dimension[dim];
			}

			for (size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if (tag == boundary_ids_[b])
				{
					auto &tmp = displacements_[b].dirichlet_dimension;
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
					for (int d = 0; d < val.cols(); ++d)
					{
						val(i, d) = displacements_[0].eval(pts.row(i), d, t);
					}
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
								val(i, d) = displacements_[b].eval(pts.row(i), d, t);
							}

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
							val(i, d) = forces_[b].eval(pts.row(i), d, t);
						}

						break;
					}
				}

				for (size_t b = 0; b < pressure_boundary_ids_.size(); ++b)
				{
					if (id == pressure_boundary_ids_[b])
					{
						for (int d = 0; d < val.cols(); ++d)
						{
							val(i, d) = pressures_[b].eval(pts.row(i), t) * normals(i, d);
						}
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
			for (size_t k = 0; k < val.size(); ++k)
				displacements_.back().value[k].init(val[k]);

			displacements_.back().dirichlet_dimension << isx, isy, isz;
			displacements_.back().interpolation.push_back(interp);

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

			displacements_[index].interpolation.clear();
			displacements_[index].interpolation.push_back(interp);

			for (size_t k = 0; k < val.size(); ++k)
				displacements_[index].value[k].init(val[k]);

			displacements_[index].dirichlet_dimension << isx, isy, isz;

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::add_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			forces_.emplace_back();
			for (size_t k = 0; k < val.size(); ++k)
				forces_.back().value[k].init(val[k]);

			forces_.back().interpolation.push_back(interp);
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

			forces_[index].interpolation.clear();
			forces_[index].interpolation.push_back(interp);
			for (size_t k = 0; k < val.size(); ++k)
				forces_[index].value[k].init(val[k]);
		}

		void GenericTensorProblem::add_pressure_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			pressure_boundary_ids_.push_back(id);
			pressures_.emplace_back();
			pressures_.back().value.init(val);
			pressures_.back().interpolation = interp;
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
			pressures_[index].interpolation = interp;
			pressures_[index].value.init(val);
		}

		void GenericTensorProblem::add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp)
		{
			boundary_ids_.push_back(id);
			displacements_.emplace_back();
			displacements_.back().interpolation.push_back(interp);
			for (size_t k = 0; k < displacements_.back().value.size(); ++k)
				displacements_.back().value[k].init(func, k);

			displacements_.back().dirichlet_dimension << isx, isy, isz;

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
			displacements_[index].interpolation.clear();
			displacements_[index].interpolation.push_back(interp);
			for (size_t k = 0; k < displacements_.back().value.size(); ++k)
				displacements_[index].value[k].init(func, k);

			displacements_[index].dirichlet_dimension << isx, isy, isz;

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			forces_.emplace_back();
			forces_.back().interpolation.push_back(interp);
			for (size_t k = 0; k < forces_.back().value.size(); ++k)
				forces_.back().value[k].init(func, k);
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

			forces_[index].interpolation.clear();
			forces_[index].interpolation.push_back(interp);
			for (size_t k = 0; k < forces_.back().value.size(); ++k)
				forces_[index].value[k].init(func, k);
		}

		void GenericTensorProblem::add_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			pressure_boundary_ids_.push_back(id);
			pressures_.emplace_back();
			pressures_.back().value.init(func);
			pressures_.back().interpolation = interp;
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

			pressures_[index].interpolation = interp;
			pressures_[index].value.init(func);
		}

		void GenericTensorProblem::add_dirichlet_boundary(const int id, const json &val, const bool isx, const bool isy, const bool isz, const std::string &interpolation)
		{
			if (!val.is_array())
				throw "Val must be an array";

			boundary_ids_.push_back(id);
			displacements_.emplace_back();
			if (!interpolation.empty())
				displacements_.back().interpolation.push_back(Interpolation::build(interpolation));

			for (size_t k = 0; k < val.size(); ++k)
				displacements_.back().value[k].init(val[k]);

			displacements_.back().dirichlet_dimension << isx, isy, isz;

			if (!isx || !isy || !isz)
				all_dimensions_dirichlet_ = false;
		}

		void GenericTensorProblem::add_neumann_boundary(const int id, const json &val, const std::string &interpolation)
		{
			if (!val.is_array())
				throw "Val must be an array";

			neumann_boundary_ids_.push_back(id);

			forces_.emplace_back();
			if (!interpolation.empty())
				forces_.back().interpolation.push_back(Interpolation::build(interpolation));

			for (size_t k = 0; k < val.size(); ++k)
				forces_.back().value[k].init(val[k]);
		}

		void GenericTensorProblem::add_pressure_boundary(const int id, json val, const std::string &interpolation)
		{
			pressure_boundary_ids_.push_back(id);
			pressures_.emplace_back();

			if (interpolation.empty())
				pressures_.back().interpolation = std::make_shared<NoInterpolation>();
			else
				pressures_.back().interpolation = Interpolation::build(interpolation);
			pressures_.back().value.init(val);
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

			displacements_[index].interpolation.clear();
			if (!interpolation.empty())
				displacements_[index].interpolation.push_back(Interpolation::build(interpolation));

			for (size_t k = 0; k < val.size(); ++k)
				displacements_[index].value[k].init(val[k]);

			displacements_[index].dirichlet_dimension << isx, isy, isz;

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
			forces_[index].interpolation.clear();
			if (!interpolation.empty())
				forces_[index].interpolation.push_back(Interpolation::build(interpolation));

			for (size_t k = 0; k < val.size(); ++k)
				forces_[index].value[k].init(val[k]);
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
				pressures_[index].interpolation = std::make_shared<NoInterpolation>();
			else
				pressures_[index].interpolation = Interpolation::build(interpolation);
			pressures_[index].value.init(val);
		}

		void GenericTensorProblem::set_rhs(double x, double y, double z)
		{
			rhs_[0].init(x);
			rhs_[1].init(y);
			rhs_[2].init(z);
		}

		void GenericTensorProblem::dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(1, mesh.dimension());
			const int tag = mesh.get_node_id(node_id);

			if (is_all_)
			{
				assert(nodal_dirichlet_.size() == 1);
				const auto &tmp = nodal_dirichlet_.begin()->second;

				for (int d = 0; d < val.cols(); ++d)
				{
					val(d) = tmp.eval(pt, d, t);
				}

				return;
			}

			const auto it = nodal_dirichlet_.find(tag);
			if (it != nodal_dirichlet_.end())
			{
				for (int d = 0; d < val.cols(); ++d)
				{
					val(d) = it->second.eval(pt, d, t);
				}

				return;
			}

			for (const auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int i = 0; i < n_dirichlet.rows(); ++i)
				{
					if (n_dirichlet(i, 0) == node_id)
					{
						for (int d = 0; d < val.cols(); ++d)
						{
							val(d) = n_dirichlet(i, d + 1);
						}

						return;
					}
				}
			}

			assert(false);
		}

		void GenericTensorProblem::neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val) const
		{
			// TODO implement me;
			log_and_throw_error("Nodal neumann not implemented");
		}

		bool GenericTensorProblem::is_nodal_dirichlet_boundary(const int n_id, const int tag)
		{
			if (nodal_dirichlet_.find(tag) != nodal_dirichlet_.end())
				return true;

			for (const auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int i = 0; i < n_dirichlet.rows(); ++i)
				{
					if (n_dirichlet(i, 0) == n_id)
						return true;
				}
			}

			return false;
		}

		bool GenericTensorProblem::is_nodal_neumann_boundary(const int n_id, const int tag)
		{
			return nodal_neumann_.find(tag) != nodal_neumann_.end();
		}

		bool GenericTensorProblem::has_nodal_dirichlet()
		{
			return nodal_dirichlet_mat_.size() > 0;
		}

		bool GenericTensorProblem::has_nodal_neumann()
		{
			return false; // nodal_neumann_.size() > 0;
		}

		bool GenericTensorProblem::is_nodal_dimension_dirichlet(const int n_id, const int tag, const int dim) const
		{
			const auto it = nodal_dirichlet_.find(tag);
			if (it != nodal_dirichlet_.end())
			{
				return it->second.dirichlet_dimension(dim);
			}

			for (const auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int i = 0; i < n_dirichlet.rows(); ++i)
				{
					if (n_dirichlet(i, 0) == n_id)
					{
						return !std::isnan(n_dirichlet(i, dim + 1));
					}
				}
			}

			assert(false);
			return true;
		}

		void GenericTensorProblem::update_nodes(const Eigen::VectorXi &in_node_to_node)
		{
			for (auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int n = 0; n < n_dirichlet.rows(); ++n)
				{
					const int node_id = in_node_to_node[n_dirichlet(n, 0)];
					n_dirichlet(n, 0) = node_id;
				}
			}
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
				std::vector<json> j_boundary = flatten_ids(params["dirichlet_boundary"]);

				boundary_ids_.resize(offset + j_boundary.size());
				displacements_.resize(offset + j_boundary.size());

				for (size_t i = offset; i < boundary_ids_.size(); ++i)
				{
					if (j_boundary[i - offset].is_string())
					{
						const std::string path = resolve_path(j_boundary[i - offset], params["root_path"]);
						if (!std::filesystem::is_regular_file(path))
							log_and_throw_error(fmt::format("unable to open {} file", path));

						Eigen::MatrixXd tmp;
						io::read_matrix(path, tmp);
						nodal_dirichlet_mat_.emplace_back(tmp);

						continue;
					}

					int current_id = -1;

					if (j_boundary[i - offset]["id"] == "all")
					{
						assert(boundary_ids_.size() == 1);
						boundary_ids_.clear();
						is_all_ = true;
						nodal_dirichlet_[current_id] = TensorBCValue();
					}
					else
					{
						boundary_ids_[i] = j_boundary[i - offset]["id"];
						current_id = boundary_ids_[i];
						nodal_dirichlet_[current_id] = TensorBCValue();
					}

					auto ff = j_boundary[i - offset]["value"];
					if (ff.is_array())
					{
						for (size_t k = 0; k < ff.size(); ++k)
						{
							displacements_[i].value[k].init(ff[k]);
							nodal_dirichlet_[current_id].value[k].init(ff[k]);
						}
					}
					else
					{
						assert(false);
						displacements_[i].value[0].init(0);
						displacements_[i].value[1].init(0);
						displacements_[i].value[2].init(0);
					}

					displacements_[i].dirichlet_dimension.setConstant(true);
					nodal_dirichlet_[current_id].dirichlet_dimension.setConstant(true);
					if (j_boundary[i - offset].contains("dimension"))
					{
						all_dimensions_dirichlet_ = false;
						auto &tmp = j_boundary[i - offset]["dimension"];
						assert(tmp.is_array());
						for (size_t k = 0; k < tmp.size(); ++k)
						{
							displacements_[i].dirichlet_dimension[k] = tmp[k];
							nodal_dirichlet_[current_id].dirichlet_dimension[k] = tmp[k];
						}
					}

					if (j_boundary[i - offset]["interpolation"].is_array())
					{
						for (int ii = 0; ii < j_boundary[i - offset]["interpolation"].size(); ++ii)
							displacements_[i].interpolation.push_back(Interpolation::build(j_boundary[i - offset]["interpolation"][ii]));
					}
					else
						displacements_[i].interpolation.push_back(Interpolation::build(j_boundary[i - offset]["interpolation"]));

					nodal_dirichlet_[current_id].interpolation = displacements_[i].interpolation;
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

				for (size_t i = offset; i < neumann_boundary_ids_.size(); ++i)
				{
					neumann_boundary_ids_[i] = j_boundary[i - offset]["id"];

					auto ff = j_boundary[i - offset]["value"];
					assert(ff.is_array());

					for (size_t k = 0; k < ff.size(); ++k)
						forces_[i].value[k].init(ff[k]);

					if (j_boundary[i - offset]["interpolation"].is_array())
					{
						for (int ii = 0; ii < j_boundary[i - offset]["interpolation"].size(); ++ii)
							forces_[i].interpolation.push_back(Interpolation::build(j_boundary[i - offset]["interpolation"][ii]));
					}
					else
					{
						forces_[i].interpolation.push_back(Interpolation::build(j_boundary[i - offset]["interpolation"]));
					}
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

				for (size_t i = offset; i < pressure_boundary_ids_.size(); ++i)
				{
					pressure_boundary_ids_[i] = j_boundary[i - offset]["id"];

					auto ff = j_boundary[i - offset]["value"];
					pressures_[i].value.init(ff);

					if (j_boundary[i - offset].contains("interpolation"))
						pressures_[i].interpolation = Interpolation::build(j_boundary[i - offset]["interpolation"]);
					else
						pressures_[i].interpolation = std::make_shared<NoInterpolation>();
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
			displacements_.clear();
			pressures_.clear();

			nodal_dirichlet_.clear();
			nodal_neumann_.clear();
			nodal_dirichlet_mat_.clear();

			initial_position_.clear();
			initial_velocity_.clear();
			initial_acceleration_.clear();

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

		void GenericScalarProblem::set_units(const assembler::Assembler &assembler, const Units &units)
		{
			// TODO?

			for (auto &v : neumann_)
				v.set_unit_type("");

			for (auto &v : dirichlet_)
				v.set_unit_type("");

			for (auto &v : initial_solution_)
				v.second.set_unit_type("");

			rhs_.set_unit_type("");
			exact_.set_unit_type("");

			for (int i = 0; i < 3; ++i)
				exact_grad_[i].set_unit_type("");
		}

		void GenericScalarProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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
					val(i) = dirichlet_[0].eval(pts.row(i), t);
				}
				else
				{
					for (size_t b = 0; b < boundary_ids_.size(); ++b)
					{
						if (id == boundary_ids_[b])
						{
							val(i) = dirichlet_[b].eval(pts.row(i), t);
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
						val(i) = neumann_[b].eval(pts.row(i), t);
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
		void GenericScalarProblem::dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(1, 1);
			const int tag = mesh.get_node_id(node_id);

			if (is_all_)
			{
				assert(nodal_dirichlet_.size() == 1);
				const auto &tmp = nodal_dirichlet_.begin()->second;

				val(0) = tmp.eval(pt, t);
				return;
			}

			const auto it = nodal_dirichlet_.find(tag);
			if (it != nodal_dirichlet_.end())
			{
				val(0) = it->second.eval(pt, t);
				return;
			}

			for (const auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int i = 0; i < n_dirichlet.rows(); ++i)
				{
					if (n_dirichlet(i, 0) == node_id)
					{
						val(0) = n_dirichlet(i, 1);
						return;
					}
				}
			}

			assert(false);
		}

		void GenericScalarProblem::neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val) const
		{
			// TODO implement me;
			log_and_throw_error("Nodal neumann not implemented");
		}

		bool GenericScalarProblem::is_nodal_dirichlet_boundary(const int n_id, const int tag)
		{
			if (nodal_dirichlet_.find(tag) != nodal_dirichlet_.end())
				return true;

			for (const auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int i = 0; i < n_dirichlet.rows(); ++i)
				{
					if (n_dirichlet(i, 0) == n_id)
						return true;
				}
			}

			return false;
		}

		bool GenericScalarProblem::is_nodal_neumann_boundary(const int n_id, const int tag)
		{
			return nodal_neumann_.find(tag) != nodal_neumann_.end();
		}

		bool GenericScalarProblem::has_nodal_dirichlet()
		{
			return nodal_dirichlet_mat_.size() > 0;
		}

		bool GenericScalarProblem::has_nodal_neumann()
		{
			return false; // nodal_neumann_.size() > 0;
		}

		void GenericScalarProblem::update_nodes(const Eigen::VectorXi &in_node_to_node)
		{
			for (auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int n = 0; n < n_dirichlet.rows(); ++n)
				{
					const int node_id = in_node_to_node[n_dirichlet(n, 0)];
					n_dirichlet(n, 0) = node_id;
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
				std::vector<json> j_boundary = flatten_ids(params["dirichlet_boundary"]);

				boundary_ids_.resize(offset + j_boundary.size());
				dirichlet_.resize(offset + j_boundary.size());

				for (size_t i = offset; i < boundary_ids_.size(); ++i)
				{
					if (j_boundary[i - offset].is_string())
					{
						const std::string path = resolve_path(j_boundary[i - offset], params["root_path"]);
						if (!std::filesystem::is_regular_file(path))
							log_and_throw_error(fmt::format("unable to open {} file", path));

						Eigen::MatrixXd tmp;
						io::read_matrix(path, tmp);
						nodal_dirichlet_mat_.emplace_back(tmp);

						continue;
					}

					int current_id = -1;

					if (j_boundary[i - offset]["id"] == "all")
					{
						assert(boundary_ids_.size() == 1);

						is_all_ = true;
						boundary_ids_.clear();
						nodal_dirichlet_[current_id] = ScalarBCValue();
					}
					else
					{
						boundary_ids_[i] = j_boundary[i - offset]["id"];
						current_id = boundary_ids_[i];
						nodal_dirichlet_[current_id] = ScalarBCValue();
					}

					auto ff = j_boundary[i - offset]["value"];
					dirichlet_[i].value.init(ff);
					nodal_dirichlet_[current_id].value.init(ff);

					if (j_boundary[i - offset]["interpolation"].is_array())
					{
						if (j_boundary[i - offset]["interpolation"].size() == 0)
							dirichlet_[i].interpolation = std::make_shared<NoInterpolation>();
						else if (j_boundary[i - offset]["interpolation"].size() == 1)
							dirichlet_[i].interpolation = Interpolation::build(j_boundary[i - offset]["interpolation"][0]);
						else
							log_and_throw_error("Only one Dirichlet interpolation supported");
					}
					else
						dirichlet_[i].interpolation = Interpolation::build(j_boundary[i - offset]["interpolation"]);

					nodal_dirichlet_[current_id].interpolation = dirichlet_[i].interpolation;
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

				for (size_t i = offset; i < neumann_boundary_ids_.size(); ++i)
				{
					neumann_boundary_ids_[i] = j_boundary[i - offset]["id"];

					auto ff = j_boundary[i - offset]["value"];
					neumann_[i].value.init(ff);

					if (j_boundary[i - offset]["interpolation"].is_array())
					{
						if (j_boundary[i - offset]["interpolation"].size() == 0)
							neumann_[i].interpolation = std::make_shared<NoInterpolation>();
						else if (j_boundary[i - offset]["interpolation"].size() == 1)
							neumann_[i].interpolation = Interpolation::build(j_boundary[i - offset]["interpolation"][0]);
						else
							log_and_throw_error("Only one Neumann interpolation supported");
					}
					else
						neumann_[i].interpolation = Interpolation::build(j_boundary[i - offset]["interpolation"]);
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
			dirichlet_.back().value.init(val);
			dirichlet_.back().interpolation = interp;
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

			dirichlet_[index].value.init(val);
			dirichlet_[index].interpolation = interp;
		}

		void GenericScalarProblem::add_neumann_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			neumann_.emplace_back();
			neumann_.back().value.init(val);
			neumann_.back().interpolation = interp;
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

			neumann_[index].value.init(val);
			neumann_[index].interpolation = interp;
		}

		void GenericScalarProblem::add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			boundary_ids_.push_back(id);
			dirichlet_.emplace_back();
			dirichlet_.back().value.init(func);
			dirichlet_.back().interpolation = interp;
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
			dirichlet_[index].value.init(func);
			dirichlet_[index].interpolation = interp;
		}

		void GenericScalarProblem::add_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp)
		{
			neumann_boundary_ids_.push_back(id);
			neumann_.emplace_back();
			neumann_.back().value.init(func);
			neumann_.back().interpolation = interp;
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
			neumann_[index].value.init(func);
			neumann_[index].interpolation = interp;
		}

		void GenericScalarProblem::add_dirichlet_boundary(const int id, const json &val, const std::string &interp)
		{
			boundary_ids_.push_back(id);
			dirichlet_.emplace_back();
			dirichlet_.back().value.init(val);
			if (interp.empty())
				dirichlet_.back().interpolation = std::make_shared<NoInterpolation>();
			else
				dirichlet_.back().interpolation = Interpolation::build(interp);
		}

		void GenericScalarProblem::add_neumann_boundary(const int id, const json &val, const std::string &interp)
		{
			neumann_boundary_ids_.push_back(id);
			neumann_.emplace_back();
			neumann_.back().value.init(val);
			if (interp.empty())
				neumann_.back().interpolation = std::make_shared<NoInterpolation>();
			else
				neumann_.back().interpolation = Interpolation::build(interp);
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

			dirichlet_[index].value.init(val);
			if (interp.empty())
				dirichlet_[index].interpolation = std::make_shared<NoInterpolation>();
			else
				dirichlet_[index].interpolation = Interpolation::build(interp);
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
			neumann_[index].value.init(val);

			if (interp.empty())
				neumann_[index].interpolation = std::make_shared<NoInterpolation>();
			else
				neumann_[index].interpolation = Interpolation::build(interp);
		}

		void GenericScalarProblem::clear()
		{
			neumann_.clear();
			dirichlet_.clear();

			nodal_dirichlet_.clear();
			nodal_neumann_.clear();
			nodal_dirichlet_mat_.clear();

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
