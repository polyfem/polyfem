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
			bool matches_fe_space(const int entry_fe_space_id, const int fe_space_id)
			{
				return entry_fe_space_id < 0 || fe_space_id < 0 || entry_fe_space_id == fe_space_id;
			}

			int fe_space_id(const json &entry)
			{
				return entry.value("fe_space", -1);
			}

			template <typename Map>
			const typename Map::mapped_type *find_for_fe_space(const Map &values, const int fe_space_id)
			{
				if (const auto it = values.find(fe_space_id); it != values.end())
					return &it->second;
				if (const auto it = values.find(-1); it != values.end())
					return &it->second;
				if (fe_space_id < 0 && values.size() == 1)
					return &values.begin()->second;
				return nullptr;
			}

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
				for (int i = 0; i < 3; ++i)
				{
					for (auto &[fe_space_id, rhs] : rhs_)
						rhs[i].set_unit_type(units.force());
					exact_[i].set_unit_type(units.velocity());
				}
				for (int i = 0; i < 3; ++i)
					exact_grad_[i].set_unit_type("");

				for (auto &v : displacements_)
					v.set_unit_type(units.velocity());

				for (auto &v : forces_)
					v.set_unit_type(units.force());

				for (auto &v : normal_aligned_forces_)
					v.set_unit_type(units.pressure());

				for (auto &v : pressures_)
					v.set_unit_type(units.pressure());

				for (auto &v : cavity_pressures_)
					v.second.set_unit_type(units.pressure());

				for (auto &v : initial_position_)
					for (int i = 0; i < 3; ++i)
						v.value[i].set_unit_type(units.velocity());

				for (auto &v : initial_velocity_)
					for (int i = 0; i < 3; ++i)
						v.value[i].set_unit_type(units.velocity());

				for (auto &v : initial_acceleration_)
					for (int i = 0; i < 3; ++i)
						v.value[i].set_unit_type(units.acceleration());

				for (auto &v : nodal_dirichlet_)
					v.second.set_unit_type(units.velocity());

				for (auto &v : nodal_neumann_)
					v.second.set_unit_type(units.force());
			}
			else
			{
				for (int i = 0; i < 3; ++i)
				{
					for (auto &[fe_space_id, rhs] : rhs_)
						rhs[i].set_unit_type(units.acceleration());
					exact_[i].set_unit_type(units.length());
				}
				for (int i = 0; i < 3; ++i)
					exact_grad_[i].set_unit_type("");

				for (auto &v : displacements_)
					v.set_unit_type(units.length());

				for (auto &v : forces_)
					v.set_unit_type(units.force());

				for (auto &v : normal_aligned_forces_)
					v.set_unit_type(units.pressure());

				for (auto &v : pressures_)
					v.set_unit_type(units.pressure());

				for (auto &v : cavity_pressures_)
					v.second.set_unit_type(units.pressure());

				for (auto &v : initial_position_)
					for (int i = 0; i < 3; ++i)
						v.value[i].set_unit_type(units.length());

				for (auto &v : initial_velocity_)
					for (int i = 0; i < 3; ++i)
						v.value[i].set_unit_type(units.velocity());

				for (auto &v : initial_acceleration_)
					for (int i = 0; i < 3; ++i)
						v.value[i].set_unit_type(units.acceleration());

				for (auto &v : nodal_dirichlet_)
					v.second.set_unit_type(units.length());

				for (auto &v : nodal_neumann_)
					v.second.set_unit_type(units.force());
			}
		}

		void GenericTensorProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			const auto *rhs = find_for_fe_space(rhs_, fe_space_id);
			const auto *rhs_size = find_for_fe_space(rhs_size_, fe_space_id);
			const int size = rhs_size == nullptr ? pts.cols() : *rhs_size;
			val.resize(pts.rows(), size);

			if (rhs == nullptr || is_rhs_zero(fe_space_id))
			{
				val.setZero();
				return;
			}

			const bool planar = pts.cols() == 2;
			for (int i = 0; i < pts.rows(); ++i)
			{
				for (int j = 0; j < size; ++j)
				{
					double x = pts(i, 0), y = pts(i, 1), z = planar ? 0 : pts(i, 2);
					val(i, j) = (*rhs)[j](x, y, z, t);
				}
			}
		}

		bool GenericTensorProblem::is_rhs_zero(const int fe_space_id) const
		{
			const auto *rhs = find_for_fe_space(rhs_, fe_space_id);
			const auto *rhs_size = find_for_fe_space(rhs_size_, fe_space_id);
			if (rhs == nullptr || rhs_size == nullptr)
				return true;
			for (int i = 0; i < *rhs_size; ++i)
				if (!(*rhs)[i].is_zero())
					return false;
			return true;
		}

		bool GenericTensorProblem::has_boundary(const BoundaryKind kind, const int tag, const int fe_space_id)
		{
			if (tag <= 0)
				return false;

			if (kind == BoundaryKind::Dirichlet)
			{
				for (size_t i = 0; i < boundary_ids_.size(); ++i)
					if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(displacements_[i].fe_space_id, fe_space_id))
						return true;
				return false;
			}

			for (size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
				if (neumann_boundary_ids_[i] == tag && matches_fe_space(forces_[i].fe_space_id, fe_space_id))
					return true;
			for (size_t i = 0; i < normal_aligned_neumann_boundary_ids_.size(); ++i)
				if (normal_aligned_neumann_boundary_ids_[i] == tag && matches_fe_space(normal_aligned_forces_[i].fe_space_id, fe_space_id))
					return true;
			return false;
		}

		bool GenericTensorProblem::is_dimension_dirichet(const int tag, const int dim, const int fe_space_id) const
		{
			if (all_dimensions_dirichlet(fe_space_id))
				return true;

			for (size_t b = 0; b < boundary_ids_.size(); ++b)
			{
				if ((boundary_ids_[b] < 0 || tag == boundary_ids_[b]) && matches_fe_space(displacements_[b].fe_space_id, fe_space_id))
				{
					auto &tmp = displacements_[b].dirichlet_dimension;
					return tmp[dim];
				}
			}

			assert(false);
			return true;
		}

		bool GenericTensorProblem::all_dimensions_dirichlet(const int fe_space_id) const
		{
			for (const TensorBCValue &displacement : displacements_)
			{
				if (!matches_fe_space(displacement.fe_space_id, fe_space_id))
					continue;

				for (int d = 0; d < displacement.dirichlet_dimension.size(); ++d)
				{
					if (!displacement.dirichlet_dimension(d))
						return false;
				}
			}

			for (const auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int i = 0; i < n_dirichlet.rows(); ++i)
				{
					for (int d = 1; d < n_dirichlet.cols(); ++d)
					{
						if (std::isnan(n_dirichlet(i, d)))
							return false;
					}
				}
			}

			return true;
		}

		void GenericTensorProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			int value_size = mesh.dimension();
			for (const TensorBCValue &displacement : displacements_)
			{
				if (displacement.size > 0 && matches_fe_space(displacement.fe_space_id, fe_space_id))
				{
					value_size = displacement.size;
					break;
				}
			}
			val = Eigen::MatrixXd::Zero(pts.rows(), value_size);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if ((boundary_ids_[b] < 0 || id == boundary_ids_[b]) && matches_fe_space(displacements_[b].fe_space_id, fe_space_id))
					{
						for (int d = 0; d < std::min<int>(val.cols(), displacements_[b].size); ++d)
						{
							val(i, d) = displacements_[b].eval(pts.row(i), d, t);
						}
						break;
					}
				}
			}
		}

		void GenericTensorProblem::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), mesh.dimension());

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));

				for (size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
				{
					if (id == neumann_boundary_ids_[b] && matches_fe_space(forces_[b].fe_space_id, fe_space_id))
					{
						for (int d = 0; d < std::min<int>(val.cols(), forces_[b].size); ++d)
						{
							val(i, d) = forces_[b].eval(pts.row(i), d, t);
						}

						break;
					}
				}

				for (size_t b = 0; b < normal_aligned_neumann_boundary_ids_.size(); ++b)
				{
					if (id == normal_aligned_neumann_boundary_ids_[b] && matches_fe_space(normal_aligned_forces_[b].fe_space_id, fe_space_id))
					{
						for (int d = 0; d < val.cols(); ++d)
						{
							val(i, d) = normal_aligned_forces_[b].eval(pts.row(i), t) * normals(i, d);
						}
						break;
					}
				}
			}
		}

		void GenericTensorProblem::pressure_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), 1);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));

				for (size_t b = 0; b < pressure_boundary_ids_.size(); ++b)
				{
					if (id == pressure_boundary_ids_[b])
					{
						val(i) = pressures_[b].eval(pts.row(i), t);
						break;
					}
				}
			}
		}

		double GenericTensorProblem::pressure_cavity_bc(const int boundary_id, const double t) const
		{
			Eigen::VectorXd pt;
			pt.setZero(3);
			return cavity_pressures_.at(boundary_id).eval(pt, t);
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

		void GenericTensorProblem::dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			val = Eigen::MatrixXd::Zero(1, mesh.dimension());
			const int tag = mesh.get_node_id(node_id);

			for (size_t i = 0; i < boundary_ids_.size(); ++i)
			{
				if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(displacements_[i].fe_space_id, fe_space_id))
				{
					val = Eigen::MatrixXd::Zero(1, displacements_[i].size);
					for (int d = 0; d < val.cols(); ++d)
						val(d) = displacements_[i].eval(pt, d, t);
					return;
				}
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

		void GenericTensorProblem::neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			// TODO implement me;
			log_and_throw_error("Nodal neumann not implemented");
		}

		bool GenericTensorProblem::is_nodal_dirichlet_boundary(const int n_id, const int tag, const int fe_space_id)
		{
			for (size_t i = 0; i < boundary_ids_.size(); ++i)
				if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(displacements_[i].fe_space_id, fe_space_id))
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

		bool GenericTensorProblem::is_nodal_neumann_boundary(const int n_id, const int tag, const int fe_space_id)
		{
			for (size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
				if (neumann_boundary_ids_[i] == tag && matches_fe_space(forces_[i].fe_space_id, fe_space_id))
					return true;
			return false;
		}

		bool GenericTensorProblem::has_nodal_dirichlet(const int fe_space_id)
		{
			return !nodal_dirichlet_mat_.empty();
		}

		bool GenericTensorProblem::has_nodal_neumann(const int fe_space_id)
		{
			return false;
		}

		bool GenericTensorProblem::is_nodal_dimension_dirichlet(const int n_id, const int tag, const int dim, const int fe_space_id) const
		{
			for (size_t i = 0; i < boundary_ids_.size(); ++i)
				if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(displacements_[i].fe_space_id, fe_space_id))
					return displacements_[i].dirichlet_dimension(dim);

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
			if (updated_dirichlet_node_ordering_)
			{
				if (nodal_dirichlet_mat_.size() > 0)
					logger().debug("Skipping updating in nodes to nodes in problem, already done once...");
				return;
			}
			for (auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int n = 0; n < n_dirichlet.rows(); ++n)
				{
					const int node_id = in_node_to_node[n_dirichlet(n, 0)];
					n_dirichlet(n, 0) = node_id;
				}
			}
			updated_dirichlet_node_ordering_ = true;
		}

		void GenericTensorProblem::set_parameters(const json &params, const std::string &root_path)
		{
			if (is_param_valid(params, "is_time_dependent"))
			{
				is_time_dept_ = params["is_time_dependent"];
			}

			if (is_param_valid(params, "rhs"))
			{
				const json &rr = params["rhs"];
				const bool has_fe_spaces = rr.is_array() && !rr.empty() && rr.front().is_object() && rr.front().contains("fe_space") && rr.front().contains("value");
				if (has_fe_spaces)
				{
					for (const json &entry : rr)
					{
						const int id = fe_space_id(entry);
						const json &value = entry["value"];
						const int size = value.is_array() ? int(value.size()) : 1;
						if (size > 3)
							log_and_throw_error("RHS for FE space {} has {} components; at most 3 are supported.", id, size);
						rhs_size_[id] = size;
						for (int k = 0; k < size; ++k)
							rhs_[id][k].init(value.is_array() ? value[k] : value, root_path);
					}
				}
				else if (rr.is_array() && !rr.empty())
				{
					if (rr.size() > 3)
						log_and_throw_error("RHS has {} components; at most 3 are supported.", rr.size());
					rhs_size_[-1] = int(rr.size());
					for (size_t k = 0; k < rr.size(); ++k)
						rhs_[-1][k].init(rr[k], root_path);
				}
				else if (!rr.is_array())
					log_and_throw_error("Invalid tensor problem RHS: expected an array.");
			}

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "solution"))
			{
				auto ex = params["reference"]["solution"];
				has_exact_ = ex.size() > 0;
				if (ex.is_array())
				{
					for (size_t k = 0; k < ex.size(); ++k)
						exact_[k].init(ex[k], root_path);
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
						exact_grad_[k].init(ex[k], root_path);
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
							log_and_throw_error("unable to open {} file", path);

						Eigen::MatrixXd tmp;
						io::read_matrix(path, tmp);
						nodal_dirichlet_mat_.emplace_back(tmp);

						continue;
					}

					int current_id = -1;
					displacements_[i].fe_space_id = fe_space_id(j_boundary[i - offset]);

					if (j_boundary[i - offset]["id"] == "all")
					{
						boundary_ids_[i] = -1;
						is_all_ = true;
						nodal_dirichlet_[current_id] = TensorBCValue();
					}
					else
					{
						boundary_ids_[i] = j_boundary[i - offset]["id"];
						current_id = boundary_ids_[i];
						nodal_dirichlet_[current_id] = TensorBCValue();
					}
					nodal_dirichlet_[current_id].fe_space_id = displacements_[i].fe_space_id;

					auto ff = j_boundary[i - offset]["value"];
					if (ff.is_array())
					{
						if (ff.size() > 3)
							log_and_throw_error("Dirichlet condition for FE space {} has {} components; at most 3 are supported.", displacements_[i].fe_space_id, ff.size());
						displacements_[i].size = int(ff.size());
						for (size_t k = 0; k < ff.size(); ++k)
						{
							displacements_[i].value[k].init(ff[k], root_path);
							if (j_boundary[i - offset].contains("time_reference") && j_boundary[i - offset]["time_reference"].size() > 0)
								displacements_[i].value[k].set_t(j_boundary[i - offset]["time_reference"]);
							nodal_dirichlet_[current_id].value[k].init(ff[k], root_path);
						}
					}
					else
					{
						displacements_[i].size = 1;
						displacements_[i].value[0].init(ff, root_path);
						nodal_dirichlet_[current_id].value[0].init(ff, root_path);
					}
					nodal_dirichlet_[current_id].size = displacements_[i].size;

					displacements_[i].dirichlet_dimension.setConstant(true);
					nodal_dirichlet_[current_id].dirichlet_dimension.setConstant(true);
					if (j_boundary[i - offset].contains("dimension"))
					{
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
					forces_[i].fe_space_id = fe_space_id(j_boundary[i - offset]);

					auto ff = j_boundary[i - offset]["value"];
					forces_[i].size = ff.is_array() ? int(ff.size()) : 1;
					if (forces_[i].size > 3)
						log_and_throw_error("Neumann condition for FE space {} has {} components; at most 3 are supported.", forces_[i].fe_space_id, forces_[i].size);
					for (int k = 0; k < forces_[i].size; ++k)
						forces_[i].value[k].init(ff.is_array() ? ff[k] : ff, root_path);

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

			if (is_param_valid(params, "normal_aligned_neumann_boundary"))
			{
				const int offset = normal_aligned_neumann_boundary_ids_.size();

				auto j_boundary_tmp = params["normal_aligned_neumann_boundary"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

				normal_aligned_neumann_boundary_ids_.resize(offset + j_boundary.size());
				normal_aligned_forces_.resize(offset + j_boundary.size());

				for (size_t i = offset; i < normal_aligned_neumann_boundary_ids_.size(); ++i)
				{
					normal_aligned_neumann_boundary_ids_[i] = j_boundary[i - offset]["id"];
					normal_aligned_forces_[i].fe_space_id = fe_space_id(j_boundary[i - offset]);

					auto ff = j_boundary[i - offset]["value"];
					normal_aligned_forces_[i].value.init(ff, root_path);

					if (j_boundary[i - offset].contains("interpolation"))
						normal_aligned_forces_[i].interpolation = Interpolation::build(j_boundary[i - offset]["interpolation"]);
					else
						normal_aligned_forces_[i].interpolation = std::make_shared<NoInterpolation>();
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
					pressures_[i].value.init(ff, root_path);
					if (j_boundary[i - offset].contains("time_reference") && j_boundary[i - offset]["time_reference"].size() > 0)
						pressures_[i].value.set_t(j_boundary[i - offset]["time_reference"]);

					pressures_[i].interpolation = std::make_shared<NoInterpolation>();
				}
			}

			if (is_param_valid(params, "pressure_cavity"))
			{
				const int offset = pressure_cavity_ids_.size();

				auto j_boundary_tmp = params["pressure_cavity"];
				std::vector<json> j_boundary = flatten_ids(j_boundary_tmp);

				pressure_cavity_ids_.resize(offset + j_boundary.size());

				for (size_t i = offset; i < pressure_cavity_ids_.size(); ++i)
				{
					int boundary_id = j_boundary[i - offset]["id"];
					pressure_cavity_ids_[i] = boundary_id;

					if (cavity_pressures_.find(boundary_id) == cavity_pressures_.end())
					{
						cavity_pressures_[boundary_id] = ScalarBCValue();

						auto ff = j_boundary[i - offset]["value"];
						cavity_pressures_[boundary_id].value.init(ff, root_path);

						cavity_pressures_[boundary_id].interpolation = std::make_shared<NoInterpolation>();
					}
				}
			}

			if (is_param_valid(params, "solution"))
			{
				auto rr = params["solution"];
				initial_position_.resize(rr.size());
				assert(rr.is_array());

				for (size_t k = 0; k < rr.size(); ++k)
				{
					initial_position_[k].body_id = rr[k]["id"];
					initial_position_[k].fe_space_id = fe_space_id(rr[k]);
					const auto v = rr[k]["value"];
					initial_position_[k].size = v.is_array() ? int(v.size()) : 1;
					if (initial_position_[k].size > 3)
						log_and_throw_error("Initial solution for FE space {} has {} components; at most 3 are supported.", initial_position_[k].fe_space_id, initial_position_[k].size);
					for (int d = 0; d < initial_position_[k].size; ++d)
						initial_position_[k].value[d].init(v.is_array() ? v[d] : v, root_path);
				}
			}

			if (is_param_valid(params, "velocity"))
			{
				auto rr = params["velocity"];
				initial_velocity_.resize(rr.size());
				assert(rr.is_array());

				for (size_t k = 0; k < rr.size(); ++k)
				{
					initial_velocity_[k].body_id = rr[k]["id"];
					initial_velocity_[k].fe_space_id = fe_space_id(rr[k]);
					const auto v = rr[k]["value"];
					initial_velocity_[k].size = v.is_array() ? int(v.size()) : 1;
					if (initial_velocity_[k].size > 3)
						log_and_throw_error("Initial velocity for FE space {} has {} components; at most 3 are supported.", initial_velocity_[k].fe_space_id, initial_velocity_[k].size);
					for (int d = 0; d < initial_velocity_[k].size; ++d)
						initial_velocity_[k].value[d].init(v.is_array() ? v[d] : v, root_path);
				}
			}

			if (is_param_valid(params, "acceleration"))
			{
				auto rr = params["acceleration"];
				initial_acceleration_.resize(rr.size());
				assert(rr.is_array());

				for (size_t k = 0; k < rr.size(); ++k)
				{
					initial_acceleration_[k].body_id = rr[k]["id"];
					initial_acceleration_[k].fe_space_id = fe_space_id(rr[k]);
					const auto v = rr[k]["value"];
					initial_acceleration_[k].size = v.is_array() ? int(v.size()) : 1;
					if (initial_acceleration_[k].size > 3)
						log_and_throw_error("Initial acceleration for FE space {} has {} components; at most 3 are supported.", initial_acceleration_[k].fe_space_id, initial_acceleration_[k].size);
					for (int d = 0; d < initial_acceleration_[k].size; ++d)
						initial_acceleration_[k].value[d].init(v.is_array() ? v[d] : v, root_path);
				}
			}
		}

		void GenericTensorProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			int value_size = pts.cols();
			for (const TensorInitialValue &entry : initial_position_)
				if (matches_fe_space(entry.fe_space_id, fe_space_id))
				{
					value_size = entry.size;
					break;
				}
			val.resize(pts.rows(), value_size);
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
					if (initial_position_[j].body_id == id && matches_fe_space(initial_position_[j].fe_space_id, fe_space_id))
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

				for (int j = 0; j < val.cols(); ++j)
					val(i, j) = planar ? initial_position_[index].value[j](pts(i, 0), pts(i, 1)) : initial_position_[index].value[j](pts(i, 0), pts(i, 1), pts(i, 2));
			}
		}

		void GenericTensorProblem::initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			int value_size = pts.cols();
			for (const TensorInitialValue &entry : initial_velocity_)
				if (matches_fe_space(entry.fe_space_id, fe_space_id))
				{
					value_size = entry.size;
					break;
				}
			val.resize(pts.rows(), value_size);
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
					if (initial_velocity_[j].body_id == id && matches_fe_space(initial_velocity_[j].fe_space_id, fe_space_id))
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

				for (int j = 0; j < val.cols(); ++j)
					val(i, j) = planar ? initial_velocity_[index].value[j](pts(i, 0), pts(i, 1)) : initial_velocity_[index].value[j](pts(i, 0), pts(i, 1), pts(i, 2));
			}
		}

		void GenericTensorProblem::initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			int value_size = pts.cols();
			for (const TensorInitialValue &entry : initial_acceleration_)
				if (matches_fe_space(entry.fe_space_id, fe_space_id))
				{
					value_size = entry.size;
					break;
				}
			val.resize(pts.rows(), value_size);
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
					if (initial_acceleration_[j].body_id == id && matches_fe_space(initial_acceleration_[j].fe_space_id, fe_space_id))
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

				for (int j = 0; j < val.cols(); ++j)
					val(i, j) = planar ? initial_acceleration_[index].value[j](pts(i, 0), pts(i, 1)) : initial_acceleration_[index].value[j](pts(i, 0), pts(i, 1), pts(i, 2));
			}
		}

		void GenericTensorProblem::clear()
		{
			has_exact_ = false;
			has_exact_grad_ = false;
			is_time_dept_ = false;

			forces_.clear();
			displacements_.clear();
			normal_aligned_forces_.clear();
			pressures_.clear();
			cavity_pressures_.clear();

			nodal_dirichlet_.clear();
			nodal_neumann_.clear();
			nodal_dirichlet_mat_.clear();

			initial_position_.clear();
			initial_velocity_.clear();
			initial_acceleration_.clear();

			rhs_.clear();
			rhs_size_.clear();
			for (int i = 0; i < exact_.size(); ++i)
				exact_[i].clear();
			for (int i = 0; i < exact_grad_.size(); ++i)
				exact_grad_[i].clear();
			is_all_ = false;
			updated_dirichlet_node_ordering_ = false;
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
				v.value.set_unit_type("");

			for (auto &[fe_space_id, rhs] : rhs_)
				rhs.set_unit_type("");
			exact_.set_unit_type("");

			for (int i = 0; i < 3; ++i)
				exact_grad_[i].set_unit_type("");
		}

		void GenericScalarProblem::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			val.resize(pts.rows(), 1);
			const auto *rhs = find_for_fe_space(rhs_, fe_space_id);
			if (rhs == nullptr || rhs->is_zero())
			{
				val.setZero();
				return;
			}
			const bool planar = pts.cols() == 2;
			for (int i = 0; i < pts.rows(); ++i)
			{
				double x = pts(i, 0), y = pts(i, 1), z = planar ? 0 : pts(i, 2);
				val(i) = (*rhs)(x, y, z, t);
			}
		}

		bool GenericScalarProblem::is_rhs_zero(const int fe_space_id) const
		{
			const auto *rhs = find_for_fe_space(rhs_, fe_space_id);
			return rhs == nullptr || rhs->is_zero();
		}

		bool GenericScalarProblem::has_boundary(const BoundaryKind kind, const int tag, const int fe_space_id)
		{
			if (tag <= 0)
				return false;
			if (kind == BoundaryKind::Dirichlet)
			{
				for (size_t i = 0; i < boundary_ids_.size(); ++i)
					if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(dirichlet_[i].fe_space_id, fe_space_id))
						return true;
				return false;
			}
			for (size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
				if (neumann_boundary_ids_[i] == tag && matches_fe_space(neumann_[i].fe_space_id, fe_space_id))
					return true;
			return false;
		}

		void GenericScalarProblem::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), 1);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));
				for (size_t b = 0; b < boundary_ids_.size(); ++b)
				{
					if ((boundary_ids_[b] < 0 || id == boundary_ids_[b]) && matches_fe_space(dirichlet_[b].fe_space_id, fe_space_id))
					{
						val(i) = dirichlet_[b].eval(pts.row(i), t);
						break;
					}
				}
			}
		}

		void GenericScalarProblem::neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			val = Eigen::MatrixXd::Zero(pts.rows(), 1);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const int id = mesh.get_boundary_id(global_ids(i));

				for (size_t b = 0; b < neumann_boundary_ids_.size(); ++b)
				{
					if (id == neumann_boundary_ids_[b] && matches_fe_space(neumann_[b].fe_space_id, fe_space_id))
					{
						double x = pts(i, 0), y = pts(i, 1), z = pts.cols() == 2 ? 0 : pts(i, 2);
						val(i) = neumann_[b].eval(pts.row(i), t);
						break;
					}
				}
			}
		}

		void GenericScalarProblem::initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id) const
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
					if (initial_solution_[j].body_id == id && matches_fe_space(initial_solution_[j].fe_space_id, fe_space_id))
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

				val(i) = planar ? initial_solution_[index].value(pts(i, 0), pts(i, 1)) : initial_solution_[index].value(pts(i, 0), pts(i, 1), pts(i, 2));
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

		void GenericTensorProblem::update_pressure_boundary(const int id, const int time_step, const double val)
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
				throw "Invalid boundary id for pressure update";
			}

			if (pressures_[index].value.is_mat())
			{
				Eigen::MatrixXd curr_val = pressures_[index].value.get_mat();
				assert(time_step <= curr_val.size());
				assert(curr_val.cols() == 1);
				curr_val(time_step) = val;
				pressures_[index].value.set_mat(curr_val);
			}
			else
			{
				pressures_[index].value.init(val);
			}
		}

		void GenericTensorProblem::update_dirichlet_boundary(const int id, const int time_step, const Eigen::VectorXd &val)
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
				throw "Invalid boundary id for dirichlet update";
			}

			for (int i = 0; i < val.size(); ++i)
			{
				if (displacements_[index].value[i].is_mat())
				{
					Eigen::MatrixXd curr_val = displacements_[index].value[i].get_mat();
					assert(time_step <= curr_val.size());
					assert(curr_val.cols() == 1);
					curr_val(time_step) = val(i);
					displacements_[index].value[i].set_mat(curr_val);
				}
				else
				{
					displacements_[index].value[i].init(val(i));
				}
			}
		}

		void GenericTensorProblem::update_dirichlet_nodes(const Eigen::VectorXi &in_node_to_node, const Eigen::VectorXi &node_ids, const Eigen::MatrixXd &nodal_dirichlet)
		{
			assert(node_ids.size() == nodal_dirichlet.rows());
			// NOTE!!! Update nodes called in `State::build_basis()` so the first row of this mat
			// must be set to input node ordering if `build_basis` will be called, like in optimization
			for (auto &n_dirichlet : nodal_dirichlet_mat_)
				for (int i = 0; i < node_ids.size(); ++i)
				{
					int mapped_node_id = in_node_to_node(node_ids(i));
					for (int j = 0; j < n_dirichlet.rows(); ++j)
						if (mapped_node_id == n_dirichlet(j, 0))
							for (int k = 0; k < n_dirichlet.cols() - 1; ++k)
								n_dirichlet(j, k + 1) = nodal_dirichlet(i, k);
				}
		}

		void GenericScalarProblem::dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			val = Eigen::MatrixXd::Zero(1, 1);
			const int tag = mesh.get_node_id(node_id);

			for (size_t i = 0; i < boundary_ids_.size(); ++i)
			{
				if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(dirichlet_[i].fe_space_id, fe_space_id))
				{
					val(0) = dirichlet_[i].eval(pt, t);
					return;
				}
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

		void GenericScalarProblem::neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val, const int fe_space_id) const
		{
			// TODO implement me;
			log_and_throw_error("Nodal neumann not implemented");
		}

		bool GenericScalarProblem::is_nodal_dirichlet_boundary(const int n_id, const int tag, const int fe_space_id)
		{
			for (size_t i = 0; i < boundary_ids_.size(); ++i)
				if ((boundary_ids_[i] < 0 || boundary_ids_[i] == tag) && matches_fe_space(dirichlet_[i].fe_space_id, fe_space_id))
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

		bool GenericScalarProblem::is_nodal_neumann_boundary(const int n_id, const int tag, const int fe_space_id)
		{
			for (size_t i = 0; i < neumann_boundary_ids_.size(); ++i)
				if (neumann_boundary_ids_[i] == tag && matches_fe_space(neumann_[i].fe_space_id, fe_space_id))
					return true;
			return false;
		}

		bool GenericScalarProblem::has_nodal_dirichlet(const int fe_space_id)
		{
			return nodal_dirichlet_mat_.size() > 0;
		}

		bool GenericScalarProblem::has_nodal_neumann(const int fe_space_id)
		{
			return false; // nodal_neumann_.size() > 0;
		}

		void GenericScalarProblem::update_nodes(const Eigen::VectorXi &in_node_to_node)
		{
			if (updated_dirichlet_node_ordering_)
				return;
			for (auto &n_dirichlet : nodal_dirichlet_mat_)
			{
				for (int n = 0; n < n_dirichlet.rows(); ++n)
				{
					const int node_id = in_node_to_node[n_dirichlet(n, 0)];
					n_dirichlet(n, 0) = node_id;
				}
			}
			updated_dirichlet_node_ordering_ = true;
		}

		void GenericScalarProblem::set_parameters(const json &params, const std::string &root_path)
		{
			if (is_param_valid(params, "is_time_dependent"))
			{
				is_time_dept_ = params["is_time_dependent"];
			}

			if (is_param_valid(params, "rhs"))
			{
				const json &rr = params["rhs"];
				const bool has_fe_spaces = rr.is_array() && !rr.empty() && rr.front().is_object() && rr.front().contains("fe_space") && rr.front().contains("value");
				if (has_fe_spaces)
				{
					for (const json &entry : rr)
						rhs_[fe_space_id(entry)].init(entry["value"], root_path);
				}
				else if (!rr.is_array() || !rr.empty())
					rhs_[-1].init(rr, root_path);
			}

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "solution"))
			{
				has_exact_ = !params["reference"]["solution"].empty();
				exact_.init(params["reference"]["solution"], root_path);
			}

			if (is_param_valid(params, "reference") && is_param_valid(params["reference"], "gradient"))
			{
				auto ex = params["reference"]["gradient"];
				has_exact_grad_ = ex.size() > 0;
				if (ex.is_array())
				{
					for (size_t k = 0; k < ex.size(); ++k)
						exact_grad_[k].init(ex[k], root_path);
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
					dirichlet_[i].fe_space_id = fe_space_id(j_boundary[i - offset]);

					if (j_boundary[i - offset]["id"] == "all")
					{
						boundary_ids_[i] = -1;
						is_all_ = true;
						nodal_dirichlet_[current_id] = ScalarBCValue();
					}
					else
					{
						boundary_ids_[i] = j_boundary[i - offset]["id"];
						current_id = boundary_ids_[i];
						nodal_dirichlet_[current_id] = ScalarBCValue();
					}
					nodal_dirichlet_[current_id].fe_space_id = dirichlet_[i].fe_space_id;

					auto ff = j_boundary[i - offset]["value"];
					dirichlet_[i].value.init(ff, root_path);
					nodal_dirichlet_[current_id].value.init(ff, root_path);

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
					neumann_[i].fe_space_id = fe_space_id(j_boundary[i - offset]);

					auto ff = j_boundary[i - offset]["value"];
					neumann_[i].value.init(ff, root_path);

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
					initial_solution_[k].body_id = rr[k]["id"];
					initial_solution_[k].fe_space_id = fe_space_id(rr[k]);
					initial_solution_[k].value.init(rr[k]["value"], root_path);
				}
			}
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
			updated_dirichlet_node_ordering_ = false;
		}
	} // namespace assembler
} // namespace polyfem
