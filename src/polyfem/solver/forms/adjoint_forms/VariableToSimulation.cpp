#include "VariableToSimulation.hpp"
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

namespace polyfem::solver
{
	namespace
	{
		template <typename T>
		std::string to_string_with_precision(const T a_value, const int n = 6)
		{
			std::ostringstream out;
			out.precision(n);
			out << std::fixed << a_value;
			return out.str();
		}

		RowVectorNd get_barycenter(const mesh::Mesh &mesh, int e)
		{
			RowVectorNd barycenter;
			if (!mesh.is_volume())
			{
				const auto &mesh2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
				barycenter = mesh2d.face_barycenter(e);
			}
			else
			{
				const auto &mesh3d = dynamic_cast<const mesh::Mesh3D &>(mesh);
				barycenter = mesh3d.cell_barycenter(e);
			}
			return barycenter;
		}
	} // namespace

	void ShapeVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		const int dim = state_ptr_->mesh->dimension();

		// If indices include one vertex entry, we assume it include all entries of this vertex.
		for (int i = 0; i < indices.size(); i += dim)
			for (int j = 0; j < dim; j++)
				assert(indices(i + j) == indices(i) + j);

		for (int i = 0; i < indices.size(); i += dim)
			state_ptr_->set_mesh_vertex(indices(i) / dim, state_variable(Eigen::seqN(i, dim)));

		// TODO: move this to the end of all variable to simulation
		state_ptr_->build_basis();
	}
	Eigen::VectorXd ShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_shape_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			AdjointTools::dJ_shape_static_adjoint_term(get_state(), get_state().diff_cached[0].u, get_state().get_adjoint_mat(0), term);
		}
		return parametrization_.apply_jacobian(term, x);
	}

	SDFShapeVariableToSimulation::SDFShapeVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization, const json &args) : ShapeVariableToSimulation(state_ptr, parametrization), out_velocity_path_("micro-tmp-velocity.msh"), out_msh_path_("micro-tmp.msh"), isosurface_inflator_prefix_(args["isosurface_inflator_prefix"].get<std::string>()), unit_size_(args["unit_size"].get<double>()), periodic_tiling_(unit_size_ > 0)
	{
	}
	void SDFShapeVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
	}
	bool SDFShapeVariableToSimulation::generate_graph_mesh(const Eigen::VectorXd &x)
	{
		std::string shape_params = "--params \"";
		for (int i = 0; i < x.size(); i++)
			shape_params += to_string_with_precision(x(i), 16) + " ";
		shape_params += "\" ";

		std::string command = isosurface_inflator_prefix_ + " " + shape_params + " -S " + out_velocity_path_ + " " + out_msh_path_;

		int return_val;
		try
		{
			return_val = system(command.c_str());
		}
		catch (const std::exception &err)
		{
			logger().error("remesh command \"{}\" returns {}", command, return_val);

			return false;
		}

		logger().info("remesh command \"{}\" returns {}", command, return_val);

		if (periodic_tiling_)
		{
			command = "python ../tile.py " + out_msh_path_;
			try
			{
				return_val = system(command.c_str());
			}
			catch (const std::exception &err)
			{
				logger().error("tile command \"{}\" returns {}", command, return_val);

				return false;
			}

			logger().info("tile command \"{}\" returns {}", command, return_val);
		}

		return true;
	}
	void SDFShapeVariableToSimulation::compute_pattern_period()
	{
		const mesh::Mesh &mesh = *(get_state().mesh);
		int elem_period_ = 0;
		full_to_periodic_.clear();

		if (!periodic_tiling_)
		{
			full_to_periodic_.reserve(get_state().n_geom_bases);
			for (int i = 0; i < get_state().n_geom_bases; i++)
				full_to_periodic_.push_back(i);

			return;
		}

		RowVectorNd min, max;
		mesh.bounding_box(min, max);

		Eigen::VectorXi nums;
		nums.setZero(mesh.dimension());
		for (int d = 0; d < mesh.dimension(); d++)
		{
			int tmp = std::lround((max(d) - min(d)) / unit_size_);
			if (abs(tmp * unit_size_ + min(d) - max(d)) > 1e-8)
				log_and_throw_error("Mesh size is not periodic!");
			nums(d) = tmp;
		}

		for (int e = 0; e < mesh.n_elements(); e++)
		{
			if ((get_barycenter(mesh, e) - min).maxCoeff() >= unit_size_)
			{
				elem_period_ = e;
				break;
			}
		}
		if (elem_period_ == 0)
			elem_period_ = mesh.n_elements();

		// node correspondence
		{
			full_to_periodic_.assign(get_state().n_geom_bases, -1);

			utils::maybe_parallel_for(mesh.n_elements(), [&](int start, int end, int thread_id) {
				for (int e = start; e < end; e++)
				{
					RowVectorNd offset = get_barycenter(mesh, e) - get_barycenter(mesh, e % elem_period_);

					assert(!mesh.is_volume());
					for (int lv = 0; lv < mesh.n_face_vertices(e); lv++) // only 2D
					{
						int vid1 = mesh.face_vertex(e, lv);
						auto p1 = mesh.point(vid1);
						bool flag = false;

						if (e < elem_period_)
						{
							flag = true;
							full_to_periodic_[vid1] = vid1;
						}
						else
						{
							double min_diff = 1e5;
							int min_id = -1;
							for (int lv2 = 0; lv2 < mesh.n_face_vertices(e % elem_period_); lv2++)
							{
								int vid2 = mesh.face_vertex(e % elem_period_, lv2);
								auto p2 = mesh.point(vid2);

								if ((p1 - offset - p2).norm() < min_diff)
								{
									min_diff = (p1 - offset - p2).norm();
									min_id = vid2;
								}
							}
							if (min_diff > 1e-5)
								log_and_throw_error("Failed to find periodic node in periodic pattern, error = {}!", min_diff);
							full_to_periodic_[vid1] = min_id;
						}
					}
				}
			});
		}

		logger().info("Number of elements in one period: {}, number of periods: {}", elem_period_, nums.prod());
	}

	void ElasticVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		const int n_elem = state_ptr_->bases.size();
		assert(n_elem * 2 == state_variable.size());
		state_ptr_->assembler.update_lame_params(state_variable.segment(0, n_elem), state_variable.segment(n_elem, n_elem));
	}
	Eigen::VectorXd ElasticVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_material_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			AdjointTools::dJ_material_static_adjoint_term(get_state(), get_state().diff_cached[0].u, get_state().get_adjoint_mat(0), term);
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void FrictionCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 1);
		assert(state_variable(0) >= 0);
		state_ptr_->args["contact"]["friction_coefficient"] = state_variable(0);
	}
	Eigen::VectorXd FrictionCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_friction_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Friction coefficient grad in static simulations not implemented!");
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void DampingCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 2);
		json damping_param = {
			{"psi", state_variable(0)},
			{"phi", state_variable(1)},
		};
		state_ptr_->assembler.add_multimaterial(0, damping_param);
		logger().info("Current damping params: {}, {}", state_variable(0), state_variable(1));
	}
	Eigen::VectorXd DampingCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_damping_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Static damping not supported!");
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void InitialConditionVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == state_ptr_->ndof() * 2);
		state_ptr_->initial_sol_update = state_variable.head(state_ptr_->ndof());
		state_ptr_->initial_vel_update = state_variable.tail(state_ptr_->ndof());
	}
	Eigen::VectorXd InitialConditionVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_initial_condition_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Static initial condition not supported!");
		}
		return parametrization_.apply_jacobian(term, x);
	}

	void DirichletVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		log_and_throw_error("Dirichlet variable to simulation not implemented!");
		// auto &problem = *dynamic_cast<assembler::GenericTensorProblem *>(state_ptr_->problem.get());
		// // This should eventually update dirichlet boundaries per boundary element, using the shape constraint.
		// auto constraint_string = control_constraints_->constraint_to_string(state_variable);
		// for (const auto &kv : boundary_id_to_reduced_param)
		// {
		// 	json dirichlet_bc = constraint_string[kv.first];
		// 	// Need time_steps + 1 entry, though unused.
		// 	for (int k = 0; k < states_ptr_[0]->mesh->dimension(); ++k)
		// 		dirichlet_bc[k].push_back(dirichlet_bc[k][time_steps - 1]);
		// 	logger().trace("Updating boundary id {} to dirichlet bc {}", kv.first, dirichlet_bc);
		// 	problem.update_dirichlet_boundary(kv.first, dirichlet_bc, true, true, true, "");
		// }
	}
	Eigen::VectorXd DirichletVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			Eigen::MatrixXd adjoint_nu, adjoint_p;
			adjoint_nu = get_state().get_adjoint_mat(1);
			adjoint_p = get_state().get_adjoint_mat(0);
			AdjointTools::dJ_dirichlet_transient_adjoint_term(get_state(), adjoint_nu, adjoint_p, term);
		}
		else
		{
			log_and_throw_error("Static dirichlet boundary optimization not supported!");
		}
		return parametrization_.apply_jacobian(term, x);
	}
	std::string DirichletVariableToSimulation::variable_to_string(const Eigen::VectorXd &variable)
	{
		return "";
	}

	void MacroStrainVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == state_ptr_->mesh->dimension() * state_ptr_->mesh->dimension());
		state_ptr_->disp_grad = utils::unflatten(state_variable, state_ptr_->mesh->dimension());
	}
	Eigen::VectorXd MacroStrainVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term;
		if (get_state().problem->is_time_dependent())
		{
			log_and_throw_error("Transient macro strain optimization not supported!");
		}
		else
		{
			AdjointTools::dJ_macro_strain_adjoint_term(get_state(), get_state().diff_cached[0].u, get_state().get_adjoint_mat(0), term);
		}
		return parametrization_.apply_jacobian(term, x);
	}
} // namespace polyfem::solver