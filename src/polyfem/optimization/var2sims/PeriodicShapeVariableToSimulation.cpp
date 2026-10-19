#include <polyfem/optimization/var2sims/PeriodicShapeVariableToSimulation.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/VarFormDiff.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <cassert>
#include <string>
#include <utility>

namespace polyfem::solver
{
	PeriodicShapeVariableToSimulation::PeriodicShapeVariableToSimulation(
		VarFormPtrs varforms,
		DiffCachePtrs diff_caches,
		CompositeParametrization parametrizations)
		: dim_(varforms[0]->get_mesh().dimension()),
		  vertex_num_(varforms[0]->get_mesh().n_vertices()),
		  varforms_(std::move(varforms)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations))
	{
		assert(!varforms_.empty());
		assert(varforms_.size() == diff_caches_.size());

		for (const auto &varform : varforms_)
		{
			if (varform->get_mesh().dimension() != dim_)
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: mesh dimension mismatch between varforms.");
			}
			if (varform->get_mesh().n_vertices() != vertex_num_)
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: mesh vertex num mismatch between varforms.");
			}
			if (varform->get_problem().is_time_dependent())
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: transient simulations are not supported.");
			}
			if (!varform->has_periodic_boundary())
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: periodic boundary conditions are not enabled.");
			}
			const Eigen::MatrixXd tile_offsets = varform->periodic_tile_offsets();
			if (tile_offsets.rows() != dim_ || tile_offsets.cols() != dim_
				|| Eigen::FullPivLU<Eigen::MatrixXd>(tile_offsets).rank() != dim_)
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: partial periodicity is not supported.");
			}
			if (!varform->is_homogenization())
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: only homogenization problems are supported.");
			}
		}

		Eigen::MatrixXd V;
		varforms_[0]->get_vertices(V);
		periodic_mesh_map_ = std::make_unique<PeriodicMeshToMesh>(V);
	}

	std::string PeriodicShapeVariableToSimulation::name() const
	{
		return "periodic-shape";
	}

	ParameterType PeriodicShapeVariableToSimulation::parameter_type() const
	{
		return ParameterType::PeriodicShape;
	}

	bool PeriodicShapeVariableToSimulation::affects_varform(const varform::DifferentiableVarForm &target) const
	{
		for (auto &varform : varforms_)
		{
			if (varform.get() == &target)
				return true;
		}
		return false;
	}

	void PeriodicShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		Eigen::MatrixXd V = utils::unflatten(periodic_mesh_map_->eval(y), dim_);

		for (auto &varform : varforms_)
			varform->set_vertex_positions(V);
	}

	void PeriodicShapeVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd PeriodicShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < varforms_.size(); ++i)
		{
			auto &varform = varforms_[i];
			auto &diff_cache = diff_caches_[i];

			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*varform, *diff_cache, 0);

			AdjointTools::dJ_periodic_shape_adjoint_term(
				*varform,
				*diff_cache,
				*periodic_mesh_map_,
				y,
				diff_cache->u(0),
				adjoint_p,
				cur_term);

			if (term.size() != cur_term.size())
			{
				term = cur_term;
			}
			else
			{
				term += cur_term;
			}
		}

		assert(term.size() == para_out_dof());
		return parametrization_.apply_jacobian(term, x);
	}

	int PeriodicShapeVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd PeriodicShapeVariableToSimulation::inverse_eval() const
	{
		Eigen::MatrixXd V;
		varforms_[0]->get_vertices(V);

		Eigen::VectorXd y = periodic_mesh_map_->inverse_eval(utils::flatten(V));
		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd PeriodicShapeVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
	{
		assert(term.size() == vertex_num_ * dim_);

		const Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		const Eigen::VectorXd reduced_term = periodic_mesh_map_->apply_jacobian(term, y);
		assert(reduced_term.size() == para_out_dof());

		return parametrization_.apply_jacobian(reduced_term, x);
	}

	int PeriodicShapeVariableToSimulation::para_out_dof() const
	{
		return periodic_mesh_map_->input_size();
	}

} // namespace polyfem::solver
