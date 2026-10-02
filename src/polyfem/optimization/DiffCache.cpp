#include <polyfem/optimization/DiffCache.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>

#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/time_integrator/ImplicitEuler.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>

#include <polyfem/assembler/ViscousDamping.hpp>

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Logger.hpp>

#include <ipc/ipc.hpp>
#include <Eigen/Core>

#include <memory>
#include <vector>
#include <map>
#include <array>
#include <cmath>

namespace polyfem
{
	namespace
	{
		void replace_rows_by_identity(StiffnessMatrix &reduced_mat, const StiffnessMatrix &mat, const std::vector<int> &rows)
		{
			reduced_mat.resize(mat.rows(), mat.cols());

			std::vector<bool> mask(mat.rows(), false);
			for (int i : rows)
				mask[i] = true;

			std::vector<Eigen::Triplet<double>> coeffs;
			for (int k = 0; k < mat.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mat, k); it; ++it)
				{
					if (mask[it.row()])
					{
						if (it.row() == it.col())
							coeffs.emplace_back(it.row(), it.col(), 1.0);
					}
					else
						coeffs.emplace_back(it.row(), it.col(), it.value());
				}
			}
			reduced_mat.setFromTriplets(coeffs.begin(), coeffs.end());
		}

		void compute_force_jacobian(varform::DifferentiableVarForm &varform, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, StiffnessMatrix &hessian)
		{
			const solver::SolveData *solve_data = varform.solve_data();
			assert(solve_data && "Optimization varforms must expose solve data");

			if (varform.get_problem().is_time_dependent())
			{
				StiffnessMatrix tmp_hess;
				assert(solve_data->nl_problem && "Transient differentiation requires an initialized nonlinear problem");
				solve_data->nl_problem->set_project_to_psd(false);
				solve_data->nl_problem->FullNLProblem::solution_changed(sol);
				solve_data->nl_problem->FullNLProblem::hessian(sol, tmp_hess);
				hessian.setZero();
				replace_rows_by_identity(hessian, tmp_hess, varform.boundary_state().boundary_nodes);
			}
			else // static formulation
			{
				if (varform.primary_assembler().is_linear() && !varform.is_contact_enabled() && !varform.is_homogenization())
				{
					hessian.setZero();
					StiffnessMatrix stiffness;
					varform.build_stiffness_matrix(stiffness);
					replace_rows_by_identity(hessian, stiffness, varform.boundary_state().boundary_nodes);
				}
				else
				{
					assert(solve_data->nl_problem && "Nonlinear differentiation requires an initialized nonlinear problem");
					solve_data->nl_problem->set_project_to_psd(false);
					if (varform.is_homogenization())
					{
						Eigen::VectorXd reduced;
						std::shared_ptr<solver::NLHomoProblem> homo_problem = std::dynamic_pointer_cast<solver::NLHomoProblem>(solve_data->nl_problem);
						assert(homo_problem && "Homogenization requires NLHomoProblem solve data");
						reduced = homo_problem->full_to_reduced(sol, disp_grad);
						solve_data->nl_problem->solution_changed(reduced);
						solve_data->nl_problem->hessian(reduced, hessian);
					}
					else
					{
						StiffnessMatrix tmp_hess;
						solve_data->nl_problem->FullNLProblem::solution_changed(sol);
						solve_data->nl_problem->FullNLProblem::hessian(sol, tmp_hess);
						hessian.setZero();
						replace_rows_by_identity(hessian, tmp_hess, varform.boundary_state().boundary_nodes);
					}
				}
			}
		}

		StiffnessMatrix compute_basis_nodes_to_gbasis_nodes(const varform::DifferentiableVarForm &varform)
		{
			auto &gbases = varform.primary_space().geometry_basis_list();
			auto &bases = varform.primary_space().basis_list();

			std::map<std::array<int, 2>, double> pairs;
			for (int e = 0; e < gbases.size(); e++)
			{
				auto &gbs = gbases[e].bases;
				auto &bs = bases[e].bases;
				assert(!bs.empty());

				Eigen::MatrixXd local_pts;
				int order = bs.front().order();
				if (varform.get_mesh().is_volume())
				{
					if (varform.get_mesh().is_simplex(e))
					{
						autogen::p_nodes_3d(order, local_pts);
					}
					else
					{
						autogen::q_nodes_3d(order, local_pts);
					}
				}
				else
				{
					if (varform.get_mesh().is_simplex(e))
					{
						autogen::p_nodes_2d(order, local_pts);
					}
					else
					{
						autogen::q_nodes_2d(order, local_pts);
					}
				}

				assembler::ElementAssemblyValues vals;
				vals.compute(e, varform.get_mesh().is_volume(), local_pts, gbases[e], gbases[e]);

				for (int i = 0; i < bs.size(); i++)
				{
					for (int j = 0; j < gbs.size(); j++)
					{
						if (std::abs(vals.basis_values[j].val(i)) > 1e-7)
						{
							std::array<int, 2> index = {{gbs[j].global()[0].index, bs[i].global()[0].index}};
							pairs.insert({index, vals.basis_values[j].val(i)});
						}
					}
				}
			}

			int dim = varform.get_mesh().dimension();
			std::vector<Eigen::Triplet<double>> coeffs;
			coeffs.reserve(pairs.size() * dim);
			for (const auto &iter : pairs)
			{
				for (int d = 0; d < dim; d++)
				{
					coeffs.emplace_back(iter.first[0] * dim + d, iter.first[1] * dim + d, iter.second);
				}
			}

			StiffnessMatrix mapping;
			mapping.resize(varform.primary_space().geometry->n_bases * dim, varform.primary_space().n_bases * dim);
			mapping.setFromTriplets(coeffs.begin(), coeffs.end());
			return mapping;
		}
	} // namespace

	void DiffCache::init(const int dimension, const int ndof, const int n_time_steps)
	{
		cur_size_ = 0;
		n_time_steps_ = n_time_steps;

		u_.setZero(ndof, n_time_steps + 1);
		disp_grad_.assign(n_time_steps + 1, Eigen::MatrixXd::Zero(dimension, dimension));
		if (n_time_steps_ > 0)
		{
			bdf_order_.setZero(n_time_steps + 1);
			v_.setZero(ndof, n_time_steps + 1);
			acc_.setZero(ndof, n_time_steps + 1);
			// gradu_h_prev_.resize(n_time_steps + 1);
		}
		gradu_h_.resize(n_time_steps + 1);
		collision_set_.resize(n_time_steps + 1);
		smooth_collision_set_.resize(n_time_steps + 1);
		friction_collision_set_.resize(n_time_steps + 1);
		normal_adhesion_collision_set_.resize(n_time_steps + 1);
		tangential_adhesion_collision_set_.resize(n_time_steps + 1);
	}

	void DiffCache::cache_quantities_static(
		const Eigen::MatrixXd &u,
		const StiffnessMatrix &gradu_h,
		const ipc::NormalCollisions &collision_set,
		const ipc::SmoothCollisions &smooth_collision_set,
		const ipc::TangentialCollisions &friction_constraint_set,
		const ipc::NormalCollisions &normal_adhesion_set,
		const ipc::TangentialCollisions &tangential_adhesion_set,
		const Eigen::MatrixXd &disp_grad)
	{
		u_ = u;

		gradu_h_[0] = gradu_h;
		collision_set_[0] = collision_set;
		smooth_collision_set_[0] = smooth_collision_set;
		friction_collision_set_[0] = friction_constraint_set;
		normal_adhesion_collision_set_[0] = normal_adhesion_set;
		tangential_adhesion_collision_set_[0] = tangential_adhesion_set;
		disp_grad_[0] = disp_grad;

		cur_size_ = 1;
	}

	void DiffCache::cache_quantities_transient(
		const int cur_step,
		const int cur_bdf_order,
		const Eigen::MatrixXd &u,
		const Eigen::MatrixXd &v,
		const Eigen::MatrixXd &acc,
		const StiffnessMatrix &gradu_h,
		// const StiffnessMatrix &gradu_h_prev,
		const ipc::NormalCollisions &collision_set,
		const ipc::SmoothCollisions &smooth_collision_set,
		const ipc::TangentialCollisions &friction_collision_set)
	{
		bdf_order_(cur_step) = cur_bdf_order;

		u_.col(cur_step) = u;
		v_.col(cur_step) = v;
		acc_.col(cur_step) = acc;

		gradu_h_[cur_step] = gradu_h;
		// gradu_h_prev_[cur_step] = gradu_h_prev;

		collision_set_[cur_step] = collision_set;
		smooth_collision_set_[cur_step] = smooth_collision_set;
		friction_collision_set_[cur_step] = friction_collision_set;

		cur_size_++;
	}

	void DiffCache::cache_quantities_quasistatic(
		const int cur_step,
		const Eigen::MatrixXd &u,
		const StiffnessMatrix &gradu_h,
		const ipc::NormalCollisions &collision_set,
		const ipc::SmoothCollisions &smooth_collision_set,
		const ipc::NormalCollisions &normal_adhesion_set,
		const Eigen::MatrixXd &disp_grad)
	{
		u_.col(cur_step) = u;
		gradu_h_[cur_step] = gradu_h;
		collision_set_[cur_step] = collision_set;
		smooth_collision_set_[cur_step] = smooth_collision_set;
		normal_adhesion_collision_set_[cur_step] = normal_adhesion_set;
		disp_grad_[cur_step] = disp_grad;

		cur_size_++;
	}

	void DiffCache::cache_adjoints(const Eigen::MatrixXd &adjoint_mat) { adjoint_mat_ = adjoint_mat; }

	const StiffnessMatrix &DiffCache::basis_nodes_to_gbasis_nodes() const
	{
		assert(basis_nodes_to_gbasis_nodes_.size() != 0
			   && "basis_nodes_to_gbasis_nodes is empty. Expect cache_transient(step==0) to build it first.");

		return basis_nodes_to_gbasis_nodes_;
	}

	void DiffCache::cache_transient(
		int step,
		varform::DifferentiableVarForm &varform,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd *pressure)
	{
		if (pressure)
		{
			log_and_throw_adjoint_error("Navier stoke problem is not supported in adjoint optimization.");
		}

		if (step == 0)
		{
			basis_nodes_to_gbasis_nodes_ = compute_basis_nodes_to_gbasis_nodes(varform);
		}

		const Eigen::MatrixXd disp_grad = varform.displacement_gradient();

		StiffnessMatrix gradu_h(sol.size(), sol.size());
		if (step == 0)
		{
			init(varform.get_mesh().dimension(), varform.primary_space().ndof(), varform.get_problem().is_time_dependent() ? varform.get_args()["time"]["time_steps"].get<int>() : 0);
		}

		ipc::NormalCollisions cur_collision_set;
		ipc::SmoothCollisions cur_smooth_collision_set;
		ipc::TangentialCollisions cur_friction_set;
		ipc::NormalCollisions cur_normal_adhesion_set;
		ipc::TangentialCollisions cur_tangential_adhesion_set;

		if (!varform.get_problem().is_time_dependent() || step > 0)
			compute_force_jacobian(varform, sol, disp_grad, gradu_h);

		const auto *solve_data = varform.solve_data();
		assert(solve_data && "Optimization varforms must expose solve data");
		if (solve_data->contact_form)
		{
			if (const auto barrier_contact = dynamic_cast<const solver::BarrierContactForm *>(solve_data->contact_form.get()))
				cur_collision_set = barrier_contact->collision_set();
			else if (const auto smooth_contact = dynamic_cast<const solver::SmoothContactForm *>(solve_data->contact_form.get()))
				cur_smooth_collision_set = smooth_contact->collision_set();
		}
		if (solve_data->friction_form)
			cur_friction_set = solve_data->friction_form->friction_collision_set();
		if (solve_data->normal_adhesion_form)
			cur_normal_adhesion_set = solve_data->normal_adhesion_form->collision_set();
		if (solve_data->tangential_adhesion_form)
			cur_tangential_adhesion_set = solve_data->tangential_adhesion_form->tangential_collision_set();

		if (varform.get_problem().is_time_dependent())
		{
			if (varform.get_args()["time"]["quasistatic"].get<bool>())
			{
				cache_quantities_quasistatic(step, sol, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_normal_adhesion_set, disp_grad);
			}
			else
			{
				Eigen::MatrixXd vel, acc;
				if (step == 0)
				{
					if (dynamic_cast<time_integrator::BDF *>(solve_data->time_integrator.get()))
					{
						const auto bdf_integrator = dynamic_cast<time_integrator::BDF *>(solve_data->time_integrator.get());
						vel = bdf_integrator->weighted_sum_v_prevs();
					}
					else if (dynamic_cast<time_integrator::ImplicitEuler *>(solve_data->time_integrator.get()))
					{
						const auto euler_integrator = dynamic_cast<time_integrator::ImplicitEuler *>(solve_data->time_integrator.get());
						vel = euler_integrator->v_prev();
					}
					else
						log_and_throw_error("Differentiable code doesn't support this time integrator!");

					acc.setZero(varform.primary_space().ndof(), 1);
				}
				else
				{
					vel = solve_data->time_integrator->compute_velocity(sol);
					acc = solve_data->time_integrator->compute_acceleration(vel);
				}

				cache_quantities_transient(step, solve_data->time_integrator->steps(), sol, vel, acc, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_friction_set);
			}
		}
		else
		{
			cache_quantities_static(sol, gradu_h, cur_collision_set, cur_smooth_collision_set, cur_friction_set, cur_normal_adhesion_set, cur_tangential_adhesion_set, disp_grad);
		}
	}

} // namespace polyfem
