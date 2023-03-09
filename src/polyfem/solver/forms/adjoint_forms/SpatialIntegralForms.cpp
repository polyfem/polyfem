#include "SpatialIntegralForms.hpp"
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

using namespace polyfem::utils;

namespace polyfem::solver
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}

		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

		class LocalThreadScalarStorage
		{
		public:
			double val;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};
	} // namespace

	double SpatialIntegralForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(time_step_ < state_.diff_cached.size());
		return AdjointTools::integrate_objective(state_, get_integral_functional(), state_.diff_cached.u(time_step_), ids_, spatial_integral_type_, time_step_);
	}

	void SpatialIntegralForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(time_step_ < state_.diff_cached.size());
		gradv.setZero(x.size());
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &parametrization = param_map->get_parametrization();
			const auto &state = param_map->get_state();
			const auto &param_type = param_map->get_parameter_type();

			if (&state != &state_)
				continue;

			Eigen::VectorXd term;
			if (param_type == ParameterType::Shape)
				AdjointTools::compute_shape_derivative_functional_term(state_, state_.diff_cached.u(time_step_), get_integral_functional(), ids_, spatial_integral_type_, term, time_step_);
			else if (param_type == ParameterType::MacroStrain)
				AdjointTools::compute_macro_strain_derivative_functional_term(state_, state_.diff_cached.u(time_step_), get_integral_functional(), ids_, spatial_integral_type_, term, time_step_);

			if (term.size() > 0)
				gradv += parametrization.apply_jacobian(term, x);
		}
	}

	Eigen::VectorXd SpatialIntegralForm::compute_adjoint_rhs_unweighted_step(const Eigen::VectorXd &x, const State &state)
	{
		if (&state != &state_)
			return Eigen::VectorXd::Zero(state.ndof());

		assert(time_step_ < state_.diff_cached.size());

		Eigen::VectorXd rhs;
		AdjointTools::dJ_du_step(state, get_integral_functional(), state.diff_cached.u(time_step_), ids_, spatial_integral_type_, time_step_, rhs);

		return rhs;
	}

	// TODO: call local assemblers instead
	IntegrableFunctional StressNormForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		const std::string formulation = state_.formulation();
		const int power = in_power_;

		j.set_j([formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (formulation == "Laplacian")
				{
					stress = grad_u.row(q);
				}
				else if (formulation == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (formulation == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val(q) = pow(stress.squaredNorm(), power / 2.);
			}
		});

		j.set_dj_dgradu([formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			const int actual_dim = (formulation == "Laplacian") ? 1 : dim;
			Eigen::MatrixXd grad_u_q, stress, stress_dstress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (formulation == "Laplacian")
				{
					stress = grad_u.row(q);
					stress_dstress = 2 * stress;
				}
				else if (formulation == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
				}
				else if (formulation == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
					stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");

				const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
				for (int i = 0; i < actual_dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = coef * stress_dstress(i, l);
			}
		});

		return j;
	}

	void StressNormForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_unweighted(x, gradv);
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &parametrization = param_map->get_parametrization();
			const auto &state = param_map->get_state();
			const auto &param_type = param_map->get_parameter_type();

			if (&state != &state_)
				continue;

			Eigen::VectorXd term;
			if (param_type == ParameterType::Material)
				log_and_throw_error("Doesn't support stress derivative wrt. material!");

			if (term.size() > 0)
				gradv += parametrization.apply_jacobian(term, x);
		}
	}

	IntegrableFunctional ComplianceForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		const std::string formulation = state_.formulation();

		j.set_j([formulation](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				if (formulation == "LinearElasticity")
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				else
					logger().error("Unknown formulation!");
				val(q) = (stress.array() * grad_u_q.array()).sum();
			}
		});

		j.set_dj_dgradu([formulation](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());

				for (int i = 0; i < dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = 2 * stress(i, l);
			}
		});

		return j;
	}

	void ComplianceForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_unweighted(x, gradv);
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &parametrization = param_map->get_parametrization();
			const auto &state = param_map->get_state();
			const auto &param_type = param_map->get_parameter_type();

			if (&state != &state_)
				continue;

			Eigen::VectorXd term;
			if (param_type == ParameterType::Material)
			{
				term.setZero(parametrization.size(x.size()));

				const auto &bases = state.bases;
				const auto &gbases = state.geom_bases();
				auto df_dmu_dlambda_function = state.assembler.get_dstress_dmu_dlambda_function(state.formulation());
				const int dim = state.mesh->dimension();

				for (int e = 0; e < bases.size(); e++)
				{
					assembler::ElementAssemblyValues vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, state.diff_cached.u(time_step_), u, grad_u);

					Eigen::MatrixXd grad_u_q;
					for (int q = 0; q < quadrature.weights.size(); q++)
					{
						double lambda, mu;
						state.assembler.lame_params().lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

						vector2matrix(grad_u.row(q), grad_u_q);

						Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
						df_dmu_dlambda_function(e, quadrature.points.row(q), vals.val.row(q), grad_u_q, f_prime_dmu, f_prime_dlambda);

						term(e + bases.size()) += dot(f_prime_dmu, grad_u_q) * da(q);
						term(e) += dot(f_prime_dlambda, grad_u_q) * da(q);
					}
				}
			}

			if (term.size() > 0)
				gradv += parametrization.apply_jacobian(term, x);
		}
	}

	IntegrableFunctional PositionForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		const int dim = dim_;

		j.set_j([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = u.col(dim) + pts.col(dim);
		});

		j.set_dj_du([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			val.col(dim).setOnes();
		});

		j.set_dj_dx([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			val.col(dim).setOnes();
		});

		return j;
	}

	IntegrableFunctional AccelerationForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		const int dim = this->dim_;
		const int time_step = this->time_step_;

		j.set_j([dim, time_step, &state = std::as_const(this->state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			const int e = params["elem"];
			Eigen::MatrixXd acc, grad_acc;
			io::Evaluator::interpolate_at_local_vals(*(state.mesh), state.problem->is_scalar(), state.bases, state.geom_bases(), e, local_pts, state.diff_cached.acc(time_step), acc, grad_acc);

			val = acc.col(dim);
		});

		j.set_dj_du([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			log_and_throw_error("Not implemented!");
		});

		return j;
	}

	IntegrableFunctional KineticForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			
			const int e = params["elem"];
			Eigen::MatrixXd v, grad_v;
			io::Evaluator::interpolate_at_local_vals(*(state_.mesh), state_.problem->is_scalar(), state_.bases, state_.geom_bases(), e, local_pts, state_.diff_cached.v(time_step_), v, grad_v);

			val.setZero(u.rows(), 1);
			for (int q = 0; q < v.rows(); q++)
			{
				const double rho = state_.assembler.density()(local_pts.row(q), pts.row(q), e);
				val(q) = 0.5 * rho * v.row(q).squaredNorm();
			}
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			log_and_throw_error("Not implemented!");
		});

		return j;
	}

	IntegrableFunctional TargetForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		if (target_state_)
		{
			assert(target_state_->diff_cached.size() > 0);

			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);
				const int e = params["elem"];
				int e_ref;
				if (auto search = e_to_ref_e_.find(e); search != e_to_ref_e_.end())
					e_ref = search->second;
				else
					e_ref = e;
				const auto &gbase_ref = target_state_->geom_bases()[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = target_state_->problem->is_time_dependent() ? target_state_->diff_cached.u(params["step"].get<int>()) : target_state_->diff_cached.u(0);
				io::Evaluator::interpolate_at_local_vals(*(target_state_->mesh), target_state_->problem->is_scalar(), target_state_->bases, target_state_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					val(q) = ((u_ref.row(q) + pts_ref.row(q)) - (u.row(q) + pts.row(q))).squaredNorm();
				}
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				const int e = params["elem"];
				int e_ref;
				if (auto search = e_to_ref_e_.find(e); search != e_to_ref_e_.end())
					e_ref = search->second;
				else
					e_ref = e;
				const auto &gbase_ref = target_state_->geom_bases()[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = target_state_->problem->is_time_dependent() ? target_state_->diff_cached.u(params["step"].get<int>()) : target_state_->diff_cached.u(0);
				io::Evaluator::interpolate_at_local_vals(*(target_state_->mesh), target_state_->problem->is_scalar(), target_state_->bases, target_state_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					auto x = (u.row(q) + pts.row(q)) - (u_ref.row(q) + pts_ref.row(q));
					val.row(q) = 2 * x;
				}
			};

			j.set_j(j_func);
			j.set_dj_du(djdu_func);
			j.set_dj_dx(djdu_func); // only used for shape derivative
		}
		else if (have_target_func)
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);
				for (int q = 0; q < u.rows(); q++)
				{
					Eigen::VectorXd x = u.row(q) + pts.row(q);
					val(q) = target_func(x(0), x(1), x.size() == 2 ? 0 : x(2), 0, params["elem"]);
				}
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				for (int q = 0; q < u.rows(); q++)
				{
					Eigen::VectorXd x = u.row(q) + pts.row(q);
					for (int d = 0; d < val.cols(); d++)
						val(q, d) = target_func_grad[d](x(0), x(1), x.size() == 2 ? 0 : x(2), 0, params["elem"]);
				}
			};

			j.set_j(j_func);
			j.set_dj_du(djdu_func);
			j.set_dj_dx(djdu_func); // only used for shape derivative
		}
		else // error wrt. a constant displacement
		{
			if (target_disp.size() == state_.mesh->dimension())
			{
				auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
					val.setZero(u.rows(), 1);

					for (int q = 0; q < u.rows(); q++)
					{
						Eigen::VectorXd err = u.row(q) - this->target_disp.transpose();
						for (int d = 0; d < active_dimension_mask.size(); d++)
							if (!active_dimension_mask[d])
								err(d) = 0;
						val(q) = err.squaredNorm();
					}
				};
				auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
					val.setZero(u.rows(), u.cols());

					for (int q = 0; q < u.rows(); q++)
					{
						Eigen::VectorXd err = u.row(q) - this->target_disp.transpose();
						for (int d = 0; d < active_dimension_mask.size(); d++)
							if (!active_dimension_mask[d])
								err(d) = 0;
						val.row(q) = 2 * err;
					}
				};

				j.set_j(j_func);
				j.set_dj_du(djdu_func);
			}
			else
				log_and_throw_error("Only constant target displacement is supported!");
		}

		return j;
	}

	void TargetForm::set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids)
	{
		target_state_ = target_state;

		std::map<int, std::vector<int>> ref_interested_body_id_to_e;
		int ref_count = 0;
		for (int e = 0; e < target_state_->bases.size(); ++e)
		{
			int body_id = target_state_->mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (ref_interested_body_id_to_e.find(body_id) != ref_interested_body_id_to_e.end())
				ref_interested_body_id_to_e[body_id].push_back(e);
			else
				ref_interested_body_id_to_e[body_id] = {e};
			ref_count++;
		}

		std::map<int, std::vector<int>> interested_body_id_to_e;
		int count = 0;
		for (int e = 0; e < state_.bases.size(); ++e)
		{
			int body_id = state_.mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (interested_body_id_to_e.find(body_id) != interested_body_id_to_e.end())
				interested_body_id_to_e[body_id].push_back(e);
			else
				interested_body_id_to_e[body_id] = {e};
			count++;
		}

		if (count != ref_count)
			logger().error("Number of interested elements in the reference and optimization examples do not match! {} {}", count, ref_count);
		else
			logger().trace("Found {} matching elements.", count);

		for (const auto &kv : interested_body_id_to_e)
		{
			for (int i = 0; i < kv.second.size(); ++i)
			{
				e_to_ref_e_[kv.second[i]] = ref_interested_body_id_to_e[kv.first][i];
			}
		}
	}

	void TargetForm::set_reference(const json &func, const json &grad_func)
	{
		target_func.init(func);
		for (size_t k = 0; k < grad_func.size(); k++)
			target_func_grad[k].init(grad_func[k]);
		have_target_func = true;
	}

	void StressForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_unweighted(x, gradv);
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &parametrization = param_map->get_parametrization();
			const auto &state = param_map->get_state();
			const auto &param_type = param_map->get_parameter_type();

			if (&state != &state_)
				continue;

			Eigen::VectorXd term;
			if (param_type == ParameterType::Material)
				log_and_throw_error("Doesn't support stress derivative wrt. material!");

			if (term.size() > 0)
				gradv += parametrization.apply_jacobian(term, x);
		}
	}

	IntegrableFunctional StressForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		std::string formulation = state_.formulation();
		auto dimensions = dimensions_;

		j.set_j([formulation, dimensions](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (formulation == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val(q) = stress(dimensions[0], dimensions[1]);
			}
		});

		j.set_dj_dgradu([formulation, dimensions](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			Eigen::MatrixXd grad_u_q, stiffness, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				stiffness.setZero(1, dim * dim * dim * dim);
				vector2matrix(grad_u.row(q), grad_u_q);

				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					for (int i = 0, idx = 0; i < dim; i++)
						for (int j = 0; j < dim; j++)
							for (int k = 0; k < dim; k++)
								for (int l = 0; l < dim; l++)
								{
									stiffness(idx++) = mu(q) * delta(i, k) * delta(j, l) + mu(q) * delta(i, l) * delta(j, k) + lambda(q) * delta(i, j) * delta(k, l);
								}
				}
				else if (formulation == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
					Eigen::VectorXd FmT_vec = utils::flatten(FmT);
					double J = def_grad.determinant();
					double tmp1 = mu(q) - lambda(q) * std::log(J);
					for (int i = 0, idx = 0; i < dim; i++)
						for (int j = 0; j < dim; j++)
							for (int k = 0; k < dim; k++)
								for (int l = 0; l < dim; l++)
								{
									stiffness(idx++) = mu(q) * delta(i, k) * delta(j, l) + tmp1 * FmT(i, l) * FmT(k, j);
								}
					stiffness += lambda(q) * utils::flatten(FmT_vec * FmT_vec.transpose()).transpose();
				}
				else
					logger().error("Unknown formulation!");

				val.row(q) = stiffness.block(0, (dimensions[0] * dim + dimensions[1]) * dim * dim, 1, dim * dim);
			}
		});

		return j;
	}

	IntegrableFunctional VolumeForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		j.set_j([](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setOnes(grad_u.rows(), 1);
		});

		return j;
	}

	void SDFTargetForm::solution_changed(const Eigen::VectorXd &x)
	{
		assert(time_step_ < state_.diff_cached.size());

		const auto &bases = state_.bases;
		const auto &gbases = state_.geom_bases();
		const int actual_dim = state_.problem->is_scalar() ? 1 : dim;

		auto storage = utils::create_thread_storage(LocalThreadScalarStorage());
		utils::maybe_parallel_for(state_.total_local_boundary.size(), [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			Eigen::MatrixXd uv, samples, gtmp;
			Eigen::MatrixXd points, normal;
			Eigen::VectorXd weights;

			Eigen::MatrixXd u, grad_u;

			for (int lb_id = start; lb_id < end; ++lb_id)
			{
				const auto &lb = state_.total_local_boundary[lb_id];
				const int e = lb.element_id();

				for (int i = 0; i < lb.size(); i++)
				{
					const int global_primitive_id = lb.global_primitive_id(i);
					if (ids_.size() != 0 && ids_.find(state_.mesh->get_boundary_id(global_primitive_id)) == ids_.end())
						continue;

					utils::BoundarySampler::boundary_quadrature(lb, state_.n_boundary_samples(), *state_.mesh, i, false, uv, points, normal, weights);

					assembler::ElementAssemblyValues &vals = local_storage.vals;
					vals.compute(e, state_.mesh->is_volume(), points, bases[e], gbases[e]);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, state_.diff_cached.u(time_step_), u, grad_u);

					normal = normal * vals.jac_it[0]; // assuming linear geometry

					for (int q = 0; q < u.rows(); q++)
						interpolation_fn->cache_grid([this](const Eigen::MatrixXd &point, double &distance) { compute_distance(point, distance); }, vals.val.row(q) + u.row(q));
				}
			}
		});
	}

	void SDFTargetForm::set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots, const double delta)
	{
		dim = control_points.cols();
		delta_ = delta;
		assert(dim == 2);

		samples = 100;

		nanospline::BSpline<double, 2, 3> curve;
		curve.set_control_points(control_points);
		curve.set_knots(knots);

		t_or_uv_sampling = Eigen::VectorXd::LinSpaced(samples, 0, 1);
		point_sampling.setZero(samples, 2);
		for (int i = 0; i < t_or_uv_sampling.size(); ++i)
			point_sampling.row(i) = curve.evaluate(t_or_uv_sampling(i));

		Eigen::MatrixXi edges(samples - 1, 2);
		edges.col(0) = Eigen::VectorXi::LinSpaced(samples - 1, 0, samples - 2);
		edges.col(1) = Eigen::VectorXi::LinSpaced(samples - 1, 1, samples - 1);
		io::OBJWriter::write(state_.resolve_output_path(fmt::format("spline_target_{:d}.obj", rand() % 100)), point_sampling, edges);

		interpolation_fn = std::make_unique<LazyCubicInterpolator>(dim, delta_);
	}

	void SDFTargetForm::set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots_u, const Eigen::VectorXd &knots_v, const double delta)
	{

		dim = control_points.cols();
		delta_ = delta;
		assert(dim == 3);

		samples = 100;

		nanospline::BSplinePatch<double, 3, 3, 3> patch;
		patch.set_control_grid(control_points);
		patch.set_knots_u(knots_u);
		patch.set_knots_v(knots_v);
		patch.initialize();

		t_or_uv_sampling.resize(samples * samples, 2);
		for (int i = 0; i < samples; ++i)
		{
			t_or_uv_sampling.block(i * samples, 0, samples, 1) = Eigen::VectorXd::LinSpaced(samples, 0, 1);
			t_or_uv_sampling.block(i * samples, 1, samples, 1) = (double)i / (samples - 1) * Eigen::VectorXd::Ones(samples);
		}
		point_sampling.setZero(samples * samples, 3);
		for (int i = 0; i < t_or_uv_sampling.rows(); ++i)
		{
			point_sampling.row(i) = patch.evaluate(t_or_uv_sampling(i, 0), t_or_uv_sampling(i, 1));
		}

		interpolation_fn = std::make_unique<LazyCubicInterpolator>(dim, delta_);
	}

	void SDFTargetForm::compute_distance(const Eigen::MatrixXd &point, double &distance) const
	{
		distance = DBL_MAX;
		Eigen::MatrixXd p = point.transpose();

		if (dim == 2)
			for (int i = 0; i < t_or_uv_sampling.size() - 1; ++i)
			{
				const double l = (point_sampling.row(i + 1) - point_sampling.row(i)).squaredNorm();
				double distance_to_perpendicular = ((p - point_sampling.row(i)) * (point_sampling.row(i + 1) - point_sampling.row(i)).transpose())(0) / l;
				const double t = std::max(0., std::min(1., distance_to_perpendicular));
				const auto project = point_sampling.row(i) * (1 - t) + point_sampling.row(i + 1) * t;
				const double project_distance = (p - project).norm();
				if (project_distance < distance)
					distance = project_distance;
			}
		else if (dim == 3)
		{
			for (int i = 0; i < samples - 1; ++i)
				for (int j = 0; j < samples - 1; ++j)
				{
					int loc = samples * i + j;
					const double l1 = (point_sampling.row(loc + 1) - point_sampling.row(loc)).squaredNorm();
					double distance_to_perpendicular = ((p - point_sampling.row(loc)) * (point_sampling.row(loc + 1) - point_sampling.row(loc)).transpose())(0) / l1;
					const double u = std::max(0., std::min(1., distance_to_perpendicular));

					const double l2 = (point_sampling.row(loc + samples) - point_sampling.row(loc)).squaredNorm();
					distance_to_perpendicular = ((p - point_sampling.row(loc)) * (point_sampling.row(loc + samples) - point_sampling.row(loc)).transpose())(0) / l2;
					const double v = std::max(0., std::min(1., distance_to_perpendicular));

					Eigen::MatrixXd project = point_sampling.row(loc) * (1 - u) + point_sampling.row(loc + 1) * u;
					project += v * (point_sampling.row(loc + samples) - point_sampling.row(loc));
					const double project_distance = (p - project).norm();
					if (project_distance < distance)
						distance = project_distance;
				}
		}
	}

	IntegrableFunctional SDFTargetForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);

			for (int q = 0; q < u.rows(); q++)
			{
				double distance;
				Eigen::MatrixXd unused_grad;
				interpolation_fn->evaluate(u.row(q) + pts.row(q), distance, unused_grad);
				val(q) = pow(distance, 2);
			}
		};

		auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());

			for (int q = 0; q < u.rows(); q++)
			{
				double distance;
				Eigen::MatrixXd grad;
				interpolation_fn->evaluate(u.row(q) + pts.row(q), distance, grad);
				val.row(q) = 2 * distance * grad.transpose();
			}
		};

		j.set_j(j_func);
		j.set_dj_du(djdu_func);
		j.set_dj_dx(djdu_func);

		return j;
	}
} // namespace polyfem::solver