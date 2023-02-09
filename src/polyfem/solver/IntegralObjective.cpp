#include "IntegralObjective.hpp"

#include <polyfem/utils/CubicHermiteSplineParametrization.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include "HomoObjective.hpp"
#include "ControlParameter.hpp"

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

		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

		template <typename T>
		T inverse_Lp_norm(const Eigen::Matrix<T, Eigen::Dynamic, 1> &F, const double p)
		{
			T val = T(0);
			for (int i = 0; i < F.size(); i++)
			{
				val += pow(F(i), p);
			}
			return T(1) / pow(val, 1. / p);
		}

		Eigen::VectorXd inverse_Lp_norm_grad(const Eigen::VectorXd &F, const double p)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, 1> full_diff(F.size());
			for (int i = 0; i < F.size(); i++)
				full_diff(i) = Diff(i, F(i));
			auto reduced_diff = inverse_Lp_norm(full_diff, p);

			Eigen::VectorXd grad(F.size());
			for (int i = 0; i < F.size(); ++i)
				grad(i) = reduced_diff.getGradient()(i);

			return grad;
		}
    }

	SpatialIntegralObjective::SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const json &args) : state_(state), shape_param_(shape_param), macro_strain_param_(macro_strain_param)
	{
		if (shape_param_)
			assert(shape_param_->name() == "shape");

		if (macro_strain_param_)
			assert(macro_strain_param_->name() == "macro_strain");
	}

	SpatialIntegralObjective::SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : state_(state), shape_param_(shape_param)
	{
		if (shape_param_)
			assert(shape_param_->name() == "shape");

		macro_strain_param_ = NULL;
	}

	double SpatialIntegralObjective::value()
	{
		assert(time_step_ < state_.diff_cached.size());
		return AdjointTools::integrate_objective(state_, get_integral_functional(), state_.diff_cached[time_step_].u, interested_ids_, spatial_integral_type_, time_step_);
	}

	Eigen::VectorXd SpatialIntegralObjective::compute_adjoint_rhs_step(const State &state)
	{
		if (&state != &state_)
			return Eigen::VectorXd::Zero(state.ndof());

		assert(time_step_ < state_.diff_cached.size());

		Eigen::VectorXd rhs;
		AdjointTools::dJ_du_step(state, get_integral_functional(), state.diff_cached[time_step_].u, interested_ids_, spatial_integral_type_, time_step_, rhs);

		return rhs;
	}

	Eigen::VectorXd SpatialIntegralObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		if (&param == shape_param_.get())
		{
			assert(time_step_ < state_.diff_cached.size());
			AdjointTools::compute_shape_derivative_functional_term(state_, state_.diff_cached[time_step_].u, get_integral_functional(), interested_ids_, spatial_integral_type_, term, time_step_);
		}
		else if (&param == macro_strain_param_.get())
		{
			AdjointTools::compute_macro_strain_derivative_functional_term(state_, state_.diff_cached[time_step_].u, get_integral_functional(), interested_ids_, spatial_integral_type_, term, time_step_);
		}
		else
			term.setZero(param.full_dim());

		return param.map_grad(param_value, term);
	}

	StressObjective::StressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args, bool has_integral_sqrt) : SpatialIntegralObjective(state, shape_param, args), elastic_param_(elastic_param)
	{
		spatial_integral_type_ = SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		formulation_ = state.formulation();
		in_power_ = args["power"];
		out_sqrt_ = has_integral_sqrt;
	}

	IntegrableFunctional StressObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (this->formulation_ == "Laplacian")
				{
					stress = grad_u.row(q);
				}
				else if (this->formulation_ == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (this->formulation_ == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val(q) = pow(stress.squaredNorm(), this->in_power_ / 2.);
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			const int actual_dim = (this->formulation_ == "Laplacian") ? 1 : dim;
			Eigen::MatrixXd grad_u_q, stress, stress_dstress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (this->formulation_ == "Laplacian")
				{
					stress = grad_u.row(q);
					stress_dstress = 2 * stress;
				}
				else if (this->formulation_ == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
				}
				else if (this->formulation_ == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
					stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
				}
				else
					logger().error("Unknown formulation!");

				const double coef = this->in_power_ * pow(stress.squaredNorm(), this->in_power_ / 2. - 1.);
				for (int i = 0; i < actual_dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = coef * stress_dstress(i, l);
			}
		});

		return j;
	}

	double StressObjective::value()
	{
		double val = SpatialIntegralObjective::value();
		if (out_sqrt_)
			return pow(val, 1. / in_power_);
		else
			return val;
	}

	Eigen::VectorXd StressObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd rhs = SpatialIntegralObjective::compute_adjoint_rhs_step(state);

		if (out_sqrt_)
		{
			double val = SpatialIntegralObjective::value();
			if (std::abs(val) < 1e-12)
				logger().warn("stress integral too small, may result in NAN grad!");
			return (pow(val, 1. / in_power_ - 1) / in_power_) * rhs;
		}
		else
			return rhs;
	}

	Eigen::VectorXd StressObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		if (&param == elastic_param_.get())
		{
			// TODO: differentiate stress wrt. lame param
			log_and_throw_error("Not implemented!");
		}
		else if (&param == shape_param_.get())
		{
			term = SpatialIntegralObjective::compute_partial_gradient(param, param_value);
		}
		else
			term.setZero(param.optimization_dim());

		if (out_sqrt_)
		{
			double val = SpatialIntegralObjective::value();
			if (std::abs(val) < 1e-12)
				logger().warn("stress integral too small, may result in NAN grad!");
			term *= (pow(val, 1. / in_power_ - 1) / in_power_);
		}

		return term;
	}

	PositionObjective::PositionObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
	{
		spatial_integral_type_ = SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
	}

	IntegrableFunctional PositionObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = u.col(this->dim_) + pts.col(this->dim_);
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			val.col(this->dim_).setOnes();
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			val.col(this->dim_).setOnes();
		});

		return j;
	}

	ComplianceObjective::ComplianceObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args) : SpatialIntegralObjective(state, shape_param, args), elastic_param_(elastic_param)
	{
		if (elastic_param_)
			assert(elastic_param_->name() == "material");
		spatial_integral_type_ = SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		formulation_ = state.formulation();
	}

	IntegrableFunctional ComplianceObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				if (this->formulation_ == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else
					logger().error("Unknown formulation!");
				val(q) = (stress.array() * grad_u_q.array()).sum();
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
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

	Eigen::VectorXd ComplianceObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		if (&param == shape_param_.get())
			return compute_partial_gradient(param, param_value);
		else if (&param == elastic_param_.get())
		{
			Eigen::VectorXd term;
			term.setZero(param.full_dim());

			const auto &bases = state_.bases;
			const auto &gbases = state_.geom_bases();
			auto df_dmu_dlambda_function = state_.assembler.get_dstress_dmu_dlambda_function(formulation_);
			const int dim = state_.mesh->dimension();

			for (int e = 0; e < bases.size(); e++)
			{
				assembler::ElementAssemblyValues vals;
				state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), bases[e], gbases[e], vals);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd u, grad_u;
				io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, state_.diff_cached[time_step_].u, u, grad_u);

				Eigen::MatrixXd grad_u_q;
				for (int q = 0; q < quadrature.weights.size(); q++)
				{
					double lambda, mu;
					state_.assembler.lame_params().lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

					vector2matrix(grad_u.row(q), grad_u_q);

					Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
					df_dmu_dlambda_function(e, quadrature.points.row(q), vals.val.row(q), grad_u_q, f_prime_dmu, f_prime_dlambda);

					term(e + bases.size()) += dot(f_prime_dmu, grad_u_q) * da(q);
					term(e) += dot(f_prime_dlambda, grad_u_q) * da(q);
				}
			}

			return param.map_grad(param_value, term);
		}
		else
			return Eigen::VectorXd::Zero(param.optimization_dim());
	}

	IntegrableFunctional TargetObjective::get_integral_functional()
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
				const Eigen::MatrixXd &sol_ref = target_state_->problem->is_time_dependent() ? target_state_->diff_cached[params["step"].get<int>()].u : target_state_->diff_cached[0].u;
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
				const Eigen::MatrixXd &sol_ref = target_state_->problem->is_time_dependent() ? target_state_->diff_cached[params["step"].get<int>()].u : target_state_->diff_cached[0].u;
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

	void TargetObjective::set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids)
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

	void TargetObjective::set_reference(const json &func, const json &grad_func)
	{
		target_func.init(func);
		for (size_t k = 0; k < grad_func.size(); k++)
			target_func_grad[k].init(grad_func[k]);
		have_target_func = true;
	}

	void SDFTargetObjective::compute_distance(const Eigen::MatrixXd &point, double &distance)
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

	IntegrableFunctional SDFTargetObjective::get_integral_functional()
	{
		IntegrableFunctional j;
		auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);

			for (int q = 0; q < u.rows(); q++)
			{
				double distance;
				Eigen::MatrixXd unused_grad;
				evaluate(u.row(q) + pts.row(q), distance, unused_grad);
				val(q) = pow(distance, 2);
			}
		};

		auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());

			for (int q = 0; q < u.rows(); q++)
			{
				double distance;
				Eigen::MatrixXd grad;
				evaluate(u.row(q) + pts.row(q), distance, grad);
				val.row(q) = 2 * distance * grad.transpose();
			}
		};

		j.set_j(j_func);
		j.set_dj_du(djdu_func);
		j.set_dj_dx(djdu_func);

		return j;
	}

	BarycenterTargetObjective::BarycenterTargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args, const Eigen::MatrixXd &target)
	{
		dim_ = state.mesh->dimension();
		target_ = target;

		objv = std::make_shared<VolumeObjective>(shape_param, args);
		objp.resize(dim_);
		for (int d = 0; d < dim_; d++)
		{
			objp[d] = std::make_shared<PositionObjective>(state, shape_param, args);
			objp[d]->set_dim(d);
		}
	}

	Eigen::VectorXd BarycenterTargetObjective::get_target() const
	{
		assert(target_.cols() == dim_);
		if (target_.rows() > 1)
			return target_.row(get_time_step());
		else
			return target_;
	}

	void BarycenterTargetObjective::set_time_step(int time_step)
	{
		StaticObjective::set_time_step(time_step);
		for (auto &obj : objp)
			obj->set_time_step(time_step);
	}

	double BarycenterTargetObjective::value()
	{
		return (get_barycenter() - get_target()).squaredNorm();
	}
	Eigen::VectorXd BarycenterTargetObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		term.setZero(param.optimization_dim());

		Eigen::VectorXd target = get_target();

		const double volume = objv->value();
		Eigen::VectorXd center(dim_);
		for (int d = 0; d < dim_; d++)
			center(d) = objp[d]->value() / volume;

		double coeffv = 0;
		for (int d = 0; d < dim_; d++)
			coeffv += 2 * (center(d) - target(d)) * (-center(d) / volume);

		term += coeffv * objv->compute_partial_gradient(param, param_value);

		for (int d = 0; d < dim_; d++)
			term += (2.0 / volume * (center(d) - target(d))) * objp[d]->compute_partial_gradient(param, param_value);

		return term;
	}
	Eigen::VectorXd BarycenterTargetObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd term;
		term.setZero(state.ndof());

		Eigen::VectorXd target = get_target();

		const double volume = objv->value();
		Eigen::VectorXd center(dim_);
		for (int d = 0; d < dim_; d++)
			center(d) = objp[d]->value() / volume;

		for (int d = 0; d < dim_; d++)
			term += (2.0 / volume * (center(d) - target(d))) * objp[d]->compute_adjoint_rhs_step(state);

		return term;
	}

	Eigen::VectorXd BarycenterTargetObjective::get_barycenter() const
	{
		const double volume = objv->value();
		Eigen::VectorXd center(dim_);
		for (int d = 0; d < dim_; d++)
			center(d) = objp[d]->value() / volume;

		return center;
	}
}