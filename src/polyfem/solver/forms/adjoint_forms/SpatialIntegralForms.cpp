#include "SpatialIntegralForms.hpp"
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/State.hpp>
#include <polyfem/assembler/Mass.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/utils/IntegrableFunctional.hpp>

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

		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};
	} // namespace

	double SpatialIntegralForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		assert(time_step < state_.diff_cached.size());
		return AdjointTools::integrate_objective(state_, get_integral_functional(), state_.diff_cached.u(time_step), ids_, spatial_integral_type_, time_step);
	}

	void SpatialIntegralForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(time_step < state_.diff_cached.size());
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, time_step, &x]() {
			Eigen::VectorXd term;
			AdjointTools::compute_shape_derivative_functional_term(this->state_, this->state_.diff_cached.u(time_step), this->get_integral_functional(), this->ids_, this->spatial_integral_type_, term, time_step);
			return term;
		});
		gradv += variable_to_simulations_.apply_parametrization_jacobian(ParameterType::PeriodicShape, &state_, x, [this, time_step, &x]() {
			Eigen::VectorXd term;
			AdjointTools::compute_shape_derivative_functional_term(this->state_, this->state_.diff_cached.u(time_step), this->get_integral_functional(), this->ids_, this->spatial_integral_type_, term, time_step);
			term *= this->weight();

			const Eigen::VectorXd adjoint_rhs = this->compute_adjoint_rhs_step(time_step, x, state_);
			const NLHomoProblem &homo_problem = *std::dynamic_pointer_cast<NLHomoProblem>(state_.solve_data.nl_problem);
			const Eigen::VectorXd full_shape_deriv = homo_problem.reduced_to_full_shape_derivative(state_.diff_cached.disp_grad(), adjoint_rhs);
			term += utils::flatten(utils::unflatten(full_shape_deriv, state_.mesh->dimension())(state_.primitive_to_node(), Eigen::all));

			return term;
		});
	}

	Eigen::VectorXd SpatialIntegralForm::compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		if (&state != &state_)
			return Eigen::VectorXd::Zero(state.ndof());

		assert(time_step < state_.diff_cached.size());

		Eigen::VectorXd rhs;
		AdjointTools::dJ_du_step(state, get_integral_functional(), state.diff_cached.u(time_step), ids_, spatial_integral_type_, time_step, rhs);

		return rhs * weight();
	}

	IntegrableFunctional ElasticEnergyForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		const std::string formulation = state_.formulation();

		j.set_j([formulation, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			const int dim = u.cols();
			Eigen::MatrixXd grad_u_q;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_u_q = utils::unflatten(grad_u.row(q), u.cols());
				if (formulation == "LinearElasticity")
				{
					grad_u_q = (grad_u_q + grad_u_q.transpose()).eval() / 2.;
					val(q) = mu(q) * (grad_u_q.transpose() * grad_u_q).trace() + lambda(q) / 2. * grad_u_q.trace() * grad_u_q.trace();
				}
				else if (formulation == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = grad_u_q + Eigen::MatrixXd::Identity(dim, dim);
					double log_det_j = log(def_grad.determinant());
					val(q) = mu(q) / 2 * ((def_grad.transpose() * def_grad).trace() - dim - 2 * log_det_j) + lambda(q) / 2 * log_det_j * log_det_j;
				}
				else
					log_and_throw_adjoint_error("[{}] Unknown formulation {}!", name(), formulation);
			}
		});

		j.set_dj_dgradu([formulation, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			Eigen::MatrixXd grad_u_q, def_grad, FmT, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_u_q = utils::unflatten(grad_u.row(q), u.cols());
				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (formulation == "NeoHookean")
				{
					def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_adjoint_error("[{}] Unknown formulation {}!", name(), formulation);
				val.row(q) = utils::flatten(stress);
			}
		});

		return j;
	}

	void ElasticEnergyForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_step(time_step, x, gradv);
		gradv += weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::LameParameter, &state_, x, [this]() {
			log_and_throw_adjoint_error("[{}] Doesn't support derivatives wrt. material!", name());
			return Eigen::VectorXd::Zero(0).eval();
		});
	}

	IntegrableFunctional StressNormForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		const std::string formulation = state_.formulation();
		const int power = in_power_;

		j.set_j([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0;

			Eigen::MatrixXd grad_u_q, stress, grad_unused;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (formulation == "Laplacian")
					stress = grad_u.row(q);
				else
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					state.assembler->compute_stress_grad_multiply_mat(OptAssemblerData(params.t, dt, params.elem, local_pts.row(q), pts.row(q), grad_u_q), Eigen::MatrixXd::Zero(grad_u_q.rows(), grad_u_q.cols()), stress, grad_unused);
				}
				val(q) = pow(stress.squaredNorm(), power / 2.);
			}
		});

		j.set_dj_dgradu([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0;
			const int dim = state.mesh->dimension();

			if (formulation == "Laplacian")
			{
				for (int q = 0; q < grad_u.rows(); q++)
				{
					const double coef = power * pow(grad_u.row(q).squaredNorm(), power / 2. - 1.);
					val.row(q) = coef * grad_u.row(q);
				}
			}
			else
			{
				Eigen::MatrixXd grad_u_q, stress, stress_dstress;
				for (int q = 0; q < grad_u.rows(); q++)
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					state.assembler->compute_stress_grad_multiply_stress(OptAssemblerData(params.t, dt, params.elem, local_pts.row(q), pts.row(q), grad_u_q), stress, stress_dstress);

					const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
					val.row(q) = coef * utils::flatten(stress_dstress);
				}
			}
		});

		return j;
	}

	void StressNormForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_step(time_step, x, gradv);
		gradv += weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::LameParameter, &state_, x, [this]() {
			log_and_throw_adjoint_error("[{}] Doesn't support derivatives wrt. material!", name());
			return Eigen::VectorXd::Zero(0).eval();
		});
	}

	IntegrableFunctional ComplianceForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		if (state_.formulation() != "LinearElasticity")
			log_and_throw_adjoint_error("[{}] Only Linear Elasticity formulation is supported!", name());

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);

			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				val(q) = (stress.array() * grad_u_q.array()).sum();
			}
		});

		j.set_dj_dgradu([](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = u.cols();

			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				val.row(q) = 2 * utils::flatten(stress);
			}
		});

		return j;
	}

	void ComplianceForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const double dt = state_.problem->is_time_dependent() ? state_.args["time"]["dt"].get<double>() : 0;
		const double t = state_.problem->is_time_dependent() ? dt * time_step + state_.args["time"]["t0"].get<double>() : 0;

		SpatialIntegralForm::compute_partial_gradient_step(time_step, x, gradv);
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::LameParameter, &state_, x, [this, t, dt, time_step, &x]() {
			const auto &bases = state_.bases;
			Eigen::VectorXd term = Eigen::VectorXd::Zero(bases.size() * 2);
			const int dim = state_.mesh->dimension();

			for (int e = 0; e < bases.size(); e++)
			{
				assembler::ElementAssemblyValues vals;
				state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), bases[e], state_.geom_bases()[e], vals);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd u, grad_u;
				io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, state_.diff_cached.u(time_step), u, grad_u);

				Eigen::MatrixXd grad_u_q;
				for (int q = 0; q < quadrature.weights.size(); q++)
				{
					double lambda, mu;
					lambda = state_.assembler->parameters().at("lambda")(quadrature.points.row(q), vals.val.row(q), t, e);
					mu = state_.assembler->parameters().at("mu")(quadrature.points.row(q), vals.val.row(q), t, e);

					vector2matrix(grad_u.row(q), grad_u_q);

					Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
					state_.assembler->compute_dstress_dmu_dlambda(OptAssemblerData(t, dt, e, quadrature.points.row(q), vals.val.row(q), grad_u_q), f_prime_dmu, f_prime_dlambda);

					term(e + bases.size()) += dot(f_prime_dmu, grad_u_q) * da(q);
					term(e) += dot(f_prime_dlambda, grad_u_q) * da(q);
				}
			}
			return term;
		});
	}

	IntegrableFunctional PositionForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		const int dim = dim_;

		j.set_j([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val = u.col(dim) + pts.col(dim);
		});

		j.set_dj_du([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			val.col(dim).setOnes();
		});

		j.set_dj_dx([dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			val.col(dim).setOnes();
		});

		return j;
	}

	IntegrableFunctional AccelerationForm::get_integral_functional() const
	{
		IntegrableFunctional j;
		const int dim = this->dim_;

		j.set_j([dim, &state = std::as_const(this->state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			Eigen::MatrixXd acc, grad_acc;
			io::Evaluator::interpolate_at_local_vals(*(state.mesh), state.problem->is_scalar(), state.bases, state.geom_bases(), params.elem, local_pts, state.diff_cached.acc(params.step), acc, grad_acc);

			val = acc.col(dim);
		});

		j.set_dj_du([dim, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			log_and_throw_adjoint_error("[{}] Gradient not implemented!", name());
		});

		return j;
	}

	IntegrableFunctional KineticForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			Eigen::MatrixXd v, grad_v;
			io::Evaluator::interpolate_at_local_vals(*(state_.mesh), state_.problem->is_scalar(), state_.bases, state_.geom_bases(), params.elem, local_pts, state_.diff_cached.v(params.step), v, grad_v);

			val.setZero(u.rows(), 1);
			for (int q = 0; q < v.rows(); q++)
			{
				const double rho = state_.mass_matrix_assembler->density()(local_pts.row(q), pts.row(q), params.t, params.elem);
				val(q) = 0.5 * rho * v.row(q).squaredNorm();
			}
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());

			log_and_throw_adjoint_error("[{}] Gradient not implemented!", name());
		});

		return j;
	}

	void StressForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_step(time_step, x, gradv);
		gradv += weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::LameParameter, &state_, x, [this]() {
			log_and_throw_adjoint_error("[{}] Doesn't support derivatives wrt. material!", name());
			return Eigen::VectorXd::Zero(0).eval();
		});
	}

	IntegrableFunctional StressForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		std::string formulation = state_.formulation();
		auto dimensions = dimensions_;

		j.set_j([formulation, dimensions, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
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
					log_and_throw_adjoint_error("[{}] Unknown formulation {}!", name(), formulation);
				val(q) = stress(dimensions[0], dimensions[1]);
			}
		});

		j.set_dj_dgradu([formulation, dimensions, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());

			const int dim = state_.mesh->dimension();
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
					log_and_throw_adjoint_error("[{}] Unknown formulation {}!", name(), formulation);

				val.row(q) = stiffness.block(0, (dimensions[0] * dim + dimensions[1]) * dim * dim, 1, dim * dim);
			}
		});

		return j;
	}

	IntegrableFunctional VolumeForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		j.set_j([](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const IntegrableFunctional::ParameterType &params, Eigen::MatrixXd &val) {
			val.setOnes(grad_u.rows(), 1);
		});

		return j;
	}
} // namespace polyfem::solver