#include "Objective.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include "HomoObjective.hpp"

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
		T strain_norm(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &F)
		{
			T val = T(0);
			auto strain = (F.transpose() + F) / T(2.0);
			for (int i = 0; i < strain.rows(); i++)
				for (int j = 0; j < strain.cols(); j++)
					val += strain(i, j) * strain(i, j);
			return val;
		}

		Eigen::MatrixXd strain_norm_grad(const Eigen::MatrixXd &F)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic> full_diff(F.rows(), F.cols());
			for (int i = 0; i < F.rows(); i++)
				for (int j = 0; j < F.cols(); j++)
					full_diff(i, j) = Diff(i + j * F.rows(), F(i, j));
			auto reduced_diff = strain_norm(full_diff);

			Eigen::MatrixXd grad(F.rows(), F.cols());
			for (int i = 0; i < F.rows(); ++i)
				for (int j = 0; j < F.cols(); ++j)
					grad(i, j) = reduced_diff.getGradient()(i + j * F.rows());

			return grad;
		}
		
		template <typename T>
		T homo_stress_aux(const Eigen::Matrix<T, Eigen::Dynamic, 1> &F)
		{
			T val1 = F(0) * F(0) + F(1) * F(1) + F(2) * F(2);
			T val2 = F(3) * F(3);

			return sqrt(val1 / (val2 + val1));
		}

		Eigen::VectorXd homo_stress_aux_grad(const Eigen::VectorXd &F)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, 1> full_diff(F.size());
			for (int i = 0; i < F.size(); i++)
				full_diff(i) = Diff(i, F(i));
			auto reduced_diff = homo_stress_aux(full_diff);

			Eigen::VectorXd grad(F.size());
			for (int i = 0; i < F.size(); ++i)
				grad(i) = reduced_diff.getGradient()(i);

			return grad;
		}
	} // namespace

	HomogenizedEnergyObjective::HomogenizedEnergyObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args) : SpatialIntegralObjective(state, shape_param, macro_strain_param, args), elastic_param_(elastic_param)
	{
		spatial_integral_type_ = SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		formulation_ = state_.formulation();
	}

	IntegrableFunctional HomogenizedEnergyObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
            auto energy_func = this->state_.assembler.get_elastic_energy_function(this->formulation_);
			for (int q = 0; q < grad_u.rows(); q++)
			{
				val(q) = energy_func(utils::unflatten(grad_u.row(q), u.cols()), lambda(q), mu(q));
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			Eigen::MatrixXd grad_u_q, def_grad, FmT, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_u_q = utils::unflatten(grad_u.row(q), u.cols());
				if (this->formulation_ == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (this->formulation_ == "NeoHookean")
				{
					def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val.row(q) = utils::flatten(stress);
			}
		});

		return j;
	}

	double HomogenizedEnergyObjective::value()
	{
		double val = SpatialIntegralObjective::value();
		return val;
	}

	Eigen::VectorXd HomogenizedEnergyObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd rhs = SpatialIntegralObjective::compute_adjoint_rhs_step(state);
		return rhs;
	}

	Eigen::VectorXd HomogenizedEnergyObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		term.setZero(param.optimization_dim());
		if (&param == elastic_param_.get())
		{
			// TODO: differentiate stress wrt. lame param
			log_and_throw_error("Not implemented!");
		}
		else
		{
			term = SpatialIntegralObjective::compute_partial_gradient(param, param_value);
		}

		return term;
	}

	HomogenizedStressObjective::HomogenizedStressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args) : SpatialIntegralObjective(state, shape_param, macro_strain_param, args), elastic_param_(elastic_param)
	{
		spatial_integral_type_ = SpatialIntegralType::VOLUME;
		auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
		interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		id = args["id"].get<std::vector<int>>();
		assert(id.size() == 2);
		formulation_ = state.formulation();

		RowVectorNd min, max;
		state.mesh->bounding_box(min, max);
		microstructure_volume = (max - min).prod();
	}

	IntegrableFunctional HomogenizedStressObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				if (this->formulation_ == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (this->formulation_ == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val(q) = stress(id[0], id[1]);
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			Eigen::MatrixXd grad_u_q, stiffness, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				stiffness.setZero(1, dim * dim * dim * dim);
				vector2matrix(grad_u.row(q), grad_u_q);

				if (this->formulation_ == "LinearElasticity")
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
				else if (this->formulation_ == "NeoHookean")
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

				val.row(q) = stiffness.block(0, (id[0] * dim + id[1]) * dim * dim, 1, dim * dim);
			}
		});

		return j;
	}

	double HomogenizedStressObjective::value()
	{
		double val = SpatialIntegralObjective::value();
		return val / microstructure_volume;
	}

	Eigen::VectorXd HomogenizedStressObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd rhs = SpatialIntegralObjective::compute_adjoint_rhs_step(state);
		return rhs / microstructure_volume;
	}

	Eigen::VectorXd HomogenizedStressObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		term.setZero(param.optimization_dim());
		if (&param == elastic_param_.get())
		{
			// TODO: differentiate stress wrt. lame param
			log_and_throw_error("Not implemented!");
		}
		else
		{
			term = SpatialIntegralObjective::compute_partial_gradient(param, param_value);
		}

		return term / microstructure_volume;
	}

	CompositeHomogenizedStressObjective::CompositeHomogenizedStressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args)
	{
		json tmp_arg = args;
		std::vector<int> id(2);
		id[0] = 0;
		id[1] = 0;
		tmp_arg["id"] = id;
		js[0] = std::make_shared<HomogenizedStressObjective>(state, shape_param, macro_strain_param, elastic_param, tmp_arg);

		id[0] = 0;
		id[1] = 1;
		tmp_arg["id"] = id;
		js[1] = std::make_shared<HomogenizedStressObjective>(state, shape_param, macro_strain_param, elastic_param, tmp_arg);

		id[0] = 1;
		id[1] = 0;
		tmp_arg["id"] = id;
		js[2] = std::make_shared<HomogenizedStressObjective>(state, shape_param, macro_strain_param, elastic_param, tmp_arg);

		id[0] = 1;
		id[1] = 1;
		tmp_arg["id"] = id;
		js[3] = std::make_shared<HomogenizedStressObjective>(state, shape_param, macro_strain_param, elastic_param, tmp_arg);
	}
	double CompositeHomogenizedStressObjective::value()
	{
		Eigen::VectorXd F(4);
		F << js[0]->value(), js[1]->value(), js[2]->value(), js[3]->value();
		logger().debug("Current homogenized stress: {}", F.transpose());
		return homo_stress_aux(F);
	}
	Eigen::MatrixXd CompositeHomogenizedStressObjective::compute_adjoint_rhs(const State &state)
	{
		Eigen::VectorXd F(4);
		F << js[0]->value(), js[1]->value(), js[2]->value(), js[3]->value();
		Eigen::VectorXd grad_aux = homo_stress_aux_grad(F);

		Eigen::MatrixXd grad;
		grad.setZero(state.ndof(), state.diff_cached.size());
		for (int i = 0; i < F.size(); i++)
			grad += grad_aux(i) * js[i]->compute_adjoint_rhs(state);
		return grad;
	}
	Eigen::VectorXd CompositeHomogenizedStressObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd F(4);
		F << js[0]->value(), js[1]->value(), js[2]->value(), js[3]->value();
		Eigen::VectorXd grad_aux = homo_stress_aux_grad(F);

		Eigen::VectorXd grad;
		grad.setZero(param.optimization_dim());
		for (int i = 0; i < F.size(); i++)
			grad += grad_aux(i) * js[i]->compute_partial_gradient(param, param_value);
		return grad;
	}

	IntegrableFunctional StrainObjective::get_integral_functional()
	{
		IntegrableFunctional j;

		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q;
				vector2matrix(grad_u.row(q), grad_u_q);
				val(q) = strain_norm(grad_u_q);
			}
		});

		j.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q;
				vector2matrix(grad_u.row(q), grad_u_q);
				Eigen::MatrixXd grad = strain_norm_grad(grad_u_q);

				for (int i = 0; i < dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = grad(i, l);
			}
		});

		return j;
	}

	HomogenizedStiffnessObjective::HomogenizedStiffnessObjective(std::shared_ptr<State> state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args): shape_param_(shape_param), elastic_param_(elastic_param)
	{
		id = args["id"].get<std::vector<int>>();
		assert(id.size() == 4);
		formulation_ = state->formulation();

		multiscale_assembler = std::make_shared<assembler::Multiscale>(state);
	}

	double HomogenizedStiffnessObjective::value()
	{
		Eigen::MatrixXd stiffness;
		multiscale_assembler->homogenize_stiffness(multiscale_assembler->get_microstructure_state()->diff_cached[0].u, stiffness);
		return stiffness(id[0], id[1]);
	}

	Eigen::VectorXd HomogenizedStiffnessObjective::compute_adjoint_rhs_step(const State &state)
	{
		Eigen::VectorXd rhs;
		rhs.setZero(state.ndof());
		if (&state == multiscale_assembler->get_microstructure_state().get())
		{

		}
		return rhs;
	}

	Eigen::VectorXd HomogenizedStiffnessObjective::compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value)
	{
		Eigen::VectorXd term;
		term.setZero(param.optimization_dim());
		if (&param == elastic_param_.get())
		{
			// TODO: differentiate stress wrt. lame param
			log_and_throw_error("Not implemented!");
		}
		else if (&param == shape_param_.get())
		{

		}

		return term;
	}

}