#include "TractionNormForm.hpp"

#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/State.hpp>

using namespace polyfem::utils;

namespace polyfem::solver
{
	namespace
	{
		template <typename T>
		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> compute_displaced_normal(const Eigen::MatrixXd &reference_normal, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &grad_x, const Eigen::MatrixXd &grad_u_local)
		{
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> trafo = grad_x;
			for (int i = 0; i < grad_x.rows(); ++i)
				for (int j = 0; j < grad_x.cols(); ++j)
					trafo(i, j) = trafo(i, j) + grad_u_local(i, j);

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> trafo_inv = inverse(trafo);

			Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> n(reference_normal.cols(), 1);
			for (int d = 0; d < n.size(); ++d)
				n(d) = T(0);

			for (int i = 0; i < n.size(); ++i)
				for (int j = 0; j < n.size(); ++j)
					n(j) = n(j) + (reference_normal(i) * trafo_inv(i, j));
			n = n / n.norm();

			return n;
		}

	} // namespace

	typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

	IntegrableFunctional TractionNormForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		const std::string formulation = state_.formulation();
		const int power = in_power_;

		if (formulation == "Laplacian")
			log_and_throw_error("TractionNormForm is not implemented for Laplacian!");

		j.set_j([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			int el_id = params["elem"];

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u_local, grad_x;
			Eigen::MatrixXd grad_u_q, stress, traction_force, grad_unused;
			Eigen::MatrixXd reference_normal, displaced_normal;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_x = vals.jac_it[q].inverse();
				vector2matrix(grad_u.row(q), grad_u_q);
				grad_u_local = grad_u_q * grad_x;

				reference_normal = reference_normals.row(q);
				displaced_normal = compute_displaced_normal(reference_normal, grad_x, grad_u_local).transpose();
				state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, Eigen::MatrixXd::Zero(1, grad_u_q.cols()), stress, grad_unused);
				traction_force = displaced_normal * stress;
				val(q) = pow(traction_force.squaredNorm(), power / 2.);
			}
		});

		auto dj_dgradx = [formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			int el_id = params["elem"];
			const int dim = sqrt(grad_u.cols());

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u_local, grad_x;
			Eigen::MatrixXd grad_u_q, stress, traction_force, grad_unused;
			Eigen::MatrixXd reference_normal, displaced_normal;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_x = vals.jac_it[q].inverse();
				vector2matrix(grad_u.row(q), grad_u_q);
				grad_u_local = grad_u_q * grad_x;

				DiffScalarBase::setVariableCount(dim * dim);
				Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_x_auto(dim, dim);
				for (int i = 0; i < dim; i++)
					for (int j = 0; j < dim; j++)
						grad_x_auto(i, j) = Diff(i + j * dim, grad_x(i, j));

				reference_normal = reference_normals.row(q);
				auto n = compute_displaced_normal(reference_normal, grad_x_auto, grad_u_local);
				displaced_normal.resize(1, dim);
				for (int i = 0; i < dim; ++i)
					displaced_normal(i) = n(i).getValue();
				state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, Eigen::MatrixXd::Zero(1, grad_u_q.cols()), stress, grad_unused);
				traction_force = displaced_normal * stress;

				const double coef = power * pow(traction_force.squaredNorm(), power / 2. - 1.);
				for (int k = 0; k < dim; ++k)
					for (int l = 0; l < dim; ++l)
					{
						double sum_j = 0;
						for (int j = 0; j < dim; ++j)
						{
							double grad_mult_stress = 0;
							for (int i = 0; i < dim; ++i)
								grad_mult_stress *= n(i).getGradient()(k + l * dim) * stress(i, j);

							sum_j += traction_force(j) * grad_mult_stress;
						}

						val(q, k * dim + l) = coef * sum_j;
					}
			}
		};
		j.set_dj_dgradx(dj_dgradx);
		j.set_dj_dgradu_local(dj_dgradx);

		auto dj_dgradu = [formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			int el_id = params["elem"];
			const int dim = sqrt(grad_u.cols());

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u_local, grad_x;
			Eigen::MatrixXd grad_u_q, stress, traction_force, vect_mult_dstress;
			Eigen::MatrixXd reference_normal, displaced_normal;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_x = vals.jac_it[q].inverse();
				vector2matrix(grad_u.row(q), grad_u_q);
				grad_u_local = grad_u_q * grad_x;

				reference_normal = reference_normals.row(q);
				displaced_normal = compute_displaced_normal(reference_normal, grad_x, grad_u_local).transpose();
				state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, displaced_normal, stress, vect_mult_dstress);
				traction_force = displaced_normal * stress;

				const double coef = power * pow(traction_force.squaredNorm(), power / 2. - 1.);
				for (int k = 0; k < dim; ++k)
					for (int l = 0; l < dim; ++l)
					{
						double sum_j = 0;
						for (int j = 0; j < dim; ++j)
							sum_j += traction_force(j) * vect_mult_dstress(l * dim + k, j);

						val(q, k * dim + l) = coef * sum_j;
					}
			}
		};
		j.set_dj_dgradu(dj_dgradu);

		/*
		j.set_dj_dgradu([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			int el_id = params["elem"];
			const int dim = sqrt(grad_u.cols());

			Eigen::MatrixXd displaced_normals;
			compute_displaced_normals(reference_normals, vals, grad_u, displaced_normals);

			Eigen::MatrixXd grad_u_q, stress, traction_force, normal_dstress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, displaced_normals.row(q), stress, normal_dstress);
				traction_force = displaced_normals.row(q) * stress;

				const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
				for (int i = 0; i < dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = coef * (traction_force.array() * normal_dstress.row(i * dim + l).array()).sum();
			}
		});

		auto dj_du = [formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			int el_id = params["elem"];
			const int dim = sqrt(grad_u.cols());

			Eigen::MatrixXd displaced_normals;
			std::vector<Eigen::MatrixXd> normal_jacobian;
			compute_displaced_normal_jacobian(reference_normals, vals, grad_u, displaced_normals, normal_jacobian);

			Eigen::MatrixXd grad_u_q, stress, normal_duT_stress, traction_force, grad_unused;
			for (int q = 0; q < u.rows(); q++)
			{
				vector2matrix(grad_u.row(q), grad_u_q);
				state.assembler->compute_stress_grad_multiply_mat(el_id, local_pts.row(q), pts.row(q), grad_u_q, Eigen::MatrixXd::Zero(grad_u_q.rows(), grad_u_q.cols()), stress, grad_unused);
				traction_force = displaced_normals.row(q) * stress;

				Eigen::MatrixXd normal_du = normal_jacobian[q]; // compute this
				normal_duT_stress = normal_du.transpose() * stress;

				const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
				for (int i = 0; i < dim; i++)
					val(q, i) = coef * (traction_force.array() * normal_duT_stress.row(i).array()).sum();
			}
		};
		j.set_dj_du(dj_du);
		j.set_dj_dx(dj_du);
		*/

		/*
		const int normal_dim = 0;

		auto normal = [normal_dim](const Eigen::MatrixXd &reference_normal, const Eigen::MatrixXd &grad_x, const Eigen::MatrixXd &grad_u_local) {
			Eigen::MatrixXd trafo = grad_x + grad_u_local;
			Eigen::MatrixXd n = reference_normal * trafo.inverse();
			n.normalize();

			return n(normal_dim);
		};

		j.set_j([formulation, power, normal, normal_dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			int el_id = params["elem"];

			Eigen::MatrixXd grad_x, grad_u_local;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_x = vals.jac_it[q].inverse();
				vector2matrix(grad_u.row(q), grad_u_local);
				grad_u_local = grad_u_local * grad_x;
				double v = normal(reference_normals.row(q), grad_x, grad_u_local);
				val(q) = pow(v, 2);
			}
		});

		auto dj_dgradx = [formulation, power, normal](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			int el_id = params["elem"];
			const int dim = sqrt(grad_u.cols());

			Eigen::MatrixXd grad_x, grad_u_local;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				grad_x = vals.jac_it[q].inverse();
				vector2matrix(grad_u.row(q), grad_u_local);
				grad_u_local = grad_u_local * grad_x;

				const double v = normal(reference_normals.row(q), grad_x, grad_u_local);

				double eps = 1e-7;
				for (int i = 0; i < dim; ++i)
					for (int j = 0; j < dim; ++j)
					{
						Eigen::MatrixXd grad_x_copy = grad_x;
						grad_x_copy(i, j) += eps;
						const double val_plus = pow(normal(reference_normals.row(q), grad_x_copy, grad_u_local), 2);
						grad_x_copy(i, j) -= 2 * eps;
						const double val_minus = pow(normal(reference_normals.row(q), grad_x_copy, grad_u_local), 2);

						const double fd = (val_plus - val_minus) / (2 * eps);
						val(q, i * dim + j) = fd;
					}
			}
		};

		j.set_dj_dgradx(dj_dgradx);
		j.set_dj_dgradu_local(dj_dgradx);
		*/

		// auto j_func = [formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
		// 	val.setZero(grad_u.rows(), 1);
		// 	int el_id = params["elem"];

		// 	Eigen::MatrixXd grad_u_q, stress, traction_force, grad_unused;
		// 	for (int q = 0; q < grad_u.rows(); q++)
		// 	{
		// 		vector2matrix(grad_u.row(q), grad_u_q);
		// 		double value = (grad_u_q * vals.jac_it[q].inverse() + vals.jac_it[q].inverse()).squaredNorm();
		// 		val(q) = pow(value, power / 2.);
		// 	}
		// };
		// j.set_j(j_func);

		// auto dj_dgradx = [formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
		// 	val.setZero(grad_u.rows(), grad_u.cols());
		// 	int el_id = params["elem"];
		// 	const int dim = sqrt(grad_u.cols());

		// 	Eigen::MatrixXd grad_u_q;
		// 	for (int q = 0; q < grad_u.rows(); q++)
		// 	{
		// 		vector2matrix(grad_u.row(q), grad_u_q);

		// 		Eigen::MatrixXd grad = (grad_u_q * vals.jac_it[q].inverse() + vals.jac_it[q].inverse());
		// 		const double coef = power * pow(grad.squaredNorm(), power / 2. - 1.);
		// 		val.row(q) = coef * flatten(grad);
		// 	}
		// };
		// j.set_dj_dgradx(dj_dgradx);
		// j.set_dj_dgradu_local(dj_dgradx);

		return j;
	}

	void
	TractionNormForm::compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		SpatialIntegralForm::compute_partial_gradient_unweighted_step(time_step, x, gradv);
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &param_type = param_map->get_parameter_type();

			for (const auto &state : param_map->get_states())
			{
				if (state.get() != &state_)
					continue;

				Eigen::VectorXd term;
				if (param_type == ParameterType::Material)
					log_and_throw_error("Doesn't support traction derivative wrt. material!");

				if (term.size() > 0)
					gradv += param_map->apply_parametrization_jacobian(term, x);
			}
		}
	}

} // namespace polyfem::solver