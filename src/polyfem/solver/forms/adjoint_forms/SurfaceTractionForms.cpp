#include "SurfaceTractionForms.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>

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

	void ContactForceForm::build_active_nodes()
	{

		std::set<int> active_nodes_set = {};
		dim_ = state_.mesh->dimension();
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			const basis::ElementBases &bs = state_.bases[e];

			for (int i = 0; i < lb.size(); i++)
			{
				const int global_primitive_id = lb.global_primitive_id(i);
				const auto nodes = bs.local_nodes_for_primitive(global_primitive_id, *state_.mesh);
				if (ids_.size() != 0 && ids_.find(state_.mesh->get_boundary_id(global_primitive_id)) == ids_.end())
					continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const auto &b = bs.bases[nodes(n)];
					const int index = b.global()[0].index;

					for (int d = 0; d < dim_; ++d)
						active_nodes_set.insert(index * dim_ + d);
				}
			}
		}

		active_nodes_.resize(active_nodes_set.size());
		active_nodes_mat_.resize(state_.collision_mesh.full_ndof(), active_nodes_set.size());
		std::vector<Eigen::Triplet<double>> active_nodes_i;
		int count = 0;
		for (const auto node : active_nodes_set)
		{
			active_nodes_i.emplace_back(node, count, 1.0);
			active_nodes_(count++) = node;
		}
		active_nodes_mat_.setFromTriplets(active_nodes_i.begin(), active_nodes_i.end());

		epsv_ = state_.args["contact"]["epsv"];
		dhat_ = state_.args["contact"]["dhat"];
		friction_coefficient_ = state_.args["contact"]["friction_coefficient"];
		depends_on_step_prev_ = (friction_coefficient_ > 0);
	}

	double ContactForceForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

		double sum = (forces.array() * forces.array()).sum();

		return sum;
	}

	Eigen::VectorXd ContactForceForm::compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

		StiffnessMatrix hessian(collision_mesh.ndof(), collision_mesh.ndof());
		hessian -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_hessian(collision_mesh, displaced_surface, dhat_, false);
		if (state_.solve_data.friction_form && time_step > 0)
		{
			const double dv_du = 1 / state_.solve_data.time_integrator->dt();
			hessian += state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::VELOCITIES);
		}
		hessian = collision_mesh.to_full_dof(hessian);

		Eigen::VectorXd gradu = 2 * hessian.transpose() * active_nodes_mat_ * forces;

		return gradu;
	}

	Eigen::VectorXd ContactForceForm::compute_adjoint_rhs_unweighted_step_prev(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

		StiffnessMatrix hessian_prev(collision_mesh.ndof(), collision_mesh.ndof());
		if (state_.solve_data.friction_form && time_step > 0)
		{
			const double dv_du = -1 / state_.solve_data.time_integrator->dt();
			hessian_prev += state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::LAGGED_DISPLACEMENTS);
			hessian_prev += dv_du * state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::VELOCITIES);
		}
		hessian_prev = collision_mesh.to_full_dof(hessian_prev);

		Eigen::VectorXd gradu = 2 * hessian_prev.transpose() * active_nodes_mat_ * forces;

		return gradu;
	}

	void ContactForceForm::compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

		StiffnessMatrix shape_derivative(collision_mesh.ndof(), collision_mesh.ndof());
		shape_derivative -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_shape_derivative(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			shape_derivative += state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::REST_POSITIONS);
		shape_derivative = collision_mesh.to_full_dof(shape_derivative);

		Eigen::VectorXd grads = 2 * shape_derivative.transpose() * active_nodes_mat_ * forces;
		grads = state_.gbasis_nodes_to_basis_nodes * grads;
		grads = AdjointTools::map_node_to_primitive_order(state_, grads);

		gradv.setZero(x.size());

		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &param_type = param_map->get_parameter_type();

			for (const auto &state : param_map->get_states())
			{
				if (state.get() != &state_)
					continue;

				if (param_type != ParameterType::Shape)
					log_and_throw_error("Only support contact force derivative wrt. shape!");

				if (grads.size() > 0)
					gradv += param_map->apply_parametrization_jacobian(grads, x);
			}
		}
	}

	void ContactForceMatchForm::build_active_nodes(const std::vector<std::string> &closed_form_forces)
	{

		std::set<int> active_nodes_set = {};
		std::map<int, double> node_area_scaling_map;
		dim_ = state_.mesh->dimension();
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			const basis::ElementBases &bs = state_.bases[e];

			for (int i = 0; i < lb.size(); i++)
			{
				const int global_primitive_id = lb.global_primitive_id(i);
				const auto nodes = bs.local_nodes_for_primitive(global_primitive_id, *state_.mesh);

				double area_scaling;
				if (state_.mesh->is_volume())
					area_scaling = state_.mesh->tri_area(global_primitive_id) / 3;
				else
					area_scaling = state_.mesh->edge_length(global_primitive_id) / 2;
				// std::cout << "area scaling: " << area_scaling << std::endl;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const auto &b = bs.bases[nodes(n)];
					const int index = b.global()[0].index;

					for (int d = 0; d < dim_; ++d)
					{
						if (node_area_scaling_map.count(index * dim_ + d) == 0)
							node_area_scaling_map[index * dim_ + d] = area_scaling;
						else
							node_area_scaling_map[index * dim_ + d] += area_scaling;
					}
				}

				if (ids_.size() != 0 && ids_.find(state_.mesh->get_boundary_id(global_primitive_id)) == ids_.end())
					continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const auto &b = bs.bases[nodes(n)];
					const int index = b.global()[0].index;

					for (int d = 0; d < dim_; ++d)
					{
						active_nodes_set.insert(index * dim_ + d);
					}
				}
			}
		}

		active_nodes_.resize(active_nodes_set.size());
		// node_area_scaling_.resize(active_nodes_set.size());
		active_nodes_mat_.resize(state_.collision_mesh.full_ndof(), active_nodes_set.size());
		std::vector<Eigen::Triplet<double>> active_nodes_i;
		int count = 0;
		for (const auto node : active_nodes_set)
		{
			active_nodes_i.emplace_back(node, count, 1.0);
			active_nodes_(count) = node;
			// node_area_scaling_(count) = node_area_scaling_map[node];
			count++;
		}
		active_nodes_mat_.setFromTriplets(active_nodes_i.begin(), active_nodes_i.end());
		// for (const auto kv : node_area_scaling_map)
		// 	std::cout << "k: " << kv.first << " v: " << kv.second << std::endl;

		if (closed_form_forces.size() != dim_)
			log_and_throw_error("Specified function dim for force matching function does not match problem dim!");
		std::vector<ExpressionValue> matching_expr;
		for (int i = 0; i < dim_; ++i)
		{
			matching_expr.push_back(ExpressionValue());
			matching_expr[i].init(closed_form_forces[i]);
		}

		Eigen::MatrixXd V;
		int dim = state_.collision_mesh.dim();
		V.setZero(active_nodes_.size() / dim, dim);
		{
			const Eigen::MatrixXd &rest_positions = state_.collision_mesh.rest_positions();
			for (int i = 0; i < state_.collision_mesh.num_vertices(); ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					int dof_idx = state_.collision_mesh.to_full_vertex_id(i) * dim + j;
					auto it = std::find(active_nodes_.begin(), active_nodes_.end(), dof_idx);
					if (it != active_nodes_.end())
					{
						int idx = it - active_nodes_.begin();
						V(idx / dim, idx % dim) = rest_positions(i, j);
					}
				}
			}
		}

		// state_.get_vertices(V);
		// V = utils::unflatten(AdjointTools::map_primitive_to_node_order(state_, utils::flatten(V)), dim_);
		matched_forces_.resize(active_nodes_.size(), 1);
		for (int i = 0; i < active_nodes_.size(); ++i)
		{
			int n = i / dim_;
			int d = i % dim_;
			if (state_.mesh->is_volume())
				matched_forces_(i) = matching_expr[d](V(n, 0), V(n, 1), V(n, 2));
			else
				matched_forces_(i) = matching_expr[d](V(n, 0), V(n, 1));

			if (d == 0)
				std::cout << "[";
			std::cout << V(n, d);
			if (d < dim_ - 1)
				std::cout << ", ";
			else if (d == dim_ - 1)
				std::cout << "],\n";
		}
		std::cout << "matched forces:\n"
				  << utils::unflatten(matched_forces_, dim_).col(1) << std::endl;

		epsv_ = state_.args["contact"]["epsv"];
		dhat_ = state_.args["contact"]["dhat"];
		friction_coefficient_ = state_.args["contact"]["friction_coefficient"];
		depends_on_step_prev_ = (friction_coefficient_ > 0);
	}

	double ContactForceMatchForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);
		// forces = forces.array() / node_area_scaling_.array();
		// std::cout << "actual forces:\n"
		// 		  << utils::unflatten(forces, dim_) << std::endl;
		forces -= matched_forces_;

		double sum = (forces.array() * forces.array()).sum();

		return sum;
	}

	Eigen::VectorXd ContactForceMatchForm::compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);
		// forces = forces.array() / node_area_scaling_.array() / node_area_scaling_.array();
		forces -= matched_forces_;

		StiffnessMatrix hessian(collision_mesh.ndof(), collision_mesh.ndof());
		hessian -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_hessian(collision_mesh, displaced_surface, dhat_, false);
		if (state_.solve_data.friction_form && time_step > 0)
		{
			const double dv_du = 1 / state_.solve_data.time_integrator->dt();
			hessian += state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::VELOCITIES);
		}
		hessian = collision_mesh.to_full_dof(hessian);

		Eigen::VectorXd gradu = 2 * hessian.transpose() * active_nodes_mat_ * forces;

		return gradu;
	}

	Eigen::VectorXd ContactForceMatchForm::compute_adjoint_rhs_unweighted_step_prev(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);
		// forces = forces.array() / node_area_scaling_.array() / node_area_scaling_.array();
		forces -= matched_forces_;

		StiffnessMatrix hessian_prev(collision_mesh.ndof(), collision_mesh.ndof());
		if (state_.solve_data.friction_form && time_step > 0)
		{
			const double dv_du = -1 / state_.solve_data.time_integrator->dt();
			hessian_prev += state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::LAGGED_DISPLACEMENTS);
			hessian_prev += dv_du * state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::VELOCITIES);
		}
		hessian_prev = collision_mesh.to_full_dof(hessian_prev);

		Eigen::VectorXd gradu = 2 * hessian_prev.transpose() * active_nodes_mat_ * forces;

		return gradu;
	}

	void ContactForceMatchForm::compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		double barrier_stiffness = state_.solve_data.contact_form->weight();

		const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
		Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
		forces -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_potential_gradient(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			forces += state_.diff_cached.friction_constraint_set(time_step).compute_force(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt());
		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);
		// forces = forces.array() / node_area_scaling_.array() / node_area_scaling_.array();
		forces -= matched_forces_;

		StiffnessMatrix shape_derivative(collision_mesh.ndof(), collision_mesh.ndof());
		shape_derivative -= barrier_stiffness * state_.diff_cached.contact_set(time_step).compute_shape_derivative(collision_mesh, displaced_surface, dhat_);
		if (state_.solve_data.friction_form && time_step > 0)
			shape_derivative += state_.diff_cached.friction_constraint_set(time_step).compute_force_jacobian(collision_mesh, collision_mesh.rest_positions(), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim())), collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim())), dhat_, barrier_stiffness, epsv_ * state_.solve_data.time_integrator->dt(), ipc::FrictionConstraint::DiffWRT::REST_POSITIONS);
		shape_derivative = collision_mesh.to_full_dof(shape_derivative);

		Eigen::VectorXd grads = 2 * shape_derivative.transpose() * active_nodes_mat_ * forces;
		grads = state_.gbasis_nodes_to_basis_nodes * grads;
		grads = AdjointTools::map_node_to_primitive_order(state_, grads);

		gradv.setZero(x.size());

		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &param_type = param_map->get_parameter_type();

			for (const auto &state : param_map->get_states())
			{
				if (state.get() != &state_)
					continue;

				if (param_type != ParameterType::Shape)
					log_and_throw_error("Only support contact force derivative wrt. shape!");

				if (grads.size() > 0)
					gradv += param_map->apply_parametrization_jacobian(grads, x);
			}
		}
	}

} // namespace polyfem::solver