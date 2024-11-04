#include "SurfaceTractionForms.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>

#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/State.hpp>

// #include <finitediff.hpp>

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

		class QuadraticBarrier : public ipc::Barrier
		{
		public:
			QuadraticBarrier(const double weight = 1) : weight_(weight) {}

			double operator()(const double d, const double dhat) const override
			{
				if (d > dhat)
					return 0;
				else
					return weight_ * (d - dhat) * (d - dhat);
			}
			double first_derivative(const double d, const double dhat) const override
			{
				if (d > dhat)
					return 0;
				else
					return 2 * weight_ * (d - dhat);
			}
			double second_derivative(const double d, const double dhat) const override
			{
				if (d > dhat)
					return 0;
				else
					return 2 * weight_;
			}

		private:
			const double weight_;
		};

		void compute_collision_mesh_quantities(
			const State &state,
			const std::set<int> &boundary_ids,
			const ipc::CollisionMesh &collision_mesh,
			Eigen::MatrixXd &node_positions,
			Eigen::MatrixXi &boundary_edges,
			Eigen::MatrixXi &boundary_triangles,
			Eigen::SparseMatrix<double> &displacement_map,
			std::vector<bool> &is_on_surface,
			Eigen::MatrixXi &can_collide_cache)
		{
			std::vector<Eigen::Triplet<double>> displacement_map_entries;
			io::OutGeometryData::extract_boundary_mesh(*state.mesh, state.n_bases, state.bases, state.total_local_boundary,
													   node_positions, boundary_edges, boundary_triangles, displacement_map_entries);

			is_on_surface.resize(node_positions.rows(), false);

			std::map<int, std::set<int>> boundary_ids_to_dof;
			assembler::ElementAssemblyValues vals;
			Eigen::MatrixXd points, uv, normals;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_primitive_ids;
			for (const auto &lb : state.total_local_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, state.n_boundary_samples(), *state.mesh, false, uv, points, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				const basis::ElementBases &bs = state.bases[e];
				const basis::ElementBases &gbs = state.geom_bases()[e];

				vals.compute(e, state.mesh->is_volume(), points, bs, gbs);

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, *state.mesh);
					const int boundary_id = state.mesh->get_boundary_id(primitive_global_id);

					if (!std::count(boundary_ids.begin(), boundary_ids.end(), boundary_id))
						continue;

					for (long n = 0; n < nodes.size(); ++n)
					{
						const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
						is_on_surface[v.global[0].index] = true;
						if (v.global[0].index >= node_positions.rows())
							log_and_throw_adjoint_error("Error building collision mesh in ProxyContactForceForm!");
						boundary_ids_to_dof[boundary_id].insert(v.global[0].index);
					}
				}
			}

			if (!displacement_map_entries.empty())
			{
				displacement_map.resize(node_positions.rows(), state.n_bases);
				displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
			}

			// Fix boundary edges and boundary triangles to exclude vertices not on triangles
			Eigen::MatrixXi boundary_edges_alt(0, 2), boundary_triangles_alt(0, 3);
			{
				for (int i = 0; i < boundary_edges.rows(); ++i)
				{
					bool on_surface = true;
					for (int j = 0; j < boundary_edges.cols(); ++j)
						on_surface &= is_on_surface[boundary_edges(i, j)];
					if (on_surface)
					{
						boundary_edges_alt.conservativeResize(boundary_edges_alt.rows() + 1, 2);
						boundary_edges_alt.row(boundary_edges_alt.rows() - 1) = boundary_edges.row(i);
					}
				}

				if (state.mesh->is_volume())
				{
					for (int i = 0; i < boundary_triangles.rows(); ++i)
					{
						bool on_surface = true;
						for (int j = 0; j < boundary_triangles.cols(); ++j)
							on_surface &= is_on_surface[boundary_triangles(i, j)];
						if (on_surface)
						{
							boundary_triangles_alt.conservativeResize(boundary_triangles_alt.rows() + 1, 3);
							boundary_triangles_alt.row(boundary_triangles_alt.rows() - 1) = boundary_triangles.row(i);
						}
					}
				}
				else
					boundary_triangles_alt.resize(0, 0);
			}
			boundary_edges = boundary_edges_alt;
			boundary_triangles = boundary_triangles_alt;

			can_collide_cache.resize(0, 0);
			can_collide_cache.resize(collision_mesh.num_vertices(), collision_mesh.num_vertices());
			for (int i = 0; i < can_collide_cache.rows(); ++i)
			{
				int dof_idx_i = collision_mesh.to_full_vertex_id(i);
				if (!is_on_surface[dof_idx_i])
					continue;
				for (int j = 0; j < can_collide_cache.cols(); ++j)
				{
					int dof_idx_j = collision_mesh.to_full_vertex_id(j);
					if (!is_on_surface[dof_idx_j])
						continue;

					bool collision_allowed = true;
					for (const auto &id : boundary_ids)
						if (boundary_ids_to_dof[id].count(dof_idx_i) && boundary_ids_to_dof[id].count(dof_idx_j))
							collision_allowed = false;
					can_collide_cache(i, j) = collision_allowed;
				}
			}
		}

		double compute_quantity(
			const Eigen::MatrixXd &node_positions,
			const Eigen::MatrixXi &boundary_edges,
			const Eigen::MatrixXi &boundary_triangles,
			const Eigen::SparseMatrix<double> &displacement_map,
			const std::vector<bool> &is_on_surface,
			const Eigen::MatrixXi &can_collide_cache,
			const double dhat,
			const double dmin,
			std::function<ipc::Collisions(const Eigen::MatrixXd &)> cs_func,
			const Eigen::MatrixXd &u,
			const ipc::BarrierPotential &barrier_potential)

		{
			static int count = 0;
			std::cout << "computing " << ((double)count / 2) / u.size() * 100. << std::endl;
			++count;
			ipc::CollisionMesh collision_mesh = ipc::CollisionMesh(is_on_surface,
																   node_positions,
																   boundary_edges,
																   boundary_triangles,
																   displacement_map);

			collision_mesh.can_collide = [&can_collide_cache](size_t vi, size_t vj) {
				// return true;
				return (bool)can_collide_cache(vi, vj);
			};

			collision_mesh.init_area_jacobians();

			Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(u, collision_mesh.dim()));

			ipc::Collisions cs_ = cs_func(displaced_surface);
			cs_.build(collision_mesh, displaced_surface, dhat, dmin, ipc::BroadPhaseMethod::HASH_GRID);

			Eigen::MatrixXd forces = collision_mesh.to_full_dof(barrier_potential.gradient(cs_, collision_mesh, displaced_surface));

			double sum = (forces.array() * forces.array()).sum();

			return sum;
		}

	} // namespace

	typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

	// IntegrableFunctional TractionNormForm::get_integral_functional() const
	// {
	// 	IntegrableFunctional j;

	// 	const std::string formulation = state_.formulation();
	// 	const int power = in_power_;

	// 	if (formulation == "Laplacian")
	// 		log_and_throw_error("TractionNormForm is not implemented for Laplacian!");

	// 	j.set_j([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(grad_u.rows(), 1);
	// 		int el_id = params["elem"];
	// 		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0;
	// 		const double t = params["t"];

	// 		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u_local, grad_x;
	// 		Eigen::MatrixXd grad_u_q, stress, traction_force, grad_unused;
	// 		Eigen::MatrixXd reference_normal, displaced_normal;
	// 		for (int q = 0; q < grad_u.rows(); q++)
	// 		{
	// 			grad_x = vals.jac_it[q].inverse();
	// 			vector2matrix(grad_u.row(q), grad_u_q);
	// 			grad_u_local = grad_u_q * grad_x;

	// 			reference_normal = reference_normals.row(q);
	// 			displaced_normal = compute_displaced_normal(reference_normal, grad_x, grad_u_local).transpose();
	// 			// state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, Eigen::MatrixXd::Zero(1, grad_u_q.cols()), stress, grad_unused);
	// 			state.assembler->compute_stress_grad_multiply_vect(OptAssemblerData(t, dt, el_id, local_pts.row(q), pts.row(q), grad_u_q), Eigen::MatrixXd::Zero(1, grad_u_q.cols()), stress, grad_unused);
	// 			traction_force = displaced_normal * stress;
	// 			val(q) = pow(traction_force.squaredNorm(), power / 2.);
	// 		}
	// 	});

	// 	auto dj_dgradx = [formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(grad_u.rows(), grad_u.cols());
	// 		int el_id = params["elem"];
	// 		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0;
	// 		const double t = params["t"];
	// 		const int dim = sqrt(grad_u.cols());

	// 		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u_local, grad_x;
	// 		Eigen::MatrixXd grad_u_q, stress, traction_force, grad_unused;
	// 		Eigen::MatrixXd reference_normal, displaced_normal;
	// 		for (int q = 0; q < grad_u.rows(); q++)
	// 		{
	// 			grad_x = vals.jac_it[q].inverse();
	// 			vector2matrix(grad_u.row(q), grad_u_q);
	// 			grad_u_local = grad_u_q * grad_x;

	// 			DiffScalarBase::setVariableCount(dim * dim);
	// 			Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_x_auto(dim, dim);
	// 			for (int i = 0; i < dim; i++)
	// 				for (int j = 0; j < dim; j++)
	// 					grad_x_auto(i, j) = Diff(i + j * dim, grad_x(i, j));

	// 			reference_normal = reference_normals.row(q);
	// 			auto n = compute_displaced_normal(reference_normal, grad_x_auto, grad_u_local);
	// 			displaced_normal.resize(1, dim);
	// 			for (int i = 0; i < dim; ++i)
	// 				displaced_normal(i) = n(i).getValue();
	// 			// state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, Eigen::MatrixXd::Zero(1, grad_u_q.cols()), stress, grad_unused);
	// 			state.assembler->compute_stress_grad_multiply_vect(OptAssemblerData(t, dt, el_id, local_pts.row(q), pts.row(q), grad_u_q), Eigen::MatrixXd::Zero(1, grad_u_q.cols()), stress, grad_unused);
	// 			traction_force = displaced_normal * stress;

	// 			const double coef = power * pow(traction_force.squaredNorm(), power / 2. - 1.);
	// 			for (int k = 0; k < dim; ++k)
	// 				for (int l = 0; l < dim; ++l)
	// 				{
	// 					double sum_j = 0;
	// 					for (int j = 0; j < dim; ++j)
	// 					{
	// 						double grad_mult_stress = 0;
	// 						for (int i = 0; i < dim; ++i)
	// 							grad_mult_stress *= n(i).getGradient()(k + l * dim) * stress(i, j);

	// 						sum_j += traction_force(j) * grad_mult_stress;
	// 					}

	// 					val(q, k * dim + l) = coef * sum_j;
	// 				}
	// 		}
	// 	};
	// 	j.set_dj_dgradx(dj_dgradx);
	// 	j.set_dj_dgradu_local(dj_dgradx);

	// 	auto dj_dgradu = [formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(grad_u.rows(), grad_u.cols());
	// 		int el_id = params["elem"];
	// 		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0;
	// 		const double t = params["t"];
	// 		const int dim = sqrt(grad_u.cols());

	// 		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u_local, grad_x;
	// 		Eigen::MatrixXd grad_u_q, stress, traction_force, vect_mult_dstress;
	// 		Eigen::MatrixXd reference_normal, displaced_normal;
	// 		for (int q = 0; q < grad_u.rows(); q++)
	// 		{
	// 			grad_x = vals.jac_it[q].inverse();
	// 			vector2matrix(grad_u.row(q), grad_u_q);
	// 			grad_u_local = grad_u_q * grad_x;

	// 			reference_normal = reference_normals.row(q);
	// 			displaced_normal = compute_displaced_normal(reference_normal, grad_x, grad_u_local).transpose();
	// 			// state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, displaced_normal, stress, vect_mult_dstress);
	// 			state.assembler->compute_stress_grad_multiply_vect(OptAssemblerData(t, dt, el_id, local_pts.row(q), pts.row(q), grad_u_q), displaced_normal, stress, vect_mult_dstress);
	// 			traction_force = displaced_normal * stress;

	// 			const double coef = power * pow(traction_force.squaredNorm(), power / 2. - 1.);
	// 			for (int k = 0; k < dim; ++k)
	// 				for (int l = 0; l < dim; ++l)
	// 				{
	// 					double sum_j = 0;
	// 					for (int j = 0; j < dim; ++j)
	// 						sum_j += traction_force(j) * vect_mult_dstress(l * dim + k, j);

	// 					val(q, k * dim + l) = coef * sum_j;
	// 				}
	// 		}
	// 	};
	// 	j.set_dj_dgradu(dj_dgradu);

	// 	/*
	// 	j.set_dj_dgradu([formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(grad_u.rows(), grad_u.cols());
	// 		int el_id = params["elem"];
	// 		const int dim = sqrt(grad_u.cols());

	// 		Eigen::MatrixXd displaced_normals;
	// 		compute_displaced_normals(reference_normals, vals, grad_u, displaced_normals);

	// 		Eigen::MatrixXd grad_u_q, stress, traction_force, normal_dstress;
	// 		for (int q = 0; q < grad_u.rows(); q++)
	// 		{
	// 			vector2matrix(grad_u.row(q), grad_u_q);
	// 			state.assembler->compute_stress_grad_multiply_vect(el_id, local_pts.row(q), pts.row(q), grad_u_q, displaced_normals.row(q), stress, normal_dstress);
	// 			traction_force = displaced_normals.row(q) * stress;

	// 			const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
	// 			for (int i = 0; i < dim; i++)
	// 				for (int l = 0; l < dim; l++)
	// 					val(q, i * dim + l) = coef * (traction_force.array() * normal_dstress.row(i * dim + l).array()).sum();
	// 		}
	// 	});

	// 	auto dj_du = [formulation, power, &state = std::as_const(state_)](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(u.rows(), u.cols());
	// 		int el_id = params["elem"];
	// 		const int dim = sqrt(grad_u.cols());

	// 		Eigen::MatrixXd displaced_normals;
	// 		std::vector<Eigen::MatrixXd> normal_jacobian;
	// 		compute_displaced_normal_jacobian(reference_normals, vals, grad_u, displaced_normals, normal_jacobian);

	// 		Eigen::MatrixXd grad_u_q, stress, normal_duT_stress, traction_force, grad_unused;
	// 		for (int q = 0; q < u.rows(); q++)
	// 		{
	// 			vector2matrix(grad_u.row(q), grad_u_q);
	// 			state.assembler->compute_stress_grad_multiply_mat(el_id, local_pts.row(q), pts.row(q), grad_u_q, Eigen::MatrixXd::Zero(grad_u_q.rows(), grad_u_q.cols()), stress, grad_unused);
	// 			traction_force = displaced_normals.row(q) * stress;

	// 			Eigen::MatrixXd normal_du = normal_jacobian[q]; // compute this
	// 			normal_duT_stress = normal_du.transpose() * stress;

	// 			const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
	// 			for (int i = 0; i < dim; i++)
	// 				val(q, i) = coef * (traction_force.array() * normal_duT_stress.row(i).array()).sum();
	// 		}
	// 	};
	// 	j.set_dj_du(dj_du);
	// 	j.set_dj_dx(dj_du);
	// 	*/

	// 	/*
	// 	const int normal_dim = 0;

	// 	auto normal = [normal_dim](const Eigen::MatrixXd &reference_normal, const Eigen::MatrixXd &grad_x, const Eigen::MatrixXd &grad_u_local) {
	// 		Eigen::MatrixXd trafo = grad_x + grad_u_local;
	// 		Eigen::MatrixXd n = reference_normal * trafo.inverse();
	// 		n.normalize();

	// 		return n(normal_dim);
	// 	};

	// 	j.set_j([formulation, power, normal, normal_dim](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(grad_u.rows(), 1);
	// 		int el_id = params["elem"];

	// 		Eigen::MatrixXd grad_x, grad_u_local;
	// 		for (int q = 0; q < grad_u.rows(); q++)
	// 		{
	// 			grad_x = vals.jac_it[q].inverse();
	// 			vector2matrix(grad_u.row(q), grad_u_local);
	// 			grad_u_local = grad_u_local * grad_x;
	// 			double v = normal(reference_normals.row(q), grad_x, grad_u_local);
	// 			val(q) = pow(v, 2);
	// 		}
	// 	});

	// 	auto dj_dgradx = [formulation, power, normal](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(grad_u.rows(), grad_u.cols());
	// 		int el_id = params["elem"];
	// 		const int dim = sqrt(grad_u.cols());

	// 		Eigen::MatrixXd grad_x, grad_u_local;
	// 		for (int q = 0; q < grad_u.rows(); q++)
	// 		{
	// 			grad_x = vals.jac_it[q].inverse();
	// 			vector2matrix(grad_u.row(q), grad_u_local);
	// 			grad_u_local = grad_u_local * grad_x;

	// 			const double v = normal(reference_normals.row(q), grad_x, grad_u_local);

	// 			double eps = 1e-7;
	// 			for (int i = 0; i < dim; ++i)
	// 				for (int j = 0; j < dim; ++j)
	// 				{
	// 					Eigen::MatrixXd grad_x_copy = grad_x;
	// 					grad_x_copy(i, j) += eps;
	// 					const double val_plus = pow(normal(reference_normals.row(q), grad_x_copy, grad_u_local), 2);
	// 					grad_x_copy(i, j) -= 2 * eps;
	// 					const double val_minus = pow(normal(reference_normals.row(q), grad_x_copy, grad_u_local), 2);

	// 					const double fd = (val_plus - val_minus) / (2 * eps);
	// 					val(q, i * dim + j) = fd;
	// 				}
	// 		}
	// 	};

	// 	j.set_dj_dgradx(dj_dgradx);
	// 	j.set_dj_dgradu_local(dj_dgradx);
	// 	*/

	// 	// auto j_func = [formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 	// 	val.setZero(grad_u.rows(), 1);
	// 	// 	int el_id = params["elem"];

	// 	// 	Eigen::MatrixXd grad_u_q, stress, traction_force, grad_unused;
	// 	// 	for (int q = 0; q < grad_u.rows(); q++)
	// 	// 	{
	// 	// 		vector2matrix(grad_u.row(q), grad_u_q);
	// 	// 		double value = (grad_u_q * vals.jac_it[q].inverse() + vals.jac_it[q].inverse()).squaredNorm();
	// 	// 		val(q) = pow(value, power / 2.);
	// 	// 	}
	// 	// };
	// 	// j.set_j(j_func);

	// 	// auto dj_dgradx = [formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const Eigen::MatrixXd &reference_normals, const assembler::ElementAssemblyValues &vals, const json &params, Eigen::MatrixXd &val) {
	// 	// 	val.setZero(grad_u.rows(), grad_u.cols());
	// 	// 	int el_id = params["elem"];
	// 	// 	const int dim = sqrt(grad_u.cols());

	// 	// 	Eigen::MatrixXd grad_u_q;
	// 	// 	for (int q = 0; q < grad_u.rows(); q++)
	// 	// 	{
	// 	// 		vector2matrix(grad_u.row(q), grad_u_q);

	// 	// 		Eigen::MatrixXd grad = (grad_u_q * vals.jac_it[q].inverse() + vals.jac_it[q].inverse());
	// 	// 		const double coef = power * pow(grad.squaredNorm(), power / 2. - 1.);
	// 	// 		val.row(q) = coef * flatten(grad);
	// 	// 	}
	// 	// };
	// 	// j.set_dj_dgradx(dj_dgradx);
	// 	// j.set_dj_dgradu_local(dj_dgradx);

	// 	return j;
	// }

	// void TrueContactForceForm::build_active_nodes()
	// {

	// 	std::set<int> active_nodes_set = {};
	// 	dim_ = state_.mesh->dimension();
	// 	for (const auto &lb : state_.total_local_boundary)
	// 	{
	// 		const int e = lb.element_id();
	// 		const basis::ElementBases &bs = state_.bases[e];

	// 		for (int i = 0; i < lb.size(); i++)
	// 		{
	// 			const int global_primitive_id = lb.global_primitive_id(i);
	// 			const auto nodes = bs.local_nodes_for_primitive(global_primitive_id, *state_.mesh);
	// 			if (ids_.size() != 0 && ids_.find(state_.mesh->get_boundary_id(global_primitive_id)) == ids_.end())
	// 				continue;

	// 			for (long n = 0; n < nodes.size(); ++n)
	// 			{
	// 				const auto &b = bs.bases[nodes(n)];
	// 				const int index = b.global()[0].index;

	// 				for (int d = 0; d < dim_; ++d)
	// 					active_nodes_set.insert(index * dim_ + d);
	// 			}
	// 		}
	// 	}

	// 	active_nodes_.resize(active_nodes_set.size());
	// 	active_nodes_mat_.resize(state_.collision_mesh.full_ndof(), active_nodes_set.size());
	// 	std::vector<Eigen::Triplet<double>> active_nodes_i;
	// 	int count = 0;
	// 	for (const auto node : active_nodes_set)
	// 	{
	// 		active_nodes_i.emplace_back(node, count, 1.0);
	// 		active_nodes_(count++) = node;
	// 	}
	// 	active_nodes_mat_.setFromTriplets(active_nodes_i.begin(), active_nodes_i.end());

	// 	epsv_ = state_.args["contact"]["epsv"];
	// 	dhat_ = state_.args["contact"]["dhat"];
	// 	friction_coefficient_ = state_.args["contact"]["friction_coefficient"];
	//  static_friction_coefficient_ = state.args["contact"].contains("static_friction_coefficient") ? state.args["contact"]["static_friction_coefficient"]: friction_coefficient_;
	//  static_friction_coefficient_ = state.args["contact"].contains("kinetic_friction_coefficient") ? state.args["contact"]["kinetic_friction_coefficient"] : friction_coefficient_;
	// 	depends_on_step_prev_ = (friction_coefficient_ > 0);
	// }

	// double TrueContactForceForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	// {
	// 	assert(state_.solve_data.time_integrator != nullptr);
	// 	assert(state_.solve_data.contact_form != nullptr);

	// 	double barrier_stiffness = state_.solve_data.contact_form->weight();

	// 	const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
	// 	Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

	// 	Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
	// 	forces -= barrier_stiffness
	// 			  * state_.solve_data.contact_form->barrier_potential().gradient(
	// 				  state_.diff_cached.collision_set(time_step), collision_mesh, displaced_surface);
	// 	if (state_.solve_data.friction_form && time_step > 0)
	// 	{
	// 		Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim()));
	// 		Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim()));
	// 		const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / state_.solve_data.time_integrator->dt();

	// 		forces += state_.solve_data.friction_form->friction_potential().force(
	// 			state_.diff_cached.friction_collision_set(time_step),
	// 			collision_mesh,
	// 			collision_mesh.rest_positions(),
	// 			surface_solution_prev,
	// 			surface_velocities,
	// 			state_.solve_data.contact_form->barrier_potential(),
	// 			state_.solve_data.contact_form->barrier_stiffness(),
	// 			state_.solve_data.friction_form->epsv());
	// 	}
	// 	forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

	// 	double sum = (forces.array() * forces.array()).sum();

	// 	return sum;
	// }

	// Eigen::VectorXd TrueContactForceForm::compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	// {
	// 	assert(state_.solve_data.time_integrator != nullptr);
	// 	assert(state_.solve_data.contact_form != nullptr);

	// 	double barrier_stiffness = state_.solve_data.contact_form->weight();

	// 	const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
	// 	Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

	// 	Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
	// 	forces -= barrier_stiffness
	// 			  * state_.solve_data.contact_form->barrier_potential().gradient(
	// 				  state_.diff_cached.collision_set(time_step), collision_mesh, displaced_surface);
	// 	if (state_.solve_data.friction_form && time_step > 0)
	// 	{
	// 		Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim()));
	// 		Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim()));
	// 		const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / state_.solve_data.time_integrator->dt();

	// 		forces += state_.solve_data.friction_form->friction_potential().force(
	// 			state_.diff_cached.friction_collision_set(time_step),
	// 			collision_mesh,
	// 			collision_mesh.rest_positions(),
	// 			surface_solution_prev,
	// 			surface_velocities,
	// 			state_.solve_data.contact_form->barrier_potential(),
	// 			state_.solve_data.contact_form->barrier_stiffness(),
	// 			state_.solve_data.friction_form->epsv());
	// 	}
	// 	forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

	// 	StiffnessMatrix hessian(collision_mesh.ndof(), collision_mesh.ndof());
	// 	hessian -= barrier_stiffness
	// 			   * state_.solve_data.contact_form->barrier_potential().hessian(
	// 				   state_.diff_cached.collision_set(time_step), collision_mesh, displaced_surface, false);
	// 	if (state_.solve_data.friction_form && time_step > 0)
	// 	{
	// 		Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim()));
	// 		Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim()));
	// 		const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / state_.solve_data.time_integrator->dt();

	// 		const double dv_du = -1 / state_.solve_data.time_integrator->dt();
	// 		hessian += state_.solve_data.friction_form->friction_potential().force_jacobian(
	// 			state_.diff_cached.friction_collision_set(time_step),
	// 			collision_mesh,
	// 			collision_mesh.rest_positions(),
	// 			surface_solution_prev,
	// 			surface_velocities,
	// 			state_.solve_data.contact_form->barrier_potential(),
	// 			state_.solve_data.contact_form->barrier_stiffness(),
	// 			ipc::FrictionPotential::DiffWRT::VELOCITIES);
	// 	}
	// 	hessian = collision_mesh.to_full_dof(hessian);

	// 	Eigen::VectorXd gradu = 2 * hessian.transpose() * active_nodes_mat_ * forces;

	// 	return gradu;
	// }

	// Eigen::VectorXd TrueContactForceForm::compute_adjoint_rhs_unweighted_step_prev(const int time_step, const Eigen::VectorXd &x, const State &state) const
	// {
	// 	assert(state_.solve_data.time_integrator != nullptr);
	// 	assert(state_.solve_data.contact_form != nullptr);

	// 	double barrier_stiffness = state_.solve_data.contact_form->weight();

	// 	const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
	// 	Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

	// 	StiffnessMatrix hessian_prev(collision_mesh.ndof(), collision_mesh.ndof());
	// 	Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);

	// 	if (state_.solve_data.friction_form && time_step > 0)
	// 	{
	// 		Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim()));
	// 		Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim()));
	// 		const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / state_.solve_data.time_integrator->dt();

	// 		forces -= barrier_stiffness
	// 				  * state_.solve_data.contact_form->barrier_potential().gradient(
	// 					  state_.diff_cached.collision_set(time_step), collision_mesh, displaced_surface);
	// 		if (state_.solve_data.friction_form && time_step > 0)
	// 			forces += state_.solve_data.friction_form->friction_potential().force(
	// 				state_.diff_cached.friction_collision_set(time_step),
	// 				collision_mesh,
	// 				collision_mesh.rest_positions(),
	// 				surface_solution_prev,
	// 				surface_velocities,
	// 				state_.solve_data.contact_form->barrier_potential(),
	// 				state_.solve_data.contact_form->barrier_stiffness(),
	// 				state_.solve_data.friction_form->epsv());
	// 		forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

	// 		const double dv_du = -1 / state_.solve_data.time_integrator->dt();
	// 		hessian_prev += state_.solve_data.friction_form->friction_potential().force_jacobian(
	// 			state_.diff_cached.friction_collision_set(time_step),
	// 			collision_mesh,
	// 			collision_mesh.rest_positions(),
	// 			surface_solution_prev,
	// 			surface_velocities,
	// 			state_.solve_data.contact_form->barrier_potential(),
	// 			state_.solve_data.contact_form->barrier_stiffness(),
	// 			ipc::FrictionPotential::DiffWRT::LAGGED_DISPLACEMENTS);
	// 		hessian_prev += dv_du
	// 						* state_.solve_data.friction_form->friction_potential().force_jacobian(
	// 							state_.diff_cached.friction_collision_set(time_step),
	// 							collision_mesh,
	// 							collision_mesh.rest_positions(),
	// 							surface_solution_prev,
	// 							surface_velocities,
	// 							state_.solve_data.contact_form->barrier_potential(),
	// 							state_.solve_data.contact_form->barrier_stiffness(),
	// 							ipc::FrictionPotential::DiffWRT::VELOCITIES);
	// 	}
	// 	hessian_prev = collision_mesh.to_full_dof(hessian_prev);

	// 	Eigen::VectorXd gradu = 2 * hessian_prev.transpose() * active_nodes_mat_ * forces;

	// 	return gradu;
	// }

	// void TrueContactForceForm::compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	// {
	// 	assert(state_.solve_data.time_integrator != nullptr);
	// 	assert(state_.solve_data.contact_form != nullptr);

	// 	double barrier_stiffness = state_.solve_data.contact_form->weight();

	// 	const ipc::CollisionMesh &collision_mesh = state_.collision_mesh;
	// 	Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), dim_));

	// 	Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh.ndof(), 1);
	// 	forces -= barrier_stiffness
	// 			  * state_.solve_data.contact_form->barrier_potential().gradient(
	// 				  state_.diff_cached.collision_set(time_step), collision_mesh, displaced_surface);
	// 	if (state_.solve_data.friction_form && time_step > 0)
	// 	{
	// 		Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim()));
	// 		Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim()));
	// 		const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / state_.solve_data.time_integrator->dt();

	// 		forces += state_.solve_data.friction_form->friction_potential().force(
	// 			state_.diff_cached.friction_collision_set(time_step),
	// 			collision_mesh,
	// 			collision_mesh.rest_positions(),
	// 			surface_solution_prev,
	// 			surface_velocities,
	// 			state_.solve_data.contact_form->barrier_potential(),
	// 			state_.solve_data.contact_form->barrier_stiffness(),
	// 			state_.solve_data.friction_form->epsv());
	// 	}
	// 	forces = collision_mesh.to_full_dof(forces)(active_nodes_, Eigen::all);

	// 	StiffnessMatrix shape_derivative(collision_mesh.ndof(), collision_mesh.ndof());
	// 	shape_derivative -= barrier_stiffness
	// 						* state_.solve_data.contact_form->barrier_potential().shape_derivative(
	// 							state_.diff_cached.collision_set(time_step), collision_mesh, displaced_surface);
	// 	if (state_.solve_data.friction_form && time_step > 0)
	// 	{
	// 		Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step - 1), collision_mesh.dim()));
	// 		Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh.dim()));
	// 		const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / state_.solve_data.time_integrator->dt();

	// 		shape_derivative += state_.solve_data.friction_form->friction_potential().force_jacobian(
	// 			state_.diff_cached.friction_collision_set(time_step),
	// 			collision_mesh,
	// 			collision_mesh.rest_positions(),
	// 			surface_solution_prev,
	// 			surface_velocities,
	// 			state_.solve_data.contact_form->barrier_potential(),
	// 			state_.solve_data.contact_form->barrier_stiffness(),
	// 			ipc::FrictionPotential::DiffWRT::REST_POSITIONS);
	// 	}
	// 	shape_derivative = collision_mesh.to_full_dof(shape_derivative);

	// 	Eigen::VectorXd grads = 2 * shape_derivative.transpose() * active_nodes_mat_ * forces;
	// 	grads = state_.basis_nodes_to_gbasis_nodes * grads;
	// 	grads = AdjointTools::map_node_to_primitive_order(state_, grads);

	// 	gradv.setZero(x.size());

	// 	for (const auto &param_map : variable_to_simulations_)
	// 	{
	// 		const auto &param_type = param_map->get_parameter_type();

	// 		for (const auto &state : param_map->get_states())
	// 		{
	// 			if (state.get() != &state_)
	// 				continue;

	// 			if (param_type != ParameterType::Shape)
	// 				log_and_throw_error("Only support contact force derivative wrt. shape!");

	// 			if (grads.size() > 0)
	// 				gradv += param_map->apply_parametrization_jacobian(grads, x);
	// 		}
	// 	}
	// }

	ProxyContactForceForm::ProxyContactForceForm(
		const VariableToSimulationGroup &variable_to_simulations,
		const State &state,
		const double dhat,
		const bool quadratic_potential,
		const json &args)
		: StaticForm(variable_to_simulations),
		  state_(state),
		  dhat_(dhat),
		  dmin_(0),
		  barrier_potential_(dhat)
	{
		auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
		boundary_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		build_collision_mesh();

		broad_phase_method_ = ipc::BroadPhaseMethod::HASH_GRID;

		if (state.problem->is_time_dependent())
		{
			int time_steps = state.args["time"]["time_steps"].get<int>() + 1;
			collision_set_indicator_.setZero(time_steps);
			for (int i = 0; i < time_steps + 1; ++i)
			{
				collision_sets_.push_back(std::make_shared<ipc::Collisions>());
				collision_sets_.back()->set_use_convergent_formulation(true);
				collision_sets_.back()->set_are_shape_derivatives_enabled(true);
			}
		}
		else
		{
			collision_set_indicator_.setZero(1);
			collision_sets_.push_back(std::make_shared<ipc::Collisions>());
			collision_sets_.back()->set_use_convergent_formulation(true);
			collision_sets_.back()->set_are_shape_derivatives_enabled(true);
		}

		if (quadratic_potential)
			barrier_potential_.set_barrier(std::make_shared<QuadraticBarrier>());
	}

	void ProxyContactForceForm::build_collision_mesh()
	{
		boundary_ids_to_dof_.clear();
		can_collide_cache_.resize(0, 0);
		node_positions_.resize(0, 0);

		// Eigen::MatrixXd node_positions;
		Eigen::MatrixXi boundary_edges, boundary_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;
		io::OutGeometryData::extract_boundary_mesh(*state_.mesh, state_.n_bases, state_.bases, state_.total_local_boundary,
												   node_positions_, boundary_edges, boundary_triangles, displacement_map_entries);

		std::vector<bool> is_on_surface;
		is_on_surface.resize(node_positions_.rows(), false);

		assembler::ElementAssemblyValues vals;
		Eigen::MatrixXd points, uv, normals;
		Eigen::VectorXd weights;
		Eigen::VectorXi global_primitive_ids;
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, state_.n_boundary_samples(), *state_.mesh, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const basis::ElementBases &bs = state_.bases[e];
			const basis::ElementBases &gbs = state_.geom_bases()[e];

			vals.compute(e, state_.mesh->is_volume(), points, bs, gbs);

			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, *state_.mesh);
				const int boundary_id = state_.mesh->get_boundary_id(primitive_global_id);

				if (!std::count(boundary_ids_.begin(), boundary_ids_.end(), boundary_id))
					continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
					is_on_surface[v.global[0].index] = true;
					if (v.global[0].index >= node_positions_.rows())
						log_and_throw_adjoint_error("Error building collision mesh in ProxyContactForceForm!");
					boundary_ids_to_dof_[boundary_id].insert(v.global[0].index);
				}
			}
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(node_positions_.rows(), state_.n_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		// Fix boundary edges and boundary triangles to exclude vertices not on triangles
		Eigen::MatrixXi boundary_edges_alt(0, 2), boundary_triangles_alt(0, 3);
		{
			for (int i = 0; i < boundary_edges.rows(); ++i)
			{
				bool on_surface = true;
				for (int j = 0; j < boundary_edges.cols(); ++j)
					on_surface &= is_on_surface[boundary_edges(i, j)];
				if (on_surface)
				{
					boundary_edges_alt.conservativeResize(boundary_edges_alt.rows() + 1, 2);
					boundary_edges_alt.row(boundary_edges_alt.rows() - 1) = boundary_edges.row(i);
				}
			}

			if (state_.mesh->is_volume())
			{
				for (int i = 0; i < boundary_triangles.rows(); ++i)
				{
					bool on_surface = true;
					for (int j = 0; j < boundary_triangles.cols(); ++j)
						on_surface &= is_on_surface[boundary_triangles(i, j)];
					if (on_surface)
					{
						boundary_triangles_alt.conservativeResize(boundary_triangles_alt.rows() + 1, 3);
						boundary_triangles_alt.row(boundary_triangles_alt.rows() - 1) = boundary_triangles.row(i);
					}
				}
			}
			else
				boundary_triangles_alt.resize(0, 0);
		}

		collision_mesh_ = ipc::CollisionMesh(is_on_surface,
											 node_positions_,
											 boundary_edges_alt,
											 boundary_triangles_alt,
											 displacement_map);

		can_collide_cache_.resize(collision_mesh_.num_vertices(), collision_mesh_.num_vertices());
		for (int i = 0; i < can_collide_cache_.rows(); ++i)
		{
			int dof_idx_i = collision_mesh_.to_full_vertex_id(i);
			if (!is_on_surface[dof_idx_i])
				continue;
			for (int j = 0; j < can_collide_cache_.cols(); ++j)
			{
				int dof_idx_j = collision_mesh_.to_full_vertex_id(j);
				if (!is_on_surface[dof_idx_j])
					continue;

				bool collision_allowed = true;
				for (const auto &id : boundary_ids_)
					if (boundary_ids_to_dof_[id].count(dof_idx_i) && boundary_ids_to_dof_[id].count(dof_idx_j))
						collision_allowed = false;
				can_collide_cache_(i, j) = collision_allowed;
			}
		}

		collision_mesh_.can_collide = [this](size_t vi, size_t vj) {
			// return true;
			return (bool)can_collide_cache_(vi, vj);
		};

		collision_mesh_.init_area_jacobians();
	}

	const ipc::Collisions &ProxyContactForceForm::get_or_compute_collision_set(const int time_step, const Eigen::MatrixXd &displaced_surface) const
	{
		if (!collision_set_indicator_(time_step))
		{
			collision_sets_[time_step]->build(
				collision_mesh_, displaced_surface, dhat_, dmin_, broad_phase_method_);
			collision_set_indicator_(time_step) = 1;
		}
		return *collision_sets_[time_step];
	}

	double ProxyContactForceForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));
		auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

		Eigen::MatrixXd forces = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));

		double sum = (forces.array() * forces.array()).sum();

		return sum;
	}

	Eigen::VectorXd ProxyContactForceForm::compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));
		auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

		Eigen::MatrixXd forces = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));
		StiffnessMatrix hessian = collision_mesh_.to_full_dof(barrier_potential_.hessian(collision_set, collision_mesh_, displaced_surface, false));

		Eigen::VectorXd gradu = 2 * hessian.transpose() * forces;

		// std::cout << "u norm " << state_.diff_cached.u(time_step).norm() << std::endl;

		// Eigen::VectorXd G;
		// fd::finite_gradient(
		// 	state_.diff_cached.u(time_step), [this, time_step](const Eigen::VectorXd &x_) {
		// 		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(x_, collision_mesh_.dim()));

		// 		auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

		// 		Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(collision_mesh_.ndof(), 1);
		// 		forces -= barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface);
		// 		forces = collision_mesh_.to_full_dof(forces);

		// 		double sum = (forces.array() * forces.array()).sum();

		// 		return sum; },
		// 	G);

		// std::cout << "gradu difference norm " << (G - gradu).norm() << std::endl;

		return weight() * gradu;
	}

	void ProxyContactForceForm::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, time_step, &x]() {
			const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));

			auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

			Eigen::MatrixXd forces = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));

			StiffnessMatrix shape_derivative = collision_mesh_.to_full_dof(barrier_potential_.shape_derivative(collision_set, collision_mesh_, displaced_surface));

			Eigen::VectorXd grads = 2 * shape_derivative.transpose() * forces;

			// Eigen::VectorXd G;
			// fd::finite_gradient(
			// 	utils::flatten(node_positions_), [this, time_step](const Eigen::VectorXd &x_) {
			// 	Eigen::MatrixXd n;
			// 	Eigen::MatrixXi e, t;
			// 	Eigen::SparseMatrix<double> d_m;
			// 	std::vector<bool> i_o_s;
			// 	Eigen::MatrixXi c_c_c;
			// 	compute_forward_collision_mesh_quantities(state_, boundary_ids_, collision_mesh_, n, e, t, d_m, i_o_s, c_c_c);

			// 	return compute_forward_quantity(
			// 		utils::unflatten(x_, n.cols()),
			// 		e,
			// 		t,
			// 		d_m,
			// 		i_o_s,
			// 		c_c_c,
			// 		dhat_,
			// 		dmin_,
			// 		[this, time_step](const Eigen::MatrixXd &displaced_surface) {return get_or_compute_collision_set(time_step, displaced_surface);},
			// 		state_.diff_cached.u(time_step),
			// 		barrier_potential_); },
			// 	G, fd::AccuracyOrder::SECOND, 1e-7);

			// Eigen::MatrixXd diff(G.size(), 2);
			// diff.col(0) = G;
			// diff.col(1) = grads;
			// std::cout << "diff " << diff << std::endl;
			// std::cout << "size " << G.size() << " " << grads.size() << std::endl;
			// std::cout << "fd norm " << G.norm() << std::endl;
			// std::cout << "grads difference norm " << (G - grads).norm() / G.norm() << std::endl;

			grads = state_.basis_nodes_to_gbasis_nodes * grads;

			return AdjointTools::map_node_to_primitive_order(state_, grads);
		});
	}

} // namespace polyfem::solver