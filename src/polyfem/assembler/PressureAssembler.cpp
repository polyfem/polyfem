#include "PressureAssembler.hpp"

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/io/Evaluator.hpp>

namespace polyfem
{
	using namespace polysolve;
	using namespace mesh;
	using namespace quadrature;
	using namespace utils;

	namespace assembler
	{
		namespace
		{
			class LocalThreadScalarStorage
			{
			public:
				double val;

				LocalThreadScalarStorage()
				{
					val = 0;
				}
			};

			class LocalThreadVecStorage
			{
			public:
				Eigen::MatrixXd vec;

				LocalThreadVecStorage(const int size)
				{
					vec.resize(size, 1);
					vec.setZero();
				}
			};

			class LocalThreadMatStorage
			{
			public:
				std::vector<Eigen::Triplet<double>> entries;
			};

			Eigen::Matrix3d wedge_product(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
			{
				return a * b.transpose() - b * a.transpose();
			}

			void g_dual(const Eigen::Vector3d &g1, const Eigen::Vector3d &g2, Eigen::Vector3d &g1_up, Eigen::Vector3d &g2_up)
			{
				g1_up = (wedge_product(g1, g2) * g2) / (g1.dot(wedge_product(g1, g2) * g2));
				g2_up = (wedge_product(g2, g1) * g1) / (g2.dot(wedge_product(g2, g1) * g1));
			}
		} // namespace

		double PressureAssembler::compute_volume(const Eigen::MatrixXd &displacement, const std::vector<mesh::LocalBoundary> &local_boundary, const int resolution, const int boundary_id) const
		{
			double volume = 0;

			ElementAssemblyValues vals;
			Eigen::MatrixXd uv, samples, gtmp, rhs_fun, deform_mat, jac_mat, trafo, g_3;
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd points, normals;
			Eigen::VectorXd weights;

			for (const auto &lb : local_boundary)
			{
				const int e = lb.element_id();
				const basis::ElementBases &gbs = gbases_[e];
				const basis::ElementBases &bs = bases_[e];

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

					if (boundary_id != mesh_.get_boundary_id(primitive_global_id))
						continue;

					bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, i, false, uv, points, normals, weights);
					g_3.setZero(normals.rows(), normals.cols());

					if (!has_samples)
						continue;

					global_primitive_ids.setConstant(weights.size(), primitive_global_id);

					Eigen::MatrixXd reference_normals = normals;

					vals.compute(e, mesh_.is_volume(), points, bs, gbs);

					std::vector<std::vector<Eigen::MatrixXd>> grad_normal;
					for (int n = 0; n < vals.jac_it.size(); ++n)
					{
						trafo = vals.jac_it[n].inverse();

						if (displacement.size() > 0)
						{

							assert(size_ == 2 || size_ == 3);
							deform_mat.resize(size_, size_);
							deform_mat.setZero();
							jac_mat.resize(size_, vals.basis_values.size());
							int b_idx = 0;
							for (const auto &b : vals.basis_values)
							{
								jac_mat.col(b_idx++) = b.grad.row(n);

								for (const auto &g : b.global)
									for (int d = 0; d < size_; ++d)
										deform_mat.row(d) += displacement(g.index * size_ + d) * b.grad.row(n);
							}

							trafo += deform_mat;
						}

						Eigen::VectorXd displaced_normal = normals.row(n) * trafo.inverse();
						normals.row(n) = displaced_normal / displaced_normal.norm();

						Eigen::Vector3d g1, g2, g3;
						auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
						g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
						g2 = trafo * (endpoints.row(0) - endpoints.row(2)).transpose();
						if (lb[i] == 0)
							g1 *= -1;
						g3 = g1.cross(g2);
						g_3.row(n) = g3.transpose();
					}

					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(mesh_, problem_.is_scalar(), bases_, gbases_, e, points, displacement, u, grad_u);

					u += vals.val;

					for (long p = 0; p < weights.size(); ++p)
						for (int d = 0; d < size_; ++d)
							volume += g_3(p, d) * u(p, d) * weights(p);
				}
			}

			return (1. / 3.) * volume;
		}

		void PressureAssembler::compute_grad_volume(const Eigen::MatrixXd &displacement, const std::vector<mesh::LocalBoundary> &local_boundary, const int resolution, const int boundary_id, Eigen::MatrixXd &grad) const
		{
			grad.setZero(n_basis_ * size_, 1);

			ElementAssemblyValues vals;
			Eigen::MatrixXd uv, samples, gtmp, rhs_fun, deform_mat, jac_mat, trafo, g_3;
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd points, normals;
			Eigen::VectorXd weights;

			for (const auto &lb : local_boundary)
			{
				const int e = lb.element_id();
				const basis::ElementBases &gbs = gbases_[e];
				const basis::ElementBases &bs = bases_[e];

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

					if (boundary_id != mesh_.get_boundary_id(primitive_global_id))
						continue;

					bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, i, false, uv, points, normals, weights);
					g_3.setZero(normals.rows(), normals.cols());

					if (!has_samples)
						continue;

					global_primitive_ids.setConstant(weights.size(), primitive_global_id);

					Eigen::MatrixXd reference_normals = normals;

					vals.compute(e, mesh_.is_volume(), points, bs, gbs);

					std::vector<std::vector<Eigen::MatrixXd>> grad_normal;
					for (int n = 0; n < vals.jac_it.size(); ++n)
					{
						trafo = vals.jac_it[n].inverse();

						if (displacement.size() > 0)
						{

							assert(size_ == 2 || size_ == 3);
							deform_mat.resize(size_, size_);
							deform_mat.setZero();
							jac_mat.resize(size_, vals.basis_values.size());
							int b_idx = 0;
							for (const auto &b : vals.basis_values)
							{
								jac_mat.col(b_idx++) = b.grad.row(n);

								for (const auto &g : b.global)
									for (int d = 0; d < size_; ++d)
										deform_mat.row(d) += displacement(g.index * size_ + d) * b.grad.row(n);
							}

							trafo += deform_mat;
						}

						Eigen::VectorXd displaced_normal = normals.row(n) * trafo.inverse();
						normals.row(n) = displaced_normal / displaced_normal.norm();

						Eigen::Vector3d g1, g2, g3;
						auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
						g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
						g2 = trafo * (endpoints.row(0) - endpoints.row(2)).transpose();
						if (lb[i] == 0)
							g1 *= -1;
						g3 = g1.cross(g2);
						g_3.row(n) = g3.transpose();
					}

					for (long n = 0; n < nodes.size(); ++n)
					{
						const AssemblyValues &v = vals.basis_values[nodes(n)];
						for (int d = 0; d < size_; ++d)
							for (size_t g = 0; g < v.global.size(); ++g)
							{
								const int g_index = v.global[g].index * size_ + d;

								for (long p = 0; p < weights.size(); ++p)
									grad(g_index) += g_3(p, d) * v.val(p) * weights(p);
							}
					}
				}
			}
		}

		PressureAssembler::PressureAssembler(const Assembler &assembler, const Mesh &mesh, const Obstacle &obstacle,
											 const std::vector<mesh::LocalBoundary> &local_pressure_boundary, const std::vector<int> &primitive_to_nodes, const std::vector<int> &node_to_primitives,
											 const int n_basis, const int size,
											 const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases,
											 const Problem &problem)
			: assembler_(assembler),
			  mesh_(mesh),
			  obstacle_(obstacle),
			  n_basis_(n_basis),
			  size_(size),
			  bases_(bases),
			  gbases_(gbases),
			  problem_(problem),
			  primitive_to_nodes_(primitive_to_nodes),
			  node_to_primitives_(node_to_primitives)
		{
			if (size_ != 3 && (local_pressure_boundary.size() > 0))
				log_and_throw_error("Pressure is only supported for 3d!");
		}

		double PressureAssembler::compute_energy(
			const Eigen::MatrixXd &displacement,
			const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
			const int resolution,
			const double t) const
		{
			double res = 0;

			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

			utils::maybe_parallel_for(local_pressure_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd pressure_vals, g_3;
				ElementAssemblyValues vals;
				Eigen::MatrixXd points, uv, normals, deform_mat, trafo;
				Eigen::VectorXd weights;
				Eigen::VectorXi global_primitive_ids;
				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = local_pressure_boundary[lb_id];
					const int e = lb.element_id();
					const basis::ElementBases &gbs = gbases_[e];
					const basis::ElementBases &bs = bases_[e];

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

						bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, i, false, uv, points, normals, weights);
						if (mesh_.is_volume())
							weights /= 2 * mesh_.tri_area(primitive_global_id);
						else
							weights /= mesh_.edge_length(primitive_global_id);
						g_3.setZero(normals.rows(), normals.cols());

						if (!has_samples)
							continue;

						global_primitive_ids.setConstant(weights.size(), primitive_global_id);

						vals.compute(e, mesh_.is_volume(), points, bs, gbs);

						for (int n = 0; n < vals.jac_it.size(); ++n)
						{
							trafo = vals.jac_it[n].inverse();

							if (displacement.size() > 0)
							{
								assert(size_ == 2 || size_ == 3);
								deform_mat.resize(size_, size_);
								deform_mat.setZero();
								for (const auto &b : vals.basis_values)
								{
									for (const auto &g : b.global)
									{
										for (int d = 0; d < size_; ++d)
										{
											deform_mat.row(d) += displacement(g.index * size_ + d) * b.grad.row(n);
										}
									}
								}

								trafo += deform_mat;
							}

							normals.row(n) = normals.row(n) * trafo.inverse();
							normals.row(n).normalize();

							if (mesh_.is_volume())
							{
								Eigen::Vector3d g1, g2, g3;
								auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
								g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
								g2 = trafo * (endpoints.row(0) - endpoints.row(2)).transpose();
								if (lb[i] == 0)
									g1 *= -1;
								g3 = g1.cross(g2);
								g_3.row(n) = g3.transpose();
							}
							else
							{
								assert(false);
								// Eigen::Vector2d g1, g3;
								// auto endpoints = utils::BoundarySampler::tri_local_node_coordinates_from_edge(lb[i]);
								// g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
								// g3(0) = -g1(0);
								// g3(1) = g1(1);
								// g_3.row(n) = g3.transpose();
							}
						}
						problem_.pressure_bc(mesh_, global_primitive_ids, uv, vals.val, normals, t, pressure_vals);

						Eigen::MatrixXd u, grad_u;
						io::Evaluator::interpolate_at_local_vals(mesh_, problem_.is_scalar(), bases_, gbases_, e, points, displacement, u, grad_u);
						u += vals.val;

						for (long p = 0; p < weights.size(); ++p)
							for (int d = 0; d < size_; ++d)
								local_storage.val += pressure_vals(p) * g_3(p, d) * u(p, d) * weights(p);
					}
				}
			});

			for (const LocalThreadScalarStorage &local_storage : storage)
				res += local_storage.val;

			res *= 1. / size_;

			return res;
		}

		void PressureAssembler::compute_energy_grad(
			const Eigen::MatrixXd &displacement,
			const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
			const std::vector<int> dirichlet_nodes,
			const int resolution,
			const double t,
			Eigen::VectorXd &grad) const
		{
			grad.setZero(n_basis_ * size_);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(grad.size()));

			utils::maybe_parallel_for(local_pressure_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd pressure_vals, g_3;
				ElementAssemblyValues vals;
				Eigen::MatrixXd points, uv, normals, deform_mat, trafo;
				Eigen::VectorXd weights;
				Eigen::VectorXi global_primitive_ids;
				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = local_pressure_boundary[lb_id];
					const int e = lb.element_id();
					const basis::ElementBases &gbs = gbases_[e];
					const basis::ElementBases &bs = bases_[e];

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

						bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, i, false, uv, points, normals, weights);
						if (mesh_.is_volume())
							weights /= 2 * mesh_.tri_area(primitive_global_id);
						else
							weights /= mesh_.edge_length(primitive_global_id);
						g_3.setZero(normals.rows(), normals.cols());

						if (!has_samples)
							continue;

						global_primitive_ids.setConstant(weights.size(), primitive_global_id);

						vals.compute(e, mesh_.is_volume(), points, bs, gbs);
						for (int n = 0; n < vals.jac_it.size(); ++n)
						{
							trafo = vals.jac_it[n].inverse();

							if (displacement.size() > 0)
							{
								assert(size_ == 2 || size_ == 3);
								deform_mat.resize(size_, size_);
								deform_mat.setZero();
								for (const auto &b : vals.basis_values)
								{
									for (const auto &g : b.global)
									{
										for (int d = 0; d < size_; ++d)
										{
											deform_mat.row(d) += displacement(g.index * size_ + d) * b.grad.row(n);
										}
									}
								}

								trafo += deform_mat;
							}

							normals.row(n) = normals.row(n) * trafo.inverse();
							normals.row(n).normalize();

							if (mesh_.is_volume())
							{
								Eigen::Vector3d g1, g2, g3;
								auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
								g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
								g2 = trafo * (endpoints.row(0) - endpoints.row(2)).transpose();
								if (lb[i] == 0)
									g1 *= -1;
								g3 = g1.cross(g2);
								g_3.row(n) = g3.transpose();
							}
							else
							{
								assert(false);
								// Eigen::Vector2d g1, g3;
								// auto endpoints = utils::BoundarySampler::tri_local_node_coordinates_from_edge(lb[i]);
								// g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
								// g3(0) = -g1(0);
								// g3(1) = g1(1);
								// g_3.row(n) = g3.transpose();
							}
						}
						problem_.pressure_bc(mesh_, global_primitive_ids, uv, vals.val, normals, t, pressure_vals);

						for (long n = 0; n < nodes.size(); ++n)
						{
							const AssemblyValues &v = vals.basis_values[nodes(n)];
							for (int d = 0; d < size_; ++d)
							{
								for (size_t g = 0; g < v.global.size(); ++g)
								{
									const int g_index = v.global[g].index * size_ + d;
									const bool is_dof_dirichlet = std::find(dirichlet_nodes.begin(), dirichlet_nodes.end(), g_index) != dirichlet_nodes.end();
									if (is_dof_dirichlet)
										continue;

									for (long p = 0; p < weights.size(); ++p)
									{
										// grad(g_index) += pressure_vals(p) * g_3(p, d) * v.val(p) * weights(p);
										local_storage.vec(g_index) += pressure_vals(p) * g_3(p, d) * v.val(p) * weights(p);
									}
								}
							}
						}
					}
				}
			});
			for (const LocalThreadVecStorage &local_storage : storage)
				grad += local_storage.vec;
		}

		void PressureAssembler::compute_energy_hess(
			const Eigen::MatrixXd &displacement,
			const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
			const std::vector<int> dirichlet_nodes,
			const int resolution,
			const double t,
			const bool project_to_psd,
			StiffnessMatrix &hess) const
		{
			hess.resize(n_basis_ * size_, n_basis_ * size_);

			auto storage = create_thread_storage(LocalThreadMatStorage());

			utils::maybe_parallel_for(local_pressure_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadMatStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd pressure_vals, g_1, g_2, g_3, local_hessian;
				ElementAssemblyValues vals;
				Eigen::MatrixXd points, uv, normals, deform_mat, trafo;
				Eigen::VectorXd weights;
				Eigen::VectorXi global_primitive_ids;
				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = local_pressure_boundary[lb_id];
					const int e = lb.element_id();
					const basis::ElementBases &gbs = gbases_[e];
					const basis::ElementBases &bs = bases_[e];

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

						bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, i, false, uv, points, normals, weights);
						std::vector<Eigen::VectorXd> param_chain_rule;
						if (mesh_.is_volume())
						{
							weights /= 2 * mesh_.tri_area(primitive_global_id);
							g_1.setZero(normals.rows(), normals.cols());
							g_2.setZero(normals.rows(), normals.cols());
							g_3.setZero(normals.rows(), normals.cols());

							auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
							param_chain_rule = {(endpoints.row(0) - endpoints.row(1)).transpose(), (endpoints.row(0) - endpoints.row(2)).transpose()};
							if (lb[i] == 0)
								param_chain_rule[0] *= -1;
						}
						else
						{
							assert(false);
							// 	weights /= mesh_.edge_length(primitive_global_id);
							// 	g_1.setZero(normals.rows(), normals.cols());
							// 	g_2.setZero(0, 0);
							// 	g_3.setZero(normals.rows(), normals.cols());

							// 	auto endpoints = utils::BoundarySampler::tri_local_node_coordinates_from_edge(lb[i]);
							// 	param_chain_rule = {(-endpoints.row(0) - endpoints.row(1)).transpose()};
						}

						if (!has_samples)
							continue;

						global_primitive_ids.setConstant(weights.size(), primitive_global_id);

						vals.compute(e, mesh_.is_volume(), points, bs, gbs);
						for (int n = 0; n < vals.jac_it.size(); ++n)
						{
							trafo = vals.jac_it[n].inverse();

							if (displacement.size() > 0)
							{
								assert(size_ == 2 || size_ == 3);
								deform_mat.resize(size_, size_);
								deform_mat.setZero();
								for (const auto &b : vals.basis_values)
								{
									for (const auto &g : b.global)
									{
										for (int d = 0; d < size_; ++d)
										{
											deform_mat.row(d) += displacement(g.index * size_ + d) * b.grad.row(n);
										}
									}
								}

								trafo += deform_mat;
							}

							normals.row(n) = normals.row(n) * trafo.inverse();
							normals.row(n).normalize();

							if (mesh_.is_volume())
							{
								Eigen::Vector3d g1, g2, g3;
								auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
								g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
								g2 = trafo * (endpoints.row(0) - endpoints.row(2)).transpose();
								if (lb[i] == 0)
									g1 *= -1;
								g3 = g1.cross(g2);

								g_1.row(n) = g1.transpose();
								g_2.row(n) = g2.transpose();
								g_3.row(n) = g3.transpose();
							}
							else
							{
								assert(false);
								// Eigen::Vector2d g1, g3;
								// auto endpoints = utils::BoundarySampler::tri_local_node_coordinates_from_edge(lb[i]);
								// g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
								// g3(0) = -g1(0);
								// g3(1) = g1(1);

								// g_1.row(n) = g1.transpose();
								// g_3.row(n) = g3.transpose();
							}
						}
						problem_.pressure_bc(mesh_, global_primitive_ids, uv, vals.val, normals, t, pressure_vals);

						local_hessian.setZero(vals.basis_values.size() * size_, vals.basis_values.size() * size_);

						for (long p = 0; p < weights.size(); ++p)
						{
							Eigen::Vector3d g_up_1, g_up_2;
							g_dual(g_1.row(p).transpose(), g_2.row(p).transpose(), g_up_1, g_up_2);
							std::vector<Eigen::MatrixXd> g_3_wedge_g_up = {wedge_product(g_3.row(p), g_up_1), wedge_product(g_3.row(p), g_up_2)};

							for (long ni = 0; ni < nodes.size(); ++ni)
							{
								const AssemblyValues &vi = vals.basis_values[nodes(ni)];
								for (int di = 0; di < size_; ++di)
								{
									const int gi_index = vi.global[0].index * size_ + di;
									const bool is_dof_i_dirichlet = std::find(dirichlet_nodes.begin(), dirichlet_nodes.end(), gi_index) != dirichlet_nodes.end();
									if (is_dof_i_dirichlet)
										continue;

									Eigen::MatrixXd grad_phi_i;
									grad_phi_i.setZero(3, 3);
									grad_phi_i.row(di) = vi.grad.row(p);

									for (long nj = 0; nj < nodes.size(); ++nj)
									{
										const AssemblyValues &vj = vals.basis_values[nodes(nj)];
										for (int dj = 0; dj < size_; ++dj)
										{
											const int gj_index = vj.global[0].index * size_ + dj;
											const bool is_dof_j_dirichlet = std::find(dirichlet_nodes.begin(), dirichlet_nodes.end(), gj_index) != dirichlet_nodes.end();
											if (is_dof_j_dirichlet)
												continue;

											Eigen::MatrixXd grad_phi_j;
											grad_phi_j.setZero(3, 3);
											grad_phi_j.row(dj) = vj.grad.row(p);

											double value = 0;
											for (int alpha = 0; alpha < size_ - 1; ++alpha)
											{
												auto a = vj.val(p) * g_3_wedge_g_up[alpha].row(di) * (grad_phi_i * param_chain_rule[alpha]);
												auto b = (g_3_wedge_g_up[alpha] * (grad_phi_j * param_chain_rule[alpha])).row(di) * vi.val(p);
												value += pressure_vals(p) * (a(0) + b(0)) * weights(p);
											}
											local_hessian(nodes(ni) * size_ + di, nodes(nj) * size_ + dj) += value;
										}
									}
								}
							}
						}

						if (project_to_psd)
							local_hessian = ipc::project_to_psd(local_hessian);

						for (long ni = 0; ni < nodes.size(); ++ni)
						{
							const AssemblyValues &vi = vals.basis_values[nodes(ni)];
							for (int di = 0; di < size_; ++di)
							{
								const int gi_index = vi.global[0].index * size_ + di;
								for (long nj = 0; nj < nodes.size(); ++nj)
								{
									const AssemblyValues &vj = vals.basis_values[nodes(nj)];
									for (int dj = 0; dj < size_; ++dj)
									{
										const int gj_index = vj.global[0].index * size_ + dj;

										local_storage.entries.push_back(Eigen::Triplet<double>(gi_index, gj_index, local_hessian(nodes(ni) * size_ + di, nodes(nj) * size_ + dj)));
									}
								}
							}
						}
					}
				}
			});

			std::vector<Eigen::Triplet<double>> all_entries;
			// Serially merge local storages
			for (LocalThreadMatStorage &local_storage : storage)
			{
				all_entries.insert(all_entries.end(), local_storage.entries.begin(), local_storage.entries.end());
			}

			hess.setFromTriplets(all_entries.begin(), all_entries.end());
		}

		void PressureAssembler::compute_force_jacobian(
			const Eigen::MatrixXd &displacement,
			const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
			const std::vector<int> dirichlet_nodes,
			const int resolution,
			const double t,
			const int n_vertices,
			StiffnessMatrix &hess) const
		{
			compute_energy_hess(displacement, local_pressure_boundary, dirichlet_nodes, resolution, t, false, hess);

			// hess.resize(n_basis_ * size_, n_vertices * size_);

			// std::vector<Eigen::Triplet<double>> entries;

			// Eigen::MatrixXd pressure_vals, g_1, g_2, g_3;
			// ElementAssemblyValues vals, gvals;
			// Eigen::MatrixXd points, uv, normals, deform_mat, trafo;
			// Eigen::VectorXd weights;
			// Eigen::VectorXi global_primitive_ids;
			// for (const auto &lb : local_pressure_boundary)
			// {
			// 	const int e = lb.element_id();
			// 	const basis::ElementBases &gbs = gbases_[e];
			// 	const basis::ElementBases &bs = bases_[e];

			// 	for (int i = 0; i < lb.size(); ++i)
			// 	{
			// 		const int primitive_global_id = lb.global_primitive_id(i);
			// 		const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);
			// 		const auto gnodes = gbs.local_nodes_for_primitive(primitive_global_id, mesh_);

			// 		bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, i, false, uv, points, normals, weights);
			// 		std::vector<Eigen::VectorXd> param_chain_rule;
			// 		if (mesh_.is_volume())
			// 		{
			// 			weights /= 2 * mesh_.tri_area(primitive_global_id);
			// 			g_1.setZero(normals.rows(), normals.cols());
			// 			g_2.setZero(normals.rows(), normals.cols());
			// 			g_3.setZero(normals.rows(), normals.cols());

			// 			auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
			// 			param_chain_rule = {(endpoints.row(0) - endpoints.row(1)).transpose(), (endpoints.row(0) - endpoints.row(2)).transpose()};
			// 			if (lb[i] == 0)
			// 				param_chain_rule[0] *= -1;
			// 		}
			// 		else
			// 		{
			// 			assert(false);
			// 			// 	weights /= mesh_.edge_length(primitive_global_id);
			// 			// 	g_1.setZero(normals.rows(), normals.cols());
			// 			// 	g_2.setZero(0, 0);
			// 			// 	g_3.setZero(normals.rows(), normals.cols());

			// 			// 	auto endpoints = utils::BoundarySampler::tri_local_node_coordinates_from_edge(lb[i]);
			// 			// 	param_chain_rule = {(-endpoints.row(0) - endpoints.row(1)).transpose()};
			// 		}

			// 		if (!has_samples)
			// 			continue;

			// 		global_primitive_ids.setConstant(weights.size(), primitive_global_id);

			// 		vals.compute(e, mesh_.is_volume(), points, bs, gbs);
			// 		gvals.compute(e, mesh_.is_volume(), points, gbs, gbs);
			// 		for (int n = 0; n < vals.jac_it.size(); ++n)
			// 		{
			// 			trafo = vals.jac_it[n].inverse();

			// 			if (displacement.size() > 0)
			// 			{
			// 				assert(size_ == 2 || size_ == 3);
			// 				deform_mat.resize(size_, size_);
			// 				deform_mat.setZero();
			// 				for (const auto &b : vals.basis_values)
			// 				{
			// 					for (const auto &g : b.global)
			// 					{
			// 						for (int d = 0; d < size_; ++d)
			// 						{
			// 							deform_mat.row(d) += displacement(g.index * size_ + d) * b.grad.row(n);
			// 						}
			// 					}
			// 				}

			// 				trafo += deform_mat;
			// 			}

			// 			normals.row(n) = normals.row(n) * trafo.inverse();
			// 			normals.row(n).normalize();

			// 			if (mesh_.is_volume())
			// 			{
			// 				Eigen::Vector3d g1, g2, g3;
			// 				auto endpoints = utils::BoundarySampler::tet_local_node_coordinates_from_face(lb[i]);
			// 				g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
			// 				g2 = trafo * (endpoints.row(0) - endpoints.row(2)).transpose();
			// 				if (lb[i] == 0)
			// 					g1 *= -1;
			// 				g3 = g1.cross(g2);

			// 				g_1.row(n) = g1.transpose();
			// 				g_2.row(n) = g2.transpose();
			// 				g_3.row(n) = g3.transpose();
			// 			}
			// 			else
			// 			{
			// 				assert(false);
			// 				// Eigen::Vector2d g1, g3;
			// 				// auto endpoints = utils::BoundarySampler::tri_local_node_coordinates_from_edge(lb[i]);
			// 				// g1 = trafo * (endpoints.row(0) - endpoints.row(1)).transpose();
			// 				// g3(0) = -g1(0);
			// 				// g3(1) = g1(1);

			// 				// g_1.row(n) = g1.transpose();
			// 				// g_3.row(n) = g3.transpose();
			// 			}
			// 		}
			// 		problem_.pressure_bc(mesh_, global_primitive_ids, uv, vals.val, normals, t, pressure_vals);

			// 		for (long p = 0; p < weights.size(); ++p)
			// 		{
			// 			Eigen::Vector3d g_up_1, g_up_2;
			// 			g_dual(g_1.row(p).transpose(), g_2.row(p).transpose(), g_up_1, g_up_2);
			// 			std::vector<Eigen::MatrixXd> g_3_wedge_g_up = {wedge_product(g_3.row(p), g_up_1), wedge_product(g_3.row(p), g_up_2)};

			// 			for (long ni = 0; ni < nodes.size(); ++ni)
			// 			{
			// 				const AssemblyValues &vi = vals.basis_values[nodes(ni)];
			// 				for (int di = 0; di < size_; ++di)
			// 				{
			// 					const int gi_index = vi.global[0].index * size_ + di;
			// 					const bool is_dof_i_dirichlet = std::find(dirichlet_nodes.begin(), dirichlet_nodes.end(), gi_index) != dirichlet_nodes.end();
			// 					if (is_dof_i_dirichlet)
			// 						continue;

			// 					Eigen::MatrixXd grad_phi_i;
			// 					grad_phi_i.setZero(3, 3);
			// 					grad_phi_i.row(di) = vi.grad.row(p);

			// 					for (long nj = 0; nj < gnodes.size(); ++nj)
			// 					{
			// 						const AssemblyValues &vj = gvals.basis_values[gnodes(nj)];
			// 						for (int dj = 0; dj < size_; ++dj)
			// 						{
			// 							const int gj_index = vj.global[0].index * size_ + dj;
			// 							const bool is_dof_j_dirichlet = std::find(dirichlet_nodes.begin(), dirichlet_nodes.end(), gj_index) != dirichlet_nodes.end();
			// 							if (is_dof_j_dirichlet)
			// 								continue;

			// 							Eigen::MatrixXd grad_phi_j;
			// 							grad_phi_j.setZero(3, 3);
			// 							grad_phi_j.row(dj) = vj.grad.row(p);

			// 							double value = 0;
			// 							for (int alpha = 0; alpha < size_ - 1; ++alpha)
			// 							{
			// 								auto a = vj.val(p) * g_3_wedge_g_up[alpha].row(di) * (grad_phi_i * param_chain_rule[alpha]);
			// 								auto b = (g_3_wedge_g_up[alpha] * (grad_phi_j * param_chain_rule[alpha])).row(di) * vi.val(p);
			// 								value += pressure_vals(p) * (a(0) + b(0)) * weights(p);
			// 							}
			// 							entries.push_back(Eigen::Triplet<double>(gi_index, gj_index, value));
			// 						}
			// 					}
			// 				}
			// 			}
			// 		}
			// 	}
			// }

			// hess.setFromTriplets(entries.begin(), entries.end());
		}

	} // namespace assembler
} // namespace polyfem
