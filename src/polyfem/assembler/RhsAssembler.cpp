#include "RhsAssembler.hpp"
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/Logger.hpp>

#include <Eigen/Sparse>

#include <iostream>
#include <map>
#include <memory>

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
				ElementAssemblyValues vals;

				LocalThreadScalarStorage()
				{
					val = 0;
				}
			};
		} // namespace

		RhsAssembler::RhsAssembler(const AssemblerUtils &assembler, const Mesh &mesh, const Obstacle &obstacle,
								   const std::vector<int> &dirichlet_nodes, const std::vector<int> &neumann_nodes,
								   const std::vector<RowVectorNd> &dirichlet_nodes_position, const std::vector<RowVectorNd> &neumann_nodes_position,
								   const int n_basis, const int size,
								   const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const AssemblyValsCache &ass_vals_cache,
								   const std::string &formulation, const Problem &problem,
								   const std::string bc_method,
								   const std::string &solver, const std::string &preconditioner, const json &solver_params)
			: assembler_(assembler), mesh_(mesh), obstacle_(obstacle),
			  n_basis_(n_basis), size_(size),
			  bases_(bases), gbases_(gbases), ass_vals_cache_(ass_vals_cache),
			  formulation_(formulation), problem_(problem),
			  bc_method_(bc_method),
			  solver_(solver), preconditioner_(preconditioner), solver_params_(solver_params),
			  dirichlet_nodes_(dirichlet_nodes), neumann_nodes_(neumann_nodes),
			  dirichlet_nodes_position_(dirichlet_nodes_position), neumann_nodes_position_(neumann_nodes_position)
		{
			assert(ass_vals_cache_.is_mass());
		}

		void RhsAssembler::assemble(const Density &density, Eigen::MatrixXd &rhs, const double t) const
		{
			rhs = Eigen::MatrixXd::Zero(n_basis_ * size_, 1);
			if (!problem_.is_rhs_zero())
			{
				Eigen::MatrixXd rhs_fun;

				const int n_elements = int(bases_.size());
				ElementAssemblyValues vals;
				for (int e = 0; e < n_elements; ++e)
				{
					// vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);
					ass_vals_cache_.compute(e, mesh_.is_volume(), bases_[e], gbases_[e], vals);

					const Quadrature &quadrature = vals.quadrature;

					problem_.rhs(assembler_, formulation_, vals.val, t, rhs_fun);

					for (int d = 0; d < size_; ++d)
					{
						// rhs_fun.col(d) = rhs_fun.col(d).array() * vals.det.array() * quadrature.weights.array();
						for (int q = 0; q < quadrature.weights.size(); ++q)
						{
							// const double rho = problem_.is_time_dependent() ? density(vals.quadrature.points.row(q), vals.val.row(q), vals.element_id) : 1;
							const double rho = density(vals.quadrature.points.row(q), vals.val.row(q), vals.element_id);
							rhs_fun(q, d) *= vals.det(q) * quadrature.weights(q) * rho;
						}
					}

					const int n_loc_bases_ = int(vals.basis_values.size());
					for (int i = 0; i < n_loc_bases_; ++i)
					{
						const AssemblyValues &v = vals.basis_values[i];

						for (int d = 0; d < size_; ++d)
						{
							const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();
							for (std::size_t ii = 0; ii < v.global.size(); ++ii)
								rhs(v.global[ii].index * size_ + d) += rhs_value * v.global[ii].val;
						}
					}
				}
			}
		}

		void RhsAssembler::initial_solution(Eigen::MatrixXd &sol) const
		{
			time_bc([&](const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) {
				problem_.initial_solution(mesh, global_ids, pts, val);
			},
					sol);
		}

		void RhsAssembler::initial_velocity(Eigen::MatrixXd &sol) const
		{
			time_bc([&](const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) {
				problem_.initial_velocity(mesh, global_ids, pts, val);
			},
					sol);
		}

		void RhsAssembler::initial_acceleration(Eigen::MatrixXd &sol) const
		{
			time_bc([&](const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) {
				problem_.initial_acceleration(mesh, global_ids, pts, val);
			},
					sol);
		}

		void RhsAssembler::time_bc(const std::function<void(const Mesh &, const Eigen::MatrixXi &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &fun, Eigen::MatrixXd &sol) const
		{
			sol = Eigen::MatrixXd::Zero(n_basis_ * size_, 1);
			Eigen::MatrixXd loc_sol;

			const int n_elements = int(bases_.size());
			ElementAssemblyValues vals;
			Eigen::MatrixXi ids;

			if (bc_method_ == "sample")
			{
				for (int e = 0; e < n_elements; ++e)
				{
					const basis::ElementBases &bs = bases_[e];
					// vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);
					ass_vals_cache_.compute(e, mesh_.is_volume(), bases_[e], gbases_[e], vals);
					ids.resize(1, 1);
					ids.setConstant(e);

					for (long i = 0; i < bs.bases.size(); ++i)
					{
						const auto &b = bs.bases[i];
						const auto &glob = b.global();
						// assert(glob.size() == 1);
						for (size_t ii = 0; ii < glob.size(); ++ii)
						{
							fun(mesh_, ids, glob[ii].node, loc_sol);

							for (int d = 0; d < size_; ++d)
							{
								sol(glob[ii].index * size_ + d) = loc_sol(d) * glob[ii].val;
							}
						}
					}
				}
			}
			else
			{

				for (int e = 0; e < n_elements; ++e)
				{
					// vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);
					ass_vals_cache_.compute(e, mesh_.is_volume(), bases_[e], gbases_[e], vals);
					ids.resize(vals.val.rows(), 1);
					ids.setConstant(e);

					const Quadrature &quadrature = vals.quadrature;
					// problem_.initial_solution(vals.val, loc_sol);
					fun(mesh_, ids, vals.val, loc_sol);

					for (int d = 0; d < size_; ++d)
						loc_sol.col(d) = loc_sol.col(d).array() * vals.det.array() * quadrature.weights.array();

					const int n_loc_bases_ = int(vals.basis_values.size());
					for (int i = 0; i < n_loc_bases_; ++i)
					{
						const AssemblyValues &v = vals.basis_values[i];

						for (int d = 0; d < size_; ++d)
						{
							const double sol_value = (loc_sol.col(d).array() * v.val.array()).sum();
							for (std::size_t ii = 0; ii < v.global.size(); ++ii)
								sol(v.global[ii].index * size_ + d) += sol_value * v.global[ii].val;
						}
					}
				}

				Eigen::MatrixXd b = sol;
				sol.setZero();

				const double mmin = b.minCoeff();
				const double mmax = b.maxCoeff();

				if (fabs(mmin) > 1e-8 || fabs(mmax) > 1e-8)
				{
					StiffnessMatrix mass;
					const int n_fe_basis = n_basis_ - obstacle_.n_vertices();
					assembler_.assemble_mass_matrix(formulation_, size_ == 3, n_fe_basis, false, bases_, gbases_, ass_vals_cache_, mass);
					assert(mass.rows() == n_basis_ * size_ - obstacle_.ndof() && mass.cols() == n_basis_ * size_ - obstacle_.ndof());

					auto solver = LinearSolver::create(solver_, preconditioner_);
					solver->setParameters(solver_params_);
					solver->analyzePattern(mass, mass.rows());
					solver->factorize(mass);

					for (long i = 0; i < b.cols(); ++i)
					{
						solver->solve(b.block(0, i, mass.rows(), 1), sol.block(0, i, mass.rows(), 1));
					}
					logger().trace("mass matrix error {}", (mass * sol - b).norm());
				}
			}
		}

		void RhsAssembler::lsq_bc(const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &df,
								  const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, Eigen::MatrixXd &rhs) const
		{
			const int n_el = int(bases_.size());

			Eigen::MatrixXd uv, samples, gtmp, rhs_fun;
			Eigen::VectorXi global_primitive_ids;

			const int actual_dim = problem_.is_scalar() ? 1 : mesh_.dimension();

			Eigen::Matrix<bool, Eigen::Dynamic, 1> is_boundary(n_basis_);
			is_boundary.setConstant(false);
			int skipped_count = 0;
			for (int b : bounday_nodes)
			{
				int bindex = b / actual_dim;

				if (bindex < is_boundary.size())
					is_boundary[bindex] = true;
				else
					skipped_count++;
			}
			assert(skipped_count <= 1);

			for (int d = 0; d < size_; ++d)
			{
				int index = 0;
				std::vector<int> indices;
				indices.reserve(n_el * 10);
				std::vector<int> tags;
				tags.reserve(n_el * 10);

				long total_size = 0;

				Eigen::VectorXi global_index_to_col(n_basis_);
				global_index_to_col.setConstant(-1);

				std::vector<AssemblyValues> tmp_val;

				for (const auto &lb : local_boundary)
				{
					const int e = lb.element_id();
					bool has_samples = utils::BoundarySampler::sample_boundary(lb, resolution, mesh_, true, uv, samples, global_primitive_ids);

					if (!has_samples)
						continue;

					const basis::ElementBases &bs = bases_[e];
					bs.evaluate_bases(samples, tmp_val);
					const int n_local_bases = int(bs.bases.size());
					assert(global_primitive_ids.size() == samples.rows());

					for (int s = 0; s < samples.rows(); ++s)
					{
						const int tag = mesh_.get_boundary_id(global_primitive_ids(s));
						if (!problem_.all_dimensions_dirichlet() && !problem_.is_dimension_dirichet(tag, d))
							continue;

						total_size++;

						for (int j = 0; j < n_local_bases; ++j)
						{
							const basis::Basis &b = bs.bases[j];
							const double tmp = tmp_val[j].val(s);

							if (fabs(tmp) < 1e-10)
								continue;

							for (std::size_t ii = 0; ii < b.global().size(); ++ii)
							{
								// pt found
								if (is_boundary[b.global()[ii].index])
								{
									if (global_index_to_col(b.global()[ii].index) == -1)
									{
										global_index_to_col(b.global()[ii].index) = index++;
										indices.push_back(b.global()[ii].index);
										tags.push_back(tag);
										assert(indices.size() == size_t(index));
									}
								}
							}
						}
					}
				}

				Eigen::MatrixXd global_rhs = Eigen::MatrixXd::Zero(total_size, 1);

				const long buffer_size = total_size * long(indices.size());
				std::vector<Eigen::Triplet<double>> entries, entries_t;

				index = 0;

				int global_counter = 0;
				Eigen::MatrixXd mapped;

				for (const auto &lb : local_boundary)
				{
					const int e = lb.element_id();
					bool has_samples = utils::BoundarySampler::sample_boundary(lb, resolution, mesh_, false, uv, samples, global_primitive_ids);

					if (!has_samples)
						continue;

					const basis::ElementBases &bs = bases_[e];
					const basis::ElementBases &gbs = gbases_[e];
					const int n_local_bases = int(bs.bases.size());

					gbs.eval_geom_mapping(samples, mapped);

					bs.evaluate_bases(samples, tmp_val);
					df(global_primitive_ids, uv, mapped, rhs_fun);

					for (int s = 0; s < samples.rows(); ++s)
					{
						const int tag = mesh_.get_boundary_id(global_primitive_ids(s));
						if (!problem_.all_dimensions_dirichlet() && !problem_.is_dimension_dirichet(tag, d))
							continue;

						for (int j = 0; j < n_local_bases; ++j)
						{
							const basis::Basis &b = bs.bases[j];
							const double tmp = tmp_val[j].val(s);

							for (std::size_t ii = 0; ii < b.global().size(); ++ii)
							{
								auto item = global_index_to_col(b.global()[ii].index);
								if (item != -1)
								{
									entries.push_back(Eigen::Triplet<double>(global_counter, item, tmp * b.global()[ii].val));
									entries_t.push_back(Eigen::Triplet<double>(item, global_counter, tmp * b.global()[ii].val));
								}
							}
						}

						global_rhs(global_counter) = rhs_fun(s, d);
						global_counter++;
					}
				}

				assert(global_counter == total_size);

				if (total_size > 0)
				{
					const double mmin = global_rhs.minCoeff();
					const double mmax = global_rhs.maxCoeff();

					if (fabs(mmin) < 1e-8 && fabs(mmax) < 1e-8)
					{
						for (size_t i = 0; i < indices.size(); ++i)
						{
							const int tag = tags[i];
							if (problem_.all_dimensions_dirichlet() || problem_.is_dimension_dirichet(tag, d))
								rhs(indices[i] * size_ + d) = 0;
						}
					}
					else
					{
						StiffnessMatrix mat(int(total_size), int(indices.size()));
						mat.setFromTriplets(entries.begin(), entries.end());

						StiffnessMatrix mat_t(int(indices.size()), int(total_size));
						mat_t.setFromTriplets(entries_t.begin(), entries_t.end());

						StiffnessMatrix A = mat_t * mat;
						Eigen::VectorXd b = mat_t * global_rhs;

						Eigen::VectorXd coeffs(b.rows(), 1);
						auto solver = LinearSolver::create(solver_, preconditioner_);
						solver->setParameters(solver_params_);
						solver->analyzePattern(A, A.rows());
						solver->factorize(A);
						coeffs.setZero();
						solver->solve(b, coeffs);

						logger().trace("RHS solve error {}", (A * coeffs - b).norm());

						for (long i = 0; i < coeffs.rows(); ++i)
						{
							const int tag = tags[i];
							if (problem_.all_dimensions_dirichlet() || problem_.is_dimension_dirichet(tag, d))
								rhs(indices[i] * size_ + d) = coeffs(i);
						}
					}
				}
			}
		}

		void RhsAssembler::sample_bc(const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &df,
									 const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, Eigen::MatrixXd &rhs) const
		{
			const int n_el = int(bases_.size());

			Eigen::MatrixXd rhs_fun;
			Eigen::VectorXi global_primitive_ids(1);
			Eigen::MatrixXd nans(1, 1);
			nans(0) = std::nan("");

#ifndef NDEBUG
			Eigen::Matrix<bool, Eigen::Dynamic, 1> is_boundary(n_basis_);
			is_boundary.setConstant(false);

			const int actual_dim = problem_.is_scalar() ? 1 : mesh_.dimension();

			int skipped_count = 0;
			for (int b : bounday_nodes)
			{
				int bindex = b / actual_dim;

				if (bindex < is_boundary.size())
					is_boundary[bindex] = true;
				else
					skipped_count++;
			}
			assert(skipped_count <= 1);
#endif

			for (const auto &lb : local_boundary)
			{
				const int e = lb.element_id();
				const basis::ElementBases &bs = bases_[e];

				for (int i = 0; i < lb.size(); ++i)
				{
					global_primitive_ids(0) = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(global_primitive_ids(0), mesh_);
					assert(global_primitive_ids.size() == 1);
					const int tag = mesh_.get_boundary_id(global_primitive_ids(0));

					for (long n = 0; n < nodes.size(); ++n)
					{
						const auto &b = bs.bases[nodes(n)];
						const auto &glob = b.global();

						for (size_t ii = 0; ii < glob.size(); ++ii)
						{
							assert(is_boundary[glob[ii].index]);

							// TODO, missing UV!!!!
							df(global_primitive_ids, nans, glob[ii].node, rhs_fun);

							for (int d = 0; d < size_; ++d)
							{
								if (problem_.all_dimensions_dirichlet() || problem_.is_dimension_dirichet(tag, d))
								{
									assert(problem_.all_dimensions_dirichlet() || std::find(bounday_nodes.begin(), bounday_nodes.end(), glob[ii].index * size_ + d) != bounday_nodes.end());
									rhs(glob[ii].index * size_ + d) = rhs_fun(0, d);
								}
							}
						}
					}
				}
			}
		}

		void RhsAssembler::integrate_bc(const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &df,
										const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, Eigen::MatrixXd &rhs) const
		{
			assert(false);
			Eigen::MatrixXd uv, samples, rhs_fun, normals, mapped;
			Eigen::VectorXd weights;

			Eigen::VectorXi global_primitive_ids;
			std::vector<AssemblyValues> tmp_val;

			Eigen::Matrix<bool, Eigen::Dynamic, 1> is_boundary(n_basis_);
			is_boundary.setConstant(false);

			Eigen::MatrixXd areas(rhs.rows(), 1);
			areas.setZero();

			const int actual_dim = problem_.is_scalar() ? 1 : mesh_.dimension();

			int skipped_count = 0;
			for (int b : bounday_nodes)
			{
				rhs(b) = 0;
				int bindex = b / actual_dim;

				if (bindex < is_boundary.size())
					is_boundary[bindex] = true;
				else
					skipped_count++;
			}
			assert(skipped_count <= 1);
			ElementAssemblyValues vals;

			for (const auto &lb : local_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, false, uv, samples, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				const basis::ElementBases &bs = bases_[e];
				const basis::ElementBases &gbs = gbases_[e];

				vals.compute(e, mesh_.is_volume(), samples, bs, gbs);

				df(global_primitive_ids, uv, vals.val, rhs_fun);

				for (int d = 0; d < size_; ++d)
					rhs_fun.col(d) = rhs_fun.col(d).array() * weights.array();

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

					for (long n = 0; n < nodes.size(); ++n)
					{
						// const auto &b = bs.bases[nodes(n)];
						const AssemblyValues &v = vals.basis_values[nodes(n)];
						const double area = (weights.array() * v.val.array()).sum();
						for (int d = 0; d < size_; ++d)
						{
							const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();

							for (size_t g = 0; g < v.global.size(); ++g)
							{
								const int g_index = v.global[g].index * size_ + d;
								if (problem_.all_dimensions_dirichlet() || std::find(bounday_nodes.begin(), bounday_nodes.end(), g_index) != bounday_nodes.end())
								{
									rhs(g_index) += rhs_value * v.global[g].val;
									areas(g_index) += area * v.global[g].val;
								}
							}
						}
					}
				}
			}

			for (int b : bounday_nodes)
			{
				assert(areas(b) != 0);
				rhs(b) /= areas(b);
			}
		}

		void RhsAssembler::set_bc(
			const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &df,
			const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &nf,
			const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes,
			const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary,
			const Eigen::MatrixXd &displacement, const double t,
			Eigen::MatrixXd &rhs) const
		{
			if (bc_method_ == "sample")
				sample_bc(df, local_boundary, bounday_nodes, rhs);
			else if (bc_method_ == "integrate")
				integrate_bc(df, local_boundary, bounday_nodes, resolution, rhs);
			else
				lsq_bc(df, local_boundary, bounday_nodes, resolution, rhs);

			if (bounday_nodes.size() > 0)
			{
				Eigen::MatrixXd tmp_val;
				for (int n = 0; n < dirichlet_nodes_.size(); ++n)
				{
					const auto &n_id = dirichlet_nodes_[n];
					const auto &pt = dirichlet_nodes_position_[n];

					const int tag = mesh_.get_node_id(n_id);
					problem_.dirichlet_nodal_value(mesh_, n_id, pt, t, tmp_val);
					assert(tmp_val.size() == size_);

					for (int d = 0; d < size_; ++d)
					{
						if (!problem_.is_nodal_dimension_dirichlet(n_id, tag, d))
							continue;
						const int g_index = n_id * size_ + d;
						rhs(g_index) = tmp_val(d);
					}
				}
			}

			// Neumann
			Eigen::MatrixXd uv, samples, gtmp, rhs_fun, deform_mat, trafo;
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd points, normals;
			Eigen::VectorXd weights;

			ElementAssemblyValues vals;

			for (const auto &lb : local_neumann_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, false, uv, points, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				const basis::ElementBases &gbs = gbases_[e];
				const basis::ElementBases &bs = bases_[e];

				vals.compute(e, mesh_.is_volume(), points, bs, gbs);

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					Eigen::MatrixXd ppp(1, size_);
					ppp = vals.val.row(n);

					trafo = vals.jac_it[n];

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
									deform_mat.col(d) += displacement(g.index * size_ + d) * b.grad_t_m.row(n);

									ppp(d) += displacement(g.index * size_ + d) * b.val(n);
								}
							}
						}

						trafo += deform_mat;
					}

					normals.row(n) = normals.row(n) * trafo;
					normals.row(n).normalize();
				}

				// problem_.neumann_bc(mesh_, global_primitive_ids, vals.val, t, rhs_fun);
				nf(global_primitive_ids, uv, vals.val, normals, rhs_fun);

				// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(0,1,0));

				for (int d = 0; d < size_; ++d)
					rhs_fun.col(d) = rhs_fun.col(d).array() * weights.array();

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

					for (long n = 0; n < nodes.size(); ++n)
					{
						// const auto &b = bs.bases[nodes(n)];
						const AssemblyValues &v = vals.basis_values[nodes(n)];
						for (int d = 0; d < size_; ++d)
						{
							const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();

							for (size_t g = 0; g < v.global.size(); ++g)
							{
								const int g_index = v.global[g].index * size_ + d;
								const bool is_neumann = std::find(bounday_nodes.begin(), bounday_nodes.end(), g_index) == bounday_nodes.end();

								if (is_neumann)
								{
									rhs(g_index) += rhs_value * v.global[g].val;
								}
							}
						}
					}
				}
			}

			// TODO add nodal neumann
		}

		void RhsAssembler::set_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const Eigen::MatrixXd &displacement, const double t) const
		{
			set_bc(
				[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) {
					problem_.dirichlet_bc(mesh_, global_ids, uv, pts, t, val);
				},
				[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, Eigen::MatrixXd &val) {
					problem_.neumann_bc(mesh_, global_ids, uv, pts, normals, t, val);
				},
				local_boundary, bounday_nodes, resolution, local_neumann_boundary, displacement, t, rhs);

			obstacle_.update_displacement(t, rhs);
		}

		void RhsAssembler::compute_energy_grad(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const Density &density, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, const Eigen::MatrixXd &final_rhs, const double t, Eigen::MatrixXd &rhs) const
		{
			if (problem_.is_constant_in_time())
			{
				rhs = final_rhs;
			}
			else
			{
				assemble(density, rhs, t);
				rhs *= -1;

				if (rhs.size() != final_rhs.size())
				{
					const int prev_size = rhs.size();
					rhs.conservativeResize(final_rhs.size(), rhs.cols());
					// Zero initial pressure
					rhs.block(prev_size, 0, final_rhs.size() - prev_size, rhs.cols()).setZero();
					rhs(rhs.size() - 1) = 0;
				}

				assert(rhs.size() == final_rhs.size());
			}
		}

		double RhsAssembler::compute_energy(const Eigen::MatrixXd &displacement, const std::vector<LocalBoundary> &local_neumann_boundary, const Density &density, const int resolution, const double t) const
		{

			double res = 0;

			if (!problem_.is_rhs_zero())
			{
				auto storage = create_thread_storage(LocalThreadScalarStorage());
				const int n_bases = int(bases_.size());

				maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
					LocalThreadScalarStorage &local_storage = get_local_thread_storage(storage, thread_id);
					VectorNd local_displacement(size_);
					Eigen::MatrixXd forces;

					for (int e = start; e < end; ++e)
					{
						ElementAssemblyValues &vals = local_storage.vals;
						// vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);
						ass_vals_cache_.compute(e, mesh_.is_volume(), bases_[e], gbases_[e], vals);

						const Quadrature &quadrature = vals.quadrature;
						const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

						problem_.rhs(assembler_, formulation_, vals.val, t, forces);
						assert(forces.rows() == da.size());
						assert(forces.cols() == size_);

						for (long p = 0; p < da.size(); ++p)
						{
							local_displacement.setZero();

							for (size_t i = 0; i < vals.basis_values.size(); ++i)
							{
								const auto &bs = vals.basis_values[i];
								assert(bs.val.size() == da.size());
								const double b_val = bs.val(p);

								for (int d = 0; d < size_; ++d)
								{
									for (std::size_t ii = 0; ii < bs.global.size(); ++ii)
									{
										local_displacement(d) += (bs.global[ii].val * b_val) * displacement(bs.global[ii].index * size_ + d);
									}
								}
							}
							// const double rho = problem_.is_time_dependent() ? density(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id) : 1;
							const double rho = density(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id);

							for (int d = 0; d < size_; ++d)
							{
								local_storage.val += forces(p, d) * local_displacement(d) * da(p) * rho;
								// res += forces(p, d) * local_displacement(d) * da(p);
							}
						}
					}
				});

				// Serially merge local storages
				for (const LocalThreadScalarStorage &local_storage : storage)
					res += local_storage.val;
			}

			VectorNd local_displacement(size_);
			Eigen::MatrixXd forces;

			ElementAssemblyValues vals;
			// Neumann
			Eigen::MatrixXd points, uv, normals, deform_mat;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_primitive_ids;
			for (const auto &lb : local_neumann_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, false, uv, points, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				const basis::ElementBases &gbs = gbases_[e];
				const basis::ElementBases &bs = bases_[e];

				vals.compute(e, mesh_.is_volume(), points, bs, gbs);

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					normals.row(n) = normals.row(n) * vals.jac_it[n];
					normals.row(n).normalize();
				}
				problem_.neumann_bc(mesh_, global_primitive_ids, uv, vals.val, normals, t, forces);

				// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

				for (long p = 0; p < weights.size(); ++p)
				{
					local_displacement.setZero();

					for (size_t i = 0; i < vals.basis_values.size(); ++i)
					{
						const auto &vv = vals.basis_values[i];
						assert(vv.val.size() == weights.size());
						const double b_val = vv.val(p);

						for (int d = 0; d < size_; ++d)
						{
							for (std::size_t ii = 0; ii < vv.global.size(); ++ii)
							{
								local_displacement(d) += (vv.global[ii].val * b_val) * displacement(vv.global[ii].index * size_ + d);
							}
						}
					}

					for (int d = 0; d < size_; ++d)
						res -= forces(p, d) * local_displacement(d) * weights(p);
				}
			}

			return res;
		}
	} // namespace assembler
} // namespace polyfem
