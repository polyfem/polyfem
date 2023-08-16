#include "OperatorSplittingSolver.hpp"
#include <unsupported/Eigen/SparseExtra>

#ifdef POLYFEM_WITH_OPENVDB
#include <openvdb/openvdb.h>
#endif

namespace polyfem::solver
{
	using namespace assembler;

	void OperatorSplittingSolver::initialize_grid(const mesh::Mesh &mesh,
												  const std::vector<basis::ElementBases> &gbases,
												  const std::vector<basis::ElementBases> &bases,
												  const double &density_dx)
	{
		resolution = density_dx;

		grid_cell_num = RowVectorNd::Zero(dim);
		for (int d = 0; d < dim; d++)
		{
			grid_cell_num(d) = ceil((max_domain(d) - min_domain(d)) / resolution);
		}
		if (dim == 2)
			density = Eigen::VectorXd::Zero((grid_cell_num(0) + 1) * (grid_cell_num(1) + 1));
		else
			density = Eigen::VectorXd::Zero((grid_cell_num(0) + 1) * (grid_cell_num(1) + 1) * (grid_cell_num(2) + 1));
	}

	void OperatorSplittingSolver::initialize_mesh(const mesh::Mesh &mesh,
												  const int shape, const int n_el,
												  const std::vector<mesh::LocalBoundary> &local_boundary)
	{
		dim = mesh.dimension();
		mesh.bounding_box(min_domain, max_domain);

		const int size = local_boundary.size();
		boundary_elem_id.reserve(size);
		for (int e = 0; e < size; e++)
		{
			boundary_elem_id.push_back(local_boundary[e].element_id());
		}

		T.resize(n_el, shape);
		for (int e = 0; e < n_el; e++)
		{
			for (int i = 0; i < shape; i++)
			{
				T(e, i) = mesh.cell_vertex(e, i);
			}
		}
		V = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
		for (int i = 0; i < V.rows(); i++)
		{
			auto p = mesh.point(i);
			for (int d = 0; d < dim; d++)
			{
				V(i, d) = p(d);
			}
			if (dim == 2)
				V(i, 2) = 0;
		}
	}

	void OperatorSplittingSolver::initialize_hashtable(const mesh::Mesh &mesh)
	{
		Eigen::MatrixXd p0, p1, p;
		mesh.get_edges(p0, p1);
		p = p0 - p1;
		double avg_edge_length = p.rowwise().norm().mean();

		long total_cell_num = 1;
		hash_table_cell_num.setZero(dim);
		for (int d = 0; d < dim; d++)
		{
			hash_table_cell_num(d) = (long)std::round((max_domain(d) - min_domain(d)) / avg_edge_length) * 4;
			logger().debug("hash grid in {} dimension: {}", d, hash_table_cell_num(d));
			total_cell_num *= hash_table_cell_num(d);
		}
		hash_table.resize(total_cell_num);
		for (int e = 0; e < T.rows(); e++)
		{
			Eigen::VectorXd min_ = V.row(T(e, 0));
			Eigen::VectorXd max_ = min_;

			for (int i = 1; i < T.cols(); i++)
			{
				Eigen::VectorXd p = V.row(T(e, i));
				min_ = min_.cwiseMin(p);
				max_ = max_.cwiseMax(p);
			}

			Eigen::Matrix<long, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> min_int(dim), max_int(dim);

			for (int d = 0; d < dim; d++)
			{
				double temp = hash_table_cell_num(d) / (max_domain(d) - min_domain(d));
				min_int(d) = floor((min_(d) * (1 - 1e-14) - min_domain(d)) * temp);
				max_int(d) = ceil((max_(d) * (1 + 1e-14) - min_domain(d)) * temp);

				min_int = min_int.cwiseMax(0);
				max_int = max_int.cwiseMin(hash_table_cell_num);
			}

			for (long x = min_int(0); x < max_int(0); x++)
			{
				for (long y = min_int(1); y < max_int(1); y++)
				{
					if (dim == 2)
					{
						long idx = x + y * hash_table_cell_num(0);
						hash_table[idx].push_back(e);
					}
					else
					{
						for (long z = min_int(2); z < max_int(2); z++)
						{
							long idx = x + (y + z * hash_table_cell_num(1)) * hash_table_cell_num(0);
							hash_table[idx].push_back(e);
						}
					}
				}
			}
		}

		float average_intersection_num = 0;
		int max_intersection_num = 0;
		for (int i = 0; i < hash_table.size(); i++)
		{
			average_intersection_num += hash_table[i].size();
			(hash_table[i].size() > max_intersection_num) ? (max_intersection_num = hash_table[i].size()) : 1;
		}
		average_intersection_num /= hash_table.size();
		logger().debug("average intersection number for hash grid: {}", average_intersection_num);
		logger().debug("max intersection number for hash grid: {}", max_intersection_num);
	}

	void OperatorSplittingSolver::initialize_solver(const mesh::Mesh &mesh,
													const int shape_, const int n_el_,
													const std::vector<mesh::LocalBoundary> &local_boundary,
													const std::vector<int> &bnd_nodes)
	{
		shape = shape_;
		n_el = n_el_;
		boundary_nodes = bnd_nodes;

		initialize_mesh(mesh, shape, n_el, local_boundary);
		initialize_hashtable(mesh);
	}

	OperatorSplittingSolver::OperatorSplittingSolver(const mesh::Mesh &mesh,
													 const int shape, const int n_el,
													 const std::vector<mesh::LocalBoundary> &local_boundary,
													 const std::vector<int> &bnd_nodes)
	{
		initialize_solver(mesh, shape, n_el, local_boundary, bnd_nodes);
	}

	OperatorSplittingSolver::OperatorSplittingSolver(const mesh::Mesh &mesh,
													 const int shape, const int n_el,
													 const std::vector<mesh::LocalBoundary> &local_boundary,
													 const std::vector<int> &boundary_nodes_,
													 const std::vector<int> &pressure_boundary_nodes,
													 const std::vector<int> &bnd_nodes,
													 const StiffnessMatrix &mass,
													 const StiffnessMatrix &stiffness_viscosity,
													 const StiffnessMatrix &stiffness_velocity,
													 const StiffnessMatrix &mass_velocity,
													 const double &dt,
													 const double &viscosity_,
													 const std::string &solver_type,
													 const std::string &precond,
													 const json &params) : solver_type(solver_type)
	{
		initialize_solver(mesh, shape, n_el, local_boundary, bnd_nodes);

		logger().info("Prefactorization begins...");

		solver_mass = polysolve::LinearSolver::create(solver_type, precond);
		solver_mass->setParameters(params);
		// if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
		{
			StiffnessMatrix mat1 = mass_velocity;
			prefactorize(*solver_mass, mat1, boundary_nodes_, mat1.rows(), "");
		}

		mat_diffusion = mass + viscosity_ * dt * stiffness_viscosity;

		solver_diffusion = polysolve::LinearSolver::create(solver_type, precond);
		solver_diffusion->setParameters(params);
		// if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
		{
			StiffnessMatrix mat1 = mat_diffusion;
			prefactorize(*solver_diffusion, mat1, bnd_nodes, mat1.rows(), "");
		}

		if (pressure_boundary_nodes.size() == 0)
			mat_projection.resize(stiffness_velocity.rows() + 1, stiffness_velocity.cols() + 1);
		else
			mat_projection.resize(stiffness_velocity.rows(), stiffness_velocity.cols());

		std::vector<Eigen::Triplet<double>> coefficients;
		coefficients.reserve(stiffness_velocity.nonZeros() + 2 * stiffness_velocity.rows());

		for (int i = 0; i < stiffness_velocity.outerSize(); i++)
		{
			for (StiffnessMatrix::InnerIterator it(stiffness_velocity, i); it; ++it)
			{
				coefficients.emplace_back(it.row(), it.col(), it.value());
			}
		}

		if (pressure_boundary_nodes.size() == 0)
		{
			const double val = 1. / (mat_projection.rows() - 1);
			for (int i = 0; i < mat_projection.rows() - 1; i++)
			{
				coefficients.emplace_back(i, mat_projection.cols() - 1, val);
				coefficients.emplace_back(mat_projection.rows() - 1, i, val);
			}
			coefficients.emplace_back(mat_projection.rows() - 1, mat_projection.cols() - 1, 2);
		}

		mat_projection.setFromTriplets(coefficients.begin(), coefficients.end());
		solver_projection = polysolve::LinearSolver::create(solver_type, precond);
		solver_projection->setParameters(params);
		logger().info("{}...", solver_projection->name());
		// if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
		{
			StiffnessMatrix mat2 = mat_projection;
			prefactorize(*solver_projection, mat2, pressure_boundary_nodes, mat2.rows(), "");
		}
		logger().info("Prefactorization ends!");
	}

	int OperatorSplittingSolver::handle_boundary_advection(RowVectorNd &pos)
	{
		double dist = 1e10;
		int idx = -1, local_idx = -1;
		const int size = boundary_elem_id.size();
#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, size, 1, [&](int e)
#else
		for (int e = 0; e < size; e++)
#endif
						  {
							  int elem_idx = boundary_elem_id[e];

							  for (int i = 0; i < shape; i++)
							  {
								  double dist_ = 0;
								  for (int d = 0; d < dim; d++)
								  {
									  dist_ += pow(pos(d) - V(T(elem_idx, i), d), 2);
								  }
								  dist_ = sqrt(dist_);
								  if (dist_ < dist)
								  {
									  dist = dist_;
									  idx = elem_idx;
									  local_idx = i;
								  }
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
		for (int d = 0; d < dim; d++)
			pos(d) = V(T(idx, local_idx), d);
		return idx;
	}

	int OperatorSplittingSolver::trace_back(const std::vector<basis::ElementBases> &gbases,
											const std::vector<basis::ElementBases> &bases,
											const RowVectorNd &pos_1,
											const RowVectorNd &vel_1,
											RowVectorNd &pos_2,
											RowVectorNd &vel_2,
											Eigen::MatrixXd &local_pos,
											const Eigen::MatrixXd &sol,
											const double dt)
	{
		pos_2 = pos_1 - vel_1 * dt;

		return interpolator(gbases, bases, pos_2, vel_2, local_pos, sol);
	}

	int OperatorSplittingSolver::interpolator(const std::vector<basis::ElementBases> &gbases,
											  const std::vector<basis::ElementBases> &bases,
											  const RowVectorNd &pos,
											  RowVectorNd &vel,
											  Eigen::MatrixXd &local_pos,
											  const Eigen::MatrixXd &sol)
	{
		bool insideDomain = true;

		int new_elem;
		if ((new_elem = search_cell(gbases, pos, local_pos)) == -1)
		{
			insideDomain = false;
			RowVectorNd pos_ = pos;
			new_elem = handle_boundary_advection(pos_);
			calculate_local_pts(gbases[new_elem], new_elem, pos_, local_pos);
		}

		// interpolation
		vel = RowVectorNd::Zero(dim);
		assembler::ElementAssemblyValues vals;
		vals.compute(new_elem, dim == 3, local_pos, bases[new_elem], gbases[new_elem]);
		for (int d = 0; d < dim; d++)
		{
			for (int i = 0; i < vals.basis_values.size(); i++)
			{
				vel(d) += vals.basis_values[i].val(0) * sol(bases[new_elem].bases[i].global()[0].index * dim + d);
			}
		}
		if (insideDomain)
			return new_elem;
		else
			return -1;
	}

	void OperatorSplittingSolver::interpolator(const RowVectorNd &pos, double &val)
	{
		val = 0;

		Eigen::Matrix<long, Eigen::Dynamic, 1> int_pos(dim);
		Eigen::MatrixXd weights(2, dim);
		for (int d = 0; d < dim; d++)
		{
			int_pos(d) = floor((pos(d) - min_domain(d)) / resolution);
			if (int_pos(d) < 0 || int_pos(d) >= grid_cell_num(d))
				return;
			weights(1, d) = (pos(d) - min_domain(d)) / resolution - int_pos(d);
			weights(0, d) = 1 - weights(1, d);
		}

		for (int d1 = 0; d1 < 2; d1++)
			for (int d2 = 0; d2 < 2; d2++)
			{
				if (dim == 2)
				{
					const long idx = (int_pos(0) + d1) + (int_pos(1) + d2) * (grid_cell_num(0) + 1);
					val += density(idx) * weights(d1, 0) * weights(d2, 1);
				}
				else
				{
					for (int d3 = 0; d3 < 2; d3++)
					{
						const long idx = (int_pos(0) + d1) + (int_pos(1) + d2 + (int_pos(2) + d3) * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);
						val += density(idx) * weights(d1, 0) * weights(d2, 1) * weights(d3, 2);
					}
				}
			}
	}

	void OperatorSplittingSolver::advection(const mesh::Mesh &mesh,
											const std::vector<basis::ElementBases> &gbases,
											const std::vector<basis::ElementBases> &bases,
											Eigen::MatrixXd &sol,
											const double dt,
											const Eigen::MatrixXd &local_pts,
											const int order,
											const int RK)
	{
		// to store new velocity
		Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
		// number of FEM nodes
		const int n_vert = sol.size() / dim;
		Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_vert);

#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, n_el, 1, [&](int e)
#else
		for (int e = 0; e < n_el; ++e)
#endif
						  {
							  // to compute global position with barycentric coordinate
							  Eigen::MatrixXd mapped;
							  gbases[e].eval_geom_mapping(local_pts, mapped);

							  for (int i = 0; i < local_pts.rows(); i++)
							  {
								  // global index of this FEM node
								  int global = bases[e].bases[i].global()[0].index;

								  if (traversed(global))
									  continue;
								  traversed(global) = 1;

								  // velocity of this FEM node
								  RowVectorNd vel_ = sol.block(global * dim, 0, dim, 1).transpose();

								  // global position of this FEM node
								  RowVectorNd pos_ = RowVectorNd::Zero(1, dim);
								  for (int d = 0; d < dim; d++)
									  pos_(d) = mapped(i, d) - vel_(d) * dt;

								  Eigen::MatrixXd local_pos;
								  interpolator(gbases, bases, pos_, vel_, local_pos, sol);

								  new_sol.block(global * dim, 0, dim, 1) = vel_.transpose();
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
		sol.swap(new_sol);
	}

	void OperatorSplittingSolver::advect_density_exact(const std::vector<basis::ElementBases> &gbases,
													   const std::vector<basis::ElementBases> &bases,
													   const std::shared_ptr<assembler::Problem> problem,
													   const double t,
													   const double dt,
													   const int RK)
	{
		Eigen::VectorXd new_density = Eigen::VectorXd::Zero(density.size());
		const int Nx = grid_cell_num(0);
#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, Nx + 1, 1, [&](int i)
#else
		for (int i = 0; i <= Nx; i++)
#endif
						  {
							  for (int j = 0; j <= grid_cell_num(1); j++)
							  {
								  if (dim == 2)
								  {
									  Eigen::MatrixXd pos(1, dim);
									  pos(0) = i * resolution + min_domain(0);
									  pos(1) = j * resolution + min_domain(1);
									  const long idx = i + (long)j * (grid_cell_num(0) + 1);

									  Eigen::MatrixXd vel1, pos_;
									  problem->exact(pos, t, vel1);
									  if (RK > 1)
									  {
										  Eigen::MatrixXd vel2, vel3;
										  problem->exact(pos - 0.5 * dt * vel1, t, vel2);
										  problem->exact(pos - 0.75 * dt * vel2, t, vel3);
										  pos_ = pos - (2 * vel1 + 3 * vel2 + 4 * vel3) * dt / 9;
									  }
									  else
									  {
										  pos_ = pos - vel1 * dt;
									  }
									  interpolator(pos_, new_density[idx]);
								  }
								  else
								  {
									  for (int k = 0; k <= grid_cell_num(2); k++)
									  {
										  RowVectorNd pos(1, dim);
										  pos(0) = i * resolution + min_domain(0);
										  pos(1) = j * resolution + min_domain(1);
										  pos(2) = k * resolution + min_domain(2);
										  const long idx = i + (j + (long)k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);

										  Eigen::MatrixXd vel1, pos_;
										  problem->exact(pos, t, vel1);
										  if (RK > 1)
										  {
											  Eigen::MatrixXd vel2, vel3;
											  problem->exact(pos - 0.5 * dt * vel1, t, vel2);
											  problem->exact(pos - 0.75 * dt * vel2, t, vel3);
											  pos_ = pos - (2 * vel1 + 3 * vel2 + 4 * vel3) * dt / 9;
										  }
										  else
										  {
											  pos_ = pos - vel1 * dt;
										  }
										  interpolator(pos_, new_density[idx]);
									  }
								  }
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
		density.swap(new_density);
	}

	void OperatorSplittingSolver::advect_density(const std::vector<basis::ElementBases> &gbases,
												 const std::vector<basis::ElementBases> &bases,
												 const Eigen::MatrixXd &sol,
												 const double dt,
												 const int RK)
	{
		Eigen::VectorXd new_density = Eigen::VectorXd::Zero(density.size());
		const int Nx = grid_cell_num(0);
#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, Nx + 1, 1, [&](int i)
#else
		for (int i = 0; i <= Nx; i++)
#endif
						  {
							  for (int j = 0; j <= grid_cell_num(1); j++)
							  {
								  Eigen::MatrixXd local_pos;
								  if (dim == 2)
								  {
									  RowVectorNd pos(1, dim);
									  pos(0) = i * resolution + min_domain(0);
									  pos(1) = j * resolution + min_domain(1);
									  const long idx = i + (long)j * (grid_cell_num(0) + 1);

									  RowVectorNd vel1, pos_;
									  interpolator(gbases, bases, pos, vel1, local_pos, sol);
									  if (RK > 1)
									  {
										  RowVectorNd vel2, vel3;
										  interpolator(gbases, bases, pos - 0.5 * dt * vel1, vel2, local_pos, sol);
										  interpolator(gbases, bases, pos - 0.75 * dt * vel2, vel3, local_pos, sol);
										  pos_ = pos - (2 * vel1 + 3 * vel2 + 4 * vel3) * dt / 9;
									  }
									  else
									  {
										  pos_ = pos - vel1 * dt;
									  }
									  interpolator(pos_, new_density[idx]);
								  }
								  else
								  {
									  for (int k = 0; k <= grid_cell_num(2); k++)
									  {
										  RowVectorNd pos(1, dim);
										  pos(0) = i * resolution + min_domain(0);
										  pos(1) = j * resolution + min_domain(1);
										  pos(2) = k * resolution + min_domain(2);
										  const long idx = i + (j + (long)k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);

										  RowVectorNd vel1, pos_;
										  interpolator(gbases, bases, pos, vel1, local_pos, sol);
										  if (RK > 1)
										  {
											  RowVectorNd vel2, vel3;
											  interpolator(gbases, bases, pos - 0.5 * dt * vel1, vel2, local_pos, sol);
											  interpolator(gbases, bases, pos - 0.75 * dt * vel2, vel3, local_pos, sol);
											  pos_ = pos - (2 * vel1 + 3 * vel2 + 4 * vel3) * dt / 9;
										  }
										  else
										  {
											  pos_ = pos - vel1 * dt;
										  }
										  interpolator(pos_, new_density[idx]);
									  }
								  }
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
		density.swap(new_density);
	}

	void OperatorSplittingSolver::advection_FLIP(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &gbases, const std::vector<basis::ElementBases> &bases, Eigen::MatrixXd &sol, const double dt, const Eigen::MatrixXd &local_pts, const int order)
	{
		const int ppe = shape; // particle per element
		const double FLIPRatio = 1;
		// initialize or resample particles and update velocity via g2p
		if (position_particle.empty())
		{
			// initialize particles
			position_particle.resize(n_el * ppe);
			velocity_particle.resize(n_el * ppe);
			cellI_particle.resize(n_el * ppe);
#ifdef POLYFEM_WITH_TBB
			tbb::parallel_for(0, n_el, 1, [&](int e)
#else
			for (int e = 0; e < n_el; ++e)
#endif
							  {
								  // sample particle in element e
								  Eigen::MatrixXd local_pts_particle;
								  local_pts_particle.setRandom(ppe, dim);
								  local_pts_particle.array() += 1;
								  local_pts_particle.array() /= 2;
								  for (int ppeI = 0; ppeI < ppe; ++ppeI)
								  {
									  if (shape == 3 && dim == 2 && local_pts_particle.row(ppeI).sum() > 1)
									  {
										  double x = 1 - local_pts_particle(ppeI, 1);
										  local_pts_particle(ppeI, 1) = 1 - local_pts_particle(ppeI, 0);
										  local_pts_particle(ppeI, 0) = x;
										  // TODO: dim == 3
									  }
								  }

								  for (int i = 0; i < shape; ++i)
									  cellI_particle[e * ppe + i] = e;

								  // compute global position and velocity of particles
								  // construct interpolant (linear for position)
								  Eigen::MatrixXd mapped;
								  gbases[e].eval_geom_mapping(local_pts_particle, mapped);
								  // construct interpolant (for velocity)
								  assembler::ElementAssemblyValues vals;
								  vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
								  for (int j = 0; j < ppe; ++j)
								  {
									  position_particle[ppe * e + j].setZero(1, dim);
									  for (int d = 0; d < dim; d++)
									  {
										  position_particle[ppe * e + j](d) = mapped(j, d);
									  }

									  velocity_particle[e * ppe + j].setZero(1, dim);
									  for (int i = 0; i < vals.basis_values.size(); ++i)
									  {
										  velocity_particle[e * ppe + j] += vals.basis_values[i].val(j) * sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
									  }
								  }
							  }
#ifdef POLYFEM_WITH_TBB
			);
#endif
		}
		else
		{
			// resample particles
			// count particle per cell
			std::vector<int> counter(n_el, 0);
			std::vector<int> redundantPI;
			std::vector<bool> isRedundant(cellI_particle.size(), false);
			for (int pI = 0; pI < cellI_particle.size(); ++pI)
			{
				if (cellI_particle[pI] < 0)
				{
					redundantPI.emplace_back(pI);
					isRedundant[pI] = true;
				}
				else
				{
					++counter[cellI_particle[pI]];
					if (counter[cellI_particle[pI]] > ppe)
					{
						redundantPI.emplace_back(pI);
						isRedundant[pI] = true;
					}
				}
			}
			// g2p -- update velocity
#ifdef POLYFEM_WITH_TBB
			tbb::parallel_for(0, (int)cellI_particle.size(), 1, [&](int pI)
#else
			for (int pI = 0; pI < cellI_particle.size(); ++pI)
#endif
							  {
								  if (!isRedundant[pI])
								  {
									  int e = cellI_particle[pI];
									  Eigen::MatrixXd local_pts_particle;
									  calculate_local_pts(gbases[e], e, position_particle[pI], local_pts_particle);

									  assembler::ElementAssemblyValues vals;
									  vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
									  RowVectorNd FLIPdVel, PICVel;
									  FLIPdVel.setZero(1, dim);
									  PICVel.setZero(1, dim);
									  for (int i = 0; i < vals.basis_values.size(); ++i)
									  {
										  FLIPdVel += vals.basis_values[i].val(0) * (sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1) - new_sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1)).transpose();
										  PICVel += vals.basis_values[i].val(0) * sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
									  }
									  velocity_particle[pI] = (1.0 - FLIPRatio) * PICVel + FLIPRatio * (velocity_particle[pI] + FLIPdVel);
								  }
							  }
#ifdef POLYFEM_WITH_TBB
			);
#endif
			// resample
			for (int e = 0; e < n_el; ++e)
			{
				if (counter[e] >= ppe)
				{
					continue;
				}

				while (counter[e] < ppe)
				{
					int pI = redundantPI.back();
					redundantPI.pop_back();

					cellI_particle[pI] = e;

					// sample particle in element e
					Eigen::MatrixXd local_pts_particle;
					local_pts_particle.setRandom(1, dim);
					local_pts_particle.array() += 1;
					local_pts_particle.array() /= 2;
					if (shape == 3 && dim == 2 && local_pts_particle.sum() > 1)
					{
						double x = 1 - local_pts_particle(0, 1);
						local_pts_particle(0, 1) = 1 - local_pts_particle(0, 0);
						local_pts_particle(0, 0) = x;
						// TODO: dim == 3
					}

					// compute global position and velocity of particles
					// construct interpolant (linear for position)
					Eigen::MatrixXd mapped;
					gbases[e].eval_geom_mapping(local_pts_particle, mapped);
					for (int d = 0; d < dim; d++)
						position_particle[pI](d) = mapped(0, d);

					// construct interpolant (for velocity)
					assembler::ElementAssemblyValues vals;
					vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
					velocity_particle[pI].setZero(1, dim);
					for (int i = 0; i < vals.basis_values.size(); ++i)
					{
						velocity_particle[pI] += vals.basis_values[i].val(0) * sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
					}

					++counter[e];
				}
			}
		}

		// advect
		std::vector<assembler::ElementAssemblyValues> velocity_interpolator(ppe * n_el);
#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, (int)(ppe * n_el), 1, [&](int pI)
#else
		for (int pI = 0; pI < ppe * n_el; ++pI)
#endif
						  {
							  // update particle position via advection
							  RowVectorNd newvel;
							  Eigen::MatrixXd local_pos;
							  cellI_particle[pI] = trace_back(gbases, bases, position_particle[pI], velocity_particle[pI],
															  position_particle[pI], newvel, local_pos, sol, -dt);

							  // RK3:
							  // RowVectorNd bypass, vel2, vel3;
							  // trace_back( gbases, bases, position_particle[pI], velocity_particle[pI],
							  //     bypass, vel2, sol, -0.5 * dt);
							  // trace_back( gbases, bases, position_particle[pI], vel2,
							  //     bypass, vel3, sol, -0.75 * dt);
							  // trace_back( gbases, bases, position_particle[pI],
							  //     2 * velocity_particle[pI] + 3 * vel2 + 4 * vel3,
							  //     position_particle[pI], bypass, sol, -dt / 9);

							  // prepare P2G
							  if (cellI_particle[pI] >= 0)
							  {
								  // construct interpolator (always linear for P2G, can use gaussian or bspline later)
								  velocity_interpolator[pI].compute(cellI_particle[pI], dim == 3, local_pos,
																	gbases[cellI_particle[pI]], gbases[cellI_particle[pI]]);
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif

		// P2G
		new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
		new_sol_w = Eigen::MatrixXd::Zero(sol.size() / dim, 1);
		new_sol_w.array() += 1e-13;
		for (int pI = 0; pI < ppe * n_el; ++pI)
		{
			int cellI = cellI_particle[pI];
			if (cellI == -1)
			{
				continue;
			}
			assembler::ElementAssemblyValues &vals = velocity_interpolator[pI];
			for (int i = 0; i < vals.basis_values.size(); ++i)
			{
				new_sol.block(bases[cellI].bases[i].global()[0].index * dim, 0, dim, 1) +=
					vals.basis_values[i].val(0) * velocity_particle[pI].transpose();
				new_sol_w(bases[cellI].bases[i].global()[0].index) += vals.basis_values[i].val(0);
			}
		}
		// TODO: need to add up boundary velocities and weights because of perodic BC

#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, (int)new_sol.rows() / dim, 1, [&](int i)
#else
		for (int i = 0; i < new_sol.rows() / dim; ++i)
#endif
						  {
							  new_sol.block(i * dim, 0, dim, 1) /= new_sol_w(i, 0);
							  sol.block(i * dim, 0, dim, 1) = new_sol.block(i * dim, 0, dim, 1);
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
	}

	void OperatorSplittingSolver::advection_PIC(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &gbases, const std::vector<basis::ElementBases> &bases, Eigen::MatrixXd &sol, const double dt, const Eigen::MatrixXd &local_pts, const int order)
	{
		// to store new velocity and weights for particle grid transfer
		Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
		Eigen::MatrixXd new_sol_w = Eigen::MatrixXd::Zero(sol.size() / dim, 1);
		new_sol_w.array() += 1e-13;

		const int ppe = shape; // particle per element
		std::vector<assembler::ElementAssemblyValues> velocity_interpolator(ppe * n_el);
		position_particle.resize(ppe * n_el);
		velocity_particle.resize(ppe * n_el);
		cellI_particle.resize(ppe * n_el);
#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, n_el, 1, [&](int e)
#else
		for (int e = 0; e < n_el; ++e)
#endif
						  {
							  // resample particle in element e
							  Eigen::MatrixXd local_pts_particle;
							  local_pts_particle.setRandom(ppe, dim);
							  local_pts_particle.array() += 1;
							  local_pts_particle.array() /= 2;

							  // geometry vertices of element e
							  std::vector<RowVectorNd> vert(shape);
							  for (int i = 0; i < shape; ++i)
							  {
								  vert[i] = mesh.point(mesh.cell_vertex(e, i));
							  }

							  // construct interpolant (linear for position)
							  assembler::ElementAssemblyValues gvals;
							  gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);

							  // compute global position of particles
							  for (int i = 0; i < ppe; ++i)
							  {
								  position_particle[ppe * e + i].setZero(1, dim);
								  for (int j = 0; j < shape; ++j)
								  {
									  position_particle[ppe * e + i] += gvals.basis_values[j].val(i) * vert[j];
								  }
							  }

							  // compute velocity
							  assembler::ElementAssemblyValues vals;
							  vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
							  for (int j = 0; j < ppe; ++j)
							  {
								  velocity_particle[e * ppe + j].setZero(1, dim);
								  for (int i = 0; i < vals.basis_values.size(); ++i)
								  {
									  velocity_particle[e * ppe + j] += vals.basis_values[i].val(j) * sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
								  }
							  }

							  // update particle position via advection
							  for (int j = 0; j < ppe; ++j)
							  {
								  RowVectorNd newvel;
								  Eigen::MatrixXd local_pos;
								  cellI_particle[ppe * e + j] = trace_back(gbases, bases, position_particle[ppe * e + j], velocity_particle[e * ppe + j],
																		   position_particle[ppe * e + j], newvel, local_pos, sol, -dt);

								  // RK3:
								  // RowVectorNd bypass, vel2, vel3;
								  // trace_back( gbases, bases, position_particle[ppe * e + i], velocity_particle[e * ppe + i],
								  //     bypass, vel2, sol, -0.5 * dt);
								  // trace_back( gbases, bases, position_particle[ppe * e + i], vel2,
								  //     bypass, vel3, sol, -0.75 * dt);
								  // trace_back( gbases, bases, position_particle[ppe * e + i],
								  //     2 * velocity_particle[e * ppe + i] + 3 * vel2 + 4 * vel3,
								  //     position_particle[ppe * e + i], bypass, sol, -dt / 9);

								  // prepare P2G
								  if (cellI_particle[ppe * e + j] >= 0)
								  {
									  // construct interpolator (always linear for P2G, can use gaussian or bspline later)
									  velocity_interpolator[ppe * e + j].compute(cellI_particle[ppe * e + j], dim == 3, local_pos,
																				 gbases[cellI_particle[ppe * e + j]], gbases[cellI_particle[ppe * e + j]]);
								  }
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif

		// P2G
		for (int e = 0; e < n_el; ++e)
		{
			for (int j = 0; j < ppe; ++j)
			{
				int cellI = cellI_particle[ppe * e + j];
				if (cellI == -1)
				{
					continue;
				}
				assembler::ElementAssemblyValues &vals = velocity_interpolator[ppe * e + j];
				for (int i = 0; i < vals.basis_values.size(); ++i)
				{
					new_sol.block(bases[cellI].bases[i].global()[0].index * dim, 0, dim, 1) +=
						vals.basis_values[i].val(0) * velocity_particle[ppe * e + j].transpose();
					new_sol_w(bases[cellI].bases[i].global()[0].index) += vals.basis_values[i].val(0);
				}
			}
		}
		// TODO: need to add up boundary velocities and weights because of perodic BC

#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, (int)new_sol.rows() / dim, 1, [&](int i)
#else
		for (int i = 0; i < new_sol.rows() / dim; ++i)
#endif
						  {
							  sol.block(i * dim, 0, dim, 1) = new_sol.block(i * dim, 0, dim, 1) / new_sol_w(i, 0);
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
		// TODO: need to think about what to do with negative quadratic weight
	}

	void OperatorSplittingSolver::solve_diffusion_1st(const StiffnessMatrix &mass, const std::vector<int> &bnd_nodes, Eigen::MatrixXd &sol)
	{
		Eigen::VectorXd rhs;

		for (int d = 0; d < dim; d++)
		{
			Eigen::VectorXd x(sol.size() / dim);
			for (int j = 0; j < x.size(); j++)
			{
				x(j) = sol(j * dim + d);
			}
			rhs = mass * x;

			// keep dirichlet bc
			for (int i = 0; i < bnd_nodes.size(); i++)
			{
				rhs(bnd_nodes[i]) = x(bnd_nodes[i]);
			}

			// if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
			{
				dirichlet_solve_prefactorized(*solver_diffusion, mat_diffusion, rhs, bnd_nodes, x);
			}
			// else
			// {
			// dirichlet_solve(*solver_diffusion, mat_diffusion, rhs, bnd_nodes, x, mat_diffusion.rows(), "", false, false, false);
			// }

			for (int j = 0; j < x.size(); j++)
			{
				sol(j * dim + d) = x(j);
			}
		}
	}

	void OperatorSplittingSolver::external_force(const mesh::Mesh &mesh,
												 const assembler::Assembler &assembler,
												 const std::vector<basis::ElementBases> &gbases,
												 const std::vector<basis::ElementBases> &bases,
												 const double dt,
												 Eigen::MatrixXd &sol,
												 const Eigen::MatrixXd &local_pts,
												 const std::shared_ptr<assembler::Problem> problem,
												 const double time)
	{
#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for(0, n_el, 1, [&](int e)
#else
		for (int e = 0; e < n_el; e++)
#endif
						  {
							  Eigen::MatrixXd mapped;
							  gbases[e].eval_geom_mapping(local_pts, mapped);

							  for (int local_idx = 0; local_idx < bases[e].bases.size(); local_idx++)
							  {
								  int global_idx = bases[e].bases[local_idx].global()[0].index;

								  Eigen::MatrixXd pos = Eigen::MatrixXd::Zero(1, dim);
								  for (int d = 0; d < dim; d++)
									  pos(0, d) = mapped(local_idx, d);

								  Eigen::MatrixXd val;
								  problem->rhs(assembler, pos, time, val);

								  for (int d = 0; d < dim; d++)
								  {
									  sol(global_idx * dim + d) += val(d) * dt;
								  }
							  }
						  }
#ifdef POLYFEM_WITH_TBB
		);
#endif
	}

	void OperatorSplittingSolver::solve_pressure(const StiffnessMatrix &mixed_stiffness, const std::vector<int> &pressure_boundary_nodes, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		Eigen::VectorXd rhs;
		if (pressure_boundary_nodes.size() == 0)
			rhs = Eigen::VectorXd::Zero(mixed_stiffness.rows() + 1); // mixed_stiffness * sol;
		else
			rhs = Eigen::VectorXd::Zero(mixed_stiffness.rows());     // mixed_stiffness * sol;

		Eigen::VectorXd temp = mixed_stiffness * sol;
		for (int i = 0; i < temp.rows(); i++)
		{
			rhs(i) = temp(i);
		}

		Eigen::VectorXd x = Eigen::VectorXd::Zero(rhs.size());
		for (int i = 0; i < pressure.size(); i++)
		{
			x(i) = pressure(i);
		}

		for (int i = 0; i < pressure_boundary_nodes.size(); i++)
		{
			rhs(pressure_boundary_nodes[i]) = 0;
			x(pressure_boundary_nodes[i]) = 0;
		}
		// if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
		// {
		dirichlet_solve_prefactorized(*solver_projection, mat_projection, rhs, pressure_boundary_nodes, x);
		// }
		// else
		// {
		// dirichlet_solve(*solver_projection, mat_projection, rhs, std::vector<int>(), x, mat_projection.rows() - 1, "", false, false, false);
		// }

		if (pressure_boundary_nodes.size() == 0)
			pressure = x.head(x.size() - 1);
		else
			pressure = x;
	}

	void OperatorSplittingSolver::projection(const StiffnessMatrix &velocity_mass, const StiffnessMatrix &mixed_stiffness, const std::vector<int> &boundary_nodes_, Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
	{
		Eigen::VectorXd rhs = mixed_stiffness.transpose() * pressure;
		Eigen::VectorXd dx = Eigen::VectorXd::Zero(sol.size());

		for (int i = 0; i < boundary_nodes_.size(); i++)
		{
			rhs(boundary_nodes_[i]) = 0;
		}

		if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
		{
			dirichlet_solve_prefactorized(*solver_mass, velocity_mass, rhs, boundary_nodes_, dx);
		}
		else
		{
			auto mat = velocity_mass;
			dirichlet_solve(*solver_mass, mat, rhs, boundary_nodes_, dx, velocity_mass.rows(), "", false, false, false);
		}

		sol -= dx;
	}

	void OperatorSplittingSolver::projection(int n_bases,
											 const std::vector<basis::ElementBases> &gbases,
											 const std::vector<basis::ElementBases> &bases,
											 const std::vector<basis::ElementBases> &pressure_bases,
											 const Eigen::MatrixXd &local_pts,
											 Eigen::MatrixXd &pressure,
											 Eigen::MatrixXd &sol)
	{
		Eigen::VectorXd grad_pressure = Eigen::VectorXd::Zero(n_bases * dim);
		Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_bases);

		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < n_el; ++e)
		{
			vals.compute(e, dim == 3, local_pts, pressure_bases[e], gbases[e]);
			for (int j = 0; j < local_pts.rows(); j++)
			{
				int global_ = bases[e].bases[j].global()[0].index;
				for (int i = 0; i < vals.basis_values.size(); i++)
				{
					for (int d = 0; d < dim; d++)
					{
						assert(pressure_bases[e].bases[i].global().size() == 1);
						grad_pressure(global_ * dim + d) += vals.basis_values[i].grad_t_m(j, d) * pressure(pressure_bases[e].bases[i].global()[0].index);
					}
				}
				traversed(global_)++;
			}
		}
		for (int i = 0; i < traversed.size(); i++)
		{
			for (int d = 0; d < dim; d++)
			{
				sol(i * dim + d) -= grad_pressure(i * dim + d) / traversed(i);
			}
		}
	}

	void OperatorSplittingSolver::initialize_density(const std::shared_ptr<assembler::Problem> &problem)
	{
		Eigen::MatrixXd pts(1, dim);
		Eigen::MatrixXd tmp;
		for (long i = 0; i <= grid_cell_num(0); i++)
		{
			pts(0, 0) = i * resolution + min_domain(0);
			for (long j = 0; j <= grid_cell_num(1); j++)
			{
				pts(0, 1) = j * resolution + min_domain(1);
				if (dim == 2)
				{
					const long idx = i + j * (grid_cell_num(0) + 1);
					problem->initial_density(pts, tmp);
					density(idx) = tmp(0);
				}
				else
				{
					for (long k = 0; k <= grid_cell_num(2); k++)
					{
						pts(0, 2) = k * resolution + min_domain(2);
						const long idx = i + (j + k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);
						problem->initial_density(pts, tmp);
						density(idx) = tmp(0);
					}
				}
			}
		}
	}

	long OperatorSplittingSolver::search_cell(const std::vector<basis::ElementBases> &gbases, const RowVectorNd &pos, Eigen::MatrixXd &local_pts)
	{
		Eigen::Matrix<long, Eigen::Dynamic, 1> pos_int(dim);
		for (int d = 0; d < dim; d++)
		{
			pos_int(d) = floor((pos(d) - min_domain(d)) / (max_domain(d) - min_domain(d)) * hash_table_cell_num(d));
			if (pos_int(d) < 0)
				return -1;
			else if (pos_int(d) >= hash_table_cell_num(d))
				return -1;
		}

		long idx = 0, dim_num = 1;
		for (int d = 0; d < dim; d++)
		{
			idx += pos_int(d) * dim_num;
			dim_num *= hash_table_cell_num(d);
		}

		const std::vector<long> &list = hash_table[idx];
		for (auto it = list.begin(); it != list.end(); it++)
		{
			calculate_local_pts(gbases[*it], *it, pos, local_pts);

			if (shape == dim + 1)
			{
				if (local_pts.minCoeff() > -1e-13 && local_pts.sum() < 1 + 1e-13)
					return *it;
			}
			else
			{
				if (local_pts.minCoeff() > -1e-13 && local_pts.maxCoeff() < 1 + 1e-13)
					return *it;
			}
		}
		return -1; // not inside any elem
	}

	bool OperatorSplittingSolver::outside_quad(const std::vector<RowVectorNd> &vert, const RowVectorNd &pos)
	{
		double a = (vert[1](0) - vert[0](0)) * (pos(1) - vert[0](1)) - (vert[1](1) - vert[0](1)) * (pos(0) - vert[0](0));
		double b = (vert[2](0) - vert[1](0)) * (pos(1) - vert[1](1)) - (vert[2](1) - vert[1](1)) * (pos(0) - vert[1](0));
		double c = (vert[3](0) - vert[2](0)) * (pos(1) - vert[2](1)) - (vert[3](1) - vert[2](1)) * (pos(0) - vert[2](0));
		double d = (vert[0](0) - vert[3](0)) * (pos(1) - vert[3](1)) - (vert[0](1) - vert[3](1)) * (pos(0) - vert[3](0));

		if ((a > 0 && b > 0 && c > 0 && d > 0) || (a < 0 && b < 0 && c < 0 && d < 0))
			return false;
		return true;
	}

	void OperatorSplittingSolver::compute_gbase_val(const int elem_idx, const Eigen::MatrixXd &local_pos, Eigen::MatrixXd &pos)
	{
		pos = Eigen::MatrixXd::Zero(1, dim);
		Eigen::VectorXd weights(shape);
		const double x = local_pos(0);
		const double y = local_pos(1);
		if (dim == 2)
		{
			if (shape == 3)
			{
				weights << 1 - x - y, x, y;
			}
			else
			{
				weights << (1 - x) * (1 - y), x * (1 - y),
					x * y, y * (1 - x);
			}
		}
		else
		{
			const double z = local_pos(2);
			if (shape == 4)
			{
				weights << 1 - x - y - z, x, y, z;
			}
			else
			{
				weights << (1 - x) * (1 - y) * (1 - z),
					x * (1 - y) * (1 - z),
					x * y * (1 - z),
					(1 - x) * y * (1 - z),
					(1 - x) * (1 - y) * z,
					x * (1 - y) * z,
					x * y * z,
					(1 - x) * y * z;
			}
		}

		for (int d = 0; d < dim; d++)
		{
			for (int j = 0; j < shape; j++)
			{
				pos(d) += V(T(elem_idx, j), d) * weights(j);
			}
		}
	}

	void OperatorSplittingSolver::compute_gbase_jacobi(const int elem_idx, const Eigen::MatrixXd &local_pos, Eigen::MatrixXd &jacobi)
	{
		jacobi = Eigen::MatrixXd::Zero(dim, dim);
		Eigen::MatrixXd weights(shape, dim);
		const double x = local_pos(0);
		const double y = local_pos(1);
		if (dim == 2)
		{
			if (shape == 3)
			{
				weights << -1, -1,
					1, 0,
					0, 1;
			}
			else
			{
				weights << y - 1, x - 1,
					1 - y, -x,
					y, x,
					-y, 1 - x;
			}
		}
		else
		{
			const double z = local_pos(2);
			if (shape == 4)
			{
				weights << -1, -1, -1,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1;
			}
			else
			{
				weights << -(1 - y) * (1 - z), -(1 - x) * (1 - z), -(1 - x) * (1 - y),
					(1 - y) * (1 - z), -x * (1 - z), -x * (1 - y),
					y * (1 - z), x * (1 - z), -x * y,
					-y * (1 - z), (1 - x) * (1 - z), -(1 - x) * y,
					-(1 - y) * z, -(1 - x) * z, (1 - x) * (1 - y),
					(1 - y) * z, -x * z, x * (1 - y),
					y * z, x * z, x * y,
					-y * z, (1 - x) * z, (1 - x) * y;
			}
		}

		for (int d1 = 0; d1 < dim; d1++)
		{
			for (int d2 = 0; d2 < dim; d2++)
			{
				for (int i = 0; i < shape; i++)
				{
					jacobi(d1, d2) += weights(i, d2) * V(T(elem_idx, i), d1);
				}
			}
		}
	}

	void OperatorSplittingSolver::calculate_local_pts(const basis::ElementBases &gbase,
													  const int elem_idx,
													  const RowVectorNd &pos,
													  Eigen::MatrixXd &local_pos)
	{
		local_pos = Eigen::MatrixXd::Zero(1, dim);

		// if(shape == 4 && dim == 2 && outside_quad(vert, pos))
		// {
		//     local_pos(0) = local_pos(1) = -1;
		//     return;
		// }
		Eigen::MatrixXd res;
		int iter_times = 0;
		int max_iter = 20;
		do
		{
			res = -pos;

			Eigen::MatrixXd mapped;
			gbase.eval_geom_mapping(local_pos, mapped);
			for (int d = 0; d < dim; d++)
				res(d) += mapped(0, d);

			std::vector<Eigen::MatrixXd> grads;
			gbase.eval_geom_mapping_grads(local_pos, grads);
			Eigen::MatrixXd jacobi = grads[0].transpose();

			Eigen::VectorXd delta = jacobi.colPivHouseholderQr().solve(res.transpose());
			for (int d = 0; d < dim; d++)
			{
				local_pos(d) -= delta(d);
			}
			iter_times++;
			if (shape == dim + 1)
				break;
		} while (res.norm() > 1e-12 && iter_times < max_iter);

		if (iter_times >= max_iter)
		{
			for (int d = 0; d < dim; d++)
				local_pos(d) = -1;
		}
	}

	void OperatorSplittingSolver::save_density()
	{
#ifdef POLYFEM_WITH_OPENVDB
		openvdb::initialize();
		openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
		openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

		for (int i = 0; i <= grid_cell_num(0); i++)
		{
			for (int j = 0; j <= grid_cell_num(1); j++)
			{
				if (dim == 2)
				{
					const int idx = i + j * (grid_cell_num(0) + 1);
					openvdb::Coord xyz(i, j, 0);
					if (density(idx) > 1e-8)
						accessor.setValue(xyz, density(idx));
				}
				else
				{
					for (int k = 0; k <= grid_cell_num(2); k++)
					{
						const int idx = i + (j + k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);
						openvdb::Coord xyz(i, j, k);
						if (density(idx) > 1e-8)
							accessor.setValue(xyz, density(idx));
					}
				}
			}
		}
		grid->setName("density_smoke");
		grid->setGridClass(openvdb::GRID_FOG_VOLUME);

		static int num_frame = 0;
		const std::string filename = "density" + std::to_string(num_frame) + ".vdb";
		openvdb::io::File file(filename.c_str());
		num_frame++;

		openvdb::GridPtrVec(grids);
		grids.push_back(grid);
		file.write(grids);
		file.close();
#else
		static int num_frame = 0;
		std::string name = "density" + std::to_string(num_frame) + ".txt";
		std::ofstream file(name.c_str());
		num_frame++;
		for (int i = 0; i <= grid_cell_num(0); i++)
		{
			for (int j = 0; j <= grid_cell_num(1); j++)
			{
				if (dim == 2)
				{
					const int idx = i + j * (grid_cell_num(0) + 1);
					if (density(idx) < 1e-10)
						continue;
					file << i << " " << j << " " << density(idx) << std::endl;
				}
				else
				{
					for (int k = 0; k <= grid_cell_num(2); k++)
					{
						const int idx = i + (j + k * (grid_cell_num(1) + 1)) * (grid_cell_num(0) + 1);
						if (density(idx) < 1e-10)
							continue;
						file << i << " " << j << " " << k << " " << density(idx) << std::endl;
					}
				}
			}
		}
		file.close();
#endif
	};
} // namespace polyfem::solver