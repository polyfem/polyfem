#include <polyfem/RhsAssembler.hpp>

#include <polyfem/BoundarySampler.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/Logger.hpp>

#include <Eigen/Sparse>

#ifdef POLYFEM_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#endif

#include <iostream>
#include <map>
#include <memory>

namespace polyfem
{
	using namespace polysolve;
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

	RhsAssembler::RhsAssembler(const Mesh &mesh, const int n_basis, const int size, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases, const std::string &formulation, const Problem &problem)
		: mesh_(mesh), n_basis_(n_basis), size_(size), bases_(bases), gbases_(gbases), formulation_(formulation), problem_(problem)
	{
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
				vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);

				const Quadrature &quadrature = vals.quadrature;

				problem_.rhs(formulation_, vals.val, t, rhs_fun);

				for (int d = 0; d < size_; ++d)
				{
					//rhs_fun.col(d) = rhs_fun.col(d).array() * vals.det.array() * quadrature.weights.array();
					for (int q = 0; q < quadrature.weights.size(); ++q)
					{
						const double rho = density(vals.val(q, 0), vals.val(q, 1), size_ == 2 ? 0. : vals.val(q, 2), vals.element_id);
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
		time_bc([&](const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) { problem_.initial_solution(pts, val); }, sol);
	}

	void RhsAssembler::initial_velocity(Eigen::MatrixXd &sol) const
	{
		time_bc([&](const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) { problem_.initial_velocity(pts, val); }, sol);
	}

	void RhsAssembler::initial_acceleration(Eigen::MatrixXd &sol) const
	{
		time_bc([&](const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) { problem_.initial_acceleration(pts, val); }, sol);
	}

	void RhsAssembler::time_bc(const std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &)> &fun, Eigen::MatrixXd &sol) const
	{
		sol = Eigen::MatrixXd::Zero(n_basis_ * size_, 1);
		Eigen::MatrixXd loc_sol;

		const int n_elements = int(bases_.size());
		ElementAssemblyValues vals;
		for (int e = 0; e < n_elements; ++e)
		{
			vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);

			const Quadrature &quadrature = vals.quadrature;
			//problem_.initial_solution(vals.val, loc_sol);
			fun(vals.val, loc_sol);

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

		StiffnessMatrix mass;
		json params = {
			{"mtype", -2}, // matrix type for Pardiso (2 = SPD)
						   // {"max_iter", 0}, // for iterative solvers
						   // {"tolerance", 1e-9}, // for iterative solvers
		};
		const auto &assembler = AssemblerUtils::instance();
		Density d;
		assembler.assemble_mass_matrix(formulation_, size_ == 3, n_basis_, d, bases_, gbases_, mass);
		auto solver = LinearSolver::create(LinearSolver::defaultSolver(), LinearSolver::defaultPrecond());
		solver->setParameters(params);
		solver->analyzePattern(mass, mass.rows());
		solver->factorize(mass);
		Eigen::MatrixXd b = sol;
		sol.setZero();
		for (long i = 0; i < b.cols(); ++i)
		{
			solver->solve(b.col(i), sol.col(i));
		}
		logger().trace("mass matrix error {}", (mass * sol - b).norm());
	}

	void RhsAssembler::set_bc(
		const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &df,
		const std::function<void(const Eigen::MatrixXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::MatrixXd &)> &nf,
		const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs) const
	{
		const int n_el = int(bases_.size());

		Eigen::MatrixXd uv, samples, gtmp, rhs_fun;
		Eigen::VectorXi global_primitive_ids;

		int index = 0;
		std::vector<int> indices;
		indices.reserve(n_el * 10);
		// std::map<int, int> global_index_to_col;

		long total_size = 0;

		Eigen::Matrix<bool, Eigen::Dynamic, 1> is_boundary(n_basis_);
		is_boundary.setConstant(false);
		Eigen::VectorXi global_index_to_col(n_basis_);
		global_index_to_col.setConstant(-1);

		const int actual_dim = problem_.is_scalar() ? 1 : mesh_.dimension();

		// assert((bounday_nodes.size()/actual_dim)*actual_dim == bounday_nodes.size());

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

		for (const auto &lb : local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = sample_boundary(lb, resolution, true, uv, samples, global_primitive_ids);

			if (!has_samples)
				continue;

			const ElementBases &bs = bases_[e];
			const int n_local_bases = int(bs.bases.size());

			total_size += samples.rows();

			for (int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b = bs.bases[j];

				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					//pt found
					// if(std::find(bounday_nodes.begin(), bounday_nodes.end(), size_ * b.global()[ii].index) != bounday_nodes.end())
					if (is_boundary[b.global()[ii].index])
					{
						//if(global_index_to_col.find( b.global()[ii].index ) == global_index_to_col.end())
						if (global_index_to_col(b.global()[ii].index) == -1)
						{
							// global_index_to_col[b.global()[ii].index] = index++;
							global_index_to_col(b.global()[ii].index) = index++;
							indices.push_back(b.global()[ii].index);
							assert(indices.size() == size_t(index));
						}
					}
				}
			}
		}

		// Eigen::MatrixXd global_mat = Eigen::MatrixXd::Zero(total_size, indices.size());
		Eigen::MatrixXd global_rhs = Eigen::MatrixXd::Zero(total_size, size_);

		const long buffer_size = total_size * long(indices.size());
		std::vector<Eigen::Triplet<double>> entries, entries_t;
		// entries.reserve(buffer_size);
		// entries_t.reserve(buffer_size);

		index = 0;

		int global_counter = 0;
		Eigen::MatrixXd mapped;

		std::vector<AssemblyValues> tmp_val;

		for (const auto &lb : local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = sample_boundary(lb, resolution, false, uv, samples, global_primitive_ids);

			if (!has_samples)
				continue;

			const ElementBases &bs = bases_[e];
			const ElementBases &gbs = gbases_[e];
			const int n_local_bases = int(bs.bases.size());

			gbs.eval_geom_mapping(samples, mapped);

			bs.evaluate_bases(samples, tmp_val);
			for (int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &tmp = tmp_val[j].val;

				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					// auto item = global_index_to_col.find(b.global()[ii].index);
					// if(item != global_index_to_col.end()){
					auto item = global_index_to_col(b.global()[ii].index);
					if (item != -1)
					{
						for (int k = 0; k < int(tmp.size()); ++k)
						{
							// entries.push_back(Eigen::Triplet<double>(global_counter+k, item->second, tmp(k, j) * b.global()[ii].val));
							// entries_t.push_back(Eigen::Triplet<double>(item->second, global_counter+k, tmp(k, j) * b.global()[ii].val));
							entries.push_back(Eigen::Triplet<double>(global_counter + k, item, tmp(k) * b.global()[ii].val));
							entries_t.push_back(Eigen::Triplet<double>(item, global_counter + k, tmp(k) * b.global()[ii].val));
						}
						// global_mat.block(global_counter, item->second, tmp.size(), 1) = tmp;
					}
				}
			}

			// problem_.bc(mesh_, global_primitive_ids, mapped, t, rhs_fun);
			df(global_primitive_ids, uv, mapped, rhs_fun);
			global_rhs.block(global_counter, 0, rhs_fun.rows(), rhs_fun.cols()) = rhs_fun;
			global_counter += rhs_fun.rows();

			//UIState::ui_state().debug_data().add_points(mapped, Eigen::MatrixXd::Constant(1, 3, 0));

			//Eigen::MatrixXd asd(mapped.rows(), 3);
			//asd.col(0)=mapped.col(0);
			//asd.col(1)=mapped.col(1);
			//asd.col(2)=rhs_fun;
			//UIState::ui_state().debug_data().add_points(asd, Eigen::MatrixXd::Constant(1, 3, 0));
		}

		assert(global_counter == total_size);

		if (total_size > 0)
		{
			const double mmin = global_rhs.minCoeff();
			const double mmax = global_rhs.maxCoeff();

			if (fabs(mmin) < 1e-8 && fabs(mmax) < 1e-8)
			{
				// std::cout<<"is all zero, skipping"<<std::endl;
				for (size_t i = 0; i < indices.size(); ++i)
				{
					for (int d = 0; d < size_; ++d)
					{
						if (problem_.all_dimentions_dirichelt() || std::find(bounday_nodes.begin(), bounday_nodes.end(), indices[i] * size_ + d) != bounday_nodes.end())
							rhs(indices[i] * size_ + d) = 0;
					}
				}
			}
			else
			{
				StiffnessMatrix mat(int(total_size), int(indices.size()));
				mat.setFromTriplets(entries.begin(), entries.end());

				StiffnessMatrix mat_t(int(indices.size()), int(total_size));
				mat_t.setFromTriplets(entries_t.begin(), entries_t.end());

				StiffnessMatrix A = mat_t * mat;
				Eigen::MatrixXd b = mat_t * global_rhs;

				Eigen::MatrixXd coeffs(b.rows(), b.cols());

				json params = {
					{"mtype", -2}, // matrix type for Pardiso (2 = SPD)
								   // {"max_iter", 0}, // for iterative solvers
								   // {"tolerance", 1e-9}, // for iterative solvers
				};

				// auto solver = LinearSolver::create("", "");
				auto solver = LinearSolver::create(LinearSolver::defaultSolver(), LinearSolver::defaultPrecond());
				solver->setParameters(params);
				solver->analyzePattern(A, A.rows());
				solver->factorize(A);
				coeffs.setZero();
				for (long i = 0; i < b.cols(); ++i)
				{
					solver->solve(b.col(i), coeffs.col(i));
				}
				logger().trace("RHS solve error {}", (A * coeffs - b).norm());

				for (long i = 0; i < coeffs.rows(); ++i)
				{
					for (int d = 0; d < size_; ++d)
					{
						if (problem_.all_dimentions_dirichelt() || std::find(bounday_nodes.begin(), bounday_nodes.end(), indices[i] * size_ + d) != bounday_nodes.end())
							rhs(indices[i] * size_ + d) = coeffs(i, d);
					}
				}
			}
		}

		//Neumann
		Eigen::MatrixXd points, normals;
		Eigen::VectorXd weights;

		ElementAssemblyValues vals;

		// std::ofstream fff("text.obj");
		// int aaaa = 0;
		for (const auto &lb : local_neumann_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = boundary_quadrature(lb, resolution, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const ElementBases &gbs = gbases_[e];
			const ElementBases &bs = bases_[e];

			vals.compute(e, mesh_.is_volume(), points, bs, gbs);

			for (int n = 0; n < vals.jac_it.size(); ++n)
			{
				normals.row(n) = normals.row(n) * vals.jac_it[n];
				normals.row(n).normalize();
			}

			// for (int aa = 0; aa < normals.rows(); ++aa, ++aaaa)
			// {
			// 	fff << "v " <<vals.val(aa, 0) << " " << vals.val(aa, 1);
			// 	if(vals.val.cols() == 2)
			// 		fff<<" 0\n";
			// 	else
			// 		fff << " " << vals.val(aa, 2) << "\n";

			// 	fff << "v " << vals.val(aa, 0) + normals(aa, 0)/100 << " " << vals.val(aa, 1) + normals(aa, 1)/100;
			// 	if (vals.val.cols() == 2)
			// 		fff << " 0\n";
			// 	else
			// 		fff << " " << vals.val(aa, 2) + normals(aa, 2)/100 << "\n";

			// 	fff << "l " << 2 * aaaa + 1 << " " << 2 * aaaa + 2 <<"\n";
			// }

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
								// UIState::ui_state().debug_data().add_points(v.global[g].node, Eigen::RowVector3d(1,0,0));
							}
							// else
							// std::cout<<"skipping "<<g_index<<" "<<rhs_value * v.global[g].val<<std::endl;
						}
					}
				}
			}
		}

		// fff.close();
	}

	void RhsAssembler::set_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t) const
	{
		set_bc(
			[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) { problem_.bc(mesh_, global_ids, uv, pts, t, val); },
			[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, Eigen::MatrixXd &val) { problem_.neumann_bc(mesh_, global_ids, uv, pts, normals, t, val); },
			local_boundary, bounday_nodes, resolution, local_neumann_boundary, rhs);
	}

	void RhsAssembler::set_velocity_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t) const
	{
		set_bc(
			[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) { problem_.velocity_bc(mesh_, global_ids, uv, pts, t, val); },
			[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, Eigen::MatrixXd &val) { problem_.neumann_velocity_bc(mesh_, global_ids, uv, pts, normals, t, val); },
			local_boundary, bounday_nodes, resolution, local_neumann_boundary, rhs);
	}

	void RhsAssembler::set_acceleration_bc(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, Eigen::MatrixXd &rhs, const double t) const
	{
		set_bc(
			[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) { problem_.acceleration_bc(mesh_, global_ids, uv, pts, t, val); },
			[&](const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, Eigen::MatrixXd &val) { problem_.neumann_acceleration_bc(mesh_, global_ids, uv, pts, normals, t, val); },
			local_boundary, bounday_nodes, resolution, local_neumann_boundary, rhs);
	}

	void RhsAssembler::compute_energy_grad(const std::vector<LocalBoundary> &local_boundary, const std::vector<int> &bounday_nodes, const Density &density, const int resolution, const std::vector<LocalBoundary> &local_neumann_boundary, const Eigen::MatrixXd &final_rhs, const double t, Eigen::MatrixXd &rhs) const
	{
		if (problem_.is_linear_in_time())
		{
			if (problem_.is_time_dependent())
				rhs = final_rhs;
			else
				rhs = final_rhs * t;
		}
		else
		{
			assemble(density, rhs, t);
			rhs *= -1;
			set_bc(local_boundary, bounday_nodes, resolution, local_neumann_boundary, rhs, t);
		}
	}

	double RhsAssembler::compute_energy(const Eigen::MatrixXd &displacement, const std::vector<LocalBoundary> &local_neumann_boundary, const Density &density, const int resolution, const double t) const
	{

		double res = 0;
		Eigen::MatrixXd forces;

		if (!problem_.is_rhs_zero())
		{
#ifdef POLYFEM_WITH_TBB
			typedef tbb::enumerable_thread_specific<LocalThreadScalarStorage> LocalStorage;
			LocalStorage storages((LocalThreadScalarStorage()));
#else
			LocalThreadScalarStorage loc_storage;
#endif

			const int n_bases = int(bases_.size());

#ifdef POLYFEM_WITH_TBB
			tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> local_displacement(size_);

		for (int e = r.begin(); e != r.end(); ++e) {
#else
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> local_displacement(size_);

			for (int e = 0; e < n_bases; ++e)
			{
#endif
			ElementAssemblyValues &vals = loc_storage.vals;
			vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);

			const Quadrature &quadrature = vals.quadrature;
			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();


			problem_.rhs(formulation_, vals.val, t, forces);
			assert(forces.rows() == da.size());
			assert(forces.cols() == size_);

			for(long p = 0; p < da.size(); ++p)
			{
				local_displacement.setZero();

				for(size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					const auto &bs = vals.basis_values[i];
					assert(bs.val.size() == da.size());
					const double b_val = bs.val(p);

					for(int d = 0; d < size_; ++d)
					{
						for(std::size_t ii = 0; ii < bs.global.size(); ++ii)
						{
							local_displacement(d) += (bs.global[ii].val * b_val) * displacement(bs.global[ii].index*size_ + d);
						}
					}
				}
				const double rho = density(vals.val(p, 0), vals.val(p, 1), size_ == 2 ? 0. : vals.val(p, 2), vals.element_id);

				for(int d = 0; d < size_; ++d){
					loc_storage.val += forces(p, d) * local_displacement(d) * da(p) * rho;
					// res += forces(p, d) * local_displacement(d) * da(p);
				}
			}
#ifdef POLYFEM_WITH_TBB
		} });
#else
			}
#endif

#ifdef POLYFEM_WITH_TBB
			for (LocalStorage::iterator i = storages.begin(); i != storages.end(); ++i)
			{
				res += i->val;
			}
#else
			res = loc_storage.val;
#endif
		}

		if (problem_.is_linear_in_time() && !problem_.is_time_dependent())
			res *= t;

		// #ifdef POLYFEM_WITH_TBB
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> local_displacement(size_);
		// #endif

		ElementAssemblyValues vals;
		//Neumann
		Eigen::MatrixXd points, uv, normals;
		Eigen::VectorXd weights;
		Eigen::VectorXi global_primitive_ids;
		for (const auto &lb : local_neumann_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = boundary_quadrature(lb, resolution, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const ElementBases &gbs = gbases_[e];
			const ElementBases &bs = bases_[e];

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

	bool RhsAssembler::boundary_quadrature(const LocalBoundary &local_boundary, const int order, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::MatrixXd &normals, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids) const
	{
		uv.resize(0, 0);
		points.resize(0, 0);
		normals.resize(0, 0);
		weights.resize(0);
		global_primitive_ids.resize(0);

		for (int i = 0; i < local_boundary.size(); ++i)
		{
			const int gid = local_boundary.global_primitive_id(i);
			Eigen::MatrixXd tmp_p, tmp_uv, tmp_n;
			Eigen::VectorXd tmp_w;
			switch (local_boundary.type())
			{
			case BoundaryType::TriLine:
				BoundarySampler::quadrature_for_tri_edge(local_boundary[i], order, gid, mesh_, tmp_uv, tmp_p, tmp_w);
				BoundarySampler::normal_for_tri_edge(local_boundary[i], tmp_n);
				break;
			case BoundaryType::QuadLine:
				BoundarySampler::quadrature_for_quad_edge(local_boundary[i], order, gid, mesh_, tmp_uv, tmp_p, tmp_w);
				BoundarySampler::normal_for_quad_edge(local_boundary[i], tmp_n);
				break;
			case BoundaryType::Quad:
				BoundarySampler::quadrature_for_quad_face(local_boundary[i], order, gid, mesh_, tmp_uv, tmp_p, tmp_w);
				BoundarySampler::normal_for_quad_face(local_boundary[i], tmp_n);
				break;
			case BoundaryType::Tri:
				BoundarySampler::quadrature_for_tri_face(local_boundary[i], order, gid, mesh_, tmp_uv, tmp_p, tmp_w);
				BoundarySampler::normal_for_tri_face(local_boundary[i], tmp_n);
				break;
			case BoundaryType::Polygon:
				BoundarySampler::quadrature_for_polygon_edge(local_boundary.element_id(), local_boundary.global_primitive_id(i), order, mesh_, tmp_uv, tmp_p, tmp_w);
				BoundarySampler::normal_for_polygon_edge(local_boundary.element_id(), local_boundary.global_primitive_id(i), mesh_, tmp_n);
				break;
			case BoundaryType::Invalid:
				assert(false);
				break;
			default:
				assert(false);
			}

			uv.conservativeResize(uv.rows() + tmp_uv.rows(), tmp_uv.cols());
			uv.bottomRows(tmp_uv.rows()) = tmp_uv;

			points.conservativeResize(points.rows() + tmp_p.rows(), tmp_p.cols());
			points.bottomRows(tmp_p.rows()) = tmp_p;

			normals.conservativeResize(normals.rows() + tmp_p.rows(), tmp_p.cols());
			for (int k = normals.rows() - tmp_p.rows(); k < normals.rows(); ++k)
				normals.row(k) = tmp_n;

			weights.conservativeResize(weights.rows() + tmp_w.rows(), tmp_w.cols());
			weights.bottomRows(tmp_w.rows()) = tmp_w;

			global_primitive_ids.conservativeResize(global_primitive_ids.rows() + tmp_p.rows());
			global_primitive_ids.bottomRows(tmp_p.rows()).setConstant(gid);
		}

		assert(uv.rows() == global_primitive_ids.size());
		assert(points.rows() == global_primitive_ids.size());
		assert(normals.rows() == global_primitive_ids.size());
		assert(weights.size() == global_primitive_ids.size());

		return true;
	}

	bool RhsAssembler::sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids) const
	{
		uv.resize(0, 0);
		samples.resize(0, 0);
		global_primitive_ids.resize(0);

		for (int i = 0; i < local_boundary.size(); ++i)
		{
			Eigen::MatrixXd tmp, tmp_uv;
			switch (local_boundary.type())
			{
			case BoundaryType::TriLine:
				BoundarySampler::sample_parametric_tri_edge(local_boundary[i], n_samples, tmp_uv, tmp);
				break;
			case BoundaryType::QuadLine:
				BoundarySampler::sample_parametric_quad_edge(local_boundary[i], n_samples, tmp_uv, tmp);
				break;
			case BoundaryType::Quad:
				BoundarySampler::sample_parametric_quad_face(local_boundary[i], n_samples, tmp_uv, tmp);
				break;
			case BoundaryType::Tri:
				BoundarySampler::sample_parametric_tri_face(local_boundary[i], n_samples, tmp_uv, tmp);
				break;
			case BoundaryType::Polygon:
				BoundarySampler::sample_polygon_edge(local_boundary.element_id(), local_boundary.global_primitive_id(i), n_samples, mesh_, tmp_uv, tmp);
				break;
			case BoundaryType::Invalid:
				assert(false);
				break;
			default:
				assert(false);
			}

			uv.conservativeResize(uv.rows() + tmp_uv.rows(), tmp_uv.cols());
			uv.bottomRows(tmp_uv.rows()) = tmp_uv;

			samples.conservativeResize(samples.rows() + tmp.rows(), tmp.cols());
			samples.bottomRows(tmp.rows()) = tmp;

			global_primitive_ids.conservativeResize(global_primitive_ids.rows() + tmp.rows());
			global_primitive_ids.bottomRows(tmp.rows()).setConstant(local_boundary.global_primitive_id(i));
		}

		assert(uv.rows() == global_primitive_ids.size());
		assert(samples.rows() == global_primitive_ids.size());

		return true;
	}

} // namespace polyfem
