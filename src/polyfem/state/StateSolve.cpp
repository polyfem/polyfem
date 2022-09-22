#include <polyfem/State.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace io;
	using namespace utils;

	void State::init_solve()
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		logger().info("Solve using {} linear solver", args["solver"]["linear"]["solver"].get<std::string>());

		solve_data.rhs_assembler = build_rhs_assembler();

		initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.middleRows(n_bases * mesh->dimension(), n_pressure_bases).setZero();
			sol(sol.size() - 1) = 0;

			sol_to_pressure();
		}

		if (problem->is_time_dependent())
			save_timestep(0, 0, 0, 0);
	}

	void State::initial_solution(Eigen::MatrixXd &solution) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["u_path"]);
		if (!in_path.empty())
			import_matrix(in_path, args["import"], solution);
		else
		{
			if (problem->is_time_dependent())
				solve_data.rhs_assembler->initial_solution(solution);
			else
			{
				solution.resize(rhs.size(), 1);
				solution.setZero();
			}
		}
	}

	void State::initial_velocity(Eigen::MatrixXd &velocity) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["v_path"]);
		if (!in_path.empty())
			import_matrix(in_path, args["import"], velocity);
		else
			solve_data.rhs_assembler->initial_velocity(velocity);
	}

	void State::initial_acceleration(Eigen::MatrixXd &acceleration) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["a_path"]);
		if (!in_path.empty())
			import_matrix(in_path, args["import"], acceleration);
		else
			solve_data.rhs_assembler->initial_acceleration(acceleration);
	}

	void State::apply_lagrange_multipliers(StiffnessMatrix &A) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const auto& gbases = geom_bases();

		if (n_lagrange_multipliers() == 0)
			return;

		logger().debug("No Dirichlet BC, use Lagrange multiplier to find unique solution...");

		if (formulation() == "Laplacian")
		{
			Eigen::VectorXd coeffs(n_bases);
			coeffs.setZero();
			for (int e = 0; e < bases.size(); e++)
			{
				ElementAssemblyValues vals;
				vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

				const int n_loc_bases = int(vals.basis_values.size());
				for (int i = 0; i < n_loc_bases; ++i) 
				{
					const auto &val = vals.basis_values[i];
					for (size_t ii = 0; ii < val.global.size(); ++ii) 
					{
						Eigen::MatrixXd tmp = val.global[ii].val * val.val;
						coeffs(val.global[ii].index) += (tmp.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
					}
				}
			}

			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(A.nonZeros() + coeffs.size() * 2);
			for (int k = 0; k < A.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
				{
					entries.emplace_back(it.row(), it.col(), it.value());
				}
			}

			for (int k = 0; k < coeffs.size(); k++)
			{
				entries.emplace_back(k, A.cols(), coeffs(k));
				entries.emplace_back(A.rows(), k, coeffs(k));
			}

			StiffnessMatrix A_extended(A.rows()+1, A.cols()+1);
			A_extended.setFromTriplets(entries.begin(), entries.end());
			std::swap(A, A_extended);
		}
		else if (formulation() == "LinearElasticity" || formulation() == "NeoHookean")
		{
			Eigen::MatrixXd test_func;
			if (problem_dim == 2)
			{
				test_func.setZero(n_bases * problem_dim, 3);
				
				// (1, 0)
				for (int i = 0; i < n_bases; i++)
					test_func(i * problem_dim + 0, 0) = 1;

				// (0, 1)
				for (int i = 0; i < n_bases; i++)
					test_func(i * problem_dim + 1, 1) = 1;

				// (y, -x)
				for (int i = 0; i < n_bases; i++)
				{
					test_func(i * problem_dim + 0, 2) = mesh_nodes->node_position(i)(1);
					test_func(i * problem_dim + 1, 2) = -mesh_nodes->node_position(i)(0);
				}
			}
			else if (problem_dim == 3)
			{
				test_func.setZero(n_bases * problem_dim, 6);
				
				// (1, 0, 0)
				for (int i = 0; i < n_bases; i++)
					test_func(i * problem_dim + 0, 0) = 1;

				// (0, 1, 0)
				for (int i = 0; i < n_bases; i++)
					test_func(i * problem_dim + 1, 1) = 1;

				// (0, 0, 1)
				for (int i = 0; i < n_bases; i++)
					test_func(i * problem_dim + 2, 2) = 1;

				// (y, -x, 0)
				for (int i = 0; i < n_bases; i++)
				{
					test_func(i * problem_dim + 0, 3) = mesh_nodes->node_position(i)(1);
					test_func(i * problem_dim + 1, 3) = -mesh_nodes->node_position(i)(0);
				}

				// (z, 0, -x)
				for (int i = 0; i < n_bases; i++)
				{
					test_func(i * problem_dim + 0, 4) = mesh_nodes->node_position(i)(2);
					test_func(i * problem_dim + 2, 4) = -mesh_nodes->node_position(i)(0);
				}

				// (0, z, -y)
				for (int i = 0; i < n_bases; i++)
				{
					test_func(i * problem_dim + 1, 5) = mesh_nodes->node_position(i)(2);
					test_func(i * problem_dim + 2, 5) = -mesh_nodes->node_position(i)(1);
				}
			}
			else
				assert(false);

			Eigen::MatrixXd coeffs(n_bases * problem_dim, test_func.cols());
			coeffs.setZero();

			// coeffs = mass * test_func;

			for (int k = 0; k < test_func.cols(); k++)
			{
				for (int e = 0; e < bases.size(); e++)
				{
					ElementAssemblyValues vals;
					vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

					Eigen::MatrixXd result;
					result.setZero(vals.val.rows(), mesh->dimension());

					const int n_loc_bases = int(vals.basis_values.size());
					for (int i = 0; i < n_loc_bases; ++i) 
					{
						const auto &val = vals.basis_values[i];
						for (size_t ii = 0; ii < val.global.size(); ++ii) 
						{
							for (int d = 0; d < problem_dim; ++d)
							{
								result.col(d) += val.global[ii].val * test_func(val.global[ii].index * problem_dim + d, k) * val.val;
							}
						}
					}

					for (int i = 0; i < n_loc_bases; ++i) 
					{
						const auto &val = vals.basis_values[i];
						for (size_t ii = 0; ii < val.global.size(); ++ii) 
						{
							Eigen::MatrixXd tmp = val.global[ii].val * val.val;
							for (int d = 0; d < problem_dim; d++)
								coeffs(val.global[ii].index * problem_dim + d, k) += (tmp.array() * result.col(d).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
						}
					}
				}
			}
			
			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(A.nonZeros() + coeffs.size() * 2);
			for (int k = 0; k < A.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
				{
					entries.emplace_back(it.row(), it.col(), it.value());
				}
			}

			for (int i = 0; i < coeffs.rows(); i++)
			{
				for (int j = 0; j < coeffs.cols(); j++)
				{
					entries.emplace_back(i, A.cols() + j, coeffs(i, j));
					entries.emplace_back(A.rows() + j, i, coeffs(i, j));
				}
			}

			StiffnessMatrix A_extended(A.rows()+coeffs.cols(), A.cols()+coeffs.cols());
			A_extended.setFromTriplets(entries.begin(), entries.end());
			std::swap(A, A_extended);
		}
		else if (formulation() == "Stokes" || formulation() == "NavierStokes")
		{
			Eigen::MatrixXd coeffs(n_bases * problem_dim, problem_dim);
			coeffs.setZero();
			for (int e = 0; e < bases.size(); e++)
			{
				ElementAssemblyValues vals;
				vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

				const int n_loc_bases = int(vals.basis_values.size());
				for (int i = 0; i < n_loc_bases; ++i) 
				{
					const auto &val = vals.basis_values[i];
					for (size_t ii = 0; ii < val.global.size(); ++ii) 
					{
						Eigen::MatrixXd tmp = val.global[ii].val * val.val;
						for (int d = 0; d < problem_dim; d++)
							coeffs(val.global[ii].index * problem_dim + d, d) += (tmp.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
					}
				}
			}

			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(A.nonZeros() + coeffs.size() * 2 * problem_dim);
			for (int k = 0; k < A.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
				{
					entries.emplace_back(it.row(), it.col(), it.value());
				}
			}

			for (int d = 0; d < problem_dim; d++)
				for (int k = 0; k < coeffs.rows(); k++)
				{
					entries.emplace_back(k, A.cols() + d, coeffs(k, d));
					entries.emplace_back(A.rows() + d, k, coeffs(k, d));
				}

			StiffnessMatrix A_extended(A.rows() + problem_dim, A.cols() + problem_dim);
			A_extended.setFromTriplets(entries.begin(), entries.end());
			std::swap(A, A_extended);
		}
		else
			assert(false);
	}

	void State::apply_lagrange_multipliers(StiffnessMatrix &A, const Eigen::MatrixXd &coeffs) const
	{
		assert(coeffs.rows() <= A.rows());
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(A.nonZeros() + coeffs.size() * 2);
		for (int k = 0; k < A.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
			{
				entries.emplace_back(it.row(), it.col(), it.value());
			}
		}

		for (int k = 0; k < coeffs.rows(); k++)
		{
			for (int j = 0; j < coeffs.cols(); j++)
			{
				entries.emplace_back(k, A.cols() + j, coeffs(k, j));
				entries.emplace_back(A.rows() + j, k, coeffs(k, j));
			}
		}

		StiffnessMatrix A_extended(A.rows()+coeffs.cols(), A.cols()+coeffs.cols());
		A_extended.setFromTriplets(entries.begin(), entries.end());
		std::swap(A, A_extended);
	}

	int State::full_to_periodic(StiffnessMatrix &A) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int independent_dof = periodic_reduce_map.maxCoeff() + 1;
		
		// account for potential pressure block
		auto index_map = [&](int id){
			if (id < periodic_reduce_map.size())
				return periodic_reduce_map(id);
			else
				return (int)(id + independent_dof - n_bases * problem_dim);
		};

		StiffnessMatrix A_periodic(index_map(A.rows()), index_map(A.cols()));
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(A.nonZeros());
		for (int k = 0; k < A.outerSize(); k++)
		{
			for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
			{
				entries.emplace_back(index_map(it.row()), index_map(it.col()), it.value());
			}
		}
		A_periodic.setFromTriplets(entries.begin(),entries.end());

		std::swap(A_periodic, A);

		return independent_dof;
	}

	int State::full_to_periodic(Eigen::MatrixXd &b, bool force_dirichlet) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int independent_dof = periodic_reduce_map.maxCoeff() + 1;
		
		// account for potential pressure block
		auto index_map = [&](int id){
			if (id < periodic_reduce_map.size())
				return periodic_reduce_map(id);
			else
				return (int)(id + independent_dof - n_bases * problem_dim);
		};

		// rhs under periodic basis
		Eigen::MatrixXd b_periodic;
		b_periodic.setZero(index_map(b.rows()), b.cols());
		for (int d = 0; d < b_periodic.cols(); d++)
		{
			for (int k = 0; k < b.rows(); k++)
				b_periodic(index_map(k), d) += b(k, d);

			if (force_dirichlet)
				for (int k : boundary_nodes)
					b_periodic(index_map(k), d) = b(k, d);
		}

		b = b_periodic;

		return independent_dof;
	}

	Eigen::MatrixXd State::periodic_to_full(const int ndofs, const Eigen::MatrixXd &x_periodic) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int independent_dof = periodic_reduce_map.maxCoeff() + 1;
		
		auto index_map = [&](int id){
			if (id < periodic_reduce_map.size())
				return periodic_reduce_map(id);
			else
				return (int)(id + independent_dof - n_bases * problem_dim);
		};

		Eigen::MatrixXd x_full;
		x_full.resize(ndofs, x_periodic.cols());
		for (int i = 0; i < x_full.rows(); i++)
			for (int j = 0; j < x_full.cols(); j++)
				x_full(i, j) = x_periodic(index_map(i), j);

		return x_full;
	}
} // namespace polyfem
