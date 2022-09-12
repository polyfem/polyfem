#include <polyfem/State.hpp>

#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace utils;

	void State::init_solve()
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		solve_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle, input_dirichlet, n_bases, size, bases, gbases, ass_vals_cache, formulation(),
			*problem, args["space"]["advanced"]["bc_method"], args["solver"]["linear"]["solver"],
			args["solver"]["linear"]["precond"], rhs_solver_params);

		initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			const int prev_size = sol.size();
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.middleRows(prev_size, n_pressure_bases).setZero();
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

	int State::remove_pure_neumann_singularity(StiffnessMatrix &A) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const auto& gbases = iso_parametric() ? bases : geom_bases;
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
			// Eigen::VectorXd test_func;
			// test_func.setOnes(n_bases, 1);
			// Eigen::VectorXd coeffs = mass * test_func;

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

			return 1;
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

			return 3 * (problem_dim - 1);
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

			// Eigen::MatrixXd test_func;
			// test_func.setZero(n_bases * problem_dim, problem_dim);
			// for (int i = 0; i < n_bases; i++)
			// 	for (int d = 0; d < problem_dim; d++)
			// 	{
			// 		test_func(i * problem_dim + d, d) = 1;
			// 	}
			// Eigen::MatrixXd coeffs = mass.topLeftCorner(test_func.rows(), test_func.rows()) * test_func;

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

			return problem_dim;
		}
		else
			return 0;
	}

	int State::remove_pure_periodic_singularity(StiffnessMatrix &A) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const auto& gbases = iso_parametric() ? bases : geom_bases;
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
			// Eigen::VectorXd test_func;
			// test_func.setOnes(n_bases, 1);
			// Eigen::VectorXd coeffs = mass * test_func;

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

			return 1;
		}
		else if (formulation() == "LinearElasticity" || formulation() == "NeoHookean" || formulation() == "Stokes")
		{
			Eigen::MatrixXd coeffs(n_bases * problem_dim, mesh->dimension());
			coeffs.setZero();

			// coeffs = mass * test_func;

			for (int e = 0; e < bases.size(); e++)
			{
				ElementAssemblyValues vals;
				ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

				const int n_loc_bases = int(vals.basis_values.size());
				for (int i = 0; i < n_loc_bases; ++i) 
				{
					const auto &val = vals.basis_values[i];
					for (size_t ii = 0; ii < val.global.size(); ++ii) 
					{
						Eigen::MatrixXd tmp = val.global[ii].val * val.val;
						for (int k = 0; k < mesh->dimension(); k++)
							coeffs(val.global[ii].index * problem_dim + k, k) += (tmp.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
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

			return problem_dim;
		}
		else
			return 0;
	}
} // namespace polyfem
