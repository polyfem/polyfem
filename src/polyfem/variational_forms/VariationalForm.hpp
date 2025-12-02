#pragma once

#include <polyfem/Common.hpp>

#include <memory>
#include <vector>

namespace polyfem
{
	class VariationalForm
	{
	public:
		virtual ~VariationalForm() = default;

		/// check if using iso parametric bases
		/// @return if basis are isoparametric
		bool iso_parametric() const;

		/// @brief Get a constant reference to the geometry mapping bases.
		/// @return A constant reference to the geometry mapping bases.
		const std::vector<basis::ElementBases> &geom_bases() const
		{
			return iso_parametric() ? bases() : geom_bases_;
		}

		virtual std::vector<basis::ElementBases> &bases() const = 0;
		virtual int n_bases() const = 0;

		/// build a RhsAssembler for the problem
		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler(
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const assembler::AssemblyValsCache &ass_vals_cache) const;

		/// build a RhsAssembler for the problem
		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler() const
		{
			return build_rhs_assembler(n_bases(), bases(), mass_ass_vals_cache);
		}

		/// builds the bases step 2 of solve
		/// modifies bases, pressure_bases, geom_bases_, boundary_nodes,
		/// dirichlet_nodes, neumann_nodes, local_boundary, total_local_boundary
		/// local_neumann_boundary, polys, poly_edge_to_data, rhs
		virtual void build_basis() = 0;
		/// builds bases for polygons, called inside build_basis
		virtual void build_polygonal_basis() = 0;

		/// compute rhs, step 3 of solve
		/// build rhs vector based on defined basis and given rhs of the problem
		/// modifies rhs (and maybe more?)
		void assemble_rhs();
		/// assemble mass, step 4 of solve
		/// build mass matrix based on defined basis
		/// modifies mass (and maybe more?)
		void assemble_mass_mat();

		virtual void clear() = 0;

		/// return the formulation (checks if the problem is scalar or not and deals with multiphysics)
		/// @return formulation
		std::string formulation() const;

		/// quadrature used for projecting boundary conditions
		/// @return the quadrature used for projecting boundary conditions
		QuadratureOrders n_boundary_samples() const
		{
			using assembler::AssemblerUtils;
			const int n_b_samples_j = args["space"]["advanced"]["n_boundary_samples"];
			const int gdiscr_order = mesh->orders().size() <= 0 ? 1 : mesh->orders().maxCoeff();
			const int discr_order = std::max(disc_orders.maxCoeff(), gdiscr_order);

			const int n_b_samples = std::max(n_b_samples_j, AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::POLY, mesh->dimension()));
			// todo prism
			return {{n_b_samples, n_b_samples}};
		}

		/// set the material and the problem dimension
		/// @param[in/out] list of assembler to set
		void set_materials(std::vector<std::shared_ptr<assembler::Assembler>> &assemblers) const;
		/// utility to set the material and the problem dimension to only 1 assembler
		/// @param[in/out] assembler to set
		void set_materials(assembler::Assembler &assembler) const;
		/// @param[in/out] assembler to set
		void set_materials() const;

		/// initialize solver
		/// @param[out] sol solution
		void init_solve(Eigen::MatrixXd &sol);

		/// solves a linear problem
		/// @param[out] sol solution
		void solve_static(Eigen::MatrixXd &sol);

		virtual void set_parameters(const json &args, bool has_constraints) = 0;
		/*
		const std::string formulation = this->formulation();
		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		assert(assembler->name() == formulation);
		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		const auto other_name = assembler::AssemblerUtils::other_assembler_name(formulation);

		if (!other_name.empty())
		{
			mixed_assembler = assembler::AssemblerUtils::make_mixed_assembler(formulation);
			pressure_assembler = assembler::AssemblerUtils::make_assembler(other_name);
		}

		if (args["solver"]["advanced"]["check_inversion"] == "Conservative")
		{
			if (auto elastic_assembler = std::dynamic_pointer_cast<assembler::ElasticityAssembler>(assembler))
				elastic_assembler->set_use_robust_jacobian();
		}

		if (!args.contains("preset_problem"))
		{
			if (!assembler->is_tensor())
				problem = std::make_shared<assembler::GenericScalarProblem>("GenericScalar");
			else
				problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

			problem->clear();
			if (!args["time"].is_null())
			{
				const auto tmp = R"({"is_time_dependent": true})"_json;
				problem->set_parameters(tmp);
			}
			// important for the BC

			auto bc = args["boundary_conditions"];
			bc["root_path"] = root_path();
			problem->set_parameters(bc);
			problem->set_parameters(args["initial_conditions"]);

			problem->set_parameters(args["output"]);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<KernelProblem>("Kernel", *assembler);
				problem->clear();
				KernelProblem &kprob = *dynamic_cast<KernelProblem *>(problem.get());
			}
			else
			{
				problem = ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"]);
		}

		problem->set_units(*assembler, units);
			*/

		// 	/// @brief Solve the linear problem with the given solver and system.
		// /// @param solver Linear solver.
		// /// @param A Linear system matrix.
		// /// @param b Right-hand side.
		// /// @param compute_spectrum If true, compute the spectrum.
		// /// @param[out] sol solution
		// /// @param[out] pressure pressure
		// void solve_linear(
		// 	const std::unique_ptr<polysolve::linear::Solver> &solver,
		// 	StiffnessMatrix &A,
		// 	Eigen::VectorXd &b,
		// 	const bool compute_spectrum,
		// 	Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);

		/// periodic BC and periodic mesh utils
		std::shared_ptr<utils::PeriodicBoundary> periodic_bc;
		bool has_periodic_bc() const
		{
			return args["boundary_conditions"]["periodic_boundary"]["enabled"].get<bool>();
		}

		/// @brief Returns whether the system is linear. Collisions and pressure add nonlinearity to the problem.
		bool is_problem_linear() const { return assembler->is_linear() && !is_contact_enabled() && !is_pressure_enabled() && !has_constraints(); }

		bool has_constraints() const
		{
			return has_constraints_;
		}

		//---------------------------------------------------
		//-----------------nodes flags-----------------------
		//---------------------------------------------------
		std::vector<int> primitive_to_node() const;
		std::vector<int> node_to_primitive() const;

		/// build the mapping from input nodes to polyfem nodes
		void build_node_mapping();

		int ndof() const
		{
			const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
			if (mixed_assembler == nullptr)
				return actual_dim * n_bases;
			else
				return actual_dim * n_bases + n_pressure_bases;
		}

		void compute_errors(const Eigen::MatrixXd &sol);

	protected:
		/// current problem, it contains rhs and bc
		std::shared_ptr<assembler::Problem> problem;
		/// System right-hand side.
		Eigen::MatrixXd rhs;

		/// vector of discretization orders, used when not all elements have the same degree, one per element
		Eigen::VectorXi disc_orders, disc_ordersq;

		/// Geometric mapping bases, if the elements are isoparametric, this list is empty
		std::vector<basis::ElementBases> geom_bases_;
		/// number of geometric bases
		int n_geom_bases;
		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> geom_mesh_nodes;

		std::shared_ptr<assembler::Mass> mass_matrix_assembler = nullptr;
		/// used to store assembly values for small problems
		assembler::AssemblyValsCache mass_ass_vals_cache;
		/// Mass matrix, it is computed only for time dependent problems
		StiffnessMatrix mass;
		/// average system mass, used for contact with IPC
		double avg_mass;

		/// timedependent stuff cached
		solver::SolveData solve_data;

	private:
		bool has_constraints_;
	};
} // namespace polyfem