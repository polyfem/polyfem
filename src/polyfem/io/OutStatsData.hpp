#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace polyfem::io
{
	/// @brief timers from polyfem.
	class OutRuntimeData
	{
	public:
		/// time to construct the basis
		double building_basis_time = 0;
		/// time to load the mesh
		double loading_mesh_time = 0;
		/// time to build the polygonal/polyhedral bases
		double computing_poly_basis_time = 0;
		/// time to assembly
		double assembling_stiffness_mat_time = 0;
		/// time to assembly mass
		double assembling_mass_mat_time = 0;
		/// time to computing the rhs
		double assigning_rhs_time = 0;
		/// time to solve
		double solving_time = 0;

		/// @brief computes total time
		/// @return total time
		double total_time() const
		{
			return building_basis_time
				   + loading_mesh_time
				   + computing_poly_basis_time
				   + assembling_stiffness_mat_time
				   + assembling_mass_mat_time
				   + assigning_rhs_time
				   + solving_time;
		}
	};

	/// @brief all stats from polyfem
	class OutStatsData
	{
	public:
		/// spectrum of the stiffness matrix, enable only if POLYSOLVE_WITH_SPECTRA is ON (off by default)
		Eigen::Vector4d spectrum = Eigen::Vector4d::Zero();

		/// information of the solver, eg num iteration, time, errors, etc
		/// the informations varies depending on the solver
		json solver_info;

		/// max edge lenght
		double mesh_size = 0;
		/// min edge lenght
		double min_edge_length = 0;
		/// avg edge lenght
		double average_edge_length = 0;

		/// errors, lp_err is in fact an L8 error
		double l2_err = 0, linf_err = 0, lp_err = 0, h1_err = 0, h1_semi_err = 0, grad_max_err = 0;

		/// non zeros and sytem matrix size
		/// num dof is the total dof in the system
		long long nn_zero = 0, mat_size = 0, num_dofs = 0;

		/// statiscs on angle, compute only when using p_ref (false by default)
		double max_angle = 0;
		/// statiscs on tri/tet quality, compute only when using p_ref (false by default)
		double sigma_max = 0, sigma_min = 0, sigma_avg = 0;

		/// number of flipped elements, compute only when using count_flipped_els (false by default)
		int n_flipped = 0;

		/// statiscs on the mesh (simplices)
		int simplex_count = 0;
		/// statiscs on the mesh (simplices)
		int prism_count = 0;
		/// statiscs on the mesh (simplices)
		int pyramid_count = 0;
		/// statiscs on the mesh (regular quad/hex part of the mesh), see Polyspline paper for desciption
		int regular_count = 0;
		/// statiscs on the mesh (regular quad/hex boundary part of the mesh), see Polyspline paper for desciption
		int regular_boundary_count = 0;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int simple_singular_count = 0;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int multi_singular_count = 0;
		/// statiscs on the mesh (boundary quads/hexs), see Polyspline paper for desciption
		int boundary_count = 0;
		/// statiscs on the mesh (irregular boundary quad/hex part of the mesh), see Polyspline paper for desciption
		int non_regular_boundary_count = 0;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int non_regular_count = 0;
		/// statiscs on the mesh (not quad/hex simplex), see Polyspline paper for desciption
		int undefined_count = 0;
		/// statiscs on the mesh (irregular boundary quad/hex part of the mesh), see Polyspline paper for desciption
		int multi_singular_boundary_count = 0;

		/// @brief compute errors
		/// @param[in] n_bases number of base
		/// @param[in] bases bases
		/// @param[in] gbases geometric bases
		/// @param[in] mesh mesh
		/// @param[in] problem problem
		/// @param[in] tend end time step
		/// @param[in] sol solution
		void compute_errors(const int n_bases,
							const std::vector<polyfem::basis::ElementBases> &bases,
							const std::vector<polyfem::basis::ElementBases> &gbases,
							const polyfem::mesh::Mesh &mesh,
							const assembler::Problem &problem,
							const double tend,
							const Eigen::MatrixXd &sol);

		/// @brief compute stats (counts els type, mesh lenght, etc), step 1 of solve
		/// @param mesh mesh
		void compute_mesh_stats(const polyfem::mesh::Mesh &mesh);

		/// computes the mesh size, it samples every edges n_samples times
		/// uses curved_mesh_size (false by default) to compute the size of
		/// the linear mesh
		/// @param[in] mesh to compute stats
		/// @param[in] bases geom bases
		/// @param[in] n_samples used for curved meshes
		/// @param[in] use_curved_mesh_size use curved edges to compute mesh size
		void compute_mesh_size(const polyfem::mesh::Mesh &mesh_in, const std::vector<polyfem::basis::ElementBases> &bases_in, const int n_samples, const bool use_curved_mesh_size);

		/// @brief clears all stats
		void reset();

		/// @brief counts the number of flipped elements
		/// @param[in] mesh mesh
		/// @param[in] gbases geometric bases
		void count_flipped_elements(const polyfem::mesh::Mesh &mesh, const std::vector<polyfem::basis::ElementBases> &gbases);

		/// saves the output statistic to a json object
		/// @param[in] j output json

		/// @brief save json
		/// @param[in] args input argumeents
		/// @param[in] n_bases number of bases
		/// @param[in] n_pressure_bases number fo pressure bases
		/// @param[in] sol solution
		/// @param[in] mesh mesh
		/// @param[in] disc_orders discretization order
		/// @param[in] disc_ordersq discretization order
		/// @param[in] problem problem
		/// @param[in] runtime rumtime
		/// @param[in] formulation formulation
		/// @param[in] isoparametric if isoparametric
		/// @param[in] sol_at_node_id export solution at node
		/// @param[out] j output json
		void save_json(const nlohmann::json &args,
					   const int n_bases, const int n_pressure_bases,
					   const Eigen::MatrixXd &sol,
					   const mesh::Mesh &mesh,
					   const Eigen::VectorXi &disc_orders,
					   const Eigen::VectorXi &disc_ordersq,
					   const assembler::Problem &problem,
					   const OutRuntimeData &runtime,
					   const std::string &formulation,
					   const bool isoparametric,
					   const int sol_at_node_id,
					   nlohmann::json &j) const;
	};
} // namespace polyfem::io
