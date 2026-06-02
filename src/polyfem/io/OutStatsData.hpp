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
		double building_basis_time;
		/// time to load the mesh
		double loading_mesh_time;
		/// time to build the polygonal/polyhedral bases
		double computing_poly_basis_time;
		/// time to assembly
		double assembling_stiffness_mat_time;
		/// time to assembly mass
		double assembling_mass_mat_time;
		/// time to computing the rhs
		double assigning_rhs_time;
		/// time to solve
		double solving_time;

		/// @brief computes total time
		/// @return total time
		double total_time()
		{
			return building_basis_time + assembling_mass_mat_time + assembling_stiffness_mat_time + solving_time;
		}
	};

	/// @brief all stats from polyfem
	class OutStatsData
	{
	public:
		/// spectrum of the stiffness matrix, enable only if POLYSOLVE_WITH_SPECTRA is ON (off by default)
		Eigen::Vector4d spectrum;

		/// information of the solver, eg num iteration, time, errors, etc
		/// the informations varies depending on the solver
		json solver_info;

		/// max edge lenght
		double mesh_size;
		/// min edge lenght
		double min_edge_length;
		/// avg edge lenght
		double average_edge_length;

		/// errors, lp_err is in fact an L8 error
		double l2_err, linf_err, lp_err, h1_err, h1_semi_err, grad_max_err;

		/// non zeros and sytem matrix size
		/// num dof is the total dof in the system
		long long nn_zero, mat_size, num_dofs;

		/// statiscs on angle, compute only when using p_ref (false by default)
		double max_angle;
		/// statiscs on tri/tet quality, compute only when using p_ref (false by default)
		double sigma_max, sigma_min, sigma_avg;

		/// number of flipped elements, compute only when using count_flipped_els (false by default)
		int n_flipped;

		/// statiscs on the mesh (simplices)
		int simplex_count;
		/// statiscs on the mesh (simplices)
		int prism_count;
		/// statiscs on the mesh (simplices)
		int pyramid_count;
		/// statiscs on the mesh (regular quad/hex part of the mesh), see Polyspline paper for desciption
		int regular_count;
		/// statiscs on the mesh (regular quad/hex boundary part of the mesh), see Polyspline paper for desciption
		int regular_boundary_count;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int simple_singular_count;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int multi_singular_count;
		/// statiscs on the mesh (boundary quads/hexs), see Polyspline paper for desciption
		int boundary_count;
		/// statiscs on the mesh (irregular boundary quad/hex part of the mesh), see Polyspline paper for desciption
		int non_regular_boundary_count;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int non_regular_count;
		/// statiscs on the mesh (not quad/hex simplex), see Polyspline paper for desciption
		int undefined_count;
		/// statiscs on the mesh (irregular boundary quad/hex part of the mesh), see Polyspline paper for desciption
		int multi_singular_boundary_count;

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
