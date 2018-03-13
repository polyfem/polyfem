#ifndef STATE_HPP
#define STATE_HPP

#include "ElementBases.hpp"
#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Mesh.hpp"
#include "Problem.hpp"
#include "LocalBoundary.hpp"
#include "InterfaceData.hpp"
#include "Common.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <string>

namespace poly_fem
{
	class State
	{
	public:
		static State &state();
		~State() = default;

		State();

		void init(const json &args);

		json args;


		std::shared_ptr<Problem> problem;


		std::vector< ElementBases >    bases;
		std::vector< ElementBases >    geom_bases;

		std::vector< int >                   boundary_nodes;
		std::vector< LocalBoundary >         local_boundary;
		std::map<int, InterfaceData> 		 poly_edge_to_data;

		std::vector<int> flipped_elements;



		std::unique_ptr<Mesh> mesh;

		std::map<int, Eigen::MatrixXd> polys;
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi> > polys_3d;
		std::vector<int> parent_elements;

		Eigen::SparseMatrix<double> stiffness;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol;


		json solver_info;

		int n_bases;

		double mesh_size;
		double min_edge_length;
		double average_edge_length;

		double l2_err, linf_err, lp_err, h1_err;

		long long nn_zero, mat_size, num_dofs;

		double building_basis_time;
		double loading_mesh_time;
		double computing_assembly_values_time;
		double assembling_stiffness_mat_time;
		double assigning_rhs_time;
		double solving_time;
		double computing_errors_time;

		int n_flipped;

		int simplex_count;
		int regular_count;
		int regular_boundary_count;
		int simple_singular_count;
		int multi_singular_count;
		int boundary_count;
		int non_regular_boundary_count;
		int non_regular_count;
		int undefined_count;
		int multi_singular_boundary_count;

		json build_json_params();

		void compute_mesh_size(const Mesh &mesh, const std::vector< ElementBases > &bases, const int n_samples);

		void load_mesh();
		void build_basis();
		void build_polygonal_basis();
		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		void compute_errors();

		void interpolate_function(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &local_pts, Eigen::MatrixXd &result);

		void save_json(std::ostream &out);

		void compute_mesh_stats();

		void save_vtu(const std::string &name);

		inline std::string mesh_path() const { return args["mesh"]; }

		inline std::string formulation() const { return problem->is_scalar() ? scalar_formulation() : tensor_formulation(); }
		inline bool iso_parametric() const { return args["iso_parametric"]; }

		inline std::string solver_type() const { return args["solver_type"]; }
		inline std::string precond_type() const { return args["precond_type"]; }
		inline const json &solver_params() const { return args["solver_params"]; }

		inline std::string scalar_formulation() const { return args["scalar_formulation"]; }
		inline std::string tensor_formulation() const { return args["tensor_formulation"]; }

	};

}
#endif //STATE_HPP

