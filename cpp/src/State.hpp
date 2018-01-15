#ifndef STATE_HPP
#define STATE_HPP

#include "ElementBases.hpp"
#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Mesh.hpp"
#include "Problem.hpp"
#include "LocalBoundary.hpp"
#include "InterfaceData.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace poly_fem
{
	class State
	{
	public:
		static State &state();
		~State() { delete mesh; }

		void init(const std::string &mesh_path, const int n_refs, const int problem_num_);

		int quadrature_order = 4;
		int discr_order = 1;
		int n_boundary_samples = 10;
		int harmonic_samples_res = 10;

		std::string mesh_path;
		int n_refs = 0;

		bool use_splines = false;
		bool iso_parametric = true;

		Problem problem;

		int n_bases;

		std::vector< ElementBases >    bases;
		std::vector< ElementBases >    geom_bases;
		// std::vector< ElementAssemblyValues > values;
		// std::vector< ElementAssemblyValues > geom_values;

		std::vector< int >                   boundary_nodes;
		std::vector< LocalBoundary >         local_boundary;
		std::map<int, InterfaceData> 		 poly_edge_to_data;

		std::vector<int> boundary_tag;

		std::vector<double> errors;


		Mesh *mesh = NULL;

		std::map<int, Eigen::MatrixXd> polys;
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi> > polys_3d;
		std::vector<int> parent_elements;

		std::string solver_type;
		std::string precond_type;

		Eigen::SparseMatrix<double> stiffness;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol;

		double lambda = 1, mu = 1;

		double mesh_size;
		double l2_err, linf_err, lp_err;
		long nn_zero, mat_size;

		double refinenemt_location = 0.5;

		double building_basis_time;
		double loading_mesh_time;
		double computing_assembly_values_time;
		double assembling_stiffness_mat_time;
		double assigning_rhs_time;
		double solving_time;
		double computing_errors_time;

		int n_flipped;

		int regular_count;
		int regular_boundary_count;
		int simple_singular_count;
		int multi_singular_count;
		int boundary_count;
		int non_regular_boundary_count;
		int non_regular_count;
		int undefined_count;
		int multi_singular_boundary_count;

		void load_mesh();
		void build_basis();
		void compute_assembly_vals();
		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		// void solve_problem_old();
		void compute_errors();

		void interpolate_function(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &local_pts, Eigen::MatrixXd &result);

		void save_json(const std::string &name);
		void sertialize(const std::string &file_name);

		void compute_mesh_stats();

	};

}
#endif //STATE_HPP

