#ifndef STATE_HPP
#define STATE_HPP

#include "ElementBases.hpp"
#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Mesh.hpp"
#include "Problem.hpp"
#include "LocalBoundary.hpp"

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

		std::string mesh_path;
		int n_refs = 0;

		bool use_splines = false;
		bool iso_parametric = false;
		// bool linear_elasticity = false;

		Problem problem;

		int n_bases, n_geom_bases;

		std::vector< ElementBases >    bases;
		std::vector< ElementBases >    geom_bases;
		std::vector< ElementAssemblyValues > values;
		std::vector< ElementAssemblyValues > geom_values;

		std::vector< int >                   bounday_nodes;
		std::vector< LocalBoundary >         local_boundary;

		std::vector<int> boundary_tag;

		std::vector<double> errors;


		Mesh *mesh = NULL;

		std::map<int, Eigen::MatrixXd> polys;
		std::vector<ElementType> els_tag;


		Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness;
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

		void load_mesh();
		void build_basis();
		void compute_assembly_vals();
		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		void compute_errors();

		void interpolate_function(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &local_pts, Eigen::MatrixXd &result);

		void save_json(const std::string &name);
		void sertialize(const std::string &file_name);

		void compute_mesh_stats();

	};

}
#endif //STATE_HPP

