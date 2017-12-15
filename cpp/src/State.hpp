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
	enum ProblemType
	{
		Linear = 0,
		Quadratic,
		Franke,
		Elastic
	};

	class State
	{
	public:
		static State &state();

		void init(const std::string &mesh_path, const int n_refs, const int problem_num_);

		void sertialize(const std::string &name);

		int quadrature_order = 4;
		int n_boundary_samples = 10;

		std::string mesh_path;
		int n_refs = 0;

		bool use_splines = false;
		bool linear_elasticity = false;

		Problem problem;

		int n_bases;

		std::vector< ElementBases >    bases;
		std::vector< ElementAssemblyValues > values;
		std::vector< int >                   bounday_nodes;
		std::vector< LocalBoundary >         local_boundary;

		std::vector<int> boundary_tag;


		Mesh mesh;

		std::map<int, Eigen::MatrixXd> polys;


		Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol;

		double mesh_size;
		double l2_err, linf_err;
		long nn_zero, mat_size;

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

		void save_json(const std::string &name);

		void interpolate_function(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &local_pts, Eigen::MatrixXd &result);
	};

}
#endif //STATE_HPP

