#ifndef STATE_HPP
#define STATE_HPP

#include "Basis.hpp"
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

		int quadrature_order = 2;
		int n_boundary_samples = 10;

		std::string mesh_path;
		int n_refs = 0;

		bool use_splines = false;
		bool linear_elasticity = false;

		Problem problem;

		int n_bases;

		std::vector< std::vector<Basis> >    bases;
		std::vector< ElementAssemblyValues > values;
		std::vector< int >                   bounday_nodes;
		std::vector< LocalBoundary >         local_boundary;

		std::vector<int> boundary_tag;


		Mesh mesh;


		Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol;

		double l2_err, linf_err;
		long nn_zero, mat_size;

		void load_mesh();
		void build_basis();
		void compute_assembly_vals();
		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		void compute_errors();

		void interpolate_function(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &local_pts, Eigen::MatrixXd &result);
	};

}
#endif //STATE_HPP

