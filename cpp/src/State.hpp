#ifndef STATE_HPP
#define STATE_HPP

#include "Basis.hpp"
#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Mesh.hpp"
#include "Problem.hpp"

#include <igl/viewer/Viewer.h>

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
		bool skip_visualization = false;

		bool linear_elasticity = false;

		int vis_basis = 0;

		Problem problem;

		int n_bases;


		igl::viewer::Viewer viewer;


		std::vector< std::vector<Basis> >    bases;
		std::vector< ElementAssemblyValues > values;
		std::vector< int >                   bounday_nodes;


		Mesh mesh;

		Eigen::MatrixXi tri_faces, local_vis_faces, vis_faces;
		Eigen::MatrixXd tri_pts, local_vis_pts, vis_pts;


		Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol;

		double l2_err, linf_err;
		long nn_zero, mat_size;

	private:
		void interpolate_function(const Eigen::MatrixXd &fun, Eigen::MatrixXd &result);
		void plot_function(const Eigen::MatrixXd &fun, double min=0, double max=-1);
	};

}
#endif //STATE_HPP

