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
		Franke
	};

	class State
	{
	public:
		static State &state();
		void init(const int n_x_el, const int n_y_el, const int n_z_el, const bool use_hex_, const int problem_num_);

		void sertialize(const std::string &name);

		int quadrature_order = 2;
		int n_boundary_samples = 10;

		int n_x_el=2;
		int n_y_el=2;
		int n_z_el=2;

		bool use_hex = false;
		bool use_splines = false;
		bool skip_visualization = false;

		Problem problem;

		int n_bases;


		igl::viewer::Viewer viewer;


		std::vector< std::vector<Basis> >    bases;
		std::vector< ElementAssemblyValues > values;
		std::vector< int >                   bounday_nodes;


		Mesh mesh;
		Mesh visualization_mesh, local_mesh;

		Eigen::Matrix<int, Eigen::Dynamic, 3> vis_faces;


		Eigen::SparseMatrix<double, Eigen::RowMajor> stiffness;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol;

		double l2_err, linf_err;

	private:
		void interpolate_function(const Eigen::MatrixXd &fun, Eigen::MatrixXd &result);
	};

}
#endif //STATE_HPP

