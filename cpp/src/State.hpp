#ifndef STATE_HPP
#define STATE_HPP

#include <polyfem/ElementBases.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/Problem.hpp>
#include <polyfem/Mesh.hpp>
#include <polyfem/Problem.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/InterfaceData.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <string>

namespace polyfem
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
		std::vector< ElementBases >    pressure_bases;
		std::vector< ElementBases >    geom_bases;

		std::vector< int >                   boundary_nodes;
		std::vector< LocalBoundary >         local_boundary;
		std::vector< LocalBoundary >         local_neumann_boundary;
		std::map<int, InterfaceData> 		 poly_edge_to_data;

		std::vector<int> flipped_elements;



		std::unique_ptr<Mesh> mesh;

		std::map<int, Eigen::MatrixXd> polys;
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi> > polys_3d;
		std::vector<int> parent_elements;

		Eigen::SparseMatrix<double> stiffness, mass;
		Eigen::MatrixXd rhs;
		Eigen::MatrixXd sol, pressure;

		Eigen::Vector4d spectrum;


		json solver_info;

		int n_bases, n_pressure_bases;
		Eigen::VectorXi disc_orders;

		double mesh_size;
		double min_edge_length;
		double average_edge_length;

		double l2_err, linf_err, lp_err, h1_err, h1_semi_err, grad_max_err;

		long long nn_zero, mat_size, num_dofs;

		double building_basis_time;
		double loading_mesh_time;
		double computing_poly_basis_time;
		double assembling_stiffness_mat_time;
		double assigning_rhs_time;
		double solving_time;
		double computing_errors_time;
		double max_angle;
		double sigma_max, sigma_min, sigma_avg;

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
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd&)> &boundary_marker);
		void build_basis();
		void build_polygonal_basis();
		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		void compute_errors();
		void export_data();

		void interpolate_function(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only = false);
		void interpolate_function(const int n_points, const int actual_dim, const std::vector< ElementBases > &basis, const MatrixXd &fun, MatrixXd &result, const bool boundary_only = false);

		void compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only = false);
		void compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only = false);
		void average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool boundary_only = false);

		void interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);


		void save_json(std::ostream &out);
		void save_json();

		void compute_mesh_stats();

		void save_vtu(const std::string &name);
		void save_wire(const std::string &name, bool isolines = false);

		inline std::string mesh_path() const { return args["mesh"]; }

		inline std::string formulation() const { return problem->is_mixed() ? mixed_formulation() : (problem->is_scalar() ? scalar_formulation() : tensor_formulation()); }
		inline bool iso_parametric() const { return (non_regular_count > 0 || non_regular_boundary_count > 0 || undefined_count > 0) || (!args["use_p_ref"] && args["discr_order"] == mesh->order()) || args["iso_parametric"]; }

		inline std::string solver_type() const { return args["solver_type"]; }
		inline std::string precond_type() const { return args["precond_type"]; }
		inline const json &solver_params() const { return args["solver_params"]; }

		inline std::string scalar_formulation() const { return args["scalar_formulation"]; }
		inline std::string tensor_formulation() const { return args["tensor_formulation"]; }
		inline std::string mixed_formulation() const { return args["mixed_formulation"]; }

		void p_refinement(const Mesh2D &mesh2d);
		void p_refinement(const Mesh3D &mesh3d);

	private:
		void sol_to_pressure();

	};

}
#endif //STATE_HPP

