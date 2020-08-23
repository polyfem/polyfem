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
#include <polyfem/Logger.hpp>

#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <string>

namespace polyfem
{
	class SolutionFrame
	{
	public:
		std::string name;
		Eigen::MatrixXd points;
		Eigen::MatrixXi connectivity;
		Eigen::MatrixXd solution;
		Eigen::MatrixXd pressure;
		Eigen::MatrixXd exact;
		Eigen::MatrixXd error;
		Eigen::MatrixXd scalar_value;
		Eigen::MatrixXd scalar_value_avg;
	};

	class State
	{
	public:
		~State() = default;

		State();
		void init_logger(const std::string &log_file, int log_level, const bool is_quiet);
		void init_logger(std::ostream &os, int log_level);
		void init_logger(std::vector<spdlog::sink_ptr> &sinks, int log_level);

		void init(const json &args);
		void init(const std::string &json);

		void set_log_level(int log_level)
		{
			log_level = std::max(0, std::min(6, log_level));
			spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
		}
		std::string get_log()
		{
			std::stringstream ss;
			save_json(ss);
			return ss.str();
		}

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

		StiffnessMatrix stiffness, mass;
		Eigen::MatrixXd rhs, rhs_in;
		Eigen::MatrixXd sol, pressure;

		Eigen::MatrixXd boundary_nodes_pos;
		Eigen::MatrixXi boundary_edges;
		Eigen::MatrixXi boundary_triangles;

		Eigen::Vector4d spectrum;


		json solver_info;

		bool use_avg_pressure;

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

		bool solve_export_to_file = true;
		std::vector<SolutionFrame> solution_frames;

		json build_json_params();

		void compute_mesh_size(const Mesh &mesh, const std::vector< ElementBases > &bases, const int n_samples);

		void load_mesh();
		void load_febio(const std::string &path);
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd&)> &boundary_marker, bool skip_boundary_sideset = false);
		void load_mesh(const std::string &path)
		{
			args["mesh"] = path;
			load_mesh();
		}
		void load_mesh(const std::string &path, const std::string &bc_tag)
		{
			args["mesh"] = path;
			args["bc_tag"] = bc_tag;
			load_mesh();
		}
		void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			if(V.cols() == 2)
				mesh = std::make_unique<polyfem::Mesh2D>();
			else
				mesh = std::make_unique<polyfem::Mesh3D>();
			mesh->build_from_matrices(V, F);

			load_mesh();
		}


		void set_boundary_side_set(const std::function<int(const polyfem::RowVectorNd&)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }
		void set_boundary_side_set(const std::function<int(const polyfem::RowVectorNd&, bool)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }
		void set_boundary_side_set(const std::function<int(const std::vector<int>&, bool)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }

		void solve()
		{
			compute_mesh_stats();

			build_basis();

			assemble_rhs();
			assemble_stiffness_mat();

			solve_export_to_file = false;
			solution_frames.clear();
			solve_problem();
			solve_export_to_file = true;
		}

		void build_basis();
		void extract_boundary_mesh();

		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		void compute_errors();
		void export_data();

		void compute_vertex_values(int actual_dim, const std::vector< ElementBases > &basis,
			const MatrixXd &fun, Eigen::MatrixXd &result);
		void compute_stress_at_quadrature_points(const MatrixXd &fun, Eigen::MatrixXd &result, Eigen::VectorXd & von_mises);

		void interpolate_function(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only = false);
		void interpolate_function(const int n_points, const int actual_dim, const std::vector< ElementBases > &basis, const MatrixXd &fun, MatrixXd &result, const bool boundary_only = false);

		void compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only = false);
		void compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only = false);
		void average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool boundary_only = false);

		void interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);
		void interpolate_boundary_function_at_vertices(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result);
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const MatrixXd &disp, const bool compute_avg, MatrixXd &result);
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);

		void get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets);


		void save_json(std::ostream &out);
		void save_json(nlohmann::json &j);
		void save_json();

		void compute_mesh_stats();

		void build_vis_mesh(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr);
		void save_vtu(const std::string &name, const double t);
		void save_wire(const std::string &name, bool isolines = false);

		const Eigen::MatrixXd &get_solution() const { return sol; }
		const Eigen::MatrixXd &get_pressure() const { return pressure; }

		void get_sampled_solution(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			Eigen::MatrixXd discr;
			Eigen::MatrixXi el_id;
			const bool tmp = args["export"]["vis_boundary_only"];
			args["export"]["vis_boundary_only"] = boundary_only;

			build_vis_mesh(points, tets, el_id, discr);
			interpolate_function(points.rows(), sol, fun, boundary_only);

			args["export"]["vis_boundary_only"] = tmp;
		}

		void get_stresses(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			Eigen::MatrixXd points;
			Eigen::MatrixXi tets;
			Eigen::MatrixXi el_id;
			Eigen::MatrixXd discr;
			const bool tmp = args["export"]["vis_boundary_only"];
			args["export"]["vis_boundary_only"] = boundary_only;

			build_vis_mesh(points, tets, el_id, discr);
			compute_tensor_value(points.rows(), sol, fun, boundary_only);

			args["export"]["vis_boundary_only"] = tmp;
		}

		void get_sampled_mises(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			Eigen::MatrixXd points;
			Eigen::MatrixXi tets;
			Eigen::MatrixXi el_id;
			Eigen::MatrixXd discr;
			const bool tmp = args["export"]["vis_boundary_only"];
			args["export"]["vis_boundary_only"] = boundary_only;

			build_vis_mesh(points, tets, el_id, discr);
			compute_scalar_value(points.rows(), sol, fun, boundary_only);

			args["export"]["vis_boundary_only"] = tmp;
		}


		void get_sampled_mises_avg(Eigen::MatrixXd &fun, Eigen::MatrixXd &tfun, bool boundary_only = false)
		{
			Eigen::MatrixXd points;
			Eigen::MatrixXi tets;
			Eigen::MatrixXi el_id;
			Eigen::MatrixXd discr;
			const bool tmp = args["export"]["vis_boundary_only"];
			args["export"]["vis_boundary_only"] = boundary_only;

			build_vis_mesh(points, tets, el_id, discr);
			average_grad_based_function(points.rows(), sol, fun, tfun, boundary_only);

			args["export"]["vis_boundary_only"] = tmp;
		}



		inline std::string mesh_path() const { return args["mesh"]; }

		inline std::string formulation() const { return problem->is_scalar() ? scalar_formulation() : tensor_formulation(); }
		inline bool iso_parametric() const {
			if(non_regular_count > 0 || non_regular_boundary_count > 0 || undefined_count > 0)
				return true;

			if(args["use_spline"])
				return true;

			if(mesh->is_rational())
				return false;

			if(args["use_p_ref"])
				return false;

			if(mesh->orders().size() <= 0){
				if(args["discr_order"] == 1)
					return true;
				else
					return args["iso_parametric"];
			}

			if(mesh->orders().minCoeff() != mesh->orders().maxCoeff())
				return false;

			if(args["discr_order"] == mesh->orders().minCoeff())
				return true;


			if( args["discr_order"] == 1 && args["force_linear_geometry"])
				return true;

			return args["iso_parametric"];
		}

		inline std::string solver_type() const { return args["solver_type"]; }
		inline std::string precond_type() const { return args["precond_type"]; }
		inline const json &solver_params() const { return args["solver_params"]; }

		inline std::string scalar_formulation() const { return args["scalar_formulation"]; }
		inline std::string tensor_formulation() const { return args["tensor_formulation"]; }
		// inline std::string mixed_formulation() const { return args["mixed_formulation"]; }

		void p_refinement(const Mesh2D &mesh2d);
		void p_refinement(const Mesh3D &mesh3d);

	private:
		void sol_to_pressure();
		void build_polygonal_basis();

	};

}
#endif //STATE_HPP

