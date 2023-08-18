#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/assembler/Problem.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <memory>

#ifdef POLYFEM_WITH_TBB
#include <tbb/tbb.h>
#endif

namespace polyfem
{
	namespace solver
	{
		class OperatorSplittingSolver
		{
		public:
			void initialize_grid(const mesh::Mesh &mesh,
								 const std::vector<basis::ElementBases> &gbases,
								 const std::vector<basis::ElementBases> &bases,
								 const double &density_dx);

			void initialize_mesh(const mesh::Mesh &mesh,
								 const int shape, const int n_el,
								 const std::vector<mesh::LocalBoundary> &local_boundary);

			void initialize_hashtable(const mesh::Mesh &mesh);

			OperatorSplittingSolver() {}

			void initialize_solver(const mesh::Mesh &mesh,
								   const int shape_, const int n_el_,
								   const std::vector<mesh::LocalBoundary> &local_boundary,
								   const std::vector<int> &bnd_nodes);

			OperatorSplittingSolver(const mesh::Mesh &mesh,
									const int shape, const int n_el,
									const std::vector<mesh::LocalBoundary> &local_boundary,
									const std::vector<int> &bnd_nodes);

			OperatorSplittingSolver(const mesh::Mesh &mesh,
									const int shape, const int n_el,
									const std::vector<mesh::LocalBoundary> &local_boundary,
									const std::vector<int> &boundary_nodes_,
									const std::vector<int> &pressure_boundary_nodes,
									const std::vector<int> &bnd_nodes,
									const StiffnessMatrix &mass,
									const StiffnessMatrix &stiffness_viscosity,
									const StiffnessMatrix &stiffness_velocity,
									const StiffnessMatrix &mass_velocity,
									const double &dt,
									const double &viscosity_,
									const std::string &solver_type,
									const std::string &precond,
									const json &params);

			int handle_boundary_advection(RowVectorNd &pos);

			int trace_back(const std::vector<basis::ElementBases> &gbases,
						   const std::vector<basis::ElementBases> &bases,
						   const RowVectorNd &pos_1,
						   const RowVectorNd &vel_1,
						   RowVectorNd &pos_2,
						   RowVectorNd &vel_2,
						   Eigen::MatrixXd &local_pos,
						   const Eigen::MatrixXd &sol,
						   const double dt);

			int interpolator(const std::vector<basis::ElementBases> &gbases,
							 const std::vector<basis::ElementBases> &bases,
							 const RowVectorNd &pos,
							 RowVectorNd &vel,
							 Eigen::MatrixXd &local_pos,
							 const Eigen::MatrixXd &sol);

			void interpolator(const RowVectorNd &pos, double &val);

		public:
			void advection(const mesh::Mesh &mesh,
						   const std::vector<basis::ElementBases> &gbases,
						   const std::vector<basis::ElementBases> &bases,
						   Eigen::MatrixXd &sol,
						   const double dt,
						   const Eigen::MatrixXd &local_pts,
						   const int order = 1,
						   const int RK = 1);

			void advect_density_exact(const std::vector<basis::ElementBases> &gbases,
									  const std::vector<basis::ElementBases> &bases,
									  const std::shared_ptr<assembler::Problem> problem,
									  const double t,
									  const double dt,
									  const int RK = 3);

			void advect_density(const std::vector<basis::ElementBases> &gbases,
								const std::vector<basis::ElementBases> &bases,
								const Eigen::MatrixXd &sol,
								const double dt,
								const int RK = 3);

			void advection_FLIP(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &gbases, const std::vector<basis::ElementBases> &bases, Eigen::MatrixXd &sol, const double dt, const Eigen::MatrixXd &local_pts, const int order = 1);

			void advection_PIC(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &gbases, const std::vector<basis::ElementBases> &bases, Eigen::MatrixXd &sol, const double dt, const Eigen::MatrixXd &local_pts, const int order = 1);

			void solve_diffusion_1st(const StiffnessMatrix &mass, const std::vector<int> &bnd_nodes, Eigen::MatrixXd &sol);

			void external_force(const mesh::Mesh &mesh,
								const assembler::Assembler &assembler,
								const std::vector<basis::ElementBases> &gbases,
								const std::vector<basis::ElementBases> &bases,
								const double dt,
								Eigen::MatrixXd &sol,
								const Eigen::MatrixXd &local_pts,
								const std::shared_ptr<assembler::Problem> problem,
								const double time);

			void solve_pressure(const StiffnessMatrix &mixed_stiffness, const std::vector<int> &pressure_boundary_nodes, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);

			void projection(const StiffnessMatrix &velocity_mass, const StiffnessMatrix &mixed_stiffness, const std::vector<int> &boundary_nodes_, Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure);

			void projection(int n_bases,
							const std::vector<basis::ElementBases> &gbases,
							const std::vector<basis::ElementBases> &bases,
							const std::vector<basis::ElementBases> &pressure_bases,
							const Eigen::MatrixXd &local_pts,
							Eigen::MatrixXd &pressure,
							Eigen::MatrixXd &sol);

			void initialize_density(const std::shared_ptr<assembler::Problem> &problem);

			long search_cell(const std::vector<basis::ElementBases> &gbases, const RowVectorNd &pos, Eigen::MatrixXd &local_pts);

			bool outside_quad(const std::vector<RowVectorNd> &vert, const RowVectorNd &pos);

			void compute_gbase_val(const int elem_idx, const Eigen::MatrixXd &local_pos, Eigen::MatrixXd &pos);

			void compute_gbase_jacobi(const int elem_idx, const Eigen::MatrixXd &local_pos, Eigen::MatrixXd &jacobi);

			void calculate_local_pts(const basis::ElementBases &gbase,
									 const int elem_idx,
									 const RowVectorNd &pos,
									 Eigen::MatrixXd &local_pos);

			void save_density();

			int dim;
			int n_el;
			int shape;

			RowVectorNd min_domain;
			RowVectorNd max_domain;

			Eigen::MatrixXd V;
			Eigen::MatrixXi T;

			std::vector<std::vector<long>> hash_table;
			Eigen::Matrix<long, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> hash_table_cell_num;

			std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3>> position_particle;
			std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3>> velocity_particle;
			std::vector<int> cellI_particle;
			Eigen::MatrixXd new_sol;
			Eigen::MatrixXd new_sol_w;

			std::vector<int> boundary_elem_id;
			std::vector<int> boundary_nodes;

			std::unique_ptr<polysolve::LinearSolver> solver_diffusion;
			std::unique_ptr<polysolve::LinearSolver> solver_projection;
			std::unique_ptr<polysolve::LinearSolver> solver_mass;

			StiffnessMatrix mat_diffusion;
			StiffnessMatrix mat_projection;

			std::string solver_type;

			Eigen::VectorXd density;
			// Eigen::VectorXi density_cell_no;
			// std::vector<ElementAssemblyValues> density_local_weights;
			Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3> grid_cell_num;
			double resolution;
		};
	} // namespace solver
} // namespace polyfem
